/**
 * CallTransport over ZEN's own WebSocket gateway (/api/vision/call).
 *
 * The session cookie authenticates the upgrade, so no token handling is
 * needed. Dropped connections are retried with backoff; the server starts a
 * fresh call session, and turn numbers keep increasing, so nothing collides.
 */
import { decodeSpeech, encodeUtterance } from "./codec";
import {
  CloseCode,
  type CallTransport,
  type ServerEvent,
  type TransportHandlers,
  type Utterance,
} from "./types";

const READY_TIMEOUT_MS = 8000;
const RETRY_DELAYS_MS = [600, 1500, 3500];

export type FailureKind = "unauthorized" | "rate_limited" | "unreachable" | "rejected";

export class CallFailure extends Error {
  readonly kind: FailureKind;

  constructor(kind: FailureKind, message: string) {
    super(message);
    this.kind = kind;
  }
}

type Ready = Extract<ServerEvent, { type: "ready" }>;

const failureFor = (code: number, reason: string): CallFailure => {
  if (code === CloseCode.unauthorized)
    return new CallFailure("unauthorized", reason || "Your session expired.");
  if (code === CloseCode.rateLimited)
    return new CallFailure("rate_limited", reason || "Too many calls right now.");
  if (code >= 4000) return new CallFailure("rejected", reason || "The call was refused.");
  return new CallFailure("unreachable", "Couldn't reach ZEN. Check your connection.");
};

export class GatewayTransport implements CallTransport {
  private ws: WebSocket | null = null;
  private handlers: TransportHandlers | null = null;
  private finished = false;

  private readonly url: string;

  constructor(path = "/api/vision/call") {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    this.url = `${scheme}//${location.host}${path}`;
  }

  connect(handlers: TransportHandlers): Promise<Ready> {
    this.handlers = handlers;
    return this.open();
  }

  sendUtterance(utterance: Utterance): void {
    this.send(encodeUtterance(utterance));
  }

  interrupt(turn: number, played: number): void {
    this.send(JSON.stringify({ type: "interrupt", turn, played }));
  }

  hangUp(): void {
    this.send(JSON.stringify({ type: "bye" }));
    this.close();
  }

  close(): void {
    this.finished = true;
    this.ws?.close(1000);
    this.ws = null;
  }

  private send(data: string | ArrayBuffer): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(data);
  }

  private open(): Promise<Ready> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let ready = false;

      const timer = setTimeout(() => {
        if (!ready) {
          ws.close();
          reject(new CallFailure("unreachable", "ZEN took too long to answer."));
        }
      }, READY_TIMEOUT_MS);

      ws.onopen = () => ws.send(JSON.stringify({ type: "hello", v: 1 }));

      ws.onmessage = (msg) => {
        if (typeof msg.data !== "string") {
          const chunk = decodeSpeech(msg.data as ArrayBuffer);
          if (chunk) this.handlers?.onSpeech(chunk);
          return;
        }
        let event: ServerEvent;
        try {
          event = JSON.parse(msg.data) as ServerEvent;
        } catch {
          return;
        }
        if (event.type === "ready" && !ready) {
          ready = true;
          clearTimeout(timer);
          resolve(event);
          return;
        }
        if (event.type === "ended") this.finished = true;
        this.handlers?.onEvent(event);
      };

      ws.onclose = (e) => {
        clearTimeout(timer);
        if (this.ws !== ws) return; // superseded by a retry or closed by us
        this.ws = null;
        if (!ready) {
          reject(failureFor(e.code, e.reason));
          return;
        }
        if (this.finished) return;
        if (e.code >= 4000) {
          this.handlers?.onStatus({ state: "closed", code: e.code, reason: e.reason });
          return;
        }
        void this.reconnect();
      };
    });
  }

  private async reconnect(): Promise<void> {
    for (const [i, delay] of RETRY_DELAYS_MS.entries()) {
      this.handlers?.onStatus({ state: "reconnecting", attempt: i + 1 });
      await new Promise((r) => setTimeout(r, delay));
      if (this.finished) return;
      try {
        await this.open();
        this.handlers?.onStatus({ state: "open" });
        return;
      } catch (err) {
        if (err instanceof CallFailure && err.kind !== "unreachable") {
          this.handlers?.onStatus({ state: "closed", code: 4000, reason: err.message });
          return;
        }
      }
    }
    this.handlers?.onStatus({ state: "closed", code: 1006, reason: "Connection lost." });
  }
}
