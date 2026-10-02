/**
 * The call's link to the server. The UI and media code only talk to this
 * interface, so the WebSocket gateway can be swapped for a WebRTC transport
 * (e.g. LiveKit) without touching anything else.
 */

export type Facing = "user" | "environment";

export type DetectionHint = { label: string; score: number };

export type Utterance = {
  turn: number;
  wav: ArrayBuffer;
  image: ArrayBuffer | null;
  camera: boolean;
  facing: Facing;
  detections: DetectionHint[];
};

export type SpeechChunk = { turn: number; seq: number; wav: ArrayBuffer };

export type EndReason = "hangup" | "time_limit" | "replaced";

export type ServerEvent =
  | { type: "ready"; max_seconds: number; voice: boolean }
  | { type: "transcript"; turn: number; text: string }
  | { type: "say"; turn: number; seq: number; text: string; audio: boolean }
  | {
      type: "turn_done";
      turn: number;
      skipped?: "no_speech" | "error";
      timings?: Record<string, number>;
    }
  | { type: "notice"; code: string; message: string }
  | { type: "error"; code: string; message: string }
  | { type: "ended"; reason: EndReason };

export type LinkStatus =
  | { state: "open" }
  | { state: "reconnecting"; attempt: number }
  | { state: "closed"; code: number; reason: string };

export type TransportHandlers = {
  onEvent: (event: ServerEvent) => void;
  onSpeech: (chunk: SpeechChunk) => void;
  onStatus: (status: LinkStatus) => void;
};

export interface CallTransport {
  /** Resolves with the server's "ready" message, or rejects with a CallFailure. */
  connect(handlers: TransportHandlers): Promise<Extract<ServerEvent, { type: "ready" }>>;
  sendUtterance(utterance: Utterance): void;
  interrupt(turn: number, played: number): void;
  hangUp(): void;
  close(): void;
}

/** Close codes the server uses (see models/vision/routes.py). */
export const CloseCode = {
  unauthorized: 4401,
  forbidden: 4403,
  rateLimited: 4429,
} as const;
