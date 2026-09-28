import type {
  ChatModelAdapter,
  ThreadAssistantMessagePart,
  ThreadMessage,
} from "@assistant-ui/react";

const TIMEZONE = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";

/** Events streamed by POST /api/chat as newline-delimited JSON. */
type StreamEvent =
  | { type: "text"; delta: string }
  | { type: "tool-call"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool-result"; id: string }
  | { type: "error"; message: string };

type Part =
  | { type: "text"; text: string }
  | {
      type: "tool-call";
      toolCallId: string;
      toolName: string;
      args: Record<string, unknown>;
      argsText: string;
      result?: string;
    };

/** The backend only needs the plain text of each turn. */
const toWire = (messages: readonly ThreadMessage[]) =>
  messages.flatMap((m) => {
    if (m.role !== "user" && m.role !== "assistant") return [];
    const text = m.content
      .map((p) => (p.type === "text" ? p.text : ""))
      .join("")
      .trim();
    return text ? [{ role: m.role, content: text }] : [];
  });

const readError = async (res: Response) => {
  try {
    const data: { error?: string } = await res.json();
    if (data.error) return data.error;
  } catch {
    // not JSON
  }
  return "ZEN is unavailable right now. Please try again.";
};

export const zenChatAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: toWire(messages), timezone: TIMEZONE }),
      signal: abortSignal,
    });

    if (res.status === 401) {
      window.location.href = "/login";
      throw new Error("Your session expired. Please sign in again.");
    }
    if (!res.ok || !res.body) throw new Error(await readError(res));

    const parts: Part[] = [];
    const apply = (event: StreamEvent) => {
      switch (event.type) {
        case "text": {
          const last = parts.at(-1);
          if (last?.type === "text") {
            parts[parts.length - 1] = { ...last, text: last.text + event.delta };
          } else {
            parts.push({ type: "text", text: event.delta });
          }
          break;
        }
        case "tool-call":
          parts.push({
            type: "tool-call",
            toolCallId: event.id,
            toolName: event.name,
            args: event.args,
            argsText: JSON.stringify(event.args),
          });
          break;
        case "tool-result": {
          const i = parts.findIndex(
            (p) => p.type === "tool-call" && p.toolCallId === event.id,
          );
          if (i !== -1) parts[i] = { ...parts[i], result: "done" } as Part;
          break;
        }
        case "error":
          throw new Error(event.message);
      }
    };

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      const events = lines.filter((l) => l.trim()).map((l) => JSON.parse(l) as StreamEvent);
      if (events.length === 0) continue;
      events.forEach(apply);
      yield { content: [...parts] as ThreadAssistantMessagePart[] };
    }
  },
};
