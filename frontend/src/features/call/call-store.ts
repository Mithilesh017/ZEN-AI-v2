import { create } from "zustand";

import type { EndReason, Facing } from "./transport/types";

export type Phase = "closed" | "preflight" | "connecting" | "live" | "ended" | "error";

/** What ZEN is doing, as the user should perceive it. */
export type ZenState = "listening" | "hearing" | "thinking" | "speaking";

export type CallErrorKind =
  | "denied"
  | "missing"
  | "insecure"
  | "unsupported"
  | "unauthorized"
  | "rate_limited"
  | "unreachable"
  | "rejected";

export type Line = { turn: number; role: "user" | "zen"; text: string };

export type CallState = {
  phase: Phase;
  zen: ZenState;
  link: "open" | "reconnecting";
  muted: boolean;
  cameraOn: boolean;
  facing: Facing;
  canFlip: boolean;
  detectionsOn: boolean;
  detectorReady: boolean;
  voice: boolean;
  userCaption: string;
  zenCaption: string;
  notice: string | null;
  startedAt: number | null;
  endedAt: number | null;
  maxSeconds: number;
  error: { kind: CallErrorKind; detail: string } | null;
  endReason: EndReason | "connection_lost" | null;
  /** label -> times it appeared, for the recap */
  seen: Record<string, number>;
  lines: Line[];
  timings: Record<string, number> | null;
};

export const initialCallState: CallState = {
  phase: "closed",
  zen: "listening",
  link: "open",
  muted: false,
  cameraOn: true,
  facing: "user",
  canFlip: false,
  detectionsOn: true,
  detectorReady: false,
  voice: true,
  userCaption: "",
  zenCaption: "",
  notice: null,
  startedAt: null,
  endedAt: null,
  maxSeconds: 0,
  error: null,
  endReason: null,
  seen: {},
  lines: [],
  timings: null,
};

export const useCallStore = create<CallState>(() => initialCallState);

export const setCall = (patch: Partial<CallState>) => useCallStore.setState(patch);

/** Append to the transcript, merging consecutive sentences of the same turn and speaker. */
export function appendLine(line: Line): void {
  useCallStore.setState((s) => {
    const last = s.lines.at(-1);
    if (last && last.turn === line.turn && last.role === line.role) {
      return { lines: [...s.lines.slice(0, -1), { ...last, text: `${last.text} ${line.text}` }] };
    }
    return { lines: [...s.lines, line] };
  });
}

export function noteSeen(label: string): void {
  useCallStore.setState((s) => ({ seen: { ...s.seen, [label]: (s.seen[label] ?? 0) + 1 } }));
}
