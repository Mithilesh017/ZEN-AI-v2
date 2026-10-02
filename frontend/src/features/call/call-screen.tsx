/**
 * Full-screen call surface. Loaded lazily from the chat header, so none of
 * the media, WebGL or detector code is in the chat bundle.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { CallController } from "./call-controller";
import { initialCallState, setCall, useCallStore } from "./call-store";
import { CallError, CallRecap } from "./components/call-outcomes";
import { LiveCall } from "./components/live-call";
import { Preflight } from "./components/preflight";

export default function CallScreen({ onClose }: { onClose: () => void }) {
  const phase = useCallStore((s) => s.phase);
  const [controller, setController] = useState<CallController | null>(null);
  const controllerRef = useRef<CallController | null>(null);

  useEffect(() => {
    setCall({ ...initialCallState, phase: "preflight" });
    const { overflow } = document.body.style;
    document.body.style.overflow = "hidden";
    return () => {
      controllerRef.current?.dispose();
      setCall(initialCallState);
      document.body.style.overflow = overflow;
    };
  }, []);

  const start = useCallback(() => {
    controllerRef.current?.dispose();
    const next = new CallController();
    controllerRef.current = next;
    setController(next);
    void next.start();
  }, []);

  const close = useCallback(() => {
    controllerRef.current?.dispose();
    onClose();
  }, [onClose]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && phase !== "live" && phase !== "connecting") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, close]);

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="call-title"
      className="fixed inset-0 z-50 h-dvh overscroll-none bg-black text-white antialiased select-none"
    >
      {phase === "preflight" && <Preflight onStart={start} onClose={close} />}
      {(phase === "connecting" || phase === "live") && controller && <LiveCall controller={controller} />}
      {phase === "ended" && <CallRecap onAgain={start} onClose={close} />}
      {phase === "error" && <CallError onRetry={start} onClose={close} />}
      <h1 id="call-title" className="sr-only">
        Video call with ZEN
      </h1>
    </div>,
    document.body,
  );
}
