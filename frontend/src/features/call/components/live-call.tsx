import { useCallback, useEffect, useState } from "react";

import { cn } from "@/lib/utils";

import type { CallController } from "../call-controller";
import { useCallStore } from "../call-store";
import { CallControls } from "./call-controls";
import { DetectionOverlay } from "./detection-overlay";
import { Orb } from "./orb";

const STATE_LABEL = {
  listening: "Listening",
  hearing: "Hearing you",
  thinking: "Thinking",
  speaking: "Speaking",
} as const;

const clock = (seconds: number) => {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

const debugEnabled = () => {
  try {
    return new URLSearchParams(location.search).has("debug") || localStorage.getItem("zen:debug") === "1";
  } catch {
    return false;
  }
};

function StatusPill() {
  const phase = useCallStore((s) => s.phase);
  const zen = useCallStore((s) => s.zen);
  const link = useCallStore((s) => s.link);
  const muted = useCallStore((s) => s.muted);
  const voice = useCallStore((s) => s.voice);
  const deviceVoice = useCallStore((s) => s.deviceVoice);
  const startedAt = useCallStore((s) => s.startedAt);
  const maxSeconds = useCallStore((s) => s.maxSeconds);
  const now = useNow(phase === "live");

  const elapsed = startedAt ? (now - startedAt) / 1000 : 0;
  const left = maxSeconds - elapsed;
  const reconnecting = link === "reconnecting";
  const label =
    phase === "connecting" ? "Connecting" : reconnecting ? "Reconnecting" : muted ? "Muted" : STATE_LABEL[zen];

  return (
    <div className="flex items-center gap-2">
      <div className="flex h-8 items-center gap-2 rounded-full bg-black/35 pr-3 pl-2.5 text-[13px] font-medium backdrop-blur-xl">
        <span
          className={cn(
            "size-2 rounded-full transition-colors duration-300",
            reconnecting ? "animate-pulse bg-amber-400" : zen === "speaking" ? "bg-brand" : "bg-white/80",
            phase === "connecting" && "animate-pulse",
          )}
        />
        <span>{label}</span>
        {phase === "live" && (
          <span className="text-white/55 tabular-nums">
            · {left < 60 ? `${clock(left)} left` : clock(elapsed)}
          </span>
        )}
      </div>
      {!voice && (
        <span className="h-8 rounded-full bg-black/35 px-3 text-[13px] leading-8 text-white/70 backdrop-blur-xl">
          {deviceVoice ? "Phone voice" : "Captions only"}
        </span>
      )}
    </div>
  );
}

function Captions() {
  const userCaption = useCallStore((s) => s.userCaption);
  const zenCaption = useCallStore((s) => s.zenCaption);
  const notice = useCallStore((s) => s.notice);

  return (
    <div className="mx-auto flex min-h-[7.5rem] w-full max-w-md flex-col items-center justify-end gap-2 px-6 text-center [text-shadow:0_1px_12px_rgb(0_0_0/0.6)]">
      {notice && (
        <p
          key={notice}
          role="status"
          className="fade-in animate-in mb-1 rounded-full bg-white/[0.14] px-3.5 py-1.5 text-[13px] text-white/90 backdrop-blur-xl duration-200 [text-shadow:none]"
        >
          {notice}
        </p>
      )}
      {userCaption && (
        <p key={`u:${userCaption}`} className="fade-in animate-in line-clamp-2 text-sm text-white/60 duration-200">
          {userCaption}
        </p>
      )}
      <p
        key={`z:${zenCaption}`}
        aria-live="polite"
        className="fade-in slide-in-from-bottom-1 animate-in line-clamp-3 text-lg leading-snug font-medium text-balance text-white duration-200"
      >
        {zenCaption}
      </p>
    </div>
  );
}

function DebugTimings() {
  const timings = useCallStore((s) => s.timings);
  if (!timings) return null;
  const parts = [
    ["stt", timings.stt_ms],
    ["first token", timings.ttft_ms],
    ["first audio", timings.first_audio_ms],
    ["total", timings.total_ms],
  ].filter(([, v]) => v !== undefined);
  return (
    <p className="mt-2 font-mono text-[11px] text-white/50">
      {parts.map(([k, v]) => `${k} ${v}ms`).join(" · ")}
    </p>
  );
}

export function LiveCall({ controller }: { controller: CallController }) {
  const phase = useCallStore((s) => s.phase);
  const zen = useCallStore((s) => s.zen);
  const cameraOn = useCallStore((s) => s.cameraOn);
  const facing = useCallStore((s) => s.facing);
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [debug] = useState(debugEnabled);

  const videoRef = useCallback(
    (el: HTMLVideoElement | null) => {
      setVideo(el);
      controller.attachVideo(el);
    },
    [controller],
  );
  const tracks = useCallback(() => controller.tracks(), [controller]);
  const level = useCallback(() => controller.orbLevel(), [controller]);

  const mirrored = facing === "user";
  const orbState = phase === "connecting" ? "thinking" : zen;

  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* camera */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={cn(
          "absolute inset-0 size-full object-cover transition-opacity duration-500",
          mirrored && "-scale-x-100",
          cameraOn ? "opacity-100" : "opacity-0",
        )}
      />
      {cameraOn && <DetectionOverlay video={video} tracks={tracks} mirrored={mirrored} />}

      {/* legibility scrims */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-36 bg-gradient-to-b from-black/55 to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-80 bg-gradient-to-t from-black/75 via-black/30 to-transparent" />

      {/* ZEN: a picture-in-picture presence with the camera on, centre stage without it */}
      <div
        className={cn(
          "absolute transition-all duration-500 ease-out",
          cameraOn
            ? "top-[max(env(safe-area-inset-top),0.75rem)] right-3 size-[5.5rem]"
            : "top-1/2 left-1/2 size-56 -translate-x-1/2 -translate-y-[60%]",
        )}
      >
        <Orb state={orbState} getLevel={level} className="size-full" />
      </div>

      <div className="absolute top-[max(env(safe-area-inset-top),0.75rem)] left-3">
        <StatusPill />
        {debug && <DebugTimings />}
      </div>

      <div className="absolute inset-x-0 bottom-0 flex flex-col gap-6 pb-[max(env(safe-area-inset-bottom),1.25rem)]">
        <Captions />
        <CallControls controller={controller} />
      </div>
    </div>
  );
}
