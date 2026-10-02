import { EyeIcon, MicIcon, ShieldCheckIcon, VideoIcon, VideoOffIcon, XIcon } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

import { setCall, useCallStore } from "../call-store";
import { Orb } from "./orb";

const Point = ({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) => (
  <li className="flex gap-3.5">
    <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-full bg-white/[0.07] text-white/80 [&_svg]:size-4">
      {icon}
    </span>
    <div className="min-w-0">
      <p className="text-[15px] font-medium text-white">{title}</p>
      <p className="text-[13px] leading-relaxed text-white/55">{children}</p>
    </div>
  </li>
);

export function Preflight({ onStart, onClose }: { onStart: () => void; onClose: () => void }) {
  const cameraOn = useCallStore((s) => s.cameraOn);

  return (
    <div className="flex h-full flex-col px-6 pt-[max(env(safe-area-inset-top),1rem)] pb-[max(env(safe-area-inset-bottom),1.5rem)]">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={onClose}
          className="flex size-10 items-center justify-center rounded-full text-white/70 transition-colors hover:bg-white/10 hover:text-white"
          aria-label="Close"
        >
          <XIcon className="size-5" />
        </button>
      </div>

      <div className="mx-auto flex w-full max-w-sm flex-1 flex-col justify-center gap-9">
        <div className="flex flex-col items-center gap-5 text-center">
          <Orb state="idle" className="fade-in zoom-in-95 animate-in size-36 duration-500" />
          <div className="space-y-2">
            <h2 className="text-[26px] font-semibold tracking-tight">
              Video call with ZEN
            </h2>
            <p className="text-[15px] leading-relaxed text-white/60">
              Show ZEN what's in front of you and talk it through, like a call with a friend who knows a lot.
            </p>
          </div>
        </div>

        <ul className="space-y-4">
          <Point icon={<EyeIcon />} title="Sees when you speak">
            One camera frame goes with each thing you say. Nothing streams in between.
          </Point>
          <Point icon={<MicIcon />} title="Talk over it any time">
            Start speaking and ZEN stops to listen, just like on a phone call.
          </Point>
          <Point icon={<ShieldCheckIcon />} title="No recordings">
            Audio and video aren't stored. As in chat, ZEN may remember what you tell it.
          </Point>
        </ul>
      </div>

      <div className="mx-auto w-full max-w-sm space-y-3 pt-6">
        <div role="radiogroup" aria-label="Camera" className="grid grid-cols-2 gap-1 rounded-full bg-white/[0.07] p-1">
          {([
            [true, "Camera on", <VideoIcon key="v" />],
            [false, "Voice only", <VideoOffIcon key="o" />],
          ] as const).map(([value, label, icon]) => (
            <button
              key={label}
              type="button"
              role="radio"
              aria-checked={cameraOn === value}
              onClick={() => setCall({ cameraOn: value })}
              className={cn(
                "flex h-9 items-center justify-center gap-2 rounded-full text-sm font-medium transition-colors [&_svg]:size-4",
                cameraOn === value ? "bg-white text-black" : "text-white/65 hover:text-white",
              )}
            >
              {icon}
              {label}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={onStart}
          autoFocus
          className="bg-brand text-brand-foreground h-12 w-full rounded-full text-[15px] font-semibold transition-[filter,transform] hover:brightness-110 active:scale-[0.98]"
        >
          Start call
        </button>
      </div>
    </div>
  );
}
