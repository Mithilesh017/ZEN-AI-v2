import {
  MicIcon,
  MicOffIcon,
  PhoneOffIcon,
  ScanSearchIcon,
  SwitchCameraIcon,
  VideoIcon,
  VideoOffIcon,
} from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

import type { CallController } from "../call-controller";
import { useCallStore } from "../call-store";

function Control({
  label,
  pressed,
  disabled,
  onClick,
  className,
  children,
}: {
  label: string;
  pressed?: boolean;
  disabled?: boolean;
  onClick: () => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={pressed}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex size-14 items-center justify-center rounded-full backdrop-blur-xl transition-[background-color,color,transform,opacity] duration-200 active:scale-95 disabled:pointer-events-none disabled:opacity-35 [&_svg]:size-[22px]",
        pressed ? "bg-white text-black" : "bg-white/[0.14] text-white hover:bg-white/[0.22]",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function CallControls({ controller }: { controller: CallController }) {
  const muted = useCallStore((s) => s.muted);
  const cameraOn = useCallStore((s) => s.cameraOn);
  const canFlip = useCallStore((s) => s.canFlip);
  const detectionsOn = useCallStore((s) => s.detectionsOn);
  const live = useCallStore((s) => s.phase === "live");

  return (
    <div className="flex items-center justify-center gap-3.5 sm:gap-4">
      <Control label={muted ? "Unmute" : "Mute"} pressed={muted} disabled={!live} onClick={() => void controller.toggleMute()}>
        {muted ? <MicOffIcon /> : <MicIcon />}
      </Control>
      <Control
        label={cameraOn ? "Turn camera off" : "Turn camera on"}
        pressed={!cameraOn}
        disabled={!live}
        onClick={() => void controller.toggleCamera()}
      >
        {cameraOn ? <VideoIcon /> : <VideoOffIcon />}
      </Control>
      <Control label="Switch camera" disabled={!live || !cameraOn || !canFlip} onClick={() => void controller.flip()}>
        <SwitchCameraIcon />
      </Control>
      <Control
        label={detectionsOn ? "Hide object labels" : "Show object labels"}
        pressed={cameraOn && detectionsOn}
        disabled={!live || !cameraOn}
        onClick={() => controller.toggleDetections()}
      >
        <ScanSearchIcon />
      </Control>
      <Control
        label="End call"
        onClick={() => controller.end()}
        className="bg-[#e5484d] text-white hover:bg-[#d93d42]"
      >
        <PhoneOffIcon />
      </Control>
    </div>
  );
}
