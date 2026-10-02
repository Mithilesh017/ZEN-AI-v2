import {
  CloudOffIcon,
  LockIcon,
  MicOffIcon,
  ShieldAlertIcon,
  TimerIcon,
  UserRoundXIcon,
} from "lucide-react";
import type { ReactNode } from "react";

import { useCallStore, type CallErrorKind } from "../call-store";
import { Orb } from "./orb";

const Shell = ({ children }: { children: ReactNode }) => (
  <div className="flex h-full flex-col overflow-y-auto overscroll-contain px-6 pt-[max(env(safe-area-inset-top),1rem)] pb-[max(env(safe-area-inset-bottom),1.5rem)]">
    <div className="fade-in zoom-in-[0.98] animate-in m-auto flex w-full max-w-sm flex-col items-center gap-6 text-center duration-300">
      {children}
    </div>
  </div>
);

const PrimaryButton = ({ onClick, children }: { onClick: () => void; children: ReactNode }) => (
  <button
    type="button"
    onClick={onClick}
    autoFocus
    className="bg-brand text-brand-foreground h-12 w-full rounded-full text-[15px] font-semibold transition-[filter,transform] hover:brightness-110 active:scale-[0.98]"
  >
    {children}
  </button>
);

const SecondaryButton = ({ onClick, children }: { onClick: () => void; children: ReactNode }) => (
  <button
    type="button"
    onClick={onClick}
    className="h-12 w-full rounded-full text-[15px] font-medium text-white/75 transition-colors hover:bg-white/[0.07] hover:text-white"
  >
    {children}
  </button>
);

const duration = (ms: number) => {
  const s = Math.max(0, Math.round(ms / 1000));
  return s < 60 ? `${s} sec` : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} sec`;
};

const END_NOTE = {
  hangup: null,
  time_limit: "Calls are limited in length for now. You can start another one.",
  replaced: "This call moved to another tab or device.",
  connection_lost: "The connection dropped and couldn't be restored.",
} as const;

export function CallRecap({ onAgain, onClose }: { onAgain: () => void; onClose: () => void }) {
  const startedAt = useCallStore((s) => s.startedAt);
  const endedAt = useCallStore((s) => s.endedAt);
  const reason = useCallStore((s) => s.endReason);
  const seen = useCallStore((s) => s.seen);
  const lines = useCallStore((s) => s.lines);

  const objects = Object.entries(seen)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([label]) => label);
  const note = reason ? END_NOTE[reason] : null;

  return (
    <Shell>
      <Orb state="idle" className="size-20" />
      <div className="space-y-1.5">
        <h2 className="text-2xl font-semibold tracking-tight">Call ended</h2>
        <p className="text-sm text-white/55 tabular-nums">
          {startedAt && endedAt ? duration(endedAt - startedAt) : null}
          {note ? ` · ${note}` : null}
        </p>
      </div>

      {objects.length > 0 && (
        <div className="w-full space-y-2.5">
          <p className="text-xs font-medium tracking-wide text-white/45 uppercase">Things you showed ZEN</p>
          <ul className="flex flex-wrap justify-center gap-1.5">
            {objects.map((label) => (
              <li key={label} className="rounded-full bg-white/[0.08] px-3 py-1 text-[13px] text-white/85">
                {label}
              </li>
            ))}
          </ul>
        </div>
      )}

      {lines.length > 0 && (
        <ol className="max-h-[32vh] w-full space-y-3 overflow-y-auto overscroll-contain rounded-2xl bg-white/[0.05] p-4 text-left text-[13px] leading-relaxed [mask-image:linear-gradient(to_bottom,black_85%,transparent)] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          {lines.map((line, i) => (
            <li key={i} className={line.role === "user" ? "text-white/55" : "text-white/90"}>
              <span className="mr-2 font-mono text-[11px] text-white/35 uppercase">
                {line.role === "user" ? "You" : "Zen"}
              </span>
              {line.text}
            </li>
          ))}
        </ol>
      )}

      <div className="w-full space-y-1.5">
        <PrimaryButton onClick={onAgain}>Call again</PrimaryButton>
        <SecondaryButton onClick={onClose}>Back to chat</SecondaryButton>
      </div>
    </Shell>
  );
}

const ERRORS: Record<CallErrorKind, { icon: ReactNode; title: string; hint?: string; retry: boolean }> = {
  denied: {
    icon: <MicOffIcon />,
    title: "Allow microphone access",
    hint: "Use the lock icon in the address bar to allow the microphone (and camera, if you'd like ZEN to see), then try again.",
    retry: true,
  },
  missing: { icon: <MicOffIcon />, title: "No microphone found", retry: true },
  insecure: { icon: <LockIcon />, title: "Secure connection needed", retry: false },
  unsupported: { icon: <ShieldAlertIcon />, title: "Calls aren't supported here", retry: false },
  unauthorized: { icon: <UserRoundXIcon />, title: "You've been signed out", retry: false },
  rate_limited: { icon: <TimerIcon />, title: "Too many calls", retry: false },
  unreachable: { icon: <CloudOffIcon />, title: "Can't reach ZEN", retry: true },
  rejected: { icon: <ShieldAlertIcon />, title: "The call couldn't start", retry: true },
};

export function CallError({ onRetry, onClose }: { onRetry: () => void; onClose: () => void }) {
  const error = useCallStore((s) => s.error);
  if (!error) return null;
  const info = ERRORS[error.kind];

  return (
    <Shell>
      <span className="flex size-14 items-center justify-center rounded-full bg-white/[0.08] text-white/85 [&_svg]:size-6">
        {info.icon}
      </span>
      <div className="space-y-2">
        <h2 className="text-xl font-semibold tracking-tight">{info.title}</h2>
        <p className="text-sm leading-relaxed text-white/60">{info.hint ?? error.detail}</p>
      </div>
      <div className="w-full space-y-1.5">
        {error.kind === "unauthorized" ? (
          <PrimaryButton onClick={() => (window.location.href = "/login")}>Sign in again</PrimaryButton>
        ) : info.retry ? (
          <PrimaryButton onClick={onRetry}>Try again</PrimaryButton>
        ) : null}
        <SecondaryButton onClick={onClose}>Back to chat</SecondaryButton>
      </div>
    </Shell>
  );
}
