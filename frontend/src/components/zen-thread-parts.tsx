import type { FC, PropsWithChildren } from "react";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { CheckIcon, ClockIcon, GlobeIcon, WrenchIcon } from "lucide-react";

import type { ThreadComponents } from "@/components/assistant-ui/elements/thread.aui";
import { ZenLogo } from "@/components/zen-logo";

/** One-line activity chip for server-side tools ("Searching the web…"). */
const ToolActivity: ToolCallMessagePartComponent = ({ toolName, args, result }) => {
  const done = result !== undefined;
  const query = typeof args.query === "string" ? args.query : "";

  let Icon = WrenchIcon;
  let label = done ? `Used ${toolName}` : `Running ${toolName}`;
  if (toolName === "search_web") {
    Icon = GlobeIcon;
    label = done ? "Searched the web" : "Searching the web";
  } else if (toolName === "get_current_datetime") {
    Icon = ClockIcon;
    label = done ? "Checked the time" : "Checking the time";
  }

  return (
    <div className="text-muted-foreground flex items-center gap-2 py-0.5 text-sm">
      <Icon className="size-3.5 shrink-0" />
      <span className={done ? undefined : "shimmer"}>{label}</span>
      {query && <span className="text-foreground/70 truncate">· {query}</span>}
      {done && <CheckIcon className="size-3.5 shrink-0 text-emerald-500" />}
    </div>
  );
};

const ToolActivityGroup: FC<PropsWithChildren> = ({ children }) => (
  <div className="mb-3 flex flex-col">{children}</div>
);

const greeting = () => {
  const h = new Date().getHours();
  if (h < 5) return "Up late";
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
};

export const makeWelcome = (name: string): FC =>
  function Welcome() {
    const first = name.split(/\s+/)[0];
    return (
      <div className="mb-8 flex flex-col items-center gap-4 px-2 text-center">
        <ZenLogo className="fade-in zoom-in-95 animate-in size-12 rounded-2xl duration-300" />
        <h1 className="fade-in slide-in-from-bottom-1 animate-in fill-mode-both text-3xl font-semibold tracking-tight duration-300">
          {greeting()}
          {first ? `, ${first}` : ""}
        </h1>
        <p className="text-muted-foreground fade-in animate-in fill-mode-both delay-100 duration-300">
          How can I help you today?
        </p>
      </div>
    );
  };

export const threadComponents = (name: string): ThreadComponents => ({
  Welcome: makeWelcome(name),
  ToolFallback: ToolActivity,
  ToolGroup: ToolActivityGroup,
});
