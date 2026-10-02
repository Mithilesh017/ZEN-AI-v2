import { Suspense, lazy, useEffect, useMemo, useState, type FC } from "react";
import {
  AssistantRuntimeProvider,
  AuiConfig,
  Suggestions,
  createSimpleTitleAdapter,
  useAui,
  useAuiState,
  useLocalRuntime,
  useRemoteThreadListRuntime,
} from "@assistant-ui/react";
import { createLocalStorageAdapter, type AsyncStorageLike } from "@assistant-ui/core/react";
import { SquarePenIcon, VideoIcon } from "lucide-react";

import { AppSidebar } from "@/components/app-sidebar";
import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { SettingsDialog } from "@/components/settings-dialog";
import { TooltipIconButton } from "@/components/tooltip-icon-button";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ZenLogo } from "@/components/zen-logo";
import { threadComponents } from "@/components/zen-thread-parts";
import { fetchMe, type Me } from "@/lib/api";
import { zenChatAdapter } from "@/lib/chat-adapter";
import { useTheme } from "@/lib/theme";

const browserStorage: AsyncStorageLike = {
  async getItem(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  async setItem(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      // storage full or unavailable: the chat still works, it just won't persist
    }
  },
  async removeItem(key) {
    try {
      localStorage.removeItem(key);
    } catch {
      // ignore
    }
  },
};

const suggestionsConfig = AuiConfig({
  suggestions: Suggestions([
    { title: "What's happening today", label: "top news headlines", prompt: "What are today's top news headlines?" },
    { title: "Explain a concept", label: "like I'm new to it", prompt: "Explain how large language models work, like I'm new to the topic." },
    { title: "Help me write", label: "a polite follow-up email", prompt: "Help me write a polite follow-up email after a job interview." },
    { title: "Solve a problem", label: "step by step", prompt: "Solve step by step: a train travels 180 km in 2.5 hours. What's its average speed in m/s?" },
  ]),
});

const CallScreen = lazy(() => import("@/features/call/call-screen"));
// Warm the call bundle on hover/focus so the screen opens instantly.
const preloadCall = () => void import("@/features/call/call-screen");

const useZenThreadRuntime = () => useLocalRuntime(zenChatAdapter);

/** Conversations are stored per Google account in this browser. */
function useZenRuntime(email: string) {
  const adapter = useMemo(
    () =>
      createLocalStorageAdapter({
        storage: browserStorage,
        prefix: `zen:${email}:`,
        titleGenerator: createSimpleTitleAdapter(),
      }),
    [email],
  );
  return useRemoteThreadListRuntime({
    runtimeHook: useZenThreadRuntime,
    adapter,
  });
}

const NewChatButton: FC = () => {
  const aui = useAui();
  const isEmpty = useAuiState((s) => s.thread.messages.length === 0);
  return (
    <TooltipIconButton
      tooltip="New chat"
      variant="ghost"
      className="size-8"
      disabled={isEmpty}
      onClick={() => aui.threads.switchToNewThread()}
    >
      <SquarePenIcon />
    </TooltipIconButton>
  );
};

const ThreadTitle: FC = () => {
  const title = useAuiState((s) => s.threadListItem.title);
  return <span className="truncate text-sm font-medium">{title || "New chat"}</span>;
};

const ChatApp: FC<{ me: Me; setMe: (me: Me) => void }> = ({ me, setMe }) => {
  const runtime = useZenRuntime(me.email);
  const { theme, setTheme } = useTheme();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [callOpen, setCallOpen] = useState(false);
  const name = me.display_name || me.name;
  const components = useMemo(() => threadComponents(name), [name]);

  return (
    <AssistantRuntimeProvider runtime={runtime} config={suggestionsConfig}>
      <SidebarProvider>
        <AppSidebar me={me} onOpenSettings={() => setSettingsOpen(true)} />
        <SidebarInset className="h-dvh overflow-hidden">
          <header className="bg-background/80 sticky top-0 z-10 flex h-14 shrink-0 items-center gap-2 px-3 backdrop-blur">
            <SidebarTrigger />
            <div className="flex min-w-0 flex-1 items-center gap-2 md:hidden">
              <ZenLogo className="size-6 rounded-md" />
              <span className="font-semibold">ZEN AI</span>
            </div>
            <div className="hidden min-w-0 flex-1 md:flex">
              <ThreadTitle />
            </div>
            <TooltipIconButton
              tooltip="Video call"
              variant="ghost"
              className="size-8"
              onClick={() => setCallOpen(true)}
              onPointerEnter={preloadCall}
              onFocus={preloadCall}
            >
              <VideoIcon />
            </TooltipIconButton>
            <NewChatButton />
          </header>
          <div className="min-h-0 flex-1">
            <Thread components={components} />
          </div>
        </SidebarInset>
      </SidebarProvider>

      <SettingsDialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        displayName={me.display_name}
        onDisplayNameSaved={(display_name) => setMe({ ...me, display_name })}
        theme={theme}
        onThemeChange={setTheme}
      />

      {callOpen && (
        <Suspense fallback={<div className="fixed inset-0 z-50 bg-black" />}>
          <CallScreen onClose={() => setCallOpen(false)} />
        </Suspense>
      )}
    </AssistantRuntimeProvider>
  );
};

export default function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMe()
      .then((m) => m && setMe(m))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Something went wrong"));
  }, []);

  if (error) {
    return (
      <div className="text-muted-foreground flex h-dvh items-center justify-center p-6 text-center">
        {error}. Please refresh the page.
      </div>
    );
  }
  if (!me) {
    return (
      <div className="flex h-dvh items-center justify-center">
        <ZenLogo className="size-10 animate-pulse rounded-xl" />
      </div>
    );
  }

  return (
    <TooltipProvider>
      <ChatApp me={me} setMe={setMe} />
    </TooltipProvider>
  );
}
