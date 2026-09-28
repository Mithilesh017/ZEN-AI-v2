import type { FC } from "react";
import { ChevronsUpDownIcon, LogOutIcon, SettingsIcon } from "lucide-react";

import { ThreadList } from "@/components/assistant-ui/elements/thread-list.aui";
import { ZenLogo } from "@/components/zen-logo";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import type { Me } from "@/lib/api";

const initials = (name: string) =>
  name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join("") || "?";

export const AppSidebar: FC<{ me: Me; onOpenSettings: () => void }> = ({
  me,
  onOpenSettings,
}) => {
  const name = me.display_name || me.name;

  return (
    <Sidebar>
      <SidebarHeader>
        <div className="flex items-center gap-2.5 px-2 py-1.5">
          <ZenLogo />
          <span className="text-base font-semibold tracking-tight">ZEN AI</span>
        </div>
      </SidebarHeader>

      <SidebarContent className="px-2">
        <ThreadList />
      </SidebarContent>

      <SidebarFooter className="border-t">
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent"
                >
                  <Avatar className="size-8 rounded-lg">
                    {me.picture && (
                      <AvatarImage src={me.picture} alt="" referrerPolicy="no-referrer" />
                    )}
                    <AvatarFallback className="rounded-lg">{initials(name)}</AvatarFallback>
                  </Avatar>
                  <div className="grid min-w-0 flex-1 text-start text-sm leading-tight">
                    <span className="truncate font-medium">{name}</span>
                    <span className="text-muted-foreground truncate text-xs">{me.email}</span>
                  </div>
                  <ChevronsUpDownIcon className="ms-auto size-4" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent side="top" align="start" className="w-(--radix-dropdown-menu-trigger-width) min-w-56">
                <DropdownMenuLabel className="text-muted-foreground font-normal">
                  {me.email}
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={onOpenSettings}>
                  <SettingsIcon />
                  Settings
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <a href="/logout">
                    <LogOutIcon />
                    Log out
                  </a>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
};
