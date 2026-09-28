import { useState, type FC } from "react";
import { MonitorIcon, MoonIcon, SunIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { saveDisplayName } from "@/lib/api";
import type { ThemePreference } from "@/lib/theme";
import { cn } from "@/lib/utils";

const THEMES: { value: ThemePreference; label: string; icon: typeof SunIcon }[] = [
  { value: "light", label: "Light", icon: SunIcon },
  { value: "dark", label: "Dark", icon: MoonIcon },
  { value: "system", label: "System", icon: MonitorIcon },
];

type SettingsProps = {
  onOpenChange: (open: boolean) => void;
  displayName: string;
  onDisplayNameSaved: (name: string) => void;
  theme: ThemePreference;
  onThemeChange: (theme: ThemePreference) => void;
};

export const SettingsDialog: FC<SettingsProps & { open: boolean }> = ({ open, ...props }) => (
  <Dialog open={open} onOpenChange={props.onOpenChange}>
    <DialogContent className="sm:max-w-md">
      <DialogHeader>
        <DialogTitle>Settings</DialogTitle>
        <DialogDescription>Personalize how ZEN talks to you.</DialogDescription>
      </DialogHeader>
      {/* Mounted only while open, so the form starts from the saved values each time. */}
      <SettingsForm {...props} />
    </DialogContent>
  </Dialog>
);

const SettingsForm: FC<SettingsProps> = ({
  onOpenChange,
  displayName,
  onDisplayNameSaved,
  theme,
  onThemeChange,
}) => {
  const [name, setName] = useState(displayName);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      onDisplayNameSaved(await saveDisplayName(name));
      onOpenChange(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form
      className="grid gap-6 py-2"
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <div className="grid gap-2">
        <Label htmlFor="display-name">What should ZEN call you?</Label>
        <Input
          id="display-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Your name"
          maxLength={32}
          autoComplete="off"
        />
      </div>

      <div className="grid gap-2">
        <Label>Appearance</Label>
        <div className="bg-muted grid grid-cols-3 gap-1 rounded-lg p-1">
          {THEMES.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              type="button"
              onClick={() => onThemeChange(value)}
              aria-pressed={theme === value}
              className={cn(
                "text-muted-foreground flex items-center justify-center gap-2 rounded-md py-1.5 text-sm transition-colors",
                theme === value
                  ? "bg-background text-foreground shadow-sm"
                  : "hover:text-foreground",
              )}
            >
              <Icon className="size-4" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {error && <p className="text-destructive text-sm">{error}</p>}

      <DialogFooter>
        <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>
          Cancel
        </Button>
        <Button type="submit" disabled={saving}>
          {saving ? "Saving…" : "Save"}
        </Button>
      </DialogFooter>
    </form>
  );
};
