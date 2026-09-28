import { cn } from "@/lib/utils";

/** ZEN mark: an ensō-style ring on the brand gradient. */
export function ZenLogo({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex aspect-square size-8 shrink-0 items-center justify-center rounded-lg bg-linear-to-br from-violet-500 to-indigo-600 text-white shadow-sm",
        className,
      )}
      aria-hidden
    >
      <svg viewBox="0 0 24 24" className="size-[60%]" fill="none">
        <path
          d="M18.4 8.2A7.5 7.5 0 1 0 19.5 12"
          stroke="currentColor"
          strokeWidth="2.6"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}
