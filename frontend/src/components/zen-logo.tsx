import { cn } from "@/lib/utils";

/** ZEN mark: an ensō-style ring on NeuZem red. */
export function ZenLogo({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "bg-brand text-brand-foreground flex aspect-square size-8 shrink-0 items-center justify-center rounded-md shadow-sm",
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

/** "by NeuZem" company wordmark, swapped for the current theme. */
export function NeuZemWordmark({ className }: { className?: string }) {
  const base = import.meta.env.BASE_URL;
  return (
    <span className={cn("inline-flex items-center", className)}>
      <img
        src={`${base}neuzem-wordmark-black.png`}
        alt="NeuZem"
        className="h-full w-auto dark:hidden"
      />
      <img
        src={`${base}neuzem-wordmark-white.png`}
        alt="NeuZem"
        className="hidden h-full w-auto dark:block"
      />
    </span>
  );
}
