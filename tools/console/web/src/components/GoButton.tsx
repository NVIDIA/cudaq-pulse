import clsx from "clsx";

/** The one main action of a view. */
const GoButton = ({ onClick, busy, label = "Go" }: { onClick: () => void; busy?: boolean; label?: string }) => (
  <button
    onClick={onClick}
    disabled={busy}
    className={clsx(
      "group relative flex items-center gap-2 rounded-md bg-[var(--accent)] px-5 py-1.5 text-sm font-black tracking-wide text-black",
      "shadow-[0_0_0_0_var(--glow-green)] transition-[box-shadow,filter] hover:shadow-[0_0_18px_0_var(--glow-green)] hover:brightness-110 disabled:opacity-60",
    )}
  >
    <svg viewBox="0 0 10 10" className={clsx("h-2.5 w-2.5", busy && "animate-spin")} aria-hidden>
      {busy ? <circle cx="5" cy="5" r="4" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="12 20" /> : <path d="M1 0.5 L9.5 5 L1 9.5 Z" fill="currentColor" />}
    </svg>
    {busy ? "Running" : label}
  </button>
);

export default GoButton;
