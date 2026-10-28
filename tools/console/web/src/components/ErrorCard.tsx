import type { ApiError } from "../api";

const ErrorCard = ({ error, onJump }: { error: ApiError; onJump?: (line: number) => void }) => (
  <div className="rounded-lg border border-[var(--danger)]/60 bg-[var(--danger)]/10 p-4">
    <div className="flex items-center gap-2 text-sm font-bold text-[var(--danger)]">
      <span className="rounded bg-[var(--danger)] px-1.5 py-0.5 text-[10px] uppercase text-black">
        {error.stage ?? "error"}
      </span>
      {error.kind}
      {error.line != null && (
        <button
          className="ml-auto rounded border border-[var(--danger)]/50 px-2 py-0.5 text-[10px] text-[var(--danger)]"
          onClick={() => onJump?.(error.line!)}
        >
          kernel.py line {error.line}
        </button>
      )}
    </div>
    <pre className="mono mt-3 whitespace-pre-wrap text-xs leading-relaxed text-[var(--text)]">
      {error.message}
    </pre>
  </div>
);

export default ErrorCard;
