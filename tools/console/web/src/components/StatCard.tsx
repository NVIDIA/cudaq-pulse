interface StatCardProps {
  label: string;
  value: string;
  delta?: string;
  accent?: boolean;
}

const StatCard = ({ label, value, delta, accent }: StatCardProps) => (
  <div className="rounded-lg border border-[var(--border)] bg-[var(--panel)] px-3 py-2">
    <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">{label}</div>
    <div
      className={`mt-1 text-xl font-black tabular-nums ${
        accent ? "text-[var(--accent)]" : "text-[var(--text)]"
      }`}
    >
      {value}
    </div>
    {delta && <div className="mt-0.5 text-[11px] text-[var(--muted)]">{delta}</div>}
  </div>
);

export default StatCard;
