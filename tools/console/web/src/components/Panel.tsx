import type { ReactNode } from "react";
import clsx from "clsx";

interface PanelProps {
  title: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
}

const Panel = ({ title, children, actions, className, bodyClassName }: PanelProps) => (
  <section
    className={clsx(
      "flex min-h-0 flex-col rounded-lg border border-[var(--border)] bg-[var(--panel)] shadow-sm",
      className,
    )}
  >
    <div className="flex items-center justify-between gap-2 border-b border-[var(--border)] px-3 py-2">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted)]">{title}</h2>
      {actions}
    </div>
    <div className={clsx("min-h-0 flex-1 p-3", bodyClassName)}>{children}</div>
  </section>
);

export default Panel;
