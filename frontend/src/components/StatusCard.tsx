import type { ReactNode } from "react";

type StatusCardProps = {
  title: string;
  children: ReactNode;
  eyebrow?: string;
  tone?: "default" | "accent" | "success" | "warning";
  footer?: ReactNode;
};

export function StatusCard({ title, children, eyebrow, tone = "default", footer }: StatusCardProps) {
  return (
    <section className={`status-card tone-${tone}`}>
      {eyebrow ? <p className="status-eyebrow">{eyebrow}</p> : null}
      <h3>{title}</h3>
      <div className="status-body">{children}</div>
      {footer ? <div className="status-footer">{footer}</div> : null}
    </section>
  );
}
