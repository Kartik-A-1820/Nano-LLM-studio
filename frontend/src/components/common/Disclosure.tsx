import { ReactNode, useState } from "react";

type DisclosureProps = {
  label: string;
  children: ReactNode;
  defaultOpen?: boolean;
  variant?: "default" | "danger";
};

export function Disclosure({
  label,
  children,
  defaultOpen = false,
  variant = "default",
}: DisclosureProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={`disclosure${variant === "danger" ? " disclosure-danger" : ""}`}>
      <button
        type="button"
        className="disclosure-toggle"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <span>{label}</span>
        <span className="disclosure-arrow">{open ? "▲" : "▼"}</span>
      </button>
      {open ? <div className="disclosure-body">{children}</div> : null}
    </div>
  );
}
