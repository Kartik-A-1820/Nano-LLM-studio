type EmptyStateProps = {
  title: string;
  description?: string;
  accent?: string;
};

export function EmptyState({ title, description, accent }: EmptyStateProps) {
  return (
    <div className="empty-state">
      {accent ? <p className="empty-state-accent">{accent}</p> : null}
      <p className="empty-state-title">{title}</p>
      {description && <p className="empty-state-desc">{description}</p>}
    </div>
  );
}
