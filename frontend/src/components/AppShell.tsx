import { ReactNode } from "react";

type TabId = "data" | "train" | "inference";

type AppShellProps = {
  currentTab: TabId;
  onTabChange: (tab: TabId) => void;
  children: ReactNode;
};

const tabs: Array<{ id: TabId; label: string; step: string }> = [
  { id: "data", label: "Data", step: "1" },
  { id: "train", label: "Train", step: "2" },
  { id: "inference", label: "Inference", step: "3" },
];

const pageMeta: Record<TabId, { kicker: string; title: string; description: string }> = {
  data: {
    kicker: "Library workspace",
    title: "Assemble a clean dataset before you spend compute.",
    description:
      "Uploads and web collection feed one shared library. Build it once, check readiness, then move on to training.",
  },
  train: {
    kicker: "Training control room",
    title: "Run the trainer with one primary decision, not fifty knobs.",
    description:
      "Pick standard or live mode, review the auto-selected setup, then monitor progress and checkpoints from one place.",
  },
  inference: {
    kicker: "Model test bench",
    title: "Talk to the model and judge the result fast.",
    description:
      "Keep the interface focused on conversation, with only the controls you actually need for quick evaluation.",
  },
};

export function AppShell({ currentTab, onTabChange, children }: AppShellProps) {
  const meta = pageMeta[currentTab];

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy-block">
          <p className="eyebrow">Nano-LLM Studio</p>
          <h1>Your own language model, from data to chat.</h1>
          <p className="hero-copy">
            A guided three-step workflow for building, training, and testing a model
            without exposing the whole research stack on day one.
          </p>
        </div>
        <div className="hero-aside">
          <div className="hero-pulse" />
          <p className="hero-kicker">{meta.kicker}</p>
          <h2>{meta.title}</h2>
          <p>{meta.description}</p>
        </div>
      </header>

      <nav className="top-tabs" aria-label="Primary navigation">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={tab.id === currentTab ? "tab active" : "tab"}
            onClick={() => onTabChange(tab.id)}
            type="button"
            aria-current={tab.id === currentTab ? "page" : undefined}
          >
            <span className="tab-step">{tab.step}</span>
            {tab.label}
          </button>
        ))}
      </nav>

      <section className="workflow-strip" aria-label="Workflow summary">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className={`workflow-step${tab.id === currentTab ? " current" : ""}`}
          >
            <span className="workflow-step-number">{tab.step}</span>
            <div>
              <p className="workflow-step-label">{tab.label}</p>
              <p className="workflow-step-copy">{pageMeta[tab.id].kicker}</p>
            </div>
          </div>
        ))}
      </section>

      <main className="page-frame">{children}</main>
    </div>
  );
}
