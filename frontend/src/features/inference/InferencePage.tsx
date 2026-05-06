import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";

import { Disclosure } from "../../components/common/Disclosure";
import { EmptyState } from "../../components/common/EmptyState";
import { StatusCard } from "../../components/StatusCard";
import { useInferenceStatus } from "../../hooks/useInferenceStatus";
import { postJson } from "../../lib/api";
import type { InferenceStatus } from "../../lib/types";

type ChatTurn = { role: "user" | "assistant"; text: string };

export function InferencePage() {
  const { status, setStatus, error, setError } = useInferenceStatus();
  const [message, setMessage] = useState("");
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(256);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [ckptStatus, setCkptStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [ckptBusy, setCkptBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const chatLogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const node = chatLogRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [history, busy]);

  async function send(event: FormEvent) {
    event.preventDefault();
    const prompt = message.trim();
    if (!prompt) return;

    setHistory((current) => [...current, { role: "user", text: prompt }]);
    setMessage("");
    setBusy(true);

    try {
      const result = await postJson<InferenceStatus>("/api/inference/chat", {
        message: prompt,
        temperature,
        max_output_tokens: maxTokens,
      });
      setStatus((current) => ({ ...current, ...result }));
      setNotice(result.message ?? "Response received.");
      setHistory((current) => [
        ...current,
        { role: "assistant", text: result.response || result.message || "" },
      ]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function handleKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  async function loadCheckpoint() {
    if (!selectedCheckpoint) return;
    setCkptBusy(true);
    try {
      const result = await postJson<InferenceStatus>("/api/inference/load-checkpoint", {
        path: selectedCheckpoint,
      });
      setStatus((current) => ({ ...current, ...result }));
      setCkptStatus(result.message ?? "Checkpoint loaded.");
      setError(null);
    } catch (err) {
      setCkptStatus(err instanceof Error ? err.message : String(err));
    } finally {
      setCkptBusy(false);
    }
  }

  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-head">
          <div>
            <p className="section-label">Step 3</p>
            <h2>Chat with your model</h2>
            <p className="panel-desc">
              Your trained model appears here automatically once training finishes.
            </p>
          </div>
          <div className="page-badge">
            <span className={`live-dot${status.has_model ? " active" : ""}`} />
            {status.has_model ? "Model ready" : "Waiting for model"}
          </div>
        </div>

        <div className="status-grid inference-status-grid">
          <StatusCard title="Model Status" eyebrow="Ready state" tone={status.has_model ? "success" : "warning"}>
            {status.readiness}
          </StatusCard>
          <StatusCard
            title="Latest Checkpoint"
            eyebrow="Current base"
            footer={`${status.checkpoint_count} checkpoint${status.checkpoint_count === 1 ? "" : "s"} available`}
          >
            {status.latest_checkpoint ?? "No checkpoint loaded yet."}
          </StatusCard>
        </div>

        {notice ? (
          <div className="info-banner">
            <div>
              <p className="info-banner-title">Conversation feedback</p>
              <p className="info-banner-copy">
                Use this page to test outputs quickly before returning to training.
              </p>
            </div>
            <p className="notice-text">{notice}</p>
          </div>
        ) : null}

        <div className="inference-grid">
          <div className="surface chat-surface chat-stage">
            <div className="chat-stage-head surface-header">
              <div>
                <p className="surface-label">Conversation</p>
                <h3>Test the current model directly</h3>
                <p className="surface-hint">
                  Keep the prompt loop tight here. Use the side rail only when you need to steer or compare checkpoints.
                </p>
              </div>
              <span className="surface-chip">{history.length} turn{history.length === 1 ? "" : "s"} in session</span>
            </div>

            <div className="chat-log" ref={chatLogRef}>
              {history.length === 0 ? (
                <EmptyState
                  accent={status.has_model ? "Model is ready" : "Training required"}
                  title={status.has_model ? "Start the first exchange" : "No model yet"}
                  description={
                    status.has_model
                      ? "Type a message below and press Enter to start."
                      : "Train a model in the Train tab, then come back here."
                  }
                />
              ) : (
                history.map((turn, index) => (
                  <div key={index} className={`chat-turn ${turn.role}`}>
                    <strong>{turn.role === "user" ? "You" : "Model"}</strong>
                    <p>{turn.text}</p>
                  </div>
                ))
              )}
              {busy ? (
                <div className="chat-turn assistant">
                  <strong>Model</strong>
                  <p className="thinking">Thinking...</p>
                </div>
              ) : null}
            </div>

            <form className="chat-form" onSubmit={send}>
              <div className="chat-composer">
                <textarea
                  value={message}
                  onChange={(event) => setMessage(event.target.value)}
                  onKeyDown={handleKey}
                  placeholder="Type a message... (Enter to send, Shift+Enter for a new line)"
                  rows={3}
                  disabled={busy}
                />
                <div className="chat-composer-actions">
                  <p className="mini-note">Enter sends immediately. Shift+Enter adds a new line.</p>
                  <div className="action-row">
                    <button className="primary" type="submit" disabled={busy || !message.trim()}>
                      Send
                    </button>
                    <button type="button" onClick={() => setHistory([])} disabled={history.length === 0}>
                      Clear Conversation
                    </button>
                  </div>
                </div>
              </div>
            </form>
          </div>

          <div className="side-stack">
            <div className="surface control-surface">
              <div className="surface-header">
                <div>
                  <p className="surface-label">Response controls</p>
                  <h3>Steer the output</h3>
                </div>
              </div>
              <div className="chat-controls vertical">
                <span className="chat-control-field stacked">
                  <span className="chat-control-label">Creativity</span>
                  <input
                    type="range"
                    min={0.1}
                    max={2.0}
                    step={0.05}
                    value={temperature}
                    onChange={(event) => setTemperature(Number(event.target.value))}
                    style={{ flex: 1, accentColor: "var(--accent)" }}
                  />
                  <span className="slider-value">{temperature.toFixed(2)}</span>
                  <span className="mini-note">Lower for steadier answers, higher for more variety.</span>
                </span>
                <span className="chat-control-field stacked">
                  <span className="chat-control-label">Max length</span>
                  <input
                    type="range"
                    min={20}
                    max={500}
                    step={10}
                    value={maxTokens}
                    onChange={(event) => setMaxTokens(Number(event.target.value))}
                    style={{ flex: 1, accentColor: "var(--accent)" }}
                  />
                  <span className="slider-value">{maxTokens}</span>
                  <span className="mini-note">Shorter replies make side-by-side checkpoint comparisons faster.</span>
                </span>
              </div>
            </div>

            <Disclosure label="Open Saved Model">
              <p className="surface-hint">
                Load a specific checkpoint to compare earlier runs without retraining first.
              </p>
              {status.available_checkpoints.length > 0 ? (
                <div className="stack-fields">
                  <label className="field">
                    <span>Checkpoint</span>
                    <select
                      value={selectedCheckpoint}
                      onChange={(event) => setSelectedCheckpoint(event.target.value)}
                    >
                      <option value="">Select a checkpoint...</option>
                      {status.available_checkpoints.map((path) => (
                        <option key={path} value={path}>
                          {path}
                        </option>
                      ))}
                    </select>
                  </label>
                  <button onClick={loadCheckpoint} disabled={ckptBusy || !selectedCheckpoint}>
                    {ckptBusy ? "Loading..." : "Load Checkpoint"}
                  </button>
                </div>
              ) : (
                <EmptyState
                  accent="Saved automatically"
                  title="No checkpoints found"
                  description="Checkpoints are saved automatically during training."
                />
              )}
              {ckptStatus ? <p className="surface-hint checkpoint-note">{ckptStatus}</p> : null}
            </Disclosure>

            <div className="surface surface-secondary">
              <div className="surface-header">
                <div>
                  <p className="surface-label">Good testing practice</p>
                  <h3>What to compare</h3>
                </div>
              </div>
              <ul className="guided-list">
                <li>Try the same prompt across checkpoints.</li>
                <li>Lower creativity for factual checks, raise it for open-ended tests.</li>
                <li>Use shorter max length when you want fast iteration.</li>
              </ul>
            </div>
          </div>
        </div>

        {error ? <p className="error-text">{error}</p> : null}
      </section>
    </div>
  );
}
