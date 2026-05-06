import { useState } from "react";

import { Disclosure } from "../../components/common/Disclosure";
import { EmptyState } from "../../components/common/EmptyState";
import { StatusCard } from "../../components/StatusCard";
import { useDataStatus } from "../../hooks/useDataStatus";
import { postForm, postJson } from "../../lib/api";
import type { DataStatus } from "../../lib/types";

const SOURCE_LABELS: Record<string, string> = {
  wikipedia: "Wikipedia",
  gutenberg: "Gutenberg",
  commoncrawl: "Common Crawl",
  archive_org: "Archive.org",
  rss: "RSS",
};

type Sources = {
  wikipedia: boolean;
  gutenberg: boolean;
  commoncrawl: boolean;
  archive_org: boolean;
  rss: boolean;
};

export function DataPage() {
  const { data, setData, error, setError } = useDataStatus();
  const [files, setFiles] = useState<FileList | null>(null);
  const [threshold, setThreshold] = useState(10);
  const [sources, setSources] = useState<Sources>({
    wikipedia: true,
    gutenberg: true,
    commoncrawl: false,
    archive_org: false,
    rss: false,
  });
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState("");

  async function handleUpload() {
    if (!files || files.length === 0) return;
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    setBusy("upload");
    try {
      const next = await postForm<DataStatus>("/api/data/upload", form);
      setData((current) => ({ ...current, ...next }));
      setNotice(next.message ?? "Uploads processed.");
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  async function handleAction(path: string, body?: unknown) {
    setBusy(path);
    try {
      const next = await postJson<DataStatus>(path, body);
      setData((current) => ({ ...current, ...next }));
      setNotice(next.message ?? "Action completed.");
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="page-grid">
      <section className="panel">
        <div className="panel-head">
          <div>
            <p className="section-label">Step 1</p>
            <h2>Build your training library</h2>
            <p className="panel-desc">
              Combine uploads and web collection into one shared library, then finalize it for training.
            </p>
          </div>
        </div>

        <div className="workspace-hero">
          <div className="workspace-hero-main">
            <p className="surface-label">Current workspace</p>
            <h3>{data.has_data ? "Your library is taking shape." : "Start building the library."}</h3>
            <p className="surface-hint">
              {data.has_data
                ? "Keep collecting or uploading until the coverage looks right, then finalize the dataset and move to Train."
                : "Bring in a few files, collect from the web, or combine both. Everything lands in one shared corpus."}
            </p>
          </div>
          <div className="workspace-hero-aside">
            <span className={`live-dot${data.is_collecting ? " active" : ""}`} />
            <strong>{data.is_collecting ? "Collector live" : "Collector idle"}</strong>
            <p>{data.active_source ? `Last active source: ${data.active_source}` : "No active source selected yet."}</p>
          </div>
        </div>

        <div className="status-grid">
          <StatusCard
            title="Training Readiness"
            eyebrow="Next step"
            tone={data.has_data ? "accent" : "warning"}
            footer={
              data.has_data
                ? "Finalize the dataset if you collected data from the web, then move to Train."
                : "Start by uploading files or collecting from public sources."
            }
          >
            {data.readiness}
          </StatusCard>
          <StatusCard
            title="Library Status"
            eyebrow="Corpus"
            footer={`Prepared chunks: ${data.prepared_chunks.toLocaleString()} | Source: ${data.active_source ?? "none"}`}
          >
            {data.library_status}
          </StatusCard>
          <StatusCard
            title="Collector Status"
            eyebrow="Live intake"
            tone={data.is_collecting ? "success" : "default"}
            footer={
              data.is_collecting
                ? "Collector is running. Finalize the dataset when the library looks large enough."
                : "Collector is idle until you start it."
            }
          >
            {data.collector_status || (data.is_collecting ? "Collecting..." : "No collector running.")}
          </StatusCard>
        </div>

        {notice && <p className="notice-text">{notice}</p>}

        <div className="metric-strip">
          <div className="metric-tile">
            <span className="metric-label">Stored docs</span>
            <strong>{data.stored_docs.toLocaleString()}</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Stored text</span>
            <strong>{data.stored_chars_m.toFixed(1)}M chars</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">JSONL size</span>
            <strong>{data.jsonl_mb.toFixed(1)} MB</strong>
          </div>
          <div className="metric-tile">
            <span className="metric-label">Binary cache</span>
            <strong>{data.bin_mb > 0 ? `${data.bin_mb.toFixed(1)} MB` : "Not built yet"}</strong>
          </div>
        </div>

        <div className="two-column">
          <div className="surface work-lane">
            <div className="surface-header">
              <div>
                <p className="surface-label">Import files</p>
                <h3>Upload source material</h3>
              </div>
              <span className="surface-chip">Best for curated docs</span>
            </div>
            <p className="surface-hint">Supported: `.txt`, `.pdf`, `.docx`, `.md`, `.jsonl`</p>
            <input
              type="file"
              multiple
              accept=".txt,.pdf,.docx,.md,.jsonl"
              onChange={(e) => setFiles(e.target.files)}
            />
            <ul className="guided-list compact">
              <li>Use this when you already have internal notes, manuals, or clean text exports.</li>
              <li>Processed uploads go into the same library used by the collector.</li>
            </ul>
            <div className="action-row">
              <button
                className="primary"
                onClick={handleUpload}
                disabled={busy !== null || !files?.length}
              >
                {busy === "upload" ? "Processing..." : "Process Uploads"}
              </button>
              <button
                onClick={() => handleAction("/api/data/import-local")}
                disabled={busy !== null}
              >
                Import Local Library
              </button>
            </div>
          </div>

          <div className="surface work-lane">
            <div className="surface-header">
              <div>
                <p className="surface-label">Collect live text</p>
                <h3>Web collector</h3>
              </div>
              <span className="surface-chip">{data.is_collecting ? "Running now" : "Optional source boost"}</span>
            </div>
            <p className="surface-hint">
              Choose sources, collect to a target size, then finalize the dataset when it looks ready.
            </p>
            <div className="checkbox-grid">
              {(Object.keys(sources) as Array<keyof Sources>).map((key) => (
                <label key={key} className="check-row">
                  <input
                    type="checkbox"
                    checked={sources[key]}
                    onChange={() => setSources((prev) => ({ ...prev, [key]: !prev[key] }))}
                  />
                  <span>{SOURCE_LABELS[key]}</span>
                </label>
              ))}
            </div>
            <label className="field">
              <span>Collect until (millions of characters)</span>
              <div className="slider-row">
                <input
                  type="range"
                  min={0.5}
                  max={100}
                  step={0.5}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
                <span className="slider-value">{threshold}M</span>
              </div>
            </label>
            <div className="action-row">
              <button
                className="primary"
                onClick={() =>
                  handleAction("/api/crawl/start", { ...sources, char_threshold_m: threshold })
                }
                disabled={busy !== null || data.is_collecting}
              >
                {data.is_collecting ? "Collecting..." : "Start Collector"}
              </button>
              <button
                onClick={() => handleAction("/api/crawl/stop")}
                disabled={busy !== null || !data.is_collecting}
              >
                Stop
              </button>
              <button
                onClick={() => handleAction("/api/data/finalize")}
                disabled={busy !== null}
              >
                Finalize Dataset
              </button>
            </div>
          </div>
        </div>

        <Disclosure label="Library Preview">
          {data.preview ? (
            <pre className="preview">{data.preview}</pre>
          ) : (
            <EmptyState
              title="No preview available yet"
              description="Upload files or run the web collector, then finalize the dataset to review it here."
            />
          )}
        </Disclosure>

        {error && <p className="error-text">{error}</p>}

        <Disclosure label="Reset Library" variant="danger">
          <p className="danger-description">
            This clears collected text, tokenizer state, and the in-memory model for a fresh start.
          </p>
          <button
            className="danger"
            onClick={() => handleAction("/api/data/reset")}
            disabled={busy !== null}
          >
            Clear Everything and Start Over
          </button>
        </Disclosure>
      </section>
    </div>
  );
}
