import { useEffect, useState } from "react";

import { getJson } from "../lib/api";
import type { TrainStatus } from "../lib/types";

const INITIAL: TrainStatus = {
  readiness: "Loading...",
  progress: "",
  sample: "",
  loss_points: [],
  is_training: false,
  mode: "idle",
  auto_config_summary: "",
  latest_checkpoint: null,
  logged_steps: 0,
  latest_loss: null,
  current_round: null,
  sample_count: 0,
  has_model: false,
  stage: "idle",
  stage_label: "",
  epoch: 0,
  total_epochs: 0,
  total_steps: 0,
  pct: 0,
  arch_name: null,
  param_count_m: null,
  batch_size: 1,
  grad_accum_steps: 4,
  device: "cpu",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function toText(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => toText(item))
      .filter(Boolean)
      .join("\n");
  }
  if (isRecord(value)) {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return fallback;
    }
  }
  return fallback;
}

function toNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function toNullableNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  return toNumber(value, 0);
}

function toBoolean(value: unknown, fallback = false): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    if (value.toLowerCase() === "true") return true;
    if (value.toLowerCase() === "false") return false;
  }
  return fallback;
}

function toLossPoints(value: unknown): TrainStatus["loss_points"] {
  if (!Array.isArray(value)) return [];

  return value
    .map((point) => {
      if (!isRecord(point)) return null;
      const step = toNumber(point.step, Number.NaN);
      const loss = toNumber(point.loss, Number.NaN);
      if (!Number.isFinite(step) || !Number.isFinite(loss)) return null;
      return { step, loss };
    })
    .filter((point): point is { step: number; loss: number } => point !== null);
}

export function normalizeTrainStatus(value: unknown): TrainStatus {
  const next = isRecord(value) ? value : {};

  return {
    readiness: toText(next.readiness, INITIAL.readiness),
    progress: toText(next.progress),
    sample: toText(next.sample),
    loss_points: toLossPoints(next.loss_points),
    is_training: toBoolean(next.is_training),
    mode: toText(next.mode, INITIAL.mode),
    auto_config_summary: toText(next.auto_config_summary),
    latest_checkpoint:
      next.latest_checkpoint === null || next.latest_checkpoint === undefined
        ? null
        : toText(next.latest_checkpoint),
    logged_steps: toNumber(next.logged_steps),
    latest_loss: toNullableNumber(next.latest_loss),
    current_round:
      next.current_round === null || next.current_round === undefined
        ? null
        : toNumber(next.current_round),
    sample_count: toNumber(next.sample_count),
    has_model: toBoolean(next.has_model),
    message: next.message === undefined ? undefined : toText(next.message),
    // Stage-level progress
    stage: toText(next.stage, "idle"),
    stage_label: toText(next.stage_label),
    epoch: toNumber(next.epoch),
    total_epochs: toNumber(next.total_epochs),
    total_steps: toNumber(next.total_steps),
    pct: toNumber(next.pct),
    // Auto-configured model details
    arch_name:
      next.arch_name === null || next.arch_name === undefined
        ? null
        : toText(next.arch_name),
    param_count_m: toNullableNumber(next.param_count_m),
    batch_size: toNumber(next.batch_size, 1),
    grad_accum_steps: toNumber(next.grad_accum_steps, 4),
    device: toText(next.device, "cpu"),
  };
}

export function useTrainingStatus() {
  const [data, setData] = useState<TrainStatus>(INITIAL);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    getJson<TrainStatus>("/api/train/status")
      .then((d) => {
        setData(normalizeTrainStatus({ ...INITIAL, ...d }));
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 2000);
    return () => window.clearInterval(id);
  }, []);

  return { data, setData, error, setError };
}
