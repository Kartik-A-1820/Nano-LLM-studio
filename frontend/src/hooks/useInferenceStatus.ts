import { useEffect, useState } from "react";

import { getJson } from "../lib/api";
import type { InferenceStatus } from "../lib/types";

const INITIAL: InferenceStatus = {
  readiness: "Loading...",
  has_model: false,
  latest_checkpoint: null,
  available_checkpoints: [],
  checkpoint_count: 0,
};

export function useInferenceStatus() {
  const [status, setStatus] = useState<InferenceStatus>(INITIAL);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    getJson<InferenceStatus>("/api/inference/status")
      .then((s) => {
        setStatus({ ...INITIAL, ...s });
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 3000);
    return () => window.clearInterval(id);
  }, []);

  return { status, setStatus, error, setError };
}
