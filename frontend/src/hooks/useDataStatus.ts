import { useEffect, useState } from "react";

import { getJson } from "../lib/api";
import type { DataStatus } from "../lib/types";

const INITIAL: DataStatus = {
  readiness: "Loading...",
  library_status: "Loading...",
  collector_status: "Loading...",
  preview: "",
  is_collecting: false,
  active_source: null,
  prepared_chunks: 0,
  stored_docs: 0,
  stored_chars: 0,
  stored_chars_m: 0,
  jsonl_mb: 0,
  bin_mb: 0,
  has_data: false,
};

export function useDataStatus() {
  const [data, setData] = useState<DataStatus>(INITIAL);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    getJson<DataStatus>("/api/data/status")
      .then((d) => {
        setData({ ...INITIAL, ...d });
        setError(null);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }

  useEffect(() => {
    refresh();
    const id = window.setInterval(refresh, 3000);
    return () => window.clearInterval(id);
  }, []);

  return { data, setData, error, setError };
}
