export type DataStatus = {
  readiness: string;
  library_status: string;
  collector_status: string;
  preview: string;
  is_collecting: boolean;
  active_source?: string | null;
  prepared_chunks: number;
  stored_docs: number;
  stored_chars: number;
  stored_chars_m: number;
  jsonl_mb: number;
  bin_mb: number;
  has_data: boolean;
  message?: string;
};

export type TrainStatus = {
  readiness: string;
  progress: string;
  sample: string;
  loss_points: Array<{ step: number; loss: number }>;
  is_training: boolean;
  mode: string;
  auto_config_summary: string;
  latest_checkpoint: string | null;
  logged_steps: number;
  latest_loss: number | null;
  current_round: number | null;
  sample_count: number;
  has_model: boolean;
  message?: string;
  // Stage-level progress
  stage: string;
  stage_label: string;
  epoch: number;
  total_epochs: number;
  total_steps: number;
  pct: number;
  // Auto-configured model details
  arch_name: string | null;
  param_count_m: number | null;
  batch_size: number;
  grad_accum_steps: number;
  device: string;
};

export type InferenceStatus = {
  readiness: string;
  has_model: boolean;
  latest_checkpoint: string | null;
  available_checkpoints: string[];
  checkpoint_count: number;
  response?: string;
  message?: string;
};
