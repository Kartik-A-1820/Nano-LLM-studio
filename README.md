# Nano-LLM Studio

Nano-LLM Studio is a local training workbench for small language models with a React frontend and a FastAPI backend.

The product flow is intentionally simple:

1. `Data`  
   Build a training library from uploads or free public crawlers.
2. `Train`  
   Train on the prepared corpus or run live training on streaming crawler data.
3. `Inference`  
   Chat with the trained model and compare `best.pt` / `last.pt` checkpoints.

## Current Stack

- Frontend: React + Vite + TypeScript
- Backend: FastAPI
- Core training/runtime: PyTorch
- Tokenizer: SentencePiece-based BPE pipeline

## Supported Data Sources

The app supports free, open sources with no API key requirement:

- Wikipedia
- Project Gutenberg
- Common Crawl
- Archive.org
- RSS feeds

For initial training quality, the default starter mix is:

- Wikipedia
- Gutenberg

## Main Features

- Unified `Data`, `Train`, and `Inference` workflow
- File upload and web collection feeding the same corpus pipeline
- Auto tokenizer preparation
- Auto architecture selection with smaller-model bias for faster iteration
- Standard training and live training modes
- Validation-based early stopping in corpus training
- Rolling live-data shard window to avoid unbounded corpus growth
- Checkpoint retention limited to:
  - `outputs/checkpoints/last.pt`
  - `outputs/checkpoints/best.pt` (selected by validation loss when a holdout exists)
- Chat-first inference UI with adjustable:
  - temperature
  - max output tokens

## Project Layout

```text
backend/      FastAPI app and browser-facing API adapter
frontend/     React app
nano_llm/     Model, tokenizer, training, crawler, and dataset logic
ui/           Shared orchestration/state layer used by the backend
scripts/      Utility scripts
train/        Local corpus storage and prepared training assets
outputs/      Checkpoints, tokenizer assets, runtime outputs
```

## Local Setup

### Backend

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -m backend.run
```

Backend runs on:

- `http://127.0.0.1:8000`

### Frontend

```powershell
cd frontend
copy .env.example .env
cmd.exe /c npm install
cmd.exe /c npm run dev
```

Frontend runs on:

- `http://127.0.0.1:5173`

## API Surface

### Data

- `GET /health`
- `GET /api/data/status`
- `POST /api/data/upload`
- `POST /api/data/import-local`
- `POST /api/data/finalize`
- `POST /api/data/reset`

### Crawl

- `POST /api/crawl/start`
- `POST /api/crawl/stop`

### Train

- `GET /api/train/status`
- `POST /api/train/start/corpus`
- `POST /api/train/start/live`
- `POST /api/train/stop`

### Inference

- `GET /api/inference/status`
- `POST /api/inference/chat`
- `POST /api/inference/load-checkpoint`

## Notes

- The app is designed for local experimentation and iterative training.
- Live training uses a bounded rolling window of text shards instead of one endlessly growing corpus file.
- Only `best.pt` and `last.pt` are retained as active checkpoints.


## Training correctness and evaluation

Corpus training (API/UI and CLI) now reserves the final 10% of token sequences
(at least one) for validation. Training uses the remaining sequences, shuffles
only that subset, and retains short final batches. At least two complete token
sequences are required. This is a contiguous **token-block holdout**, not a
claim of document-disjoint or semantic deduplication: duplicated/overlapping
source chunks still require upstream curation, and the tokenizer may have seen
held-out text while learning its vocabulary.

- `best.pt` and corpus early stopping use token-weighted validation loss.
- Training and validation loss are separate in the UI status/log records.
- Incomplete gradient-accumulation groups are flushed with their actual
  supervised-token weight; loaders smaller than an accumulation group still
  perform an update. Scheduler steps track successful optimizer updates.
- Live training has no stable validation corpus. It saves `last.pt`, does not
  select a new `best.pt`, and its outer plateau heuristic remains training-loss
  based. It must not be interpreted as held-out quality. An older `best.pt` in
  a reused directory remains an older artifact; use a fresh checkpoint directory
  for a new experiment and `last.pt` for live runs.
- Checkpoints include the scheduler, epoch, validation-selection state, Torch
  RNG state, and a schedule signature. `Trainer.load_checkpoint()` resumes at
  the next epoch with the same planned epochs/loader/accumulation configuration.
  Interrupted partial epochs repeat from their boundary. Exact replay requires
  unchanged data and deterministic loaders; custom sampler generators and data
  worker RNG are not serialized. Older checkpoints lacking the new state are
  not supported for exact training resume; model-only inference is separate.
- Model configuration is stored as a plain dictionary so new checkpoints can
  be read with `torch.load(..., weights_only=True)`.

Run the lightweight CPU regression suite after installing a CPU PyTorch build
and `requirements-test.txt`:

```bash
python -m pytest -q tests
```

The suite covers short accumulation groups, equivalence to a full batch,
validation-selected weights, epoch/scheduler resume, holdout partitioning,
empty-data errors, and a tiny real transformer training run. It does not
establish model quality, CUDA/AMP behavior on a particular GPU, or crawler/UI
end-to-end compatibility.

References: [PyTorch AMP accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).
