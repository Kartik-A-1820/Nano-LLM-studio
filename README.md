# Nano-LLM Studio

Nano-LLM Studio is a local training workbench for small language models with a React frontend and a FastAPI backend.

The workbench has three stages:

1. `Data`  
   Build a training corpus from uploads or supported web collectors.
2. `Train`  
   Train on a prepared corpus or experiment with streaming crawler data.
3. `Inference`  
   Generate text from trained models and compare checkpoint behavior.

## Current Stack

- Frontend: React + Vite + TypeScript
- Backend: FastAPI
- Core training/runtime: PyTorch
- Tokenizer: byte-level BPE with a SentencePiece training path and a Python fallback

## Supported Data Sources

Collectors are implemented for the sources below. Availability and content reuse terms depend on the source; collection support is not a blanket license to redistribute the text:

- Wikipedia
- Project Gutenberg
- Common Crawl
- Archive.org
- RSS feeds

The default starter mix uses:

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

## Try the CPU training checks

Use Python 3.11 and a separate environment:

```bash
git clone https://github.com/Kartik-A-1820/Nano-LLM-studio.git
cd Nano-LLM-studio
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python -m pytest -q tests
```

These checks exercise small training/resume cases without downloading pretrained models. They do not demonstrate useful language-model quality. For GPU work, choose a compatible PyTorch build in a separate environment.

## Run the workbench

In the activated environment, from the repository root:

```bash
python -m pip install -r requirements.txt
python -m backend.run
```

The API listens on `http://127.0.0.1:8000`. In another terminal, with Node.js and npm installed:

```bash
cd frontend
npm ci
npm run dev
```

Before launching the frontend, copy `frontend/.env.example` to `frontend/.env`. Its `VITE_API_BASE_URL` should match the backend. Open `http://127.0.0.1:5173`.

For a first UI experiment, upload a small redistributable text corpus, finalize it, train a short corpus run, and inspect separate training/validation losses before generating text. At least two complete token sequences are required; tiny or empty corpora fail explicitly. A small model trained briefly should not be expected to behave like an instruction-tuned assistant.

Runtime corpora and checkpoints live under `train/` and `outputs/`. Keep separate experiment directories when comparing runs. The app is intended for trusted local use; do not expose its data/reset/training API as a public service.

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
  Interrupted partial epochs restore an on-disk model/optimizer/scheduler/RNG boundary snapshot before replaying. This discards partial-epoch updates and avoids a second VRAM copy. Selected checkpoints are saved after sampling callbacks so their RNG state matches the next epoch. Exact replay requires
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

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md). Focused contributions could improve corpus deduplication, document-disjoint evaluation, tokenizer tests, crawler fixtures, or UI/backend error handling. Record source provenance and exact package versions for experiments; Python requirements are not locked.

No standalone LICENSE file is included. Check source-data and repository reuse terms separately.
