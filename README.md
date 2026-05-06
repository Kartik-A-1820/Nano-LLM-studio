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
- Early stopping support
- Rolling live-data shard window to avoid unbounded corpus growth
- Checkpoint retention limited to:
  - `outputs/checkpoints/last.pt`
  - `outputs/checkpoints/best.pt`
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

