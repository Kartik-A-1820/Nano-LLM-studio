"""HTTP API for the React frontend migration."""
from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    ChatRequest,
    CorpusTrainRequest,
    CrawlStartRequest,
    LiveTrainRequest,
    LoadCheckpointRequest,
)
from backend.service_adapter import (
    chat_once,
    clear_library,
    finalize_crawl_dataset,
    get_data_status,
    get_inference_status,
    get_train_status,
    import_local_library,
    load_checkpoint,
    process_uploads,
    start_crawl,
    start_corpus_train,
    start_live_train,
    stop_crawl,
    stop_current_training,
)

app = FastAPI(title="Nano-LLM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/data/status")
def data_status() -> dict:
    return get_data_status()


@app.post("/api/data/upload")
async def data_upload(files: list[UploadFile] = File(...)) -> dict:
    return await process_uploads(files)


@app.post("/api/data/import-local")
def data_import_local() -> dict:
    return import_local_library()


@app.post("/api/data/finalize")
def data_finalize() -> dict:
    return finalize_crawl_dataset()


@app.post("/api/data/reset")
def data_reset() -> dict:
    return clear_library()


@app.post("/api/crawl/start")
def crawl_start(payload: CrawlStartRequest) -> dict:
    return start_crawl(payload)


@app.post("/api/crawl/stop")
def crawl_stop() -> dict:
    return stop_crawl()


@app.get("/api/train/status")
def train_status() -> dict:
    return get_train_status()


@app.post("/api/train/start/corpus")
def train_start_corpus(payload: CorpusTrainRequest) -> dict:
    return start_corpus_train(payload)


@app.post("/api/train/start/live")
def train_start_live(payload: LiveTrainRequest) -> dict:
    return start_live_train(payload)


@app.post("/api/train/stop")
def train_stop() -> dict:
    return stop_current_training()


@app.get("/api/inference/status")
def inference_status() -> dict:
    return get_inference_status()


@app.post("/api/inference/load-checkpoint")
def inference_load_checkpoint(payload: LoadCheckpointRequest) -> dict:
    return load_checkpoint(payload.path)


@app.post("/api/inference/chat")
def inference_chat(payload: ChatRequest) -> dict:
    return chat_once(payload.message, payload.temperature, payload.max_output_tokens)
