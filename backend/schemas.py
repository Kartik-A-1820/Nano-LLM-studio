"""Pydantic request models for the browser-facing API."""
from pydantic import BaseModel, Field


class CrawlStartRequest(BaseModel):
    wikipedia: bool = True
    gutenberg: bool = True
    commoncrawl: bool = False
    archive_org: bool = False
    rss: bool = False
    char_threshold_m: float = Field(default=10.0, ge=0.5, le=100.0)


class CorpusTrainRequest(BaseModel):
    seq_len: int = Field(default=2048, ge=512, le=4096)
    epochs: int = Field(default=1, ge=1, le=200)
    epoch_mode: str = Field(default="custom", pattern="^(custom|auto)$")


class LiveTrainRequest(BaseModel):
    wikipedia: bool = True
    gutenberg: bool = True
    commoncrawl: bool = False
    archive_org: bool = False
    rss: bool = False
    seq_len: int = Field(default=2048, ge=512, le=4096)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    max_output_tokens: int = Field(default=256, ge=20, le=500)


class LoadCheckpointRequest(BaseModel):
    path: str = Field(min_length=1)
