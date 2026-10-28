# Contributing to Nano-LLM Studio

Start with the CPU test path in [README.md](README.md). A small training, data, or UI bug with a reproducible case is a good first contribution.

## Development checks

- Training/core changes: `python -m pytest -q tests` with CPU PyTorch.
- Frontend changes: from `frontend/`, run `npm ci` and `npm run build`.
- API/UI integration changes: run both services locally and describe the exact user flow tested.

Add a focused regression test for changed behavior. Documentation changes need command and source checks rather than new implementation-mirroring tests.

## Training and evaluation

Preserve response/token masks, partial accumulation updates, scheduler accounting, validation selection, and checkpoint/RNG semantics when changing the trainer. Use deterministic small-model checks before expensive runs.

Report the corpus provenance, preprocessing, tokenizer, split policy, model configuration, seed, package versions, and hardware. The current token-block holdout does not imply document-disjoint evaluation. Live-training loss is not a held-out quality metric.

Keep raw measurements for comparisons and distinguish unit-test correctness from model capability. Do not commit scraped corpora, model checkpoints, credentials, or private uploaded documents. Source collection support does not establish redistribution rights.

## Pull requests

Describe the problem, change, validation, and remaining limits. Discuss large architecture or storage-format changes in an issue first; explain checkpoint compatibility if it changes.

No standalone LICENSE file is present. Identify sources and terms for third-party additions.
