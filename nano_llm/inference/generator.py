"""
KV-cache aware text generation with MLA compressed cache.

MLA inference flow:
  - Prefill (all prompt tokens at once, past_kvs=None):
      logits, loss = model(prompt_ids)     — no cache returned yet
  - Then build initial cache by re-running prompt... actually for simplicity
    we run the model with past_kvs=[None]*n_layers to get the initial cache.
  - Decode (one token at a time):
      logits, _, new_kvs = model(next_id, past_kvs=prev_kvs)
      next_token = sample(logits[:, -1, :])

For MLA, each past_kvs[i] is (c_kv, k_pe) — the compressed latent + rope key.
c_kv has shape (B, T_cached, kv_lora_rank) — much smaller than full (K, V).
"""
from typing import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor


class Generator:
    def __init__(self, model, tokenizer, device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._use_kv_cache = True  # set False to disable for debugging

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        tokens = list(self.tokenizer.encode(prompt, add_bos=True, add_eos=False))

        if self._use_kv_cache:
            generated = self._generate_with_cache(
                tokens, max_new_tokens, temperature, top_k, top_p, repetition_penalty
            )
        else:
            generated = self._generate_no_cache(
                tokens, max_new_tokens, temperature, top_k, top_p, repetition_penalty
            )

        return self.tokenizer.decode(generated)

    # ---------------------------------------------------------------- #
    # KV-cache generation (MLA compressed latent cache)                #
    # ---------------------------------------------------------------- #

    def _generate_with_cache(
        self, prompt_ids, max_new_tokens, temperature, top_k, top_p, rep_penalty
    ) -> list[int]:
        n_layers = self.model.config.n_layers
        amp_ctx = torch.amp.autocast(
            "cuda" if self.device == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=self.device == "cuda",
        )

        # ----- Prefill: process prompt in one shot with empty cache -----
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        # Truncate if over max_seq_len
        if prompt_t.shape[1] > self.model.config.max_seq_len:
            prompt_t = prompt_t[:, -self.model.config.max_seq_len:]

        # Pass past_kvs=[None]*n_layers so the model goes through _forward_train
        # for each layer and returns the new KV caches.
        past_kvs = [None] * n_layers
        with amp_ctx:
            result = self.model(prompt_t, past_kvs=past_kvs)
        # result is (logits, None, new_kvs) when past_kvs is provided
        logits, _, past_kvs = result

        generated_ids: list[int] = []
        next_id = self._sample(logits[0, -1, :], temperature, top_k, top_p,
                               prompt_ids + generated_ids, rep_penalty)
        if next_id == self.tokenizer.EOS_ID:
            return generated_ids
        generated_ids.append(next_id)

        # ----- Decode: one token at a time, append to cache -----
        for _ in range(max_new_tokens - 1):
            tok_t = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            with amp_ctx:
                result = self.model(tok_t, past_kvs=past_kvs)
            logits, _, past_kvs = result

            all_ids = prompt_ids + generated_ids
            next_id = self._sample(
                logits[0, -1, :], temperature, top_k, top_p, all_ids, rep_penalty
            )
            if next_id == self.tokenizer.EOS_ID:
                break
            generated_ids.append(next_id)

            # Trim cache if we exceed max_seq_len
            cache_len = past_kvs[0][0].shape[1]
            if cache_len >= self.model.config.max_seq_len:
                past_kvs = self._trim_cache(past_kvs, self.model.config.max_seq_len - 1)

        return generated_ids

    def _trim_cache(self, past_kvs: list, keep_len: int) -> list:
        """Trim all cached tensors to the last keep_len positions."""
        trimmed = []
        for kv in past_kvs:
            if kv is None:
                trimmed.append(None)
            else:
                trimmed.append(tuple(t[:, -keep_len:] for t in kv))
        return trimmed

    # ---------------------------------------------------------------- #
    # Simple no-cache generation (re-runs full context each step)      #
    # ---------------------------------------------------------------- #

    def _generate_no_cache(
        self, prompt_ids, max_new_tokens, temperature, top_k, top_p, rep_penalty
    ) -> list[int]:
        amp_ctx = torch.amp.autocast(
            "cuda" if self.device == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=self.device == "cuda",
        )
        generated_ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            ctx = generated_ids[-self.model.config.max_seq_len:]
            inp = torch.tensor([ctx], dtype=torch.long, device=self.device)
            with amp_ctx:
                logits, _ = self.model(inp)
            next_id = self._sample(logits[0, -1, :], temperature, top_k, top_p,
                                   generated_ids, rep_penalty)
            if next_id == self.tokenizer.EOS_ID:
                break
            generated_ids.append(next_id)
        return generated_ids[len(prompt_ids):]

    # ---------------------------------------------------------------- #
    # Sampling                                                          #
    # ---------------------------------------------------------------- #

    def _sample(
        self,
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        prev_ids: list[int],
        rep_penalty: float,
    ) -> int:
        if rep_penalty != 1.0:
            for tid in set(prev_ids[-64:]):
                logits[tid] = logits[tid] / rep_penalty

        if temperature == 0.0:
            return int(logits.argmax().item())

        logits = logits / temperature

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            cutoff = torch.topk(logits, k).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_idx[remove]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    # ---------------------------------------------------------------- #
    # Streaming generation                                              #
    # ---------------------------------------------------------------- #

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Iterator[str]:
        """Yield one decoded token string at a time for UI streaming."""
        tokens = list(self.tokenizer.encode(prompt, add_bos=True, add_eos=False))
        n_layers = self.model.config.n_layers
        amp_ctx = torch.amp.autocast(
            "cuda" if self.device == "cuda" else "cpu",
            dtype=torch.float16,
            enabled=self.device == "cuda",
        )

        # Prefill
        prompt_t = torch.tensor([tokens], dtype=torch.long, device=self.device)
        if prompt_t.shape[1] > self.model.config.max_seq_len:
            prompt_t = prompt_t[:, -self.model.config.max_seq_len:]
        past_kvs = [None] * n_layers
        with amp_ctx:
            logits, _, past_kvs = self.model(prompt_t, past_kvs=past_kvs)

        generated: list[int] = []
        next_id = self._sample(logits[0, -1, :], temperature, top_k, top_p,
                               tokens, repetition_penalty)
        if next_id != self.tokenizer.EOS_ID:
            generated.append(next_id)
            yield self.tokenizer.decode([next_id])

        for _ in range(max_new_tokens - 1):
            tok_t = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            with amp_ctx:
                logits, _, past_kvs = self.model(tok_t, past_kvs=past_kvs)

            next_id = self._sample(
                logits[0, -1, :], temperature, top_k, top_p,
                tokens + generated, repetition_penalty
            )
            if next_id == self.tokenizer.EOS_ID:
                break
            generated.append(next_id)
            yield self.tokenizer.decode([next_id])

            cache_len = past_kvs[0][0].shape[1]
            if cache_len >= self.model.config.max_seq_len:
                past_kvs = self._trim_cache(past_kvs, self.model.config.max_seq_len - 1)
