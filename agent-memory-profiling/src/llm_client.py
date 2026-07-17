import os
import time

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .runtime_compat import load_local_qwen_causal_lm, validate_local_qwen_runtime


class MockLLMClient:
    def __init__(self):
        self.mode = "mock"

    def generate(self, prompt: str) -> str:
        return self.generate_with_stats(prompt)["text"]

    def generate_with_stats(self, prompt: str) -> dict:
        _ = prompt
        text = "mock answer"
        return {
            "text": text,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed_ms": 0.0,
            "backend_mode": self.mode,
        }


class OllamaLLMClient:
    def __init__(self, model: str = "", timeout: float = 20.0):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.timeout = timeout
        self.fallback = MockLLMClient()
        self.mode = "ollama"

    def generate(self, prompt: str) -> str:
        return self.generate_with_stats(prompt)["text"]

    def generate_with_stats(self, prompt: str) -> dict:
        try:
            start = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "").strip()
            if not text:
                raise RuntimeError("Empty response from Ollama")
            self.mode = "ollama"
            return {
                "text": text,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
                "backend_mode": self.mode,
            }
        except Exception:
            self.mode = self.fallback.mode
            return self.fallback.generate_with_stats(prompt)


class LocalQwenLLMClient:
    _model = None
    _tokenizer = None
    _device = None
    _loaded_model_dir = None

    def __init__(self, model_dir: str = "", timeout: float = 20.0, max_new_tokens: int = 24, use_chat_template: bool = False):
        self.model_dir = model_dir or os.getenv("LOCAL_QWEN_DIR", "../dir")
        self.timeout = timeout
        self.max_new_tokens = int(os.getenv("LOCAL_QWEN_MAX_NEW_TOKENS", str(max_new_tokens)))
        self.use_chat_template = use_chat_template
        self.mode = "local_qwen"
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded_common(cls, model_dir: str):
        if cls._model is not None and cls._loaded_model_dir == model_dir:
            return
        validate_local_qwen_runtime()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; local Qwen generation requires GPU")
        device_index = os.getenv("LOCAL_QWEN_CUDA_DEVICE", "0")
        cls._device = torch.device(f"cuda:{device_index}")
        cls._tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        cls._model = load_local_qwen_causal_lm(model_dir).to(cls._device)
        cls._model.eval()
        cls._loaded_model_dir = model_dir

    def _ensure_loaded(self):
        self._ensure_loaded_common(self.model_dir)

    def generate(self, prompt: str) -> str:
        return self.generate_with_stats(prompt)["text"]

    def generate_with_stats(self, prompt: str) -> dict:
        if self.use_chat_template:
            encoded = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
        else:
            encoded = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        prompt_tokens = int(encoded["input_ids"].shape[1])
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        start = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        new_tokens = output[0][encoded["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            "text": text or "empty local qwen answer",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(new_tokens.shape[0]),
            "elapsed_ms": elapsed_ms,
            "backend_mode": self.mode,
        }
