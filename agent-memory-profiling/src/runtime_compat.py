from packaging.version import Version
import torch
import transformers
from transformers import AutoModelForCausalLM


MIN_TRANSFORMERS = Version("4.51.0")
MAX_TRANSFORMERS = Version("4.52.0")


def validate_local_qwen_runtime() -> None:
    current_transformers = Version(transformers.__version__)
    if current_transformers < MIN_TRANSFORMERS:
        raise RuntimeError(
            f"transformers {transformers.__version__} is too old for Qwen3. "
            f"Install a version in [{MIN_TRANSFORMERS}, {MAX_TRANSFORMERS})."
        )
    current_torch = Version(torch.__version__.split("+")[0])
    if current_torch < Version("2.1.0"):
        raise RuntimeError(
            f"torch {torch.__version__} is too old for the validated local Qwen setup. "
            "Use torch >= 2.1.0."
        )


def load_local_qwen_causal_lm(model_dir: str):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
