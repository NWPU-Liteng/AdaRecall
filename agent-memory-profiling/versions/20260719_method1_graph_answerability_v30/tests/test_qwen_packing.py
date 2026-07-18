from __future__ import annotations

import pytest

from src.qwen_reranker import SuffixPreservingPromptPacker


INSTRUCTION = "Retrieve evidence for the query."


def test_long_document_preserves_prefix_suffix_and_budget(tokenizer) -> None:
    packer = SuffixPreservingPromptPacker(
        tokenizer,
        max_length=256,
        instruction=INSTRUCTION,
        document_head_fraction=0.5,
    )
    query = "Which medication was prescribed?"
    packed = packer.pack(query, "head evidence " + "middle " * 2000 + "tail evidence")
    prefix, suffix = packer.template_parts(query)
    assert packed.truncated
    assert len(packed.input_ids) == 256
    assert packed.input_ids[: len(prefix)] == prefix
    assert packed.input_ids[-len(suffix) :] == suffix
    assert packed.document_tokens_kept == 256 - len(prefix) - len(suffix)


def test_short_document_is_lossless(tokenizer) -> None:
    packer = SuffixPreservingPromptPacker(tokenizer, max_length=512, instruction=INSTRUCTION)
    packed = packer.pack("query", "short document")
    assert not packed.truncated
    assert packed.document_tokens_kept == packed.document_tokens
    assert packed.original_tokens == packed.packed_tokens


def test_query_cannot_silently_consume_document_budget(tokenizer) -> None:
    packer = SuffixPreservingPromptPacker(tokenizer, max_length=32, instruction=INSTRUCTION)
    with pytest.raises(ValueError, match="leaving no document budget"):
        packer.pack("query " * 200, "document")
