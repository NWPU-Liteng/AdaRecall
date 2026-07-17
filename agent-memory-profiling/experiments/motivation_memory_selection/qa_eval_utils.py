from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: Any) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    raw = raw.translate(_PUNCT_TABLE)
    return " ".join(raw.split())


def normalize_tokens(text: Any) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(str(item).strip())
    return output


def reference_variants(reference: Any) -> list[str]:
    raw = str(reference or "").strip()
    if not raw:
        return [""]

    variants = [raw]
    if re.search(r"^the information provided is not enough\b|^insufficient information\b", raw, flags=re.IGNORECASE):
        variants.append("insufficient information")

    if re.search(r"\balso acceptable\b", raw, flags=re.IGNORECASE):
        first_sentence = raw.split(".", 1)[0].strip()
        if first_sentence:
            variants.append(first_sentence)
        alt_match = re.search(
            r"\.\s*([^.]*(?:also acceptable))\.?\s*$",
            raw,
            flags=re.IGNORECASE,
        )
        if alt_match:
            alt_text = alt_match.group(1)
            alt_text = re.sub(r"\s*\([^)]*\)\s*is also acceptable\s*$", "", alt_text, flags=re.IGNORECASE).strip()
            alt_text = re.sub(r"\s*is also acceptable\s*$", "", alt_text, flags=re.IGNORECASE).strip()
            if alt_text:
                variants.append(alt_text)

    return _dedupe_preserve_order(variants)


_NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)


def extract_final_answer(text: Any, query_text: Any = None) -> str:
    raw = str(text or "").replace("\r", " ").strip()
    if not raw:
        return ""
    query = str(query_text or "").strip().lower()

    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    answer = ""
    for line in lines:
        if re.match(r"^(final answer|answer)\s*[:\-]?\s*$", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\[[A-Za-z_ ]+\]$", line):
            continue
        answer = line
        break
    if not answer:
        answer = lines[0] if lines else raw
    answer = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE)

    if re.match(r"^insufficient information\b", answer, flags=re.IGNORECASE):
        return "insufficient information"

    yes_no = re.match(r"^(yes|no)\b", answer, flags=re.IGNORECASE)
    if yes_no:
        return yes_no.group(1).lower()

    for pattern in [
        r"^(?:the answer is|answer is|the answer to .*? is|the answer)\s*[:\-]?\s*(.+)$",
        r"^(?:it is|its)\s+(.+)$",
    ]:
        match = re.match(pattern, answer, flags=re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            break

    for marker in [
        " Answer:",
        " The answer is",
        " The user ",
        " However, ",
        " Therefore, ",
        " Wait, ",
        " Note: ",
    ]:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    if query.startswith("who "):
        for pattern in [
            r"^(.+?)\s+(?:gave|recommended|bought|sent|made|ordered|told|invited|brought|wrote|attended)\b",
            r"^(.+?)\s+(?:is|was)\b",
        ]:
            match = re.match(pattern, answer, flags=re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break
    elif query.startswith("how many ") or query.startswith("how much ") or query.startswith("how long "):
        quantity_patterns = [
            r"^((?:[0-9]+(?:\.[0-9]+)?|" + "|".join(_NUMBER_WORDS) + r")(?:\s+[A-Za-z]+){0,3})\b",
            r"\b(?:has|had|is|was|were|are|ordered|wakes up|wake up|for)\s+((?:[0-9]+(?:\.[0-9]+)?|" + "|".join(_NUMBER_WORDS) + r")(?:\s+[A-Za-z]+){0,3})\b",
        ]
        for pattern in quantity_patterns:
            match = re.search(pattern, answer, flags=re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break
    elif query.startswith("when ") or "what year" in query or query.startswith("what time"):
        concise_temporal = answer.split()
        if len(concise_temporal) <= 6 and re.search(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|yesterday|today|tomorrow|last|next|week|month|year)\b|[0-9]",
            answer,
            flags=re.IGNORECASE,
        ):
            return answer.strip(" \t\n\r\"'")
        for pattern in [
            r"\bon\s+([0-9]{1,2}\s+[A-Za-z]+,?\s+[0-9]{4})\b",
            r"\b([A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})\b",
            r"\b([0-9]{1,2}:[0-9]{2}\s*(?:am|pm))\b",
            r"\b([A-Za-z]+\s+[0-9]{4})\b",
            r"\b([0-9]{4})\b",
        ]:
            match = re.search(pattern, answer, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        match = re.search(
            r"\b([0-9]{1,2}:[0-9]{2}\s*(?:am|pm)|[A-Za-z]+(?:\s+[A-Za-z]+)?\s+[0-9]{4}|[0-9]{4}|[A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4}|[A-Za-z]+\s+[0-9]{1,2}(?:st|nd|rd|th)?(?:,\s*[0-9]{4})?)\b",
            answer,
            flags=re.IGNORECASE,
        )
        if match:
            answer = match.group(1).strip()
    elif (
        query.startswith("where ")
        or query.startswith("what ")
        or "what was the name" in query
        or "remind me of the name" in query
    ):
        for pattern in [r"\b(?:is|was|were|are)\b\s+(.+)$"]:
            match = re.search(pattern, answer, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    answer = candidate
                    break

    if " me " in f" {query} " or query.startswith("who ") or query.startswith("what "):
        answer = re.sub(r"^the user's\b", "my", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^the user'?s\b", "my", answer, flags=re.IGNORECASE)
        answer = re.sub(r"^the user\b", "I", answer, flags=re.IGNORECASE)

    answer = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", answer, maxsplit=1)[0].strip()
    return answer.strip(" \t\n\r\"'")


def exact_match(prediction: Any, reference: Any) -> float:
    pred = normalize_text(prediction)
    for variant in reference_variants(reference):
        if pred == normalize_text(variant):
            return 1.0
    return 0.0


def token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = normalize_tokens(prediction)
    best = 0.0
    for variant in reference_variants(reference):
        ref_tokens = normalize_tokens(variant)
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            continue
        overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
        if overlap <= 0:
            continue
        precision = overlap / float(len(pred_tokens))
        recall = overlap / float(len(ref_tokens))
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def contains_reference(prediction: Any, reference: Any) -> float:
    pred = normalize_text(prediction)
    if not pred:
        return 0.0
    for variant in reference_variants(reference):
        ref = normalize_text(variant)
        if ref and (ref in pred or pred in ref):
            return 1.0
    return 0.0


def insufficiency_rate(prediction: Any) -> float:
    pred = str(prediction or "").strip().lower()
    return 1.0 if pred.startswith("insufficient information") else 0.0


def score_prediction(prediction: Any, reference: Any, query_text: Any = None) -> dict[str, float]:
    extracted = extract_final_answer(prediction, query_text=query_text)
    return {
        "extracted_answer": extracted,
        "answer_exact_match": exact_match(prediction, reference),
        "answer_extracted_exact_match": exact_match(extracted, reference),
        "answer_token_f1": token_f1(prediction, reference),
        "answer_extracted_token_f1": token_f1(extracted, reference),
        "answer_contains_reference": contains_reference(prediction, reference),
        "answer_extracted_contains_reference": contains_reference(extracted, reference),
        "answer_insufficient_information_rate": insufficiency_rate(prediction),
    }
