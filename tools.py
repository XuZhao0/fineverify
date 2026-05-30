"""Shared helpers for the FineVerify OpenAI and Gemini pipelines.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from .prompts import EXTRACT_GRADE_W_EXP, VERIFICATION_PROMPT_WEB, format_query
except ImportError:  # pragma: no cover - supports direct script execution.
    from prompts import EXTRACT_GRADE_W_EXP, VERIFICATION_PROMPT_WEB, format_query


load_dotenv()

VALID_JUDGMENTS = {"supported", "not_found", "contradicted"}


class GradingResponse(BaseModel):
    extracted_explanation: Optional[str] = Field(
        description="The explanation extracted from the response. Set to null if unavailable."
    )
    extracted_final_answer: Optional[str] = Field(
        description="The exact final answer extracted from the response. Set to null if no explicit answer is found."
    )
    extracted_confidence: Optional[int] = Field(
        description="The confidence score as an integer (0-100). Set to null if unavailable."
    )
    grade: Literal["correct", "incorrect", "not attempted"]
    grade_reasoning: Optional[str] = Field(
        description="A brief explanation of why the grade was assigned based on alignment with the correct answer."
    )


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")


def safe_filename(value: Any, fallback: str = "unknown") -> str:
    text = str(value) if value not in (None, "") else fallback
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip().replace("%", ""))
        except ValueError:
            return default
    return default


def parse_jsonl_line(line: str, line_num: int, path: Path) -> Optional[dict]:
    text = line.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        candidates = []
        if text.startswith("{{") and text.endswith("}}"):
            candidates.append(text[1:-1])
        if not text.startswith("{"):
            candidates.append("{" + text)
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Failed to parse JSONL line {line_num} in {path}")


def load_jsonl(path: Path | str) -> List[dict]:
    jsonl_path = Path(path)
    rows: List[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            row = parse_jsonl_line(line, idx, jsonl_path)
            if row is not None:
                rows.append(row)
    return rows


def append_jsonl(path: Path | str, row: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path | str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def first_present(row: dict, keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def get_query_id(row: dict, fallback: Any = None) -> Any:
    return first_present(row, ("query_id", "id"), fallback)


def get_question(row: dict) -> str:
    return str(
        first_present(row, ("problem", "question", "query", "query_text"), "")
    )


def get_answer(row: dict) -> str:
    return str(first_present(row, ("answer", "correct_answer"), ""))


def validate_threads(value: Any) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("Minimum threads is 1.")
    if ivalue > 100:
        raise argparse.ArgumentTypeError("Maximum recommended threads is 100.")
    return ivalue


def _slice_items(items: List[dict], start: int, end: int) -> List[dict]:
    if end == -1:
        end = len(items)
    return items[start:min(end, len(items))]


def load_decompose_dsqa_jsonl(jsonl_path: Path | str, start: int, end: int) -> List[dict]:
    """Load DSQA JSONL records as {id, question, answer} for decomposition."""
    items: List[dict] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                {
                    "id": str(obj["query_id"]),
                    "question": obj["problem"],
                    "answer": obj.get("answer", ""),
                }
            )
    return _slice_items(items, start, end)


def load_decompose_input_jsonl(jsonl_path: Path | str, start: int, end: int) -> List[dict]:
    """Load FineVerify-style JSONL records as {id, question, answer}."""
    items: List[dict] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = first_present(obj, ("query_id", "id"), line_idx)
            question = first_present(obj, ("problem", "question", "query", "query_text"), "")
            answer = first_present(obj, ("answer", "correct_answer"), "")
            if not question:
                raise ValueError(
                    f"Missing question/problem/query_text at input line {line_idx}"
                )
            if not answer:
                raise ValueError(f"Missing answer/correct_answer at input line {line_idx}")
            item = {
                "id": str(qid),
                "question": str(question),
                "answer": str(answer),
            }
            positives = obj.get("positives_for_query")
            if positives is not None:
                item["positives_for_query"] = positives
            items.append(item)
    return _slice_items(items, start, end)


def load_decompose_processed_ids(output_path: Path | str) -> set[str]:
    """Return ids already present in an existing decomposition output JSONL."""
    processed: set[str] = set()
    path = Path(output_path)
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = first_present(obj, ("id", "query_id"))
            except (json.JSONDecodeError, AttributeError):
                continue
            if qid not in (None, ""):
                processed.add(str(qid))
    return processed


def strip_code_fences(text: str) -> str:
    """Remove wrapping Markdown code fences while preserving inner content."""
    lines = text.strip().splitlines()
    if lines and lines[0].strip() == "```":
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_subquestion_items(response_text: str) -> Tuple[str, List[str]]:
    """Extract bullet subquestions and their count from model text output."""
    subquestions_text = (
        response_text.split("Checkable subquestion list:")[-1].strip()
        if "Checkable subquestion list:" in response_text
        else response_text.strip()
    )

    items: List[str] = []
    for line in subquestions_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            item = stripped[1:].strip()
            if item:
                items.append(item)

    return subquestions_text, items


def extract_decomposition_parts(response_text: str) -> Tuple[str, str, List[str]]:
    """Extract the instantiated claim and bullet statements from model text output."""
    cleaned_text = strip_code_fences(response_text)

    claim_label = "Instantiated claim:"
    statements_label = "Checkable statements list:"
    legacy_statements_label = "Checkable subquestion list:"

    statements_start = cleaned_text.find(statements_label)
    active_statements_label = statements_label
    if statements_start == -1:
        statements_start = cleaned_text.find(legacy_statements_label)
        active_statements_label = legacy_statements_label

    instantiated_claim = ""
    claim_start = cleaned_text.find(claim_label)
    if claim_start != -1:
        claim_end = (
            statements_start
            if statements_start != -1 and statements_start > claim_start
            else len(cleaned_text)
        )
        instantiated_claim = strip_code_fences(
            cleaned_text[claim_start + len(claim_label):claim_end]
        )

    if statements_start != -1:
        subquestions_text = cleaned_text[
            statements_start + len(active_statements_label):
        ].strip()
    else:
        subquestions_text = cleaned_text.strip()

    items: List[str] = []
    for line in subquestions_text.splitlines():
        stripped = line.strip()
        if stripped == "```":
            continue
        if stripped.startswith("-"):
            item = stripped[1:].strip()
            if item:
                items.append(item)

    return instantiated_claim, subquestions_text, items


def extract_openai_decompose_response(response: Any) -> Tuple[str, Any]:
    model_output: List[dict] = []
    for item in getattr(response, "output", []) or []:
        if hasattr(item, "model_dump"):
            model_output.append(item.model_dump(mode="python"))
        elif isinstance(item, dict):
            model_output.append(item)

    reasoning_summary = next(
        (item.get("summary", "") for item in model_output if item.get("type") == "reasoning"),
        "",
    )
    msg_item = next((item for item in model_output if item.get("type") == "message"), {})
    text_chunks = [
        part.get("text", "")
        for part in msg_item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "output_text"
    ]
    response_text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
    return response_text, reasoning_summary


def extract_openai_decompose_usage(response: Any) -> Tuple[int, int, int, int]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None) if usage else None
    output_details = getattr(usage, "output_tokens_details", None) if usage else None
    return (
        safe_int(getattr(usage, "input_tokens", 0) if usage else 0),
        safe_int(getattr(input_details, "cached_tokens", 0) if input_details else 0),
        safe_int(getattr(usage, "output_tokens", 0) if usage else 0),
        safe_int(
            getattr(output_details, "reasoning_tokens", 0) if output_details else 0
        ),
    )


def index_by_query_id(rows: Iterable[dict]) -> Dict[str, dict]:
    indexed: Dict[str, dict] = {}
    for row in rows:
        qid = get_query_id(row)
        if qid is not None:
            indexed[str(qid)] = row
    return indexed


def load_ground_truth_jsonl(path: Path | str) -> Dict[str, Dict[str, str]]:
    ground_truth: Dict[str, Dict[str, str]] = {}
    for row in load_jsonl(path):
        qid = get_query_id(row)
        if qid is None:
            continue
        ground_truth[str(qid)] = {
            "question": get_question(row),
            "answer": get_answer(row),
        }
    return ground_truth


def extract_output_text_from_item(item: Any) -> str:
    chunks: List[str] = []
    for part in getattr(item, "content", []) or []:
        if getattr(part, "type", None) == "output_text":
            text = getattr(part, "text", "")
            if text:
                chunks.append(str(text))
        elif isinstance(part, dict) and part.get("type") == "output_text":
            text = part.get("text", "")
            if text:
                chunks.append(str(text))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def extract_annotations_from_item(item: Any) -> List[dict]:
    annotations: List[dict] = []
    for part in getattr(item, "content", []) or []:
        for annotation in getattr(part, "annotations", []) or []:
            if hasattr(annotation, "model_dump"):
                annotations.append(annotation.model_dump(mode="python"))
            else:
                annotations.append(getattr(annotation, "__dict__", str(annotation)))
    return annotations


def normalize_response_output(response: Any) -> List[dict]:
    normalized: List[dict] = []
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            normalized.append(
                {
                    "type": "reasoning",
                    "output": getattr(item, "summary", None),
                }
            )
        elif item_type == "web_search_call":
            action = getattr(item, "action", None)
            normalized.append(
                {
                    "type": "web_search_call",
                    "action_type": getattr(action, "type", None) if action else None,
                    "query": getattr(action, "query", None) if action else None,
                    "queries": getattr(action, "queries", None) if action else None,
                    "domains": getattr(action, "domains", None) if action else None,
                    "sources": getattr(action, "sources", None) if action else None,
                    "status": getattr(item, "status", None),
                }
            )
        elif item_type == "message":
            normalized.append(
                {
                    "type": "message",
                    "output": extract_output_text_from_item(item),
                    "annotations": extract_annotations_from_item(item),
                }
            )
    return normalized


def serialize_response_output(response: Any) -> List[dict]:
    serialized: List[dict] = []
    for item in getattr(response, "output", []) or []:
        if hasattr(item, "model_dump"):
            serialized.append(item.model_dump(mode="python"))
        elif isinstance(item, dict):
            serialized.append(item)
        else:
            serialized.append(getattr(item, "__dict__", str(item)))
    return serialized


def get_text_output(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            text = extract_output_text_from_item(item)
            if text:
                chunks.append(text)
            continue

        item_dict = item.model_dump(mode="python") if hasattr(item, "model_dump") else item
        if isinstance(item_dict, dict) and item_dict.get("type") == "message":
            for part in item_dict.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        chunks.append(text)
    return "\n\n".join(chunks).strip()


def count_web_search_calls(response: Any) -> int:
    count = 0
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "web_search_call":
            continue
        if getattr(getattr(item, "action", None), "type", None) == "search":
            count += 1
    return count


def usage_from_response(response: Any, web_search_calls: int = 0) -> dict:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None) if usage else None
    output_details = getattr(usage, "output_tokens_details", None) if usage else None
    input_tokens = safe_int(getattr(usage, "input_tokens", 0) if usage else 0)
    output_tokens = safe_int(getattr(usage, "output_tokens", 0) if usage else 0)
    cached_tokens = safe_int(getattr(input_details, "cached_tokens", 0) if input_details else 0)
    reasoning_tokens = safe_int(
        getattr(output_details, "reasoning_tokens", 0) if output_details else 0
    )
    return {
        "input_tokens": input_tokens,
        "input_tokens_cached": cached_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": safe_int(getattr(usage, "total_tokens", 0) if usage else 0),
        "web_search_calls": safe_int(web_search_calls),
        "response_id": getattr(response, "id", None),
        "response_status": getattr(response, "status", None),
    }


def add_usage(total: dict, usage: Optional[dict]) -> None:
    if not usage:
        return
    for key in (
        "input_tokens",
        "input_tokens_cached",
        "output_tokens",
        "reasoning_tokens",
        "web_search_calls",
        "total_tokens",
    ):
        total[key] = safe_int(total.get(key, 0)) + safe_int(usage.get(key, 0))


def calculate_cost(
    model_name: str,
    input_tokens: int,
    cached_input: int,
    output_tokens: int,
    web_search_calls: int = 0,
) -> float:
    pricing = {
        "gpt-5.2": (1.75, 0.175, 14.00),
        "gpt-5.1": (1.25, 0.125, 10.00),
        "gpt-5": (1.25, 0.125, 10.00),
        "gpt-5-mini": (0.25, 0.025, 2.00),
        "gpt-5.4-mini": (0.75, 0.075, 4.50),
        "gemini-3-flash-preview": (0.50, 0.05, 3.00),
    }
    rates = pricing.get(model_name, (0.0, 0.0, 0.0))
    cached_input = cached_input or 0
    non_cached_input = max(0, input_tokens - cached_input)
    token_cost = (
        (non_cached_input / 1_000_000) * rates[0]
        + (cached_input / 1_000_000) * rates[1]
        + (output_tokens / 1_000_000) * rates[2]
    )
    search_cost = (web_search_calls / 1000) * 10.0
    return token_cost + search_cost


def reasoning_payload(reasoning_effort: Optional[str]) -> Optional[dict]:
    if not reasoning_effort or str(reasoning_effort).lower() in {"none", "null"}:
        return None
    return {"effort": reasoning_effort, "summary": "detailed"}


def run_search_candidate(
    client: Any,
    *,
    query_id: Any,
    question: str,
    correct_answer: Optional[str],
    model: str,
    reasoning_effort: Optional[str],
    query_template: Optional[str],
    max_output_tokens: Optional[int],
) -> dict:
    formatted = format_query(question, query_template)
    body: dict = {
        "model": model,
        "include": ["web_search_call.action.sources"],
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "input": formatted,
        "truncation": "auto",
    }
    reasoning = reasoning_payload(reasoning_effort)
    if reasoning:
        body["reasoning"] = reasoning
    if max_output_tokens:
        body["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**body)
    web_search_calls = count_web_search_calls(response)
    usage = usage_from_response(response, web_search_calls=web_search_calls)
    output_text = get_text_output(response)

    return {
        "query_id": query_id,
        "query_text": question,
        "answer": correct_answer,
        "output_text": output_text,
        "usage": usage,
        "request_body": body,
        "result": normalize_response_output(response),
        "raw_output": serialize_response_output(response),
    }


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return EXTRACT_GRADE_W_EXP.format(
        question=question, response=response, correct_answer=correct_answer
    )


def call_openai_judge(
    client: Any,
    *,
    prompt: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
) -> Any:
    body: dict = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": prompt,
        "text_format": GradingResponse,
    }
    reasoning = reasoning_payload(reasoning_effort)
    if reasoning:
        body["reasoning"] = reasoning
    return client.responses.parse(**body)


def call_judge_with_retry(
    client: Any,
    *,
    prompt: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    max_attempts: int = 4,
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_openai_judge(
                client=client,
                prompt=prompt,
                model=model,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(attempt, 3))
    raise RuntimeError(f"Judge failed after {max_attempts} attempts: {last_error}")


def normalize_extracted_explanation(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    marker = "Explanation:"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text or None


def parse_judge_response(response_parsed: Any) -> dict:
    result = {
        "extracted_explanation": None,
        "extracted_final_answer": None,
        "extracted_confidence": None,
        "grade": None,
        "grade_reasoning": None,
        "parse_error": False,
    }
    try:
        result["extracted_explanation"] = normalize_extracted_explanation(
            response_parsed.extracted_explanation
        )
        result["extracted_final_answer"] = response_parsed.extracted_final_answer
        result["extracted_confidence"] = response_parsed.extracted_confidence
        result["grade"] = response_parsed.grade
        result["grade_reasoning"] = response_parsed.grade_reasoning
    except Exception:
        result["parse_error"] = True
    return result


def evaluate_candidate(
    client: Any,
    *,
    search_record: dict,
    correct_answer: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    max_attempts: int = 4,
) -> Tuple[dict, dict]:
    query_id = search_record.get("query_id")
    question = str(search_record.get("query_text", ""))
    response_text = str(search_record.get("output_text", "") or "")
    search_usage = search_record.get("usage", {}) if isinstance(search_record.get("usage"), dict) else {}

    if not response_text.strip():
        judge_result = {
            "extracted_explanation": None,
            "extracted_final_answer": None,
            "extracted_confidence": None,
            "grade": "not attempted",
            "grade_reasoning": "Response is empty",
            "parse_error": True,
        }
        eval_row = {
            "source_line_idx": search_record.get("source_line_idx"),
            "query_id": query_id,
            "query_text": question,
            "response": response_text,
            "correct_answer": correct_answer,
            "extracted_final_answer": None,
            "extracted_confidence": None,
            "extracted_explanation": None,
            "grade": "not attempted",
            "grade_reasoning": "Response is empty",
            "web_search_calls": safe_int(search_usage.get("web_search_calls", 0)),
            "output_tokens": safe_int(search_usage.get("output_tokens", 0)),
            "reasoning_tokens": safe_int(search_usage.get("reasoning_tokens", 0)),
        }
        return eval_row, {
            "query_id": query_id,
            "query_text": question,
            "response": response_text,
            "correct_answer": correct_answer,
            "judge_result": judge_result,
            "model_info": {"judge_model": model, "skipped": True},
        }

    judge_prompt = create_judge_prompt(question, response_text, correct_answer)
    judge_response = call_judge_with_retry(
        client=client,
        prompt=judge_prompt,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        max_attempts=max_attempts,
    )
    parsed = getattr(judge_response, "output_parsed", None)
    judge_result = parse_judge_response(parsed)
    judge_usage = usage_from_response(judge_response, web_search_calls=0)
    judge_response_text = get_text_output(judge_response)

    eval_row = {
        "source_line_idx": search_record.get("source_line_idx"),
        "query_id": query_id,
        "query_text": question,
        "response": response_text,
        "correct_answer": correct_answer,
        "extracted_final_answer": judge_result.get("extracted_final_answer"),
        "extracted_confidence": judge_result.get("extracted_confidence"),
        "extracted_explanation": judge_result.get("extracted_explanation"),
        "grade": judge_result.get("grade"),
        "grade_reasoning": judge_result.get("grade_reasoning"),
        "web_search_calls": safe_int(search_usage.get("web_search_calls", 0)),
        "output_tokens": safe_int(search_usage.get("output_tokens", 0)),
        "reasoning_tokens": safe_int(search_usage.get("reasoning_tokens", 0)),
        "judge_usage": judge_usage,
    }
    raw_payload = {
        "query_id": query_id,
        "query_text": question,
        "response": response_text,
        "correct_answer": correct_answer,
        "judge_prompt": judge_prompt,
        "judge_response": judge_response_text,
        "judge_result": judge_result,
        "model_info": {
            "judge_model": model,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "judge_usage": judge_usage,
        },
    }
    return eval_row, raw_payload


def parse_subquestions(raw_subquestions: Any) -> List[str]:
    if raw_subquestions is None:
        return []
    if isinstance(raw_subquestions, list):
        return [str(x).strip() for x in raw_subquestions if str(x).strip()]
    if not isinstance(raw_subquestions, str):
        return []

    items: List[str] = []
    for line in raw_subquestions.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            content = stripped[1:].strip()
            if content:
                items.append(content)
    if not items:
        items = [ln.strip() for ln in raw_subquestions.splitlines() if ln.strip()]
    return items


def get_subquestions_from_row(row: Optional[dict]) -> Tuple[List[str], int]:
    if not row:
        return [], 0
    raw = row.get("subquestions_text", row.get("subquestions"))
    subquestions = parse_subquestions(raw)
    expected_count = safe_int(row.get("subquestion_count"), len(subquestions))
    if expected_count <= 0:
        expected_count = len(subquestions)
    return subquestions, expected_count


def format_subquestions_for_prompt(subquestions: List[str]) -> str:
    if not subquestions:
        return "1. None"
    return "\n".join(f"{idx}. {sq}" for idx, sq in enumerate(subquestions, start=1))


def normalize_label(label: str) -> Optional[str]:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in VALID_JUDGMENTS:
        return normalized
    return None


def normalize_judgment(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return normalize_label(value)


def extract_judgments(output_text: str) -> List[str]:
    judgments: List[str] = []
    for line in output_text.splitlines():
        if "judgment" not in line.lower():
            continue
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
        candidate = parts[1].strip()
        if not candidate:
            continue
        token = re.split(r"[\s,|;]+", candidate)[0]
        normalized = normalize_label(token)
        if normalized:
            judgments.append(normalized)
    return judgments


def extract_statement_texts(output_text: str) -> List[str]:
    texts: List[str] = []
    current_block: List[str] = []
    in_block = False

    def flush_block() -> None:
        nonlocal current_block, in_block
        if not current_block:
            in_block = False
            return
        for raw_line in current_block:
            stripped = raw_line.strip()
            if stripped.startswith("- Statement text:") or stripped.startswith("- Subquestion text:"):
                _, _, value = stripped.partition(":")
                text = value.strip()
                if text.startswith('"') and text.endswith('"') and len(text) >= 2:
                    text = text[1:-1]
                elif text.startswith('"'):
                    text = text[1:]
                if text:
                    texts.append(text)
                break
        current_block = []
        in_block = False

    for line in output_text.splitlines():
        stripped = line.strip()
        if re.match(r"^(Statement|Subquestion)\s+\d+\s*:\s*$", stripped):
            flush_block()
            in_block = True
            current_block = []
            continue
        if stripped.lower().startswith("overall assessment:"):
            flush_block()
            break
        if in_block:
            current_block.append(line)

    flush_block()
    return texts


def get_extracted_explanation(eval_row: dict) -> str:
    value = eval_row.get("extracted_explanation")
    if value is None:
        value = eval_row.get("extrated_explanation")
    if value is None:
        value = ""
    return str(value)


def compute_average_score(
    judgments: Any,
    *,
    score_supported: float,
    score_not_found: float,
    score_contradicted: float,
) -> float:
    if not isinstance(judgments, list):
        return 0.0

    score_map = {
        "supported": float(score_supported),
        "not_found": float(score_not_found),
        "contradicted": float(score_contradicted),
    }

    total = 0.0
    count = 0
    for judgment in judgments:
        normalized = normalize_judgment(judgment)
        if normalized is None:
            continue
        total += score_map[normalized]
        count += 1
    return total / count if count else 0.0


def build_verification_prompt(
    *,
    question: str,
    subquestions: List[str],
    candidate_answer: str,
    explanation: str,
    expected_count: int,
    attempt_idx: int,
) -> str:
    base = VERIFICATION_PROMPT_WEB.format(
        QUESTION=question,
        SUBQUESTIONS=format_subquestions_for_prompt(subquestions),
        CANDIDATE_ANSWER=candidate_answer,
        EXPLANATION=explanation,
    )
    if attempt_idx <= 2:
        return base
    return (
        f"{base}\n\n"
        "# Additional instruction:\n"
        "- You MUST evaluate ALL statements.\n"
        f"- You should output approximately {expected_count} judgment lines (one per statement).\n"
        "- Keep strict output format and ensure each statement has exactly one `Judgment:` line."
    )


def call_verification_api(
    client: Any,
    *,
    formatted_query: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
) -> dict:
    body: dict = {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "input": [{"role": "user", "content": formatted_query}],
        "include": ["web_search_call.action.sources"],
        "tools": [{"type": "web_search"}],
        "truncation": "auto",
    }
    reasoning = reasoning_payload(reasoning_effort)
    if reasoning:
        body["reasoning"] = reasoning

    response = client.responses.create(**body)
    output_text = get_text_output(response)
    web_search_calls = count_web_search_calls(response)
    usage = usage_from_response(response, web_search_calls=web_search_calls)

    return {
        "request_body": body,
        "response_id": getattr(response, "id", None),
        "response_status": getattr(response, "status", None),
        "output_text": output_text,
        "web_search_calls": web_search_calls,
        "usage": usage,
        "result": normalize_response_output(response),
        "raw_output": serialize_response_output(response),
    }


def build_skipped_verification_row(
    *,
    eval_row: dict,
    subquestions: List[str],
    expected_count: int,
    reason: str,
    note: str,
) -> dict:
    return {
        "query_id": eval_row.get("query_id"),
        "query_text": eval_row.get("query_text"),
        "correct_answer": eval_row.get("correct_answer"),
        "extracted_final_answer": eval_row.get("extracted_final_answer"),
        "grade": eval_row.get("grade"),
        "extracted_confidence": eval_row.get("extracted_confidence"),
        "extracted_explanation": get_extracted_explanation(eval_row),
        "subquestion_count": expected_count,
        "subquestions": subquestions,
        "subquestion_judgments": [],
        "average_score": 0.0,
        "web_search_calls": 0,
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "note": f"{reason}; {note}",
    }


def verify_candidate(
    client: Any,
    *,
    eval_row: dict,
    decomp_row: Optional[dict],
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    max_retries: int,
    count_tolerance: int,
    score_supported: float,
    score_not_found: float,
    score_contradicted: float,
) -> Tuple[dict, dict]:
    qid = eval_row.get("query_id")
    question = str(eval_row.get("query_text", ""))
    grade = str(eval_row.get("grade", "")).strip().lower()
    subquestions, expected_count = get_subquestions_from_row(decomp_row)
    score_note = (
        f"score function: {score_supported:g}, {score_not_found:g}, "
        f"{score_contradicted:g}"
    )

    if grade == "not attempted":
        verified_row = build_skipped_verification_row(
            eval_row=eval_row,
            subquestions=subquestions,
            expected_count=expected_count,
            reason="skipped due to abstention",
            note=score_note,
        )
        return verified_row, {
            "metadata": {"status": "skipped", "reason": "grade is not attempted"},
            "query_id": qid,
            "query_text": question,
            "verified_record": verified_row,
        }

    if not subquestions:
        verified_row = build_skipped_verification_row(
            eval_row=eval_row,
            subquestions=[],
            expected_count=0,
            reason="missing decomposition record",
            note=score_note,
        )
        return verified_row, {
            "metadata": {"status": "failed", "reason": "missing decomposition row"},
            "query_id": qid,
            "query_text": question,
            "verified_record": verified_row,
        }

    attempts_raw: List[dict] = []
    total_usage = {
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "web_search_calls": 0,
        "total_tokens": 0,
    }
    final_judgments: List[str] = []
    final_output_text = ""
    count_match = False

    for attempt_idx in range(1, max_retries + 1):
        formatted_query = build_verification_prompt(
            question=question,
            subquestions=subquestions,
            candidate_answer=str(eval_row.get("extracted_final_answer") or ""),
            explanation=get_extracted_explanation(eval_row),
            expected_count=expected_count,
            attempt_idx=attempt_idx,
        )
        attempt_result = call_verification_api(
            client=client,
            formatted_query=formatted_query,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        output_text = attempt_result["output_text"]
        judgments = extract_judgments(output_text)
        statement_texts = extract_statement_texts(output_text)
        usage = attempt_result["usage"]
        add_usage(total_usage, usage)

        attempts_raw.append(
            {
                "attempt": attempt_idx,
                "request_body": attempt_result["request_body"],
                "response_id": attempt_result["response_id"],
                "response_status": attempt_result["response_status"],
                "web_search_calls": attempt_result["web_search_calls"],
                "usage": usage,
                "result": attempt_result["result"],
                "raw_output": attempt_result["raw_output"],
                "output_text": output_text,
                "subquestions_extracted": statement_texts,
                "subquestion_judgments": judgments,
            }
        )

        final_judgments = judgments
        final_output_text = output_text

        if abs(len(judgments) - expected_count) <= count_tolerance:
            count_match = True
            break

    avg_score = compute_average_score(
        final_judgments,
        score_supported=score_supported,
        score_not_found=score_not_found,
        score_contradicted=score_contradicted,
    )
    note = score_note
    if not count_match:
        note = (
            f"judgment count mismatch after retries: expected {expected_count}, "
            f"got {len(final_judgments)}; {score_note}"
        )

    verified_row = {
        "query_id": qid,
        "query_text": question,
        "correct_answer": eval_row.get("correct_answer"),
        "extracted_final_answer": eval_row.get("extracted_final_answer"),
        "grade": eval_row.get("grade"),
        "extracted_confidence": eval_row.get("extracted_confidence"),
        "extracted_explanation": get_extracted_explanation(eval_row),
        "subquestion_count": expected_count,
        "subquestions": subquestions,
        "subquestion_judgments": final_judgments,
        "average_score": avg_score,
        "web_search_calls": total_usage["web_search_calls"],
        "input_tokens": total_usage["input_tokens"],
        "input_tokens_cached": total_usage["input_tokens_cached"],
        "output_tokens": total_usage["output_tokens"],
        "reasoning_tokens": total_usage["reasoning_tokens"],
        "note": note,
    }
    raw_payload = {
        "metadata": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "max_retries": max_retries,
            "count_tolerance": count_tolerance,
            "score_function": {
                "supported": score_supported,
                "not_found": score_not_found,
                "contradicted": score_contradicted,
            },
        },
        "query_id": qid,
        "query_text": question,
        "correct_answer": eval_row.get("correct_answer"),
        "extracted_final_answer": eval_row.get("extracted_final_answer"),
        "grade": eval_row.get("grade"),
        "subquestion_count": expected_count,
        "subquestions": subquestions,
        "attempts": attempts_raw,
        "final_output_text": final_output_text,
        "verified_record": verified_row,
    }
    return verified_row, raw_payload


def normalize_answer_for_selection(
    record: Optional[dict],
    *,
    normalize_not_attempted_to_null: bool,
) -> Any:
    if record is None:
        return None
    if normalize_not_attempted_to_null:
        grade = record.get("grade")
        if isinstance(grade, str) and grade.strip().lower() == "not attempted":
            return None
    extracted = record.get("extracted_final_answer")
    if isinstance(extracted, str):
        extracted = extracted.strip()
    return extracted


def pick_best_verified_record(
    records: List[dict],
    *,
    normalize_not_attempted_to_null: bool = True,
) -> Tuple[Optional[dict], Any, float, int]:
    best_idx = 0
    best_score = float("-inf")
    for idx, record in enumerate(records):
        score = parse_float(record.get("average_score"))
        if score > best_score:
            best_score = score
            best_idx = idx
    if not records:
        return None, None, 0.0, 0
    best_record = records[best_idx]
    return (
        best_record,
        normalize_answer_for_selection(
            best_record,
            normalize_not_attempted_to_null=normalize_not_attempted_to_null,
        ),
        parse_float(best_record.get("average_score")),
        best_idx + 1,
    )


def calculate_gemini_cost(
    model_name: str,
    input_tokens: int,
    cached_input: int,
    output_tokens: int,
    web_search_calls: int = 0,
) -> float:
    pricing = {
        "gemini-3.1-flash-lite-preview": (0.25, 0.25, 1.50),
        "gemini-3-flash-preview": (0.50, 0.05, 3.00),
        "gemini-2.5-flash": (0.30, 0.03, 2.50),
    }
    rates = pricing.get(model_name, (0.0, 0.0, 0.0))
    cached_input = cached_input or 0
    non_cached_input = max(0, input_tokens - cached_input)
    token_cost = (
        (non_cached_input / 1_000_000) * rates[0]
        + (cached_input / 1_000_000) * rates[1]
        + (output_tokens / 1_000_000) * rates[2]
    )
    search_cost = (web_search_calls / 1_000) * 14.0
    return token_cost + search_cost


def get_gemini_modules() -> Any:
    from google import genai

    return genai


def get_gemini_client(api_key_env: str = "GEMINI_API_KEY") -> Tuple[Any, Any]:
    genai = get_gemini_modules()
    api_key = os.getenv(api_key_env) or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(f"Missing API key. Set {api_key_env} or GOOGLE_API_KEY.")
    return genai.Client(api_key=api_key), genai


def gemini_to_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="python")
    return json.loads(json.dumps(obj, default=str))


def _gemini_get(mapping: dict, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _gemini_first_candidate_parts(response_dict: dict) -> List[dict]:
    candidates = response_dict.get("candidates") or [{}]
    candidate = candidates[0] if candidates else {}
    content = candidate.get("content") or {}
    return content.get("parts", []) or []


def gemini_extract_output_text_from_dict(response_dict: dict) -> str:
    chunks: List[str] = []
    for candidate in response_dict.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if part.get("thought") is True:
                continue
            text = part.get("text")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def extract_gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    texts: List[str] = []
    for part in _gemini_first_candidate_parts(gemini_to_dict(response)):
        part_text = part.get("text", "")
        if part_text and part.get("thought") is not True:
            texts.append(str(part_text).strip())
    return "\n".join(text for text in texts if text).strip()


def gemini_extract_reasoning(response_dict: dict) -> List[str]:
    thoughts: List[str] = []
    for candidate in response_dict.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if part.get("thought") is True and part.get("text"):
                thoughts.append(str(part["text"]))
    return thoughts


def extract_gemini_reasoning_summary(response: Any) -> str:
    return "\n".join(
        str(part["text"]).strip()
        for part in _gemini_first_candidate_parts(gemini_to_dict(response))
        if part.get("thought") is True and part.get("text")
    )


def extract_gemini_decompose_usage(response: Any) -> Tuple[int, int, int, int]:
    response_dict = gemini_to_dict(response)
    usage = _gemini_get(response_dict, "usage_metadata", "usageMetadata", default={}) or {}
    input_tokens = safe_int(_gemini_get(usage, "prompt_token_count", "promptTokenCount"))
    cached_tokens = safe_int(
        _gemini_get(usage, "cached_content_token_count", "cachedContentTokenCount")
    )
    candidate_tokens = safe_int(
        _gemini_get(usage, "candidates_token_count", "candidatesTokenCount")
    )
    thought_tokens = safe_int(
        _gemini_get(usage, "thoughts_token_count", "thoughtsTokenCount")
    )
    output_tokens = candidate_tokens + thought_tokens
    return input_tokens, cached_tokens, output_tokens, thought_tokens


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    return (
        exc.__class__.__name__ in {"RateLimitError", "TooManyRequests"}
        or status_code == 429
    )


def gemini_extract_grounding_metadata(response_dict: dict) -> dict:
    for candidate in response_dict.get("candidates", []) or []:
        metadata = _gemini_get(candidate, "grounding_metadata", "groundingMetadata")
        if metadata:
            return metadata
    return {}


def gemini_extract_search_queries(grounding_metadata: dict) -> List[str]:
    queries = _gemini_get(
        grounding_metadata,
        "web_search_queries",
        "webSearchQueries",
        default=[],
    )
    return [str(q) for q in queries or [] if isinstance(q, str) and q.strip()]


def gemini_extract_grounding_chunks(grounding_metadata: dict) -> List[dict]:
    chunks = _gemini_get(
        grounding_metadata,
        "grounding_chunks",
        "groundingChunks",
        default=[],
    )
    sources: List[dict] = []
    for idx, chunk in enumerate(chunks or []):
        web = chunk.get("web") or {}
        sources.append(
            {
                "index": idx,
                "title": web.get("title"),
                "url": web.get("uri"),
            }
        )
    return sources


def gemini_extract_grounding_supports(grounding_metadata: dict) -> List[dict]:
    supports = _gemini_get(
        grounding_metadata,
        "grounding_supports",
        "groundingSupports",
        default=[],
    )
    normalized: List[dict] = []
    for support in supports or []:
        segment = support.get("segment") or {}
        indices = _gemini_get(
            support,
            "grounding_chunk_indices",
            "groundingChunkIndices",
            default=[],
        )
        normalized.append(
            {
                "segment": {
                    "start_index": _gemini_get(segment, "start_index", "startIndex"),
                    "end_index": _gemini_get(segment, "end_index", "endIndex"),
                    "text": segment.get("text"),
                },
                "grounding_chunk_indices": indices or [],
            }
        )
    return normalized


def gemini_extract_annotations(grounding_metadata: dict) -> List[dict]:
    sources = gemini_extract_grounding_chunks(grounding_metadata)
    supports = gemini_extract_grounding_supports(grounding_metadata)
    annotations: List[dict] = []
    for support in supports:
        segment = support.get("segment") or {}
        for chunk_index in support.get("grounding_chunk_indices") or []:
            if not isinstance(chunk_index, int) or chunk_index >= len(sources):
                continue
            source = sources[chunk_index]
            annotations.append(
                {
                    "type": "url_citation",
                    "start_index": segment.get("start_index"),
                    "end_index": segment.get("end_index"),
                    "text": segment.get("text"),
                    "title": source.get("title"),
                    "url": source.get("url"),
                }
            )
    return annotations


def gemini_count_web_search_calls_from_dict(response_dict: dict) -> int:
    metadata = gemini_extract_grounding_metadata(response_dict)
    queries = gemini_extract_search_queries(metadata)
    return len({query.strip() for query in queries if query.strip()})


def gemini_normalize_response_from_dict(response_dict: dict) -> List[dict]:
    grounding_metadata = gemini_extract_grounding_metadata(response_dict)
    output_text = gemini_extract_output_text_from_dict(response_dict)
    results: List[dict] = []

    reasoning = gemini_extract_reasoning(response_dict)
    if reasoning:
        results.append({"type": "reasoning", "output": reasoning})

    search_queries = gemini_extract_search_queries(grounding_metadata)
    sources = gemini_extract_grounding_chunks(grounding_metadata)
    supports = gemini_extract_grounding_supports(grounding_metadata)
    if search_queries or sources:
        results.append(
            {
                "type": "web_search_call",
                "action_type": "google_search",
                "query": search_queries[0] if search_queries else None,
                "queries": search_queries,
                "domains": None,
                "sources": sources,
                "grounding_supports": supports,
                "status": "completed",
            }
        )

    results.append(
        {
            "type": "message",
            "output": output_text,
            "annotations": gemini_extract_annotations(grounding_metadata),
        }
    )
    return results


def gemini_usage_from_dict(response_dict: dict, web_search_calls: int) -> dict:
    usage_metadata = _gemini_get(
        response_dict, "usage_metadata", "usageMetadata", default={}
    ) or {}
    input_tokens = safe_int(
        _gemini_get(usage_metadata, "prompt_token_count", "promptTokenCount")
    )
    cached = safe_int(
        _gemini_get(
            usage_metadata,
            "cached_content_token_count",
            "cachedContentTokenCount",
        )
    )
    output_tokens = safe_int(
        _gemini_get(usage_metadata, "candidates_token_count", "candidatesTokenCount")
    )
    reasoning_tokens = safe_int(
        _gemini_get(usage_metadata, "thoughts_token_count", "thoughtsTokenCount")
    )
    total_tokens = safe_int(
        _gemini_get(usage_metadata, "total_token_count", "totalTokenCount")
    )
    return {
        "input_tokens": input_tokens,
        "input_tokens_cached": cached,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "web_search_calls": web_search_calls,
        "response_id": response_dict.get("response_id"),
        "response_status": gemini_status_from_dict(response_dict),
    }


def gemini_status_from_dict(response_dict: dict) -> Optional[str]:
    candidates = response_dict.get("candidates") or []
    if not candidates:
        return None
    finish_reason = candidates[0].get("finish_reason")
    if finish_reason is None:
        return "completed"
    finish_text = str(finish_reason)
    if finish_text == "STOP" or finish_text.endswith(".STOP"):
        return "completed"
    return finish_text


def gemini_thinking_payload(types_module: Any, reasoning_effort: Optional[str]) -> Optional[Any]:
    if not reasoning_effort or str(reasoning_effort).lower() in {"none", "null"}:
        return None
    return types_module.ThinkingConfig(
        include_thoughts=True,
        thinking_level=reasoning_effort,
    )


def build_gemini_generation_config(
    types_module: Any,
    *,
    max_output_tokens: Optional[int],
    reasoning_effort: Optional[str],
    system_prompt: Optional[str] = None,
    use_google_search: bool = True,
    response_schema: Any = None,
    response_mime_type: Optional[str] = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if use_google_search:
        kwargs["tools"] = [types_module.Tool(google_search=types_module.GoogleSearch())]
    if max_output_tokens:
        kwargs["max_output_tokens"] = max_output_tokens
    if system_prompt:
        kwargs["system_instruction"] = system_prompt
    thinking = gemini_thinking_payload(types_module, reasoning_effort)
    if thinking:
        kwargs["thinking_config"] = thinking
    if response_schema is not None:
        kwargs["response_schema"] = response_schema
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
    return types_module.GenerateContentConfig(**kwargs)


def serializable_gemini_request_body(
    *,
    contents: str,
    model: str,
    max_output_tokens: Optional[int],
    reasoning_effort: Optional[str],
    system_prompt: Optional[str],
    use_google_search: bool,
    response_schema: Any = None,
    response_mime_type: Optional[str] = None,
) -> dict:
    body = {
        "model": model,
        "contents": contents,
    }
    if max_output_tokens:
        body["max_output_tokens"] = max_output_tokens
    if use_google_search:
        body["tools"] = [{"google_search": {}}]
    if reasoning_effort and str(reasoning_effort).lower() not in {"none", "null"}:
        body["thinking_config"] = {
            "include_thoughts": True,
            "thinking_level": reasoning_effort,
        }
    if system_prompt:
        body["system_instruction"] = system_prompt
    if response_mime_type:
        body["response_mime_type"] = response_mime_type
    if response_schema is not None:
        body["response_schema"] = getattr(response_schema, "__name__", str(response_schema))
    return body


async def call_gemini_with_retry(
    request_fn,
    *,
    max_attempts: int,
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await request_fn()
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Gemini request failed after {max_attempts} attempts") from last_error


async def run_gemini_search_candidate(
    client: Any,
    types_module: Any,
    *,
    query_id: Any,
    question: str,
    correct_answer: Optional[str],
    model: str,
    reasoning_effort: Optional[str],
    query_template: Optional[str],
    max_output_tokens: Optional[int],
    system_prompt: Optional[str] = None,
    max_attempts: int = 7,
) -> dict:
    formatted = format_query(question, query_template)
    generation_config = build_gemini_generation_config(
        types_module,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
        use_google_search=True,
    )
    request_body = serializable_gemini_request_body(
        contents=formatted,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
        use_google_search=True,
    )

    async def _request():
        return await client.aio.models.generate_content(
            model=model,
            contents=formatted,
            config=generation_config,
        )

    response = await call_gemini_with_retry(_request, max_attempts=max_attempts)
    response_dict = gemini_to_dict(response)
    web_search_calls = gemini_count_web_search_calls_from_dict(response_dict)
    usage = gemini_usage_from_dict(response_dict, web_search_calls=web_search_calls)
    output_text = gemini_extract_output_text_from_dict(response_dict)

    return {
        "query_id": query_id,
        "query_text": question,
        "answer": correct_answer,
        "output_text": output_text,
        "usage": usage,
        "request_body": request_body,
        "result": gemini_normalize_response_from_dict(response_dict),
        "raw_output": response_dict,
    }


def parse_gemini_grading_response(response: Any, output_text: str) -> GradingResponse:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, GradingResponse):
        return parsed
    if parsed is not None:
        if hasattr(GradingResponse, "model_validate"):
            return GradingResponse.model_validate(parsed)
        return GradingResponse.parse_obj(parsed)

    if hasattr(GradingResponse, "model_validate_json"):
        try:
            return GradingResponse.model_validate_json(output_text)
        except Exception:
            pass
    try:
        return GradingResponse.parse_raw(output_text)
    except Exception:
        data = json.loads(output_text)
        if hasattr(GradingResponse, "model_validate"):
            return GradingResponse.model_validate(data)
        return GradingResponse.parse_obj(data)


async def evaluate_candidate_with_gemini(
    client: Any,
    types_module: Any,
    *,
    search_record: dict,
    correct_answer: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    max_attempts: int = 4,
) -> Tuple[dict, dict]:
    query_id = search_record.get("query_id")
    question = str(search_record.get("query_text", ""))
    response_text = str(search_record.get("output_text", "") or "")
    search_usage = search_record.get("usage", {}) if isinstance(search_record.get("usage"), dict) else {}

    if not response_text.strip():
        judge_result = {
            "extracted_explanation": None,
            "extracted_final_answer": None,
            "extracted_confidence": None,
            "grade": "not attempted",
            "grade_reasoning": "Response is empty",
            "parse_error": True,
        }
        eval_row = {
            "source_line_idx": search_record.get("source_line_idx"),
            "query_id": query_id,
            "query_text": question,
            "response": response_text,
            "correct_answer": correct_answer,
            "extracted_final_answer": None,
            "extracted_confidence": None,
            "extracted_explanation": None,
            "grade": "not attempted",
            "grade_reasoning": "Response is empty",
            "web_search_calls": safe_int(search_usage.get("web_search_calls", 0)),
            "output_tokens": safe_int(search_usage.get("output_tokens", 0)),
            "reasoning_tokens": safe_int(search_usage.get("reasoning_tokens", 0)),
        }
        return eval_row, {
            "query_id": query_id,
            "query_text": question,
            "response": response_text,
            "correct_answer": correct_answer,
            "judge_result": judge_result,
            "model_info": {"judge_model": model, "judge_provider": "gemini", "skipped": True},
        }

    judge_prompt = create_judge_prompt(question, response_text, correct_answer)
    generation_config = build_gemini_generation_config(
        types_module,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        use_google_search=False,
        response_schema=GradingResponse,
        response_mime_type="application/json",
    )
    request_body = serializable_gemini_request_body(
        contents=judge_prompt,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        system_prompt=None,
        use_google_search=False,
        response_schema=GradingResponse,
        response_mime_type="application/json",
    )

    async def _request():
        return await client.aio.models.generate_content(
            model=model,
            contents=judge_prompt,
            config=generation_config,
        )

    response = await call_gemini_with_retry(_request, max_attempts=max_attempts)
    response_dict = gemini_to_dict(response)
    output_text = gemini_extract_output_text_from_dict(response_dict)
    parsed = parse_gemini_grading_response(response, output_text)
    judge_result = parse_judge_response(parsed)
    judge_usage = gemini_usage_from_dict(response_dict, web_search_calls=0)

    eval_row = {
        "source_line_idx": search_record.get("source_line_idx"),
        "query_id": query_id,
        "query_text": question,
        "response": response_text,
        "correct_answer": correct_answer,
        "extracted_final_answer": judge_result.get("extracted_final_answer"),
        "extracted_confidence": judge_result.get("extracted_confidence"),
        "extracted_explanation": judge_result.get("extracted_explanation"),
        "grade": judge_result.get("grade"),
        "grade_reasoning": judge_result.get("grade_reasoning"),
        "web_search_calls": safe_int(search_usage.get("web_search_calls", 0)),
        "output_tokens": safe_int(search_usage.get("output_tokens", 0)),
        "reasoning_tokens": safe_int(search_usage.get("reasoning_tokens", 0)),
        "judge_usage": judge_usage,
    }
    raw_payload = {
        "query_id": query_id,
        "query_text": question,
        "response": response_text,
        "correct_answer": correct_answer,
        "judge_prompt": judge_prompt,
        "judge_request_body": request_body,
        "judge_response": output_text,
        "judge_result": judge_result,
        "model_info": {
            "judge_model": model,
            "judge_provider": "gemini",
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "judge_usage": judge_usage,
        },
    }
    return eval_row, raw_payload


async def call_gemini_verification_api(
    client: Any,
    types_module: Any,
    *,
    formatted_query: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    system_prompt: Optional[str] = None,
    max_attempts: int = 5,
) -> dict:
    generation_config = build_gemini_generation_config(
        types_module,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
        use_google_search=True,
    )
    request_body = serializable_gemini_request_body(
        contents=formatted_query,
        model=model,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        system_prompt=system_prompt,
        use_google_search=True,
    )

    async def _request():
        return await client.aio.models.generate_content(
            model=model,
            contents=formatted_query,
            config=generation_config,
        )

    response = await call_gemini_with_retry(_request, max_attempts=max_attempts)
    response_dict = gemini_to_dict(response)
    output_text = gemini_extract_output_text_from_dict(response_dict)
    web_search_calls = gemini_count_web_search_calls_from_dict(response_dict)
    usage = gemini_usage_from_dict(response_dict, web_search_calls=web_search_calls)

    return {
        "request_body": request_body,
        "response_id": response_dict.get("response_id"),
        "response_status": gemini_status_from_dict(response_dict),
        "output_text": output_text,
        "web_search_calls": web_search_calls,
        "usage": usage,
        "result": gemini_normalize_response_from_dict(response_dict),
        "raw_output": response_dict,
    }


async def verify_candidate_with_gemini(
    client: Any,
    types_module: Any,
    *,
    eval_row: dict,
    decomp_row: Optional[dict],
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    system_prompt: Optional[str],
    max_retries: int,
    request_max_attempts: int,
    count_tolerance: int,
    score_supported: float,
    score_not_found: float,
    score_contradicted: float,
) -> Tuple[dict, dict]:
    qid = eval_row.get("query_id")
    question = str(eval_row.get("query_text", ""))
    grade = str(eval_row.get("grade", "")).strip().lower()
    subquestions, expected_count = get_subquestions_from_row(decomp_row)
    score_note = (
        f"score function: {score_supported:g}, {score_not_found:g}, "
        f"{score_contradicted:g}"
    )

    if grade == "not attempted":
        verified_row = build_skipped_verification_row(
            eval_row=eval_row,
            subquestions=subquestions,
            expected_count=expected_count,
            reason="skipped due to abstention",
            note=score_note,
        )
        return verified_row, {
            "metadata": {"status": "skipped", "reason": "grade is not attempted"},
            "query_id": qid,
            "query_text": question,
            "verified_record": verified_row,
        }

    if not subquestions:
        verified_row = build_skipped_verification_row(
            eval_row=eval_row,
            subquestions=[],
            expected_count=0,
            reason="missing decomposition record",
            note=score_note,
        )
        return verified_row, {
            "metadata": {"status": "failed", "reason": "missing decomposition row"},
            "query_id": qid,
            "query_text": question,
            "verified_record": verified_row,
        }

    attempts_raw: List[dict] = []
    total_usage = {
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "web_search_calls": 0,
        "total_tokens": 0,
    }
    final_judgments: List[str] = []
    final_output_text = ""
    count_match = False

    for attempt_idx in range(1, max_retries + 1):
        formatted_query = build_verification_prompt(
            question=question,
            subquestions=subquestions,
            candidate_answer=str(eval_row.get("extracted_final_answer") or ""),
            explanation=get_extracted_explanation(eval_row),
            expected_count=expected_count,
            attempt_idx=attempt_idx,
        )
        attempt_result = await call_gemini_verification_api(
            client,
            types_module,
            formatted_query=formatted_query,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            system_prompt=system_prompt,
            max_attempts=request_max_attempts,
        )
        output_text = attempt_result["output_text"]
        judgments = extract_judgments(output_text)
        statement_texts = extract_statement_texts(output_text)
        usage = attempt_result["usage"]
        add_usage(total_usage, usage)

        attempts_raw.append(
            {
                "attempt": attempt_idx,
                "request_body": attempt_result["request_body"],
                "response_id": attempt_result["response_id"],
                "response_status": attempt_result["response_status"],
                "web_search_calls": attempt_result["web_search_calls"],
                "usage": usage,
                "result": attempt_result["result"],
                "raw_output": attempt_result["raw_output"],
                "output_text": output_text,
                "subquestions_extracted": statement_texts,
                "subquestion_judgments": judgments,
            }
        )

        final_judgments = judgments
        final_output_text = output_text
        if abs(len(judgments) - expected_count) <= count_tolerance:
            count_match = True
            break

    avg_score = compute_average_score(
        final_judgments,
        score_supported=score_supported,
        score_not_found=score_not_found,
        score_contradicted=score_contradicted,
    )
    note = score_note
    if not count_match:
        note = (
            f"judgment count mismatch after retries: expected {expected_count}, "
            f"got {len(final_judgments)}; {score_note}"
        )

    verified_row = {
        "query_id": qid,
        "query_text": question,
        "correct_answer": eval_row.get("correct_answer"),
        "extracted_final_answer": eval_row.get("extracted_final_answer"),
        "grade": eval_row.get("grade"),
        "extracted_confidence": eval_row.get("extracted_confidence"),
        "extracted_explanation": get_extracted_explanation(eval_row),
        "subquestion_count": expected_count,
        "subquestions": subquestions,
        "subquestion_judgments": final_judgments,
        "average_score": avg_score,
        "web_search_calls": total_usage["web_search_calls"],
        "input_tokens": total_usage["input_tokens"],
        "input_tokens_cached": total_usage["input_tokens_cached"],
        "output_tokens": total_usage["output_tokens"],
        "reasoning_tokens": total_usage["reasoning_tokens"],
        "note": note,
    }
    raw_payload = {
        "metadata": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "max_retries": max_retries,
            "request_max_attempts": request_max_attempts,
            "count_tolerance": count_tolerance,
            "score_function": {
                "supported": score_supported,
                "not_found": score_not_found,
                "contradicted": score_contradicted,
            },
        },
        "query_id": qid,
        "query_text": question,
        "correct_answer": eval_row.get("correct_answer"),
        "extracted_final_answer": eval_row.get("extracted_final_answer"),
        "grade": eval_row.get("grade"),
        "subquestion_count": expected_count,
        "subquestions": subquestions,
        "attempts": attempts_raw,
        "final_output_text": final_output_text,
        "verified_record": verified_row,
    }
    return verified_row, raw_payload
