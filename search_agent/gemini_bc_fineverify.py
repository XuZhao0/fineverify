"""Unified Gemini FineVerify pipeline using custom MCP search tools.

For each input JSONL question, this driver loops up to T rounds:
1. generate a search candidate answer with the search MCP tool,
2. grade the candidate against the provided ground truth,
3. skip verification when the grade is ``not attempted``,
4. otherwise verify the candidate with the verification MCP tool, reusing
   prior round verification when the candidate answer repeats,
5. score those judgments and stop early when the average score reaches 1,
6. after T rounds, select the answer from the highest-scoring round.

Final selected answers are appended to ``final_results.jsonl`` under
``--output-dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from rich import print as rprint
from tqdm import tqdm
from google import genai

SCRIPT_DIR = Path(__file__).resolve().parent


def find_fineverify_root() -> Path:
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (
            (candidate / "tools.py").is_file()
            and (candidate / "prompts.py").is_file()
            and (candidate / "gpt5_mini_fineverify.py").is_file()
        ):
            return candidate
    return SCRIPT_DIR.parent


def find_browsecomp_root() -> Path:
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if candidate.name == "BrowseComp-Plus":
            return candidate
    candidate = FINEVERIFY_ROOT / "BrowseComp-Plus"
    return candidate if candidate.is_dir() else FINEVERIFY_ROOT


def unique_paths(*paths: Path) -> Tuple[Path, ...]:
    seen: set[str] = set()
    unique: List[Path] = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return tuple(unique)


FINEVERIFY_ROOT = find_fineverify_root()
REPO_ROOT = FINEVERIFY_ROOT.parent
BROWSECOMP_ROOT = find_browsecomp_root()
PATH_SEARCH_BASES = unique_paths(
    Path.cwd(), SCRIPT_DIR, BROWSECOMP_ROOT, FINEVERIFY_ROOT, REPO_ROOT
)

for path in (REPO_ROOT, FINEVERIFY_ROOT, BROWSECOMP_ROOT, SCRIPT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from prompts import VERIFICATION_PROMPT, format_query
from utils import (
    add_mcp_usage,
    build_mcp_request,
    combine_tool_call_counts,
    empty_mcp_usage,
    generate_mcp_response_with_retry,
    get_text_output_from_items,
    make_mcp_client,
    normalize_mcp_response,
    str2bool,
)

from tools import (
    append_jsonl,
    build_skipped_verification_row,
    calculate_gemini_cost,
    compute_average_score,
    evaluate_candidate,
    evaluate_candidate_with_gemini,
    extract_judgments,
    extract_statement_texts,
    get_answer,
    get_query_id,
    get_question,
    get_subquestions_from_row,
    index_by_query_id,
    load_jsonl,
    parse_float,
    pick_best_verified_record,
    safe_filename,
    safe_int,
    utc_timestamp,
    write_json,
)

load_dotenv()

jsonl_lock = threading.Lock()
stats_lock = threading.Lock()

total_stats = {
    "input_tokens": 0,
    "input_tokens_cached": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "total_tokens": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "completed_rounds": 0,
    "early_stops": 0,
    "skipped_repetition": 0,
    "tool_call_counts": {},
}


def load_config_to_args(args: argparse.Namespace) -> argparse.Namespace:
    config_path = resolve_existing_path(getattr(args, "config", None))
    if not config_path or not config_path.is_file():
        return args
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for key, value in config.items():
        dest = key.replace("-", "_")
        if dest == "model":
            args.search_model = value
            args.verification_model = value
            if args.judge_provider == "gemini" and args.judge_model is None:
                args.judge_model = value
            continue
        if dest in {"max_tokens", "max_output_tokens"}:
            args.search_max_output_tokens = value
            args.verification_max_output_tokens = value
            continue
        if dest in {"thinking_level", "reasoning_effort"}:
            args.search_thinking_level = value
            args.verification_thinking_level = value
            if args.judge_provider == "gemini":
                args.judge_thinking_level = value
            continue
        if dest == "mcp_url":
            args.search_mcp_url = value
            args.verification_mcp_url = value
            continue
        setattr(args, dest, value)

    rprint(f"[bold cyan]Config loaded from:[/bold cyan] {config_path}")
    return args


def finalize_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.judge_model is None:
        args.judge_model = (
            args.search_model if args.judge_provider == "gemini" else "gpt-5-mini"
        )
    args.normalize_not_attempted_to_null = str2bool(args.normalize_not_attempted_to_null)
    return args


def dump_run_params(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {}
    for key, value in vars(args).items():
        if key in {"api_key", "openai_api_key"}:
            continue
        cfg[key] = str(value) if isinstance(value, Path) else value
    cfg["gemini_api_key_present"] = bool(
        args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    cfg["openai_api_key_present"] = bool(
        args.openai_api_key or os.getenv("OPENAI_API_KEY")
    )
    cfg["cwd"] = os.getcwd()
    cfg["cmdline"] = " ".join(sys.argv)
    cfg["created_at_utc"] = utc_timestamp()
    cfg["fineverify_root"] = str(FINEVERIFY_ROOT)
    cfg["browsecomp_root"] = str(BROWSECOMP_ROOT)
    cfg["query_template_demo"] = format_query("question_placeholder", args.query_template)
    cfg["verification_prompt"] = VERIFICATION_PROMPT
    append_jsonl(output_dir / "params_for_run.jsonl", cfg)


def resolve_existing_path(path_text: Optional[str]) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    if path.is_file():
        return path
    for base in PATH_SEARCH_BASES:
        candidate = (base / path_text).expanduser()
        if candidate.is_file():
            return candidate
    return path


def resolve_output_dir(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"run", "data"}:
        return (FINEVERIFY_ROOT / path).resolve()
    return (Path.cwd() / path).resolve()


def load_records_for_run(args: argparse.Namespace) -> List[dict]:
    data_path = resolve_existing_path(args.data) or Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    records = load_jsonl(data_path)
    start = max(0, safe_int(args.start, 0))
    end = safe_int(args.end, -1)
    records = records[start:end] if end >= 0 else records[start:]
    if args.limit is not None:
        records = records[: max(0, int(args.limit))]
    return records


def load_optional_jsonl_index(path_text: Optional[str]) -> Dict[str, dict]:
    if not path_text:
        return {}
    path = resolve_existing_path(path_text) or Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    return index_by_query_id(load_jsonl(path))


def load_processed_final_ids(final_path: Path) -> set[str]:
    if not final_path.exists():
        return set()
    processed: set[str] = set()
    for row in load_jsonl(final_path):
        qid = row.get("query_id")
        if qid is not None:
            processed.add(str(qid))
    return processed


def init_gemini_client(args: argparse.Namespace) -> genai.Client:
    if genai is None:
        raise RuntimeError("The google-genai package is required for Gemini runs")
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)


def init_openai_judge_client(args: argparse.Namespace) -> Any:
    if args.judge_provider != "openai":
        return None
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --judge-provider openai")
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("The openai package is required when --judge-provider openai") from exc
    return openai.OpenAI(api_key=api_key)


async def run_search_candidate(
    client: genai.Client,
    mcp_client: Client,
    *,
    query_id: Any,
    question: str,
    correct_answer: Optional[str],
    model: str,
    thinking_level: Optional[str],
    query_template: Optional[str],
    max_output_tokens: int,
    mcp_url: str,
    system_prompt: Optional[str],
    max_iterations: int,
    max_attempts: int,
) -> dict:
    formatted_query = format_query(question, query_template)
    request_body = build_mcp_request(
        formatted_query=formatted_query,
        model=model,
        max_output_tokens=max_output_tokens,
        mcp_url=mcp_url,
        system_prompt=system_prompt,
        thinking_level=thinking_level,
        max_iterations=max_iterations,
    )
    response = await generate_mcp_response_with_retry(
        genai_module=genai,
        client=client,
        mcp_client=mcp_client,
        initial_request=request_body,
        max_attempts=max_attempts,
    )
    normalized_record, _ = normalize_mcp_response(request_body, response)
    output_text = get_text_output_from_items(normalized_record["result"])

    return {
        "query_id": query_id,
        "query_text": question,
        "answer": correct_answer,
        "output_text": output_text,
        "usage": normalized_record["usage"],
        "request_body": request_body,
        "result": normalized_record["result"],
        "raw_output": normalized_record["raw_output"],
        "retrieved_docids": normalized_record["retrieved_docids"],
        "status": normalized_record["status"],
        "tool_call_counts": normalized_record["tool_call_counts"],
    }


def format_subquestions_for_prompt(subquestions: List[str]) -> str:
    if not subquestions:
        return "1. None"
    return "\n".join(f"{idx}. {sq}" for idx, sq in enumerate(subquestions, start=1))


def build_verification_prompt(
    *,
    question: str,
    subquestions: List[str],
    candidate_answer: str,
    explanation: str,
    expected_count: int,
    attempt_idx: int,
) -> str:
    base = VERIFICATION_PROMPT.format(
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
        "- You MUST evaluate ALL subquestions.\n"
        f"- You should output approximately {expected_count} judgment lines "
        "(one per subquestion).\n"
        "- Keep strict output format and ensure each subquestion has exactly one "
        "`Judgment:` line."
    )


def get_extracted_explanation(eval_row: dict) -> str:
    value = eval_row.get("extracted_explanation")
    if value is None:
        value = eval_row.get("extrated_explanation")
    if value is None:
        return ""
    return str(value)


async def verify_candidate_with_mcp(
    client: genai.Client,
    mcp_client: Client,
    *,
    eval_row: dict,
    decomp_row: Optional[dict],
    model: str,
    max_output_tokens: int,
    thinking_level: Optional[str],
    mcp_url: str,
    system_prompt: Optional[str],
    max_iterations: int,
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
        verified_row.update({"tool_call_counts": {}, "retrieved_docids": []})
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
        verified_row.update({"tool_call_counts": {}, "retrieved_docids": []})
        return verified_row, {
            "metadata": {"status": "failed", "reason": "missing decomposition row"},
            "query_id": qid,
            "query_text": question,
            "verified_record": verified_row,
        }

    attempts_raw: List[dict] = []
    merged_tool_counts: List[Dict[str, int]] = []
    total_usage = empty_mcp_usage()
    retrieved_docids: set[str] = set()
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
        request_body = build_mcp_request(
            formatted_query=formatted_query,
            model=model,
            max_output_tokens=max_output_tokens,
            mcp_url=mcp_url,
            system_prompt=system_prompt,
            thinking_level=thinking_level,
            max_iterations=max_iterations,
        )
        response = await generate_mcp_response_with_retry(
            genai_module=genai,
            client=client,
            mcp_client=mcp_client,
            initial_request=request_body,
            max_attempts=request_max_attempts,
        )
        normalized_record, _ = normalize_mcp_response(request_body, response)
        usage = normalized_record["usage"]
        output_text = get_text_output_from_items(normalized_record["result"])
        judgments = extract_judgments(output_text)
        statement_texts = extract_statement_texts(output_text)

        add_mcp_usage(total_usage, usage)
        merged_tool_counts.append(normalized_record["tool_call_counts"])
        retrieved_docids.update(normalized_record["retrieved_docids"])

        attempts_raw.append(
            {
                "attempt": attempt_idx,
                "request_body": request_body,
                "response_status": normalized_record["status"],
                "tool_call_counts": normalized_record["tool_call_counts"],
                "retrieved_docids": normalized_record["retrieved_docids"],
                "usage": usage,
                "result": normalized_record["result"],
                "raw_output": normalized_record["raw_output"],
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
    combined_tool_call_counts = combine_tool_call_counts(merged_tool_counts)
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
        "tool_call_counts": combined_tool_call_counts,
        "retrieved_docids": sorted(retrieved_docids),
        "iterations": safe_int(total_usage.get("iterations", 0)),
        "input_tokens": total_usage["input_tokens"],
        "input_tokens_cached": total_usage["input_tokens_cached"],
        "output_tokens": total_usage["output_tokens"],
        "reasoning_tokens": total_usage["reasoning_tokens"],
        "total_tokens": total_usage["total_tokens"],
        "web_search_calls": 0,
        "note": note,
    }

    raw_payload = {
        "metadata": {
            "model": model,
            "thinking_level": thinking_level,
            "max_output_tokens": max_output_tokens,
            "mcp_url": mcp_url,
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


async def evaluate_round(
    *,
    args: argparse.Namespace,
    search_record: dict,
    correct_answer: str,
    openai_client: Any,
    gemini_client: genai.Client,
) -> Tuple[dict, dict]:
    if args.judge_provider == "openai":
        return await asyncio.to_thread(
            evaluate_candidate,
            openai_client,
            search_record=search_record,
            correct_answer=correct_answer,
            model=args.judge_model,
            max_output_tokens=args.judge_max_output_tokens,
            reasoning_effort=args.judge_thinking_level,
            max_attempts=args.judge_max_attempts,
        )

    return await evaluate_candidate_with_gemini(
        gemini_client,
        genai.types,
        search_record=search_record,
        correct_answer=correct_answer,
        model=args.judge_model,
        max_output_tokens=args.judge_max_output_tokens,
        reasoning_effort=args.judge_thinking_level,
        max_attempts=args.judge_max_attempts,
    )


def normalize_answer_for_reuse(answer: Any) -> str:
    if answer is None:
        return ""
    return str(answer).strip()


def resolve_question_and_answer(record: dict) -> Tuple[str, str]:
    return get_question(record), get_answer(record)


def resolve_decomposition_row(
    record: dict,
    qid: Any,
    decomposed_map: Dict[str, dict],
) -> Optional[dict]:
    decomp_row = decomposed_map.get(str(qid)) if qid is not None else None
    if decomp_row is not None:
        return decomp_row
    subquestions, _ = get_subquestions_from_row(record)
    return record if subquestions else None


def build_reused_verification_row(
    *,
    eval_row: dict,
    previous_row: dict,
    reused_from_round: Any,
    args: argparse.Namespace,
) -> dict:
    reused_subquestions = previous_row.get("subquestions")
    if not isinstance(reused_subquestions, list):
        reused_subquestions = []

    reused_judgments = previous_row.get("subquestion_judgments")
    if not isinstance(reused_judgments, list):
        reused_judgments = []

    return {
        "query_id": eval_row.get("query_id"),
        "query_text": eval_row.get("query_text"),
        "correct_answer": eval_row.get("correct_answer"),
        "extracted_final_answer": eval_row.get("extracted_final_answer"),
        "grade": eval_row.get("grade"),
        "extracted_confidence": eval_row.get("extracted_confidence"),
        "extracted_explanation": get_extracted_explanation(eval_row),
        "subquestion_count": safe_int(
            previous_row.get("subquestion_count"),
            len(reused_subquestions),
        ),
        "subquestions": reused_subquestions,
        "subquestion_judgments": reused_judgments,
        "average_score": parse_float(previous_row.get("average_score")),
        "tool_call_counts": {},
        "retrieved_docids": previous_row.get("retrieved_docids", []),
        "iterations": 0,
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "web_search_calls": 0,
        "reused_verification": True,
        "reused_from_round": reused_from_round,
        "note": (
            f"skipped due to repetition in round {reused_from_round}; "
            f"score function: {args.score_supported:g}, "
            f"{args.score_not_found:g}, {args.score_contradicted:g}"
        ),
    }


def compact_round_summary(
    *,
    round_idx: int,
    search_record: dict,
    eval_row: dict,
    verified_row: dict,
    skipped_verification: bool,
    judge_provider: str,
) -> dict:
    search_usage = search_record.get("usage", {}) if isinstance(search_record.get("usage"), dict) else {}
    judge_usage = eval_row.get("judge_usage", {}) if isinstance(eval_row.get("judge_usage"), dict) else {}
    return {
        "round": round_idx,
        "query_id": eval_row.get("query_id"),
        "candidate_answer": eval_row.get("extracted_final_answer"),
        "grade": eval_row.get("grade"),
        "grade_reasoning": eval_row.get("grade_reasoning"),
        "extracted_confidence": eval_row.get("extracted_confidence"),
        "average_score": parse_float(verified_row.get("average_score")),
        "subquestion_judgments": verified_row.get("subquestion_judgments", []),
        "skipped_verification": skipped_verification,
        "reused_verification": bool(verified_row.get("reused_verification")),
        "reused_from_round": verified_row.get("reused_from_round"),
        "judge_provider": judge_provider,
        "search_tool_call_counts": search_usage.get("tool_call_counts", {}),
        "verification_tool_call_counts": verified_row.get("tool_call_counts", {}),
        "search_retrieved_docids": search_record.get("retrieved_docids", []),
        "verification_retrieved_docids": verified_row.get("retrieved_docids", []),
        "judge_response_id": judge_usage.get("response_id"),
        "search_output_tokens": safe_int(search_usage.get("output_tokens", 0)),
        "judge_output_tokens": safe_int(judge_usage.get("output_tokens", 0)),
        "verification_output_tokens": safe_int(verified_row.get("output_tokens", 0)),
        "note": verified_row.get("note"),
    }


def build_final_row(
    *,
    qid: Any,
    question: str,
    correct_answer: str,
    rounds: List[dict],
    verified_records: List[dict],
    total_usage: dict,
    max_rounds: int,
    stopped_early: bool,
    args: argparse.Namespace,
) -> dict:
    best_record, final_answer, best_score, best_round = pick_best_verified_record(
        verified_records,
        normalize_not_attempted_to_null=args.normalize_not_attempted_to_null,
    )
    best_record = best_record or {}
    return {
        "query_id": qid,
        "query_text": question,
        "correct_answer": correct_answer,
        "final_answer": final_answer,
        "best_verification_score": best_score,
        "best_round": best_round,
        "grade": best_record.get("grade"),
        "explanation": best_record.get("extracted_explanation"),
        "subquestion_judgments": best_record.get("subquestion_judgments", []),
        "rounds_completed": len(rounds),
        "max_rounds": max_rounds,
        "stopped_early": stopped_early,
        "rounds": rounds,
        "usage": total_usage,
        "score_function": {
            "supported": float(args.score_supported),
            "not_found": float(args.score_not_found),
            "contradicted": float(args.score_contradicted),
        },
        "models": {
            "search": args.search_model,
            "judge": args.judge_model,
            "judge_provider": args.judge_provider,
            "verification": args.verification_model,
        },
        "mcp": {
            "search_mcp_url": args.search_mcp_url,
            "verification_mcp_url": args.verification_mcp_url,
        },
    }


def write_round_outputs(
    *,
    output_dir: Path,
    qid: Any,
    line_idx: int,
    round_idx: int,
    compact: dict,
    raw_payload: dict,
) -> None:
    with jsonl_lock:
        append_jsonl(output_dir / "all_rounds.jsonl", compact)
    separated_name = f"{line_idx:06d}_{safe_filename(qid)}_round{round_idx}.json"
    write_json(output_dir / "separated" / separated_name, raw_payload)


def write_final_output(output_dir: Path, final_row: dict) -> None:
    with jsonl_lock:
        append_jsonl(output_dir / "final_results.jsonl", final_row)
    qid = final_row.get("query_id")
    write_json(output_dir / "separated" / f"{safe_filename(qid)}_final.json", final_row)


async def process_one_record(
    *,
    record: dict,
    line_idx: int,
    gemini_client: genai.Client,
    search_mcp_client: Client,
    verification_mcp_client: Client,
    openai_client: Any,
    decomposed_map: Dict[str, dict],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    qid = get_query_id(record, fallback=line_idx)
    question, correct_answer = resolve_question_and_answer(record)
    if not question:
        raise ValueError(f"query_id={qid}: missing problem/question/query_text")
    if not correct_answer:
        raise ValueError(f"query_id={qid}: missing answer/correct_answer for evaluation")

    decomp_row = resolve_decomposition_row(record, qid, decomposed_map)
    verified_records: List[dict] = []
    round_summaries: List[dict] = []
    verified_answer_cache: Dict[str, dict] = {}
    total_usage = empty_mcp_usage()
    stopped_early = False

    for round_idx in range(1, args.max_rounds + 1):
        search_record = await run_search_candidate(
            gemini_client,
            search_mcp_client,
            query_id=qid,
            question=question,
            correct_answer=correct_answer,
            model=args.search_model,
            thinking_level=args.search_thinking_level,
            query_template=args.query_template,
            max_output_tokens=args.search_max_output_tokens,
            mcp_url=args.search_mcp_url,
            system_prompt=args.search_system,
            max_iterations=args.search_max_iterations,
            max_attempts=args.search_request_max_attempts,
        )
        search_record["source_line_idx"] = line_idx
        search_record["round"] = round_idx
        add_mcp_usage(total_usage, search_record.get("usage"))

        eval_row, eval_raw = await evaluate_round(
            args=args,
            search_record=search_record,
            correct_answer=correct_answer,
            openai_client=openai_client,
            gemini_client=gemini_client,
        )
        eval_row["round"] = round_idx
        add_mcp_usage(total_usage, eval_row.get("judge_usage"))

        grade = str(eval_row.get("grade", "")).strip().lower()
        skipped_verification = grade not in {"correct", "incorrect"}
        cache_key = normalize_answer_for_reuse(eval_row.get("extracted_final_answer"))

        if skipped_verification:
            subquestions, expected_count = get_subquestions_from_row(decomp_row)
            skip_reason = (
                "skipped due to abstention"
                if grade == "not attempted"
                else f"skipped due to invalid grade: {grade or 'missing'}"
            )
            verified_row = build_skipped_verification_row(
                eval_row=eval_row,
                subquestions=subquestions,
                expected_count=expected_count,
                reason=skip_reason,
                note=(
                    f"score function: {args.score_supported:g}, "
                    f"{args.score_not_found:g}, {args.score_contradicted:g}"
                ),
            )
            verified_row.update({"tool_call_counts": {}, "retrieved_docids": []})
            verify_raw = {
                "metadata": {"status": "skipped", "reason": skip_reason},
                "query_id": qid,
                "query_text": question,
                "verified_record": verified_row,
            }
        elif cache_key and cache_key in verified_answer_cache:
            previous_verified_row = verified_answer_cache[cache_key]
            reused_from_round = previous_verified_row.get("round")
            verified_row = build_reused_verification_row(
                eval_row=eval_row,
                previous_row=previous_verified_row,
                reused_from_round=reused_from_round,
                args=args,
            )
            verify_raw = {
                "metadata": {
                    "status": "skipped",
                    "reason": "same extracted_final_answer as previous round",
                    "reused_from_round": reused_from_round,
                },
                "query_id": qid,
                "query_text": question,
                "verified_record": verified_row,
                "reused_verified_record": previous_verified_row,
            }
            with stats_lock:
                total_stats["skipped_repetition"] += 1
        else:
            verified_row, verify_raw = await verify_candidate_with_mcp(
                gemini_client,
                verification_mcp_client,
                eval_row=eval_row,
                decomp_row=decomp_row,
                model=args.verification_model,
                max_output_tokens=args.verification_max_output_tokens,
                thinking_level=args.verification_thinking_level,
                mcp_url=args.verification_mcp_url,
                system_prompt=args.verification_system,
                max_iterations=args.verification_max_iterations,
                max_retries=args.verification_max_retries,
                request_max_attempts=args.verification_request_max_attempts,
                count_tolerance=args.count_tolerance,
                score_supported=args.score_supported,
                score_not_found=args.score_not_found,
                score_contradicted=args.score_contradicted,
            )
            add_mcp_usage(total_usage, verified_row)

        verified_row["round"] = round_idx
        if not skipped_verification and cache_key and not verified_row.get("reused_verification"):
            verified_answer_cache[cache_key] = verified_row
        verified_records.append(verified_row)

        compact = compact_round_summary(
            round_idx=round_idx,
            search_record=search_record,
            eval_row=eval_row,
            verified_row=verified_row,
            skipped_verification=skipped_verification,
            judge_provider=args.judge_provider,
        )
        round_summaries.append(compact)
        write_round_outputs(
            output_dir=output_dir,
            qid=qid,
            line_idx=line_idx,
            round_idx=round_idx,
            compact=compact,
            raw_payload={
                "metadata": {
                    "query_id": qid,
                    "round": round_idx,
                    "created_at_utc": utc_timestamp(),
                },
                "search": search_record,
                "evaluation": eval_raw,
                "verification": verify_raw,
                "round_summary": compact,
            },
        )

        with stats_lock:
            total_stats["completed_rounds"] += 1

        if parse_float(verified_row.get("average_score")) >= float(args.early_stop_score):
            stopped_early = True
            break

    final_row = build_final_row(
        qid=qid,
        question=question,
        correct_answer=correct_answer,
        rounds=round_summaries,
        verified_records=verified_records,
        total_usage=total_usage,
        max_rounds=args.max_rounds,
        stopped_early=stopped_early,
        args=args,
    )
    write_final_output(output_dir, final_row)

    with stats_lock:
        add_mcp_usage(total_stats, total_usage)
        total_stats["successful_queries"] += 1
        if stopped_early:
            total_stats["early_stops"] += 1

    return final_row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unified Gemini FineVerify with custom MCP search/verification tools."
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config_gemini_bc_fineverify.yaml"),
        help="Optional YAML config file.",
    )
    parser.add_argument("--data", default="data/Browsecomp_plus.jsonl")
    parser.add_argument(
        "--decomposed-jsonl",
        default="data/decomposed/gemini_3_flash_browsecomp_decomposed.jsonl",
        dest="decomposed_jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="run/gemini-bc-fineverify/browsecomp-plus",
        dest="output_dir",
    )
    parser.add_argument("--max-rounds", "--T", type=int, default=4, dest="max_rounds")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=1, dest="num_threads")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--search-model", default="gemini-3-flash-preview", dest="search_model")
    parser.add_argument("--verification-model", default="gemini-3-flash-preview", dest="verification_model")
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "gemini"],
        default="openai",
        dest="judge_provider",
    )
    parser.add_argument("--judge-model", default=None, dest="judge_model")
    parser.add_argument(
        "--search-thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default="medium",
        dest="search_thinking_level",
    )
    parser.add_argument(
        "--judge-thinking-level",
        choices=["none", "minimal", "low", "medium", "high"],
        default="medium",
        dest="judge_thinking_level",
    )
    parser.add_argument(
        "--verification-thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default="medium",
        dest="verification_thinking_level",
    )
    parser.add_argument("--query-template", default="QUERY_TEMPLATE_NO_GET_DOCUMENT", dest="query_template")
    parser.add_argument("--search-max-output-tokens", type=int, default=40000, dest="search_max_output_tokens")
    parser.add_argument("--judge-max-output-tokens", type=int, default=20000, dest="judge_max_output_tokens")
    parser.add_argument("--verification-max-output-tokens", type=int, default=40000, dest="verification_max_output_tokens")
    parser.add_argument("--search-request-max-attempts", type=int, default=7, dest="search_request_max_attempts")
    parser.add_argument("--judge-max-attempts", type=int, default=4, dest="judge_max_attempts")
    parser.add_argument("--verification-max-retries", type=int, default=4, dest="verification_max_retries")
    parser.add_argument("--verification-request-max-attempts", type=int, default=5, dest="verification_request_max_attempts")
    parser.add_argument("--search-max-iterations", type=int, default=100, dest="search_max_iterations")
    parser.add_argument("--verification-max-iterations", type=int, default=100, dest="verification_max_iterations")
    parser.add_argument("--count-tolerance", type=int, default=1, dest="count_tolerance")
    parser.add_argument("--score-supported", type=float, default=1.0, dest="score_supported")
    parser.add_argument("--score-not-found", type=float, default=0.0, dest="score_not_found")
    parser.add_argument("--score-contradicted", type=float, default=0.0, dest="score_contradicted")
    parser.add_argument("--early-stop-score", type=float, default=1.0, dest="early_stop_score")
    parser.add_argument(
        "--normalize-not-attempted-to-null",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="normalize_not_attempted_to_null",
    )
    parser.add_argument("--search-system", default=None, dest="search_system")
    parser.add_argument("--verification-system", default=None, dest="verification_system")
    parser.add_argument("--search-mcp-url", default="http://127.0.0.1:8080/mcp", dest="search_mcp_url")
    parser.add_argument("--verification-mcp-url", default="http://127.0.0.1:8081/mcp", dest="verification_mcp_url")
    parser.add_argument("--api-key", "--api_key", default=None, dest="api_key")
    parser.add_argument("--openai-api-key", default=None, dest="openai_api_key")
    return parser


async def main_async() -> None:
    parser = build_parser()
    args = finalize_defaults(load_config_to_args(parser.parse_args()))

    if args.max_rounds <= 0:
        raise ValueError("--max-rounds/--T must be positive")
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "separated").mkdir(parents=True, exist_ok=True)
    dump_run_params(args, output_dir)

    gemini_client = init_gemini_client(args)
    openai_client = init_openai_judge_client(args)
    search_mcp_client = make_mcp_client(args.search_mcp_url)
    verification_mcp_client = make_mcp_client(args.verification_mcp_url)

    records = load_records_for_run(args)
    decomposed_map = load_optional_jsonl_index(args.decomposed_jsonl)

    final_path = output_dir / "final_results.jsonl"
    processed_ids = set() if args.force else load_processed_final_ids(final_path)
    jobs = [
        (idx, row)
        for idx, row in enumerate(records)
        if str(get_query_id(row, fallback=idx)) not in processed_ids
    ]

    print(f"Input records: {len(records)}")
    print(f"Pending records: {len(jobs)} (skipping {len(processed_ids)} already finalized)")
    print(f"Max rounds per record: {args.max_rounds}")
    print(f"Judge provider: {args.judge_provider}")
    print(f"Search MCP URL: {args.search_mcp_url}")
    print(f"Verification MCP URL: {args.verification_mcp_url}")
    print(f"Output: {output_dir}")

    start_time = datetime.now()
    semaphore = asyncio.Semaphore(args.num_threads)

    async def worker(item: tuple[int, dict]) -> Optional[dict]:
        idx, row = item
        qid = get_query_id(row, fallback=idx)
        async with semaphore:
            try:
                return await process_one_record(
                    record=row,
                    line_idx=idx,
                    gemini_client=gemini_client,
                    search_mcp_client=search_mcp_client,
                    verification_mcp_client=verification_mcp_client,
                    openai_client=openai_client,
                    decomposed_map=decomposed_map,
                    output_dir=output_dir,
                    args=args,
                )
            except Exception as exc:
                with stats_lock:
                    total_stats["failed_queries"] += 1
                error_row = {
                    "query_id": qid,
                    "source_line_idx": idx,
                    "error": str(exc),
                    "created_at_utc": utc_timestamp(),
                }
                with jsonl_lock:
                    append_jsonl(output_dir / "failed_records.jsonl", error_row)
                tqdm.write(f"[Error] query_id={qid}: {exc}")
                return None

    completed: List[dict] = []
    async with search_mcp_client:
        async with verification_mcp_client:
            tasks = [asyncio.create_task(worker(item)) for item in jobs]
            with tqdm(total=len(tasks), desc="Gemini MCP FineVerify", unit="query") as pbar:
                for task in asyncio.as_completed(tasks):
                    result = await task
                    pbar.update(1)
                    if result is not None:
                        completed.append(result)

    duration = str(datetime.now() - start_time).split(".")[0]
    estimated_cost = calculate_gemini_cost(
        args.search_model,
        total_stats["input_tokens"],
        total_stats["input_tokens_cached"],
        total_stats["output_tokens"],
        0,
    )
    summary = {
        "duration": duration,
        "records_seen": len(records),
        "records_pending": len(jobs),
        "records_completed_this_run": len(completed),
        "successful_queries": total_stats["successful_queries"],
        "failed_queries": total_stats["failed_queries"],
        "completed_rounds": total_stats["completed_rounds"],
        "early_stops": total_stats["early_stops"],
        "skipped_repetition": total_stats["skipped_repetition"],
        "input_tokens": total_stats["input_tokens"],
        "input_tokens_cached": total_stats["input_tokens_cached"],
        "output_tokens": total_stats["output_tokens"],
        "reasoning_tokens": total_stats["reasoning_tokens"],
        "total_tokens": total_stats["total_tokens"],
        "tool_call_counts": total_stats["tool_call_counts"],
        "estimated_cost_using_search_model_rates": round(estimated_cost, 6),
        "final_results": str(final_path),
    }
    write_json(output_dir / "run_summary.json", summary)

    print("\n" + "=" * 50)
    print("GEMINI MCP FINEVERIFY SUMMARY")
    print(f"Duration:          {duration}")
    print(f"Success/Fail:      {total_stats['successful_queries']}/{total_stats['failed_queries']}")
    print(f"Completed rounds:  {total_stats['completed_rounds']}")
    print(f"Early stops:       {total_stats['early_stops']}")
    print(f"Skipped reuse:     {total_stats['skipped_repetition']}")
    print(f"Tool calls:        {total_stats['tool_call_counts']}")
    print(f"Output Tokens:     {total_stats['output_tokens']:,}")
    print(f"Final JSONL:       {final_path}")
    print(f"Run summary:       {output_dir / 'run_summary.json'}")
    print("=" * 50)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
