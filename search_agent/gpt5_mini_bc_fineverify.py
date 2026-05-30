"""FineVerify pipeline using custom search/document tools for browsecomp-plus.

For each input JSONL question, this driver loops up to T rounds:
1. generate a search candidate answer with the local search tool,
2. grade the candidate against the provided ground truth,
3. skip verification when the grade is ``not attempted``,
4. otherwise verify the candidate with search/get_document-backed judgments,
   reusing prior round verification when the candidate answer repeats,
5. score those judgments and stop early when the average score reaches 1,
6. after T rounds, select the answer from the highest-scoring round.

Final selected answers are appended to ``final_results.jsonl`` under
``--output-dir``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

import openai
import yaml
from dotenv import load_dotenv
from rich import print as rprint
from tqdm import tqdm

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


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


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
SEARCHER_ROOT = (
    BROWSECOMP_ROOT / "searcher"
    if is_relative_to(SCRIPT_DIR, BROWSECOMP_ROOT)
    and (BROWSECOMP_ROOT / "searcher").is_dir()
    else FINEVERIFY_ROOT / "searcher"
)
PATH_SEARCH_BASES = unique_paths(
    Path.cwd(), SCRIPT_DIR, BROWSECOMP_ROOT, FINEVERIFY_ROOT, REPO_ROOT
)

for path in (REPO_ROOT, FINEVERIFY_ROOT, BROWSECOMP_ROOT, SCRIPT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from prompts import VERIFICATION_PROMPT, format_query
from utils import (
    SearchToolHandler,
    build_tool_request,
    calculate_cost,
    combine_tool_call_counts,
    extract_retrieved_docids_from_result,
    get_text_output_from_normalized,
    normalize_tool_outputs,
    run_conversation_with_tools,
    serialize_tool_outputs,
    str2bool,
    tool_usage_to_dict,
)

from fineverify.tools import (
    add_usage,
    append_jsonl,
    build_skipped_verification_row,
    compute_average_score,
    evaluate_candidate,
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

SEARCHER_CLASS_PATHS = {
    "bm25": ("searcher.searchers.bm25_searcher", "BM25Searcher"),
    "faiss": ("searcher.searchers.faiss_searcher", "FaissSearcher"),
    "reasonir": ("searcher.searchers.faiss_searcher", "ReasonIrSearcher"),
    "custom": ("searcher.searchers.custom_searcher", "CustomSearcher"),
}


def get_searcher_choices() -> List[str]:
    return list(SEARCHER_CLASS_PATHS.keys())


def get_searcher_class(searcher_type: str):
    try:
        module_name, class_name = SEARCHER_CLASS_PATHS[searcher_type]
    except KeyError as exc:
        raise ValueError(f"Unknown searcher type: {searcher_type}") from exc
    module_stem = module_name.rsplit(".", 1)[-1]
    module_path = SEARCHER_ROOT / "searchers" / f"{module_stem}.py"
    module = load_searcher_module_direct(module_name, module_path)
    return getattr(module, class_name)


def load_searcher_module_direct(module_name: str, module_path: Path):
    ensure_lightweight_searcher_package()
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load searcher module from {module_path}")
    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_lightweight_searcher_package() -> None:
    searcher_pkg = sys.modules.get("searcher")
    if searcher_pkg is None:
        searcher_pkg = ModuleType("searcher")
        searcher_pkg.__path__ = [str(SEARCHER_ROOT)]
        sys.modules["searcher"] = searcher_pkg

    searchers_pkg = sys.modules.get("searcher.searchers")
    if searchers_pkg is None:
        searchers_pkg = ModuleType("searcher.searchers")
        searchers_pkg.__path__ = [str(SEARCHER_ROOT / "searchers")]
        sys.modules["searcher.searchers"] = searchers_pkg

    base_name = "searcher.searchers.base"
    if base_name not in sys.modules:
        base_path = SEARCHER_ROOT / "searchers" / "base.py"
        spec = importlib_util.spec_from_file_location(base_name, base_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load searcher base module from {base_path}")
        base_module = importlib_util.module_from_spec(spec)
        sys.modules[base_name] = base_module
        spec.loader.exec_module(base_module)


def load_config_to_args(args: argparse.Namespace) -> argparse.Namespace:
    config_path = resolve_existing_path(getattr(args, "config", None))
    if not config_path or not config_path.is_file():
        return args
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    normalized_config_keys = {str(key).replace("-", "_") for key in config}
    for key, value in config.items():
        dest = key.replace("-", "_")
        if dest == "max_iterations":
            if "search_max_iterations" not in normalized_config_keys:
                args.search_max_iterations = value
            if "verification_max_iterations" not in normalized_config_keys:
                args.verification_max_iterations = value
            continue
        if dest == "snippet_max_tokens":
            if "search_snippet_max_tokens" not in normalized_config_keys:
                args.search_snippet_max_tokens = value
            if "verification_snippet_max_tokens" not in normalized_config_keys:
                args.verification_snippet_max_tokens = value
            continue
        if dest == "k":
            if "search_k" not in normalized_config_keys:
                args.search_k = value
            if "verification_k" not in normalized_config_keys:
                args.verification_k = value
            continue
        if dest == "get_document":
            args.verification_get_document = value
            continue
        setattr(args, dest, value)
    rprint(f"[bold cyan]Config loaded from:[/bold cyan] {config_path}")
    return args


def dump_run_params(args: argparse.Namespace, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = {}
    for key, value in vars(args).items():
        if key == "api_key":
            continue
        cfg[key] = str(value) if isinstance(value, Path) else value
    cfg["api_key_present"] = bool(args.api_key or os.getenv("OPENAI_API_KEY"))
    cfg["cwd"] = os.getcwd()
    cfg["cmdline"] = " ".join(sys.argv)
    cfg["created_at_utc"] = utc_timestamp()
    cfg["fineverify_root"] = str(FINEVERIFY_ROOT)
    cfg["browsecomp_root"] = str(BROWSECOMP_ROOT)
    cfg["searcher_root"] = str(SEARCHER_ROOT)
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


def resolve_glob_path(path_text: Optional[str]) -> Optional[str]:
    if not path_text:
        return path_text
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return str(path)
    if glob.glob(path_text):
        return path_text
    for base in PATH_SEARCH_BASES:
        candidate = base / path_text
        if glob.glob(str(candidate)):
            return str(candidate)
    return str(BROWSECOMP_ROOT / path_text)


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


def empty_usage() -> dict:
    return {
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "iterations": 0,
        "web_search_calls": 0,
        "tool_call_counts": {},
    }


def add_custom_usage(total: dict, usage: Optional[dict]) -> None:
    if not usage:
        return
    add_usage(total, usage)
    current_counts = total.get("tool_call_counts")
    if not isinstance(current_counts, dict):
        current_counts = {}
    incoming_counts = usage.get("tool_call_counts")
    if not isinstance(incoming_counts, dict):
        incoming_counts = {}
    total["tool_call_counts"] = combine_tool_call_counts([current_counts, incoming_counts])
    total["iterations"] = safe_int(total.get("iterations", 0)) + safe_int(
        usage.get("iterations", 0)
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
    if subquestions:
        return record
    return None


def run_search_candidate(
    client: openai.OpenAI,
    *,
    query_id: Any,
    question: str,
    correct_answer: Optional[str],
    model: str,
    reasoning_effort: Optional[str],
    query_template: Optional[str],
    max_output_tokens: int,
    tool_handler: SearchToolHandler,
    max_iterations: int,
    system_prompt: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
) -> dict:
    formatted_query = format_query(question, query_template)
    request_body = build_tool_request(
        formatted_query=formatted_query,
        model=model,
        max_tokens=max_output_tokens,
        tool_handler=tool_handler,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
    )
    response, combined_output, cumulative_usage, tool_outputs = run_conversation_with_tools(
        client=client,
        initial_request=request_body,
        tool_handler=tool_handler,
        max_iterations=max_iterations,
    )
    normalized_results, tool_call_counts = normalize_tool_outputs(
        combined_output, tool_outputs
    )
    usage = tool_usage_to_dict(cumulative_usage)
    usage["tool_call_counts"] = tool_call_counts
    usage["response_status"] = getattr(response, "status", None)
    output_text = get_text_output_from_normalized(normalized_results)
    retrieved_docids = extract_retrieved_docids_from_result(normalized_results)

    return {
        "query_id": query_id,
        "query_text": question,
        "answer": correct_answer,
        "output_text": output_text,
        "usage": usage,
        "request_body": request_body,
        "result": normalized_results,
        "raw_output": serialize_tool_outputs(combined_output),
        "retrieved_docids": retrieved_docids,
    }


def format_subquestions_for_prompt(subquestions: List[str]) -> str:
    if not subquestions:
        return "1. None"
    return "\n".join(f"{idx}. {sq}" for idx, sq in enumerate(subquestions, start=1))


def build_custom_verification_prompt(
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


def call_tool_verification_api(
    client: openai.OpenAI,
    *,
    formatted_query: str,
    model: str,
    max_output_tokens: int,
    reasoning_effort: Optional[str],
    tool_handler: SearchToolHandler,
    max_iterations: int,
    system_prompt: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
) -> dict:
    request_body = build_tool_request(
        formatted_query=formatted_query,
        model=model,
        max_tokens=max_output_tokens,
        tool_handler=tool_handler,
        system_prompt=system_prompt,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
    )
    response, combined_output, cumulative_usage, tool_outputs = run_conversation_with_tools(
        client=client,
        initial_request=request_body,
        tool_handler=tool_handler,
        max_iterations=max_iterations,
    )
    normalized_results, tool_call_counts = normalize_tool_outputs(
        combined_output, tool_outputs
    )
    usage = tool_usage_to_dict(cumulative_usage)
    usage["tool_call_counts"] = tool_call_counts

    return {
        "request_body": request_body,
        "response_ids": usage.get("response_ids", []),
        "response_status": getattr(response, "status", None),
        "output_text": get_text_output_from_normalized(normalized_results),
        "usage": usage,
        "tool_call_counts": tool_call_counts,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
        "raw_output": serialize_tool_outputs(combined_output),
    }


def verify_candidate_with_tools(
    client: openai.OpenAI,
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
    tool_handler: SearchToolHandler,
    max_iterations: int,
    system_prompt: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
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
    total_usage = empty_usage()
    retrieved_docids: set[str] = set()
    final_judgments: List[str] = []
    final_output_text = ""
    count_match = False

    for attempt_idx in range(1, max_retries + 1):
        formatted_query = build_custom_verification_prompt(
            question=question,
            subquestions=subquestions,
            candidate_answer=str(eval_row.get("extracted_final_answer") or ""),
            explanation=get_extracted_explanation(eval_row),
            expected_count=expected_count,
            attempt_idx=attempt_idx,
        )
        attempt_result = call_tool_verification_api(
            client=client,
            formatted_query=formatted_query,
            model=model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            tool_handler=tool_handler,
            max_iterations=max_iterations,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
        )
        output_text = attempt_result["output_text"]
        judgments = extract_judgments(output_text)
        statement_texts = extract_statement_texts(output_text)
        usage = attempt_result["usage"]
        add_custom_usage(total_usage, usage)
        merged_tool_counts.append(attempt_result["tool_call_counts"])
        retrieved_docids.update(attempt_result["retrieved_docids"])

        attempts_raw.append(
            {
                "attempt": attempt_idx,
                "request_body": attempt_result["request_body"],
                "response_ids": attempt_result["response_ids"],
                "response_status": attempt_result["response_status"],
                "tool_call_counts": attempt_result["tool_call_counts"],
                "retrieved_docids": attempt_result["retrieved_docids"],
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
        "search_response_ids": search_usage.get("response_ids", []),
        "judge_response_id": judge_usage.get("response_id"),
        "search_tool_call_counts": search_usage.get("tool_call_counts", {}),
        "verification_tool_call_counts": verified_row.get("tool_call_counts", {}),
        "search_retrieved_docids": search_record.get("retrieved_docids", []),
        "verification_retrieved_docids": verified_row.get("retrieved_docids", []),
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
    normalize_not_attempted_to_null: bool,
    args: argparse.Namespace,
) -> dict:
    best_record, final_answer, best_score, best_round = pick_best_verified_record(
        verified_records,
        normalize_not_attempted_to_null=normalize_not_attempted_to_null,
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
            "verification": args.verification_model,
        },
        "tools": {
            "search": {
                "searcher_type": args.searcher_type,
                "k": args.search_k,
                "include_get_document": args.search_get_document,
                "snippet_max_tokens": args.search_snippet_max_tokens,
            },
            "verification": {
                "searcher_type": args.searcher_type,
                "k": args.verification_k,
                "include_get_document": args.verification_get_document,
                "snippet_max_tokens": args.verification_snippet_max_tokens,
            },
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


def process_one_record(
    *,
    record: dict,
    line_idx: int,
    client: openai.OpenAI,
    decomposed_map: Dict[str, dict],
    output_dir: Path,
    search_tool_handler: SearchToolHandler,
    verification_tool_handler: SearchToolHandler,
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
    total_usage = empty_usage()
    stopped_early = False

    for round_idx in range(1, args.max_rounds + 1):
        search_record = run_search_candidate(
            client,
            query_id=qid,
            question=question,
            correct_answer=correct_answer,
            model=args.search_model,
            reasoning_effort=args.search_reasoning_effort,
            query_template=args.query_template,
            max_output_tokens=args.search_max_output_tokens,
            tool_handler=search_tool_handler,
            max_iterations=args.search_max_iterations,
            system_prompt=args.search_system,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        search_record["source_line_idx"] = line_idx
        search_record["round"] = round_idx
        add_custom_usage(total_usage, search_record.get("usage"))

        eval_row, eval_raw = evaluate_candidate(
            client,
            search_record=search_record,
            correct_answer=correct_answer,
            model=args.judge_model,
            max_output_tokens=args.judge_max_output_tokens,
            reasoning_effort=args.judge_reasoning_effort,
            max_attempts=args.judge_max_attempts,
        )
        eval_row["round"] = round_idx
        add_custom_usage(total_usage, eval_row.get("judge_usage"))

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
            verified_row, verify_raw = verify_candidate_with_tools(
                client,
                eval_row=eval_row,
                decomp_row=decomp_row,
                model=args.verification_model,
                max_output_tokens=args.verification_max_output_tokens,
                reasoning_effort=args.verification_reasoning_effort,
                max_retries=args.verification_max_retries,
                count_tolerance=args.count_tolerance,
                score_supported=args.score_supported,
                score_not_found=args.score_not_found,
                score_contradicted=args.score_contradicted,
                tool_handler=verification_tool_handler,
                max_iterations=args.verification_max_iterations,
                system_prompt=args.verification_system,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            add_custom_usage(total_usage, verified_row)

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
        normalize_not_attempted_to_null=args.normalize_not_attempted_to_null,
        args=args,
    )
    write_final_output(output_dir, final_row)

    with stats_lock:
        add_custom_usage(total_stats, total_usage)
        total_stats["successful_queries"] += 1
        if stopped_early:
            total_stats["early_stops"] += 1

    return final_row


def build_parser(searcher_class=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unified FineVerify with OpenAI and custom BrowseComp tools."
    )
    parser.add_argument(
        "--config",
        default=str(SCRIPT_DIR / "config_gpt5_mini_bc_fineverify.yaml"),
        help="Optional YAML config file.",
    )
    parser.add_argument(
        "--data",
        default="data/DeepsearchQA.jsonl",
        help="Input JSONL with query_id/problem/answer fields.",
    )
    parser.add_argument(
        "--decomposed-jsonl",
        default="data/decomposed/gpt_5_mini_DSQA-decomposed.jsonl",
        dest="decomposed_jsonl",
        help="JSONL with query_id/id and subquestions fields for verification.",
    )
    parser.add_argument(
        "--output-dir",
        default="run/gpt5-mini-custom-fineverify",
        dest="output_dir",
        help="Directory for final_results.jsonl and debug artifacts.",
    )
    parser.add_argument("--max-rounds", "--T", type=int, default=4, dest="max_rounds")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=1, dest="num_threads")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--search-model", default="gpt-5-mini", dest="search_model")
    parser.add_argument("--judge-model", default="gpt-5-mini", dest="judge_model")
    parser.add_argument("--verification-model", default="gpt-5.4-mini", dest="verification_model")
    parser.add_argument(
        "--search-reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="medium",
        dest="search_reasoning_effort",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="medium",
        dest="judge_reasoning_effort",
    )
    parser.add_argument(
        "--verification-reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="medium",
        dest="verification_reasoning_effort",
    )
    parser.add_argument("--query-template", default="QUERY_TEMPLATE_NO_GET_DOCUMENT", dest="query_template")
    parser.add_argument("--search-max-output-tokens", type=int, default=40000, dest="search_max_output_tokens")
    parser.add_argument("--judge-max-output-tokens", type=int, default=20000, dest="judge_max_output_tokens")
    parser.add_argument(
        "--verification-max-output-tokens",
        type=int,
        default=40000,
        dest="verification_max_output_tokens",
    )
    parser.add_argument("--judge-max-attempts", type=int, default=4, dest="judge_max_attempts")
    parser.add_argument("--verification-max-retries", type=int, default=4, dest="verification_max_retries")
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
    parser.add_argument("--search-max-iterations", type=int, default=100, dest="search_max_iterations")
    parser.add_argument(
        "--verification-max-iterations",
        type=int,
        default=100,
        dest="verification_max_iterations",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        dest="max_iterations",
        help="Deprecated alias: set both search and verification max iterations.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", "--top_p", type=float, default=None, dest="top_p")
    parser.add_argument("--search-system", default=None, dest="search_system")
    parser.add_argument("--verification-system", default=None, dest="verification_system")
    parser.add_argument("--api-key", "--api_key", default=None, dest="api_key")

    parser.add_argument(
        "--searcher-type",
        choices=get_searcher_choices(),
        required=False,
        help=f"Type of searcher to use: {', '.join(get_searcher_choices())}",
    )
    parser.add_argument("--search-snippet-max-tokens", type=int, default=512, dest="search_snippet_max_tokens")
    parser.add_argument(
        "--verification-snippet-max-tokens",
        type=int,
        default=512,
        dest="verification_snippet_max_tokens",
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=None,
        dest="snippet_max_tokens",
        help="Deprecated alias: set both search and verification snippet limits.",
    )
    parser.add_argument("--search-k", type=int, default=5, dest="search_k")
    parser.add_argument("--verification-k", type=int, default=5, dest="verification_k")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Deprecated alias: set both search and verification k.",
    )
    parser.add_argument(
        "--search-get-document",
        type=str2bool,
        default=False,
        dest="search_get_document",
        help="If true, register get_document for candidate-answer search.",
    )
    parser.add_argument(
        "--verification-get-document",
        "--get-document",
        type=str2bool,
        default=True,
        dest="verification_get_document",
        help="If true, register document retrieval for verification.",
    )
    parser.add_argument("--hf-token", type=str, default=None, dest="hf_token")
    parser.add_argument("--hf-home", type=str, default=None, dest="hf_home")

    if searcher_class is not None:
        searcher_class.parse_args(parser)

    return parser


def resolve_searcher_class_from_config(temp_args: argparse.Namespace):
    config_path = resolve_existing_path(getattr(temp_args, "config", None))
    if config_path and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if config.get("searcher_type"):
            temp_args.searcher_type = config["searcher_type"]
    if not temp_args.searcher_type:
        return None
    return get_searcher_class(temp_args.searcher_type)


def configure_environment(args: argparse.Namespace) -> None:
    if getattr(args, "index_path", None):
        args.index_path = resolve_glob_path(args.index_path)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home


def finalize_tool_args(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "max_iterations", None) is not None:
        args.search_max_iterations = args.max_iterations
        args.verification_max_iterations = args.max_iterations
    if getattr(args, "snippet_max_tokens", None) is not None:
        args.search_snippet_max_tokens = args.snippet_max_tokens
        args.verification_snippet_max_tokens = args.snippet_max_tokens
    if getattr(args, "k", None) is not None:
        args.search_k = args.k
        args.verification_k = args.k

    args.search_get_document = str2bool(args.search_get_document)
    args.verification_get_document = str2bool(args.verification_get_document)
    args.normalize_not_attempted_to_null = str2bool(args.normalize_not_attempted_to_null)
    return args


def main() -> None:
    temp_parser = build_parser()
    temp_args, _ = temp_parser.parse_known_args()
    searcher_class = resolve_searcher_class_from_config(temp_args)

    parser = build_parser(searcher_class)
    args = parser.parse_args()
    args = load_config_to_args(args)
    args = finalize_tool_args(args)

    if args.max_rounds <= 0:
        raise ValueError("--max-rounds/--T must be positive")
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")
    if not args.searcher_type:
        raise ValueError("--searcher-type must be provided in CLI or config")

    searcher_class = get_searcher_class(args.searcher_type)
    configure_environment(args)

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "separated").mkdir(parents=True, exist_ok=True)
    dump_run_params(args, output_dir)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = openai.OpenAI(api_key=api_key)

    records = load_records_for_run(args)
    decomposed_map = load_optional_jsonl_index(args.decomposed_jsonl)

    searcher = searcher_class(args)
    search_tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.search_snippet_max_tokens,
        k=args.search_k,
        include_get_document=args.search_get_document,
    )
    verification_tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.verification_snippet_max_tokens,
        k=args.verification_k,
        include_get_document=args.verification_get_document,
    )
    search_tools_registered = ["search"]
    if args.search_get_document:
        search_tools_registered.append("get_document")
    verification_tools_registered = ["search"]
    if args.verification_get_document:
        verification_tools_registered.append("get_document")

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
    print(
        f"Searcher: {searcher.search_type} "
        f"(search_k={args.search_k}, verification_k={args.verification_k})"
    )
    print(f"Search tools: {', '.join(search_tools_registered)}")
    print(f"Verification tools: {', '.join(verification_tools_registered)}")
    print(f"Output: {output_dir}")

    start_time = datetime.now()

    def worker(item: tuple[int, dict]) -> Optional[dict]:
        idx, row = item
        qid = get_query_id(row, fallback=idx)
        try:
            return process_one_record(
                record=row,
                line_idx=idx,
                client=client,
                decomposed_map=decomposed_map,
                output_dir=output_dir,
                search_tool_handler=search_tool_handler,
                verification_tool_handler=verification_tool_handler,
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
    if args.num_threads <= 1:
        for item in tqdm(jobs, desc="FineVerify", unit="query"):
            result = worker(item)
            if result is not None:
                completed.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [executor.submit(worker, item) for item in jobs]
            with tqdm(total=len(futures), desc="FineVerify", unit="query") as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    result = future.result()
                    if result is not None:
                        completed.append(result)

    duration = str(datetime.now() - start_time).split(".")[0]
    estimated_cost = calculate_cost(
        args.search_model,
        total_stats["input_tokens"],
        total_stats["input_tokens_cached"],
        total_stats["output_tokens"],
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
    print("CUSTOM FINEVERIFY SUMMARY")
    print(f"Duration:          {duration}")
    print(f"Success/Fail:      {total_stats['successful_queries']}/{total_stats['failed_queries']}")
    print(f"Completed rounds:  {total_stats['completed_rounds']}")
    print(f"Early stops:       {total_stats['early_stops']}")
    print(f"Skipped reuse:     {total_stats['skipped_repetition']}")
    print(f"Tool calls:        {total_stats['tool_call_counts']}")
    print(f"Input Tokens:      {total_stats['input_tokens']:,}")
    print(f"Cached Tokens:     {total_stats['input_tokens_cached']:,}")
    print(f"Output Tokens:     {total_stats['output_tokens']:,}")
    print(f"Final JSONL:       {final_path}")
    print(f"Run summary:       {output_dir / 'run_summary.json'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
