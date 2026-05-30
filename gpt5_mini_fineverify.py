"""GPT-5-mini FineVerify pipeline.

For each input question, this driver loops up to T rounds:
1. generate a web-search candidate answer,
2. grade the candidate against the provided ground truth,
3. skip verification when the grade is ``not attempted``,
4. otherwise verify the candidate with web-search-backed judgments against subquestions,
   reusing prior round verification when the candidate answer repeats (caching),
5. score those judgments and stop early when the average score reaches 1,
6. after T rounds, select the answer from the highest-scoring round.

Final selected answers are appended to ``final_results.jsonl`` under
``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from tools import *

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

jsonl_lock = threading.Lock()
stats_lock = threading.Lock()

total_stats = {
    "input_tokens": 0,
    "input_tokens_cached": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "web_search_calls": 0,
    "total_tokens": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "completed_rounds": 0,
    "early_stops": 0,
    "skipped_repetition": 0,
}


def load_config_to_args(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, "config", None):
        return args
    config_path = Path(args.config)
    if not config_path.exists():
        return args
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --config files")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    for key, value in config.items():
        setattr(args, key.replace("-", "_"), value)
    print(f"Config loaded from: {config_path}")
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
    append_jsonl(output_dir / "params_for_run.jsonl", cfg)


def resolve_existing_path(path_text: Optional[str]) -> Optional[Path]:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_file():
        return path
    for base in (SCRIPT_DIR, REPO_ROOT):
        candidate = (base / path_text).expanduser()
        if candidate.is_file():
            return candidate
    return path


def load_records_for_run(args: argparse.Namespace) -> List[dict]:
    data_path = resolve_existing_path(args.data) or Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    records = load_jsonl(data_path)
    start = max(0, safe_int(args.start, 0))
    end = safe_int(args.end, -1)
    if end >= 0:
        records = records[start:end]
    else:
        records = records[start:]
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


def resolve_question_and_answer(
    record: dict,
) -> tuple[str, str]:
    question = get_question(record)
    answer = get_answer(record)
    return question, answer


def resolve_decomposition_row(record: dict, qid: Any, decomposed_map: Dict[str, dict]) -> Optional[dict]:
    decomp_row = decomposed_map.get(str(qid)) if qid is not None else None
    if decomp_row is not None:
        return decomp_row
    subquestions, _ = get_subquestions_from_row(record)
    if subquestions:
        return record
    return None


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
        "search_response_id": search_usage.get("response_id"),
        "judge_response_id": judge_usage.get("response_id"),
        "search_web_search_calls": safe_int(search_usage.get("web_search_calls", 0)),
        "verification_web_search_calls": safe_int(verified_row.get("web_search_calls", 0)),
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
    write_json(
        output_dir / "separated" / f"{safe_filename(qid)}_final.json",
        final_row,
    )


def normalize_answer_for_reuse(answer: Any) -> str:
    if answer is None:
        return ""
    return str(answer).strip()


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
        "extracted_explanation": eval_row.get("extracted_explanation"),
        "subquestion_count": safe_int(
            previous_row.get("subquestion_count"),
            len(reused_subquestions),
        ),
        "subquestions": reused_subquestions,
        "subquestion_judgments": reused_judgments,
        "average_score": parse_float(previous_row.get("average_score")),
        "web_search_calls": 0,
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "reused_verification": True,
        "reused_from_round": reused_from_round,
        "note": (
            f"skipped due to repetition in round {reused_from_round}; "
            f"score function: {args.score_supported:g}, "
            f"{args.score_not_found:g}, {args.score_contradicted:g}"
        ),
    }


def process_one_record(
    *,
    record: dict,
    line_idx: int,
    client: Any,
    decomposed_map: Dict[str, dict],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    qid = get_query_id(record, fallback=line_idx)
    question, correct_answer = resolve_question_and_answer(record)
    if not question:
        raise ValueError(f"query_id={qid}: missing question/problem/query_text")
    if not correct_answer:
        raise ValueError(f"query_id={qid}: missing answer/correct_answer for evaluation")

    decomp_row = resolve_decomposition_row(record, qid, decomposed_map)
    verified_records: List[dict] = []
    round_summaries: List[dict] = []
    verified_answer_cache: Dict[str, dict] = {}
    total_usage = {
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "web_search_calls": 0,
        "total_tokens": 0,
    }
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
        )
        search_record["source_line_idx"] = line_idx
        search_record["round"] = round_idx
        add_usage(total_usage, search_record.get("usage"))

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
        add_usage(total_usage, eval_row.get("judge_usage"))

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
            verified_row, verify_raw = verify_candidate(
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
            )
            add_usage(total_usage, verified_row)

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
        add_usage(total_stats, total_usage)
        total_stats["successful_queries"] += 1
        if stopped_early:
            total_stats["early_stops"] += 1

    return final_row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unified GPT-5-mini FineVerify search/eval/verify/best-answer pipeline."
    )
    parser.add_argument(
        "--config",
        default="config_gpt5_mini_fineverify.yaml",
        help="YAML config file.",
    )
    parser.add_argument(
        "--data",
        default="fineverify/data/DeepsearchQA.jsonl",
        help="Input JSONL with query_id/problem/answer fields.",
    )
    parser.add_argument(
        "--decomposed-jsonl",
        default="fineverify/data/decomposed/gpt_5_mini_DSQA-decomposed.jsonl",
        dest="decomposed_jsonl",
        help="JSONL with query_id/id and subquestions fields for verification.",
    )
    parser.add_argument(
        "--output-dir",
        default="fineverify/run/gpt5-mini-fineverify",
        dest="output_dir",
        help="Directory for final_results.jsonl and debug artifacts.",
    )
    parser.add_argument(
        "--max-rounds",
        "--T",
        type=int,
        default=4,
        dest="max_rounds",
        help="Maximum search/eval/verify rounds per question.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in input JSONL.")
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end index; -1 means all.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of records.")
    parser.add_argument("--num-threads", type=int, default=1, dest="num_threads")
    parser.add_argument("--force", action="store_true", help="Re-run questions already in final_results.jsonl.")

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
    parser.add_argument("--query-template", default="QUERY_TEMPLATE_WEB", dest="query_template")
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
        help="Match openai_best_verification.py selection behavior.",
    )
    parser.add_argument("--api-key", default=None, dest="api_key", help="OpenAI API key or OPENAI_API_KEY env var.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = load_config_to_args(args)

    if args.max_rounds <= 0:
        raise ValueError("--max-rounds/--T must be positive")
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "separated").mkdir(parents=True, exist_ok=True)
    dump_run_params(args, output_dir)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        import openai
    except ImportError as exc:
        raise RuntimeError("The openai package is required to run API calls") from exc
    client = openai.OpenAI(api_key=api_key)

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
        total_stats["web_search_calls"],
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
        "web_search_calls": total_stats["web_search_calls"],
        "estimated_cost_using_search_model_rates": round(estimated_cost, 6),
        "final_results": str(final_path),
    }
    write_json(output_dir / "run_summary.json", summary)

    print("\n" + "=" * 50)
    print("FINEVERIFY SUMMARY")
    print(f"Duration:          {duration}")
    print(f"Success/Fail:      {total_stats['successful_queries']}/{total_stats['failed_queries']}")
    print(f"Completed rounds:  {total_stats['completed_rounds']}")
    print(f"Early stops:       {total_stats['early_stops']}")
    print(f"Skipped reuse:     {total_stats['skipped_repetition']}")
    print(f"Web Search Calls:  {total_stats['web_search_calls']:,}")
    print(f"Output Tokens:     {total_stats['output_tokens']:,}")
    print(f"Final JSONL:       {final_path}")
    print(f"Run summary:       {output_dir / 'run_summary.json'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
