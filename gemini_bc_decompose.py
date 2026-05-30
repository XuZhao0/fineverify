"""Pipeline to decompose questions into subquestions using Gemini-3-Flash for BrowseComp-Plus"""

import json
import os
import sys
import argparse
import threading
import time
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from prompts import *

try:
    from .tools import (
        calculate_gemini_cost as calculate_cost,
        extract_gemini_decompose_usage,
        extract_gemini_reasoning_summary,
        extract_gemini_response_text,
        extract_subquestion_items,
        get_gemini_client,
        is_rate_limit_error,
        load_decompose_input_jsonl,
        load_decompose_processed_ids,
        validate_threads,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from tools import (
        calculate_gemini_cost as calculate_cost,
        extract_gemini_decompose_usage,
        extract_gemini_reasoning_summary,
        extract_gemini_response_text,
        extract_subquestion_items,
        get_gemini_client,
        is_rate_limit_error,
        load_decompose_input_jsonl,
        load_decompose_processed_ids,
        validate_threads,
    )

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Thread-safe synchronization
file_lock = threading.Lock()
stats_lock = threading.Lock()
api_semaphore = None

total_stats = {
    "input_tokens": 0,
    "input_tokens_cached": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "actual_cost": 0.0,
    "successful_requests": 0,
    "failed_requests": 0,
}


def call_llm_api(queries, args):
    global api_semaphore
    if args.dry_run:
        time.sleep(0.1)
        with stats_lock:
            total_stats["input_tokens"] += 600
            total_stats["output_tokens"] += 800
            total_stats["successful_requests"] += 1
        return {**queries, "subquestions": ["Dry run sample"], "dry_run": True}

    client, genai = get_gemini_client(args.api_key_env)
    decompose_prompt = DECOMPOSE_PROMPT.format(QUESTION=queries["question"])

    max_retries = 5
    max_regenerations = 5
    regeneration_count = 0

    prompt_for_call = decompose_prompt
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0

    with api_semaphore:
        for attempt in range(max_retries):
            try:
                generation_config = genai.types.GenerateContentConfig(
                    max_output_tokens=args.max_output_tokens,
                    thinking_config=genai.types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_level=args.thinking_level,
                    ),
                )

                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt_for_call,
                    config=generation_config,
                )

                response_text = extract_gemini_response_text(response)
                reasoning_summary = extract_gemini_reasoning_summary(response)

                subquestions, subquestion_items = extract_subquestion_items(response_text)
                subquestion_count = len(subquestion_items)

                curr_input_tokens, curr_cached_tokens, curr_output_tokens, curr_reasoning_tokens = (
                    extract_gemini_decompose_usage(response)
                )

                total_input_tokens += curr_input_tokens
                total_output_tokens += curr_output_tokens
                total_reasoning_tokens += curr_reasoning_tokens

                with stats_lock:
                    total_stats["input_tokens"] += curr_input_tokens
                    total_stats["output_tokens"] += curr_output_tokens
                    total_stats["input_tokens_cached"] += curr_cached_tokens
                    total_stats["reasoning_tokens"] += curr_reasoning_tokens

                if subquestion_count < 2 and regeneration_count < max_regenerations:
                    regeneration_count += 1
                    tqdm.write(
                        f"Regenerating (ID {queries['id']}): only {subquestion_count} subquestion(s). "
                        f"Retry {regeneration_count}/{max_regenerations}."
                    )
                    prompt_for_call = (
                        f"{decompose_prompt}\n\n"
                        "Your previous output had fewer than 2 bullet subquestions. "
                        "Regenerate now and return ONLY:\n"
                        "Checkable subquestion list:\n"
                        "- ...\n"
                        "- ...\n"
                        "(at least two bullet items, each starting with '-')"
                    )
                    continue

                result = queries.copy()
                result.update(
                    {
                        "subquestions": subquestions,
                        "subquestion_count": subquestion_count,
                        "reasoning_summary": reasoning_summary,
                        "input_tokens": total_input_tokens,
                        "reasoning_tokens": total_reasoning_tokens,
                        "output_tokens": total_output_tokens,
                    }
                )

                with stats_lock:
                    total_stats["successful_requests"] += 1

                return result

            except Exception as e:
                if is_rate_limit_error(e):
                    time.sleep((2 ** attempt) + random.random())
                    continue
                tqdm.write(f"{attempt}/{max_retries}. Error (ID {queries['id']}): {e}")

    with stats_lock:
        total_stats["failed_requests"] += 1
    return None


def process_pipeline(args):
    start_time = datetime.now()
    time_str = start_time.strftime("%m%d_%H%M")

    global api_semaphore
    api_semaphore = threading.Semaphore(value=args.threads)

    all_items = load_decompose_input_jsonl(Path(args.input_file), args.start, args.end)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_decompose_processed_ids(output_path)
    data_to_process = [item for item in all_items if item["id"] not in processed_ids]

    print(
        f"Total: {len(all_items)} | Already done: {len(processed_ids)} | "
        f"To process: {len(data_to_process)}"
    )

    # --- DEBUG: Print Prompt for the first item ---
    if data_to_process:
        first_query = data_to_process[0]
        debug_prompt = DECOMPOSE_PROMPT.format(QUESTION=first_query["question"])
        print("\n" + "=" * 50)
        print(f"DEBUG: Formatted Prompt for First Item (ID: {first_query['id']})")
        print("=" * 50)
        print(debug_prompt)
        print("=" * 50 + "\n")

    if args.dry_run:
        print(f"[DRY RUN MODE ENABLED] Target: {output_path}")
    else:
        print(f"[PIPELINE START] Saving to: {output_path}")

    # Execution
    with output_path.open(mode="a", encoding="utf-8") as jsonl_file:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(call_llm_api, item, args): item for item in data_to_process
            }
            for future in tqdm(
                as_completed(futures), total=len(data_to_process), desc="Processing"
            ):
                res = future.result()
                if res and not args.dry_run:
                    with file_lock:
                        jsonl_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                        jsonl_file.flush()

    # Final Report
    end_time = datetime.now()
    estimated_cost = calculate_cost(
        args.model,
        total_stats["input_tokens"],
        total_stats["input_tokens_cached"],
        total_stats["output_tokens"],
    )

    report_title = "DRY RUN SUMMARY" if args.dry_run else "EXECUTION SUMMARY REPORT"

    report = f"""
=====================================================
            {report_title} ({time_str})
=====================================================
Timestamp:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration:       {str(end_time - start_time).split('.')[0]}
Output File:    {output_path}

[ARGS]
{json.dumps(vars(args), indent=4)}

[STATS]
Processed:      {len(data_to_process)} (skipped {len(processed_ids)})
Tokens:         Input: {total_stats['input_tokens']:,} | Output: {total_stats['output_tokens']:,}
Actual cost:    ${total_stats['actual_cost']:.6f}
ESTIMATED COST: ${estimated_cost:.4f}

Prompt: {DECOMPOSE_PROMPT}
=====================================================
"""
    print(report)

    if not args.dry_run:
        report_filename = output_path.parent / f"decompose_report_{time_str}.txt"
        with report_filename.open("a", encoding="utf-8") as f:
            f.write(report)
        print(f"Summary appended to: {report_filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="data/Browsecomp_plus.jsonl",
        help="Path to FineVerify-style JSONL with query_id/problem/answer fields",
    )
    parser.add_argument(
        "--output_file",
        default="data/decomposed/gemini_3_flash_browsecomp_decomposed.jsonl",
        help="Output JSONL path",
    )

    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--thinking_level", default="high")
    parser.add_argument("--max_output_tokens", type=int, default=30000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--threads", type=validate_threads, default=5)
    parser.add_argument("--dry_run", action="store_true", help="Simulate without API calls")

    process_pipeline(parser.parse_args())


if __name__ == "__main__":
    main()
