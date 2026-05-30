import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import yaml
from rich import print as rprint
import os

def clean_solver_response(text: str) -> str:
    """Removes boilerplate to focus the checker on the core explanation."""
    if "Explanation:" in text:
        # split by "Explanation:" and take the part after it
        cleaned = text.split("Explanation:")[-1].strip()
    else:
        cleaned = text.strip()
    # Split by common markers and take the first part
    for marker in ["Exact Answer", "Confidence"]:
        cleaned = cleaned.split(marker)[0].strip()
    return cleaned

def extract_retrieved_docids_from_result(result: List[dict]) -> List[str]:
    retrieved_docids_set = set()
    for item in result:
        if item.get("type") != "tool_call":
            continue
        tool_name = str(item.get("tool_name") or "")
        if "search" in tool_name.lower() or "retrieval" in tool_name.lower():
            output = item.get("output")
            parsed = None
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except Exception:
                    parsed = None
            elif isinstance(output, list):
                parsed = output

            if isinstance(parsed, list):
                for elem in parsed:
                    if isinstance(elem, dict) and "docid" in elem:
                        retrieved_docids_set.add(str(elem["docid"]))
            else:
                # Fallback: regex grep docids from raw string output
                raw = output if isinstance(output, str) else ""
                if raw:
                    # Quoted docid values
                    for m in re.findall(r'"docid"\s*:\s*"([^"]+)"', raw):
                        retrieved_docids_set.add(str(m))
                    # Unquoted numeric docid values
                    for m in re.findall(r'"docid"\s*:\s*(\d+)', raw):
                        retrieved_docids_set.add(str(m))
    return list(retrieved_docids_set)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


class SearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def get_tool_definitions(self):
        tools = [
            {
                "type": "function",
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "type": "function",
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )

        return tools

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["query"])
        if tool_name == "get_document":
            return self._get_document(arguments["docid"])
        raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    cand["snippet"] = self.tokenizer.decode(
                        tokens[: self.snippet_max_tokens], skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, ensure_ascii=False, indent=2)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps(
                {"error": f"Document with docid '{docid}' not found"},
                ensure_ascii=False,
            )
        return json.dumps(result, ensure_ascii=False, indent=2)


def build_tool_request(
    *,
    formatted_query: str,
    model: str,
    max_tokens: int,
    tool_handler: SearchToolHandler,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict:
    body = {
        "model": model,
        "max_output_tokens": max_tokens,
        "input": [{"role": "user", "content": formatted_query}],
        "tools": tool_handler.get_tool_definitions(),
        "tool_choice": "auto",
        "truncation": "auto",
    }

    if not model.lower().startswith("o") and "gpt-5" not in model.lower():
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p

    if system_prompt:
        body["instructions"] = system_prompt

    if reasoning_effort and str(reasoning_effort).lower() not in {"none", "null"}:
        body["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}

    return body


def run_conversation_with_tools(
    client: Any,
    initial_request: dict,
    tool_handler: SearchToolHandler,
    max_iterations: int = 100,
):
    input_messages = initial_request["input"].copy()
    global_max_tokens = initial_request["max_output_tokens"]

    cumulative_usage = {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
        "iterations": 0,
        "response_ids": [],
    }

    combined_output = []
    tool_outputs = {}
    all_responses = []

    for _ in range(max_iterations):
        cumulative_usage["iterations"] += 1
        remaining_tokens = global_max_tokens - cumulative_usage["output_tokens"]
        if remaining_tokens <= 0:
            break

        request = initial_request.copy()
        request["input"] = input_messages
        request["max_output_tokens"] = min(remaining_tokens, global_max_tokens)

        response = client.responses.create(**request)
        all_responses.append(response)
        cumulative_usage["response_ids"].append(getattr(response, "id", None))

        response_output = getattr(response, "output", []) or []
        combined_output.extend(response_output)

        usage = getattr(response, "usage", None)
        if usage:
            cumulative_usage["input_tokens"] += getattr(usage, "input_tokens", 0)
            cumulative_usage["output_tokens"] += getattr(usage, "output_tokens", 0)
            cumulative_usage["total_tokens"] += getattr(usage, "total_tokens", 0)

            input_details = getattr(usage, "input_tokens_details", None)
            if input_details:
                cumulative_usage["input_tokens_details"]["cached_tokens"] += getattr(
                    input_details, "cached_tokens", 0
                )

            output_details = getattr(usage, "output_tokens_details", None)
            if output_details:
                cumulative_usage["output_tokens_details"]["reasoning_tokens"] += getattr(
                    output_details, "reasoning_tokens", 0
                )

        function_calls = [
            item for item in response_output if getattr(item, "type", None) == "function_call"
        ]

        if not function_calls:
            return response, combined_output, cumulative_usage, tool_outputs

        for item in response_output:
            serialized_item = item.model_dump(mode="python") if hasattr(item, "model_dump") else item
            if isinstance(serialized_item, dict):
                serialized_item.pop("status", None)
            input_messages.append(serialized_item)

        for tool_call in function_calls:
            try:
                parsed_args = json.loads(tool_call.arguments)
                result = tool_handler.execute_tool(tool_call.name, parsed_args)
                payload = {"output": result, "status": "completed", "error": None}
            except Exception as exc:
                result = f"Error executing {tool_call.name}: {exc}"
                payload = {"output": None, "status": "error", "error": result}

            if getattr(tool_call, "id", None):
                tool_outputs[tool_call.id] = payload
            if getattr(tool_call, "call_id", None):
                tool_outputs[tool_call.call_id] = payload

            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                }
            )

    final_response = all_responses[-1] if all_responses else None
    return final_response, combined_output, cumulative_usage, tool_outputs


def normalize_tool_outputs(
    combined_output,
    tool_outputs: dict,
) -> Tuple[List[dict], Dict[str, int]]:
    normalized_results: List[dict] = []
    tool_call_counts: Dict[str, int] = {}

    serialized_output = [
        item.model_dump(mode="python") if hasattr(item, "model_dump") else item
        for item in (combined_output or [])
    ]

    for item in serialized_output:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "function_call":
            item_id = item.get("id")
            call_id = item.get("call_id")
            tool_name = item.get("name")
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            tool_output_payload = tool_outputs.get(item_id) or tool_outputs.get(call_id, {})
            normalized_results.append(
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": item.get("arguments"),
                    "output": tool_output_payload.get("output"),
                }
            )

        elif item_type == "reasoning":
            normalized_results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": item.get("summary"),
                }
            )

        elif item_type == "message":
            parts = item.get("content", []) or []
            text_chunks: List[str] = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_chunks.append(str(part.get("text", "")))

            text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
            if text:
                normalized_results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": text,
                    }
                )

    return normalized_results, tool_call_counts


def serialize_tool_outputs(combined_output) -> List[dict]:
    serialized: List[dict] = []
    for item in combined_output or []:
        if hasattr(item, "model_dump"):
            serialized.append(item.model_dump(mode="python"))
        elif isinstance(item, dict):
            serialized.append(item)
        else:
            serialized.append(getattr(item, "__dict__", str(item)))
    return serialized


def tool_usage_to_dict(cumulative_usage: Optional[dict]) -> dict:
    if cumulative_usage is None:
        return {
            "input_tokens": 0,
            "input_tokens_cached": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "iterations": 0,
            "response_ids": [],
        }

    return {
        "input_tokens": cumulative_usage.get("input_tokens", 0),
        "input_tokens_cached": cumulative_usage.get("input_tokens_details", {}).get(
            "cached_tokens", 0
        ),
        "output_tokens": cumulative_usage.get("output_tokens", 0),
        "reasoning_tokens": cumulative_usage.get("output_tokens_details", {}).get(
            "reasoning_tokens", 0
        ),
        "total_tokens": cumulative_usage.get("total_tokens", 0),
        "iterations": cumulative_usage.get("iterations", 0),
        "response_ids": cumulative_usage.get("response_ids", []),
    }


def get_text_output_from_normalized(items: List[dict]) -> str:
    chunks: List[str] = []
    for item in items:
        if item.get("type") != "output_text":
            continue
        out = item.get("output")
        if isinstance(out, str) and out.strip():
            chunks.append(out.strip())
    return "\n\n".join(chunks).strip()


def combine_tool_call_counts(items: List[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_mcp_client(mcp_url: str):
    try:
        from fastmcp import Client
        from fastmcp.client.transports import SSETransport
    except ImportError as exc:
        raise RuntimeError("fastmcp is required for Gemini MCP tool use") from exc

    if not mcp_url:
        raise RuntimeError("MCP URL must be provided")
    return Client(SSETransport(url=mcp_url))


def build_mcp_request(
    *,
    formatted_query: str,
    model: str,
    max_output_tokens: int,
    mcp_url: str,
    system_prompt: Optional[str],
    thinking_level: Optional[str],
    max_iterations: int,
) -> dict:
    return {
        "model": model,
        "contents": formatted_query,
        "system_instruction": system_prompt,
        "max_output_tokens": max_output_tokens,
        "thinking_level": thinking_level,
        "max_iterations": max_iterations,
        "mcp_url": mcp_url,
    }


async def run_mcp_conversation(
    *,
    genai_module: Any,
    client: Any,
    mcp_client: Any,
    initial_request: dict,
):
    generation_config = genai_module.types.GenerateContentConfig(
        tools=[mcp_client.session],
        max_output_tokens=initial_request["max_output_tokens"],
        thinking_config=genai_module.types.ThinkingConfig(
            include_thoughts=True,
            thinking_level=initial_request.get("thinking_level") or "medium",
        ),
        automatic_function_calling=genai_module.types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=initial_request["max_iterations"],
        ),
    )
    if initial_request.get("system_instruction"):
        generation_config.system_instruction = initial_request["system_instruction"]

    return await client.aio.models.generate_content(
        model=initial_request["model"],
        contents=initial_request["contents"],
        config=generation_config,
    )


async def generate_mcp_response_with_retry(
    *,
    genai_module: Any,
    client: Any,
    mcp_client: Any,
    initial_request: dict,
    max_attempts: int,
):
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await run_mcp_conversation(
                genai_module=genai_module,
                client=client,
                mcp_client=mcp_client,
                initial_request=initial_request,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Gemini MCP request failed after {max_attempts} attempts") from last_error


def _extract_function_response_output(response_payload: Any):
    if response_payload is None:
        return None
    if not isinstance(response_payload, dict):
        return response_payload
    if "error" in response_payload:
        return response_payload.get("error")
    result = response_payload.get("result")
    if result is None:
        return response_payload
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first.get("text")
    return result


def estimate_mcp_iterations(response_dict: dict) -> int:
    history = response_dict.get("automatic_function_calling_history") or []
    afc_model_turns = 0
    for content in history:
        if not isinstance(content, dict) or content.get("role") != "model":
            continue
        parts = content.get("parts") or []
        if any(isinstance(part, dict) and part.get("function_call") for part in parts):
            afc_model_turns += 1

    candidate_parts = (
        response_dict.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    )
    final_has_function_call = any(
        isinstance(part, dict) and part.get("function_call") for part in candidate_parts
    )
    if final_has_function_call:
        return max(1, afc_model_turns)
    return afc_model_turns + 1


def normalize_mcp_response(request_config: dict, response) -> Tuple[dict, dict]:
    try:
        response_dict = response.model_dump(mode="python")
    except AttributeError:
        response_dict = json.loads(json.dumps(response, default=str))

    usage_metadata = response_dict.get("usage_metadata", {}) or {}
    prompt_tokens = usage_metadata.get("prompt_token_count")
    cached_tokens = usage_metadata.get("cached_content_token_count")
    candidate_tokens = int(usage_metadata.get("candidates_token_count") or 0)
    thought_tokens = int(usage_metadata.get("thoughts_token_count") or 0)
    total_tokens = usage_metadata.get("total_token_count")

    def _part_iter():
        for content in response_dict.get("automatic_function_calling_history", []):
            for part in content.get("parts", []):
                yield part
        for part in (
            response_dict.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        ):
            yield part

    results: List[dict] = []
    tool_counts: Dict[str, int] = {}
    pending_calls: Dict[str, dict] = {}

    for part in _part_iter():
        if not isinstance(part, dict):
            continue

        function_call = part.get("function_call")
        function_response = part.get("function_response")

        if function_call:
            name = function_call.get("name")
            call_id = function_call.get("id") or f"call_{len(pending_calls)}"
            entry = {
                "type": "tool_call",
                "tool_name": name,
                "arguments": function_call.get("args"),
                "output": None,
            }
            pending_calls[call_id] = entry
            results.append(entry)
            continue

        if function_response:
            name = function_response.get("name")
            output = _extract_function_response_output(function_response.get("response"))
            response_id = function_response.get("id")
            entry = pending_calls.get(response_id) if response_id else None
            if entry is None:
                for candidate in pending_calls.values():
                    if candidate["tool_name"] == name and candidate["output"] is None:
                        entry = candidate
                        break
            if entry:
                entry["output"] = output
            else:
                results.append(
                    {
                        "type": "tool_call",
                        "tool_name": name,
                        "arguments": None,
                        "output": output,
                    }
                )
            tool_counts[name] = tool_counts.get(name, 0) + 1
            continue

        if part.get("thought") is True:
            results.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": [part.get("text", "")],
                }
            )
            continue

        text = part.get("text")
        if text:
            results.append(
                {
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": text,
                }
            )

    for entry in pending_calls.values():
        if entry["output"] is None:
            tool_counts[entry["tool_name"]] = tool_counts.get(entry["tool_name"], 0) + 1

    finish_reason = response_dict.get("candidates", [{}])[0].get("finish_reason")
    status = "completed" if finish_reason in {None, "STOP"} else str(finish_reason)

    usage = {
        "input_tokens": _safe_int(prompt_tokens),
        "input_tokens_cached": _safe_int(cached_tokens),
        "output_tokens": candidate_tokens + thought_tokens,
        "reasoning_tokens": thought_tokens,
        "total_tokens": _safe_int(total_tokens),
        "iterations": estimate_mcp_iterations(response_dict),
        "web_search_calls": 0,
        "tool_call_counts": tool_counts,
    }

    record = {
        "metadata": {
            "model": response_dict.get("model_version") or request_config.get("model"),
            "max_output_tokens": request_config.get("max_output_tokens"),
            "thinking_level": request_config.get("thinking_level"),
            "mcp_url": request_config.get("mcp_url"),
        },
        "tool_call_counts": tool_counts,
        "usage": usage,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(results),
        "result": results,
        "raw_output": response_dict,
    }
    return record, response_dict


def get_text_output_from_items(items: List[dict]) -> str:
    return get_text_output_from_normalized(items)


def empty_mcp_usage() -> dict:
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


def add_mcp_usage(total: dict, usage: Optional[dict]) -> None:
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
        total[key] = _safe_int(total.get(key, 0)) + _safe_int(usage.get(key, 0))
    total["iterations"] = _safe_int(total.get("iterations", 0)) + _safe_int(
        usage.get("iterations", 0)
    )
    current_counts = total.get("tool_call_counts")
    if not isinstance(current_counts, dict):
        current_counts = {}
    incoming_counts = usage.get("tool_call_counts")
    if not isinstance(incoming_counts, dict):
        incoming_counts = {}
    total["tool_call_counts"] = combine_tool_call_counts([current_counts, incoming_counts])

def calculate_cost(model_name, input_tokens, cached_input, output_tokens):
    pricing = {
        "gpt-5.2": (1.75, 0.175, 14.00),
        "gpt-5.1": (1.25, 0.125, 10.00),
        "gpt-5": (1.25, 0.125, 10.00),
        "gpt-5-mini": (0.25, 0.025, 2.00),
        "gpt-5.4-mini": (0.75, 0.075, 4.50),
    }
    rates = pricing.get(model_name, (0.0, 0.0, 0.0))
    
    if cached_input is None:
        cached_input = 0
    
    if cached_input != 0:
        input_tokens = input_tokens - cached_input
    
    return ((input_tokens / 1_000_000) * rates[0]) + \
              ((cached_input / 1_000_000) * rates[1]) + \
        ((output_tokens / 1_000_000) * rates[2])
        

def load_config_to_args(args):
    """
    If a config file is provided via --config, overwrite default args 
    with values from the YAML file.
    """
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            if config_dict:
                for key, value in config_dict.items():
                    setattr(args, key, value)
        rprint(f"[bold cyan]Config loaded from:[/bold cyan] {args.config}")
    return args
