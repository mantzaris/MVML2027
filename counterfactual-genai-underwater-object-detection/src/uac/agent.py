"""Frozen GPU Qwen tool controller; numeric evidence only, strict action parsing."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
from .runtime import PROJECT_ROOT, fingerprint, project_path

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
TOOLS = ("fourier", "stft", "wavelet", "hht", "verify", "stop")
REASONS = ("uncertain_detection", "representation_disagreement", "verify_candidate",
           "sufficient_evidence", "budget_exhausted", "missing_representation")
SYSTEM_PROMPT = """You select acoustic-analysis tools for an underwater acoustic detector.
You receive only measured numeric evidence and a bounded tool budget. You do not
receive true labels. Treat all candidate identifiers as opaque identifiers.
Choose exactly one available action for an existing candidate, or stop.
Allowed action names: fourier, stft, wavelet, hht, verify, stop.
Never repeat an action already performed for the same candidate. Verify at most
once per candidate, and only if remaining_verifications is positive. Stop if
remaining_actions is zero. Prefer inspecting uncertainty or disagreement before
a costly verification. Do not produce detection probabilities, class labels,
physical claims, explanations, markdown, or extra keys.
Return exactly one JSON object with keys action, candidate_id, reason_code.
For an available non-stop tool, candidate_id must be an existing identifier string.
Example: {"action":"fourier","candidate_id":"candidate0","reason_code":"missing_representation"}
For stop, candidate_id must be the JSON null literal, never the string "null".
Example: {"action":"stop","candidate_id":null,"reason_code":"sufficient_evidence"}
Choose reason_code only from uncertain_detection, representation_disagreement,
verify_candidate, sufficient_evidence, budget_exhausted, missing_representation.
Use a short enumerated reason_code, not free text. The JSON object is the entire response.
"""


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} outside allowed range")
    return value


def _numeric_tree(value, depth=0):
    if depth > 4:
        raise ValueError("Evidence nesting exceeds four levels")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _number(value, "measurement")
    if isinstance(value, list):
        if len(value) > 128:
            raise ValueError("Measurement lists must be summaries of at most 128 numbers")
        return [_numeric_tree(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 128:
            raise ValueError("Too many measurement fields")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 80:
                raise ValueError("Measurement names must be short strings")
            normalized = key.lower().replace("-", "_")
            if any(term in normalized for term in ("ground_truth", "true_label", "target_label", "y_true")):
                raise ValueError("Ground-truth information is forbidden in controller state")
            result[key] = _numeric_tree(item, depth + 1)
        return result
    raise ValueError("Evidence values must be numeric, booleans, null, or nested numeric summaries")


def validate_state(state: dict) -> dict:
    allowed = {"candidates", "observations", "history", "available_tools",
               "remaining_actions", "remaining_verifications"}
    if not isinstance(state, dict) or set(state) - allowed:
        raise ValueError("Unknown controller state fields")
    candidates = state.get("candidates", [])
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 2:
        raise ValueError("Controller requires one or two candidate events")
    identifiers = set()
    cleaned = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {
            "candidate_id", "interval_seconds", "base_probability", "uncertainty"
        }:
            raise ValueError("Candidate schema mismatch")
        identifier = candidate["candidate_id"]
        if not isinstance(identifier, str) or not identifier or len(identifier) > 160 or identifier in identifiers:
            raise ValueError("Candidate IDs must be distinct nonempty strings")
        identifiers.add(identifier)
        interval = candidate["interval_seconds"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("Expected onset/offset pair")
        start = _number(interval[0], "onset", minimum=0)
        end = _number(interval[1], "offset", minimum=0)
        if end <= start:
            raise ValueError("Event offset must exceed onset")
        cleaned.append({
            "candidate_id": identifier, "interval_seconds": [start, end],
            "base_probability": _number(candidate["base_probability"], "base_probability", 0, 1),
            "uncertainty": _number(candidate["uncertainty"], "uncertainty", 0, 1),
        })
    tools = state.get("available_tools", list(TOOLS))
    if not isinstance(tools, list) or not tools or any(tool not in TOOLS for tool in tools):
        raise ValueError("Unknown available tool")
    if len(set(tools)) != len(tools) or "stop" not in tools:
        raise ValueError("Available tools must be unique and include stop")
    history = state.get("history", [])
    if not isinstance(history, list) or len(history) > 3:
        raise ValueError("History exceeds the three-action cap")
    for entry in history:
        if not isinstance(entry, dict) or set(entry) != {"action", "candidate_id", "valid"}:
            raise ValueError("History schema mismatch")
        if entry["action"] not in TOOLS or entry["candidate_id"] not in identifiers:
            raise ValueError("History references unknown action/candidate")
        if not isinstance(entry["valid"], bool):
            raise ValueError("History validity must be Boolean")
    observations = state.get("observations", {})
    if not isinstance(observations, dict) or set(observations) - identifiers:
        raise ValueError("Observations reference unknown candidate")
    for candidate_id, by_tool in observations.items():
        if not isinstance(by_tool, dict) or any(tool not in TOOLS[:-1] for tool in by_tool):
            raise ValueError("Observation tool is unknown")
        _numeric_tree(by_tool)
    remaining = state.get("remaining_actions", 3 - len(history))
    verifies = state.get("remaining_verifications", 1)
    if isinstance(remaining, bool) or not isinstance(remaining, int) or not 0 <= remaining <= 3 - len(history):
        raise ValueError("Invalid remaining action budget")
    if isinstance(verifies, bool) or not isinstance(verifies, int) or not 0 <= verifies <= 2:
        raise ValueError("Invalid remaining verification budget")
    return {"candidates": cleaned, "observations": observations, "history": history,
            "available_tools": tools, "remaining_actions": remaining,
            "remaining_verifications": verifies}


def _reject(reason):
    return {"action": "stop", "candidate_id": None, "reason_code": "invalid_action",
            "valid": False, "validation_error": reason}


def parse_action(raw_text: str, state: dict) -> dict:
    state = validate_state(state)
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    try:
        action = json.loads(raw_text, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return _reject("response_is_not_exact_json")
    if not isinstance(action, dict) or set(action) != {"action", "candidate_id", "reason_code"}:
        return _reject("action_schema_mismatch")
    if not isinstance(action["action"], str) or not isinstance(action["reason_code"], str):
        return _reject("action_fields_must_be_strings")
    if action["candidate_id"] is not None and not isinstance(action["candidate_id"], str):
        return _reject("candidate_id_must_be_string_or_null")
    if action["action"] not in state["available_tools"] or action["reason_code"] not in REASONS:
        return _reject("unknown_action_or_reason")
    if action["action"] == "stop":
        if action["candidate_id"] is not None:
            return _reject("stop_must_have_null_candidate")
        return {**action, "valid": True, "validation_error": None}
    candidate_ids = {candidate["candidate_id"] for candidate in state["candidates"]}
    if action["candidate_id"] not in candidate_ids:
        return _reject("unknown_candidate")
    if state["remaining_actions"] <= 0:
        return _reject("action_budget_exhausted")
    if action["action"] == "verify" and state["remaining_verifications"] <= 0:
        return _reject("verification_budget_exhausted")
    performed = {(h["candidate_id"], h["action"]) for h in state["history"]}
    performed.update((candidate_id, tool) for candidate_id, obs in state["observations"].items() for tool in obs)
    if (action["candidate_id"], action["action"]) in performed:
        return _reject("repeated_tool")
    return {**action, "valid": True, "validation_error": None}


class BoundedQwenController:
    """No automatic download, CPU offload, grammar repair, or probability override."""
    def __init__(self, *, model_lock="manifests/model_lock.json",
                 max_context_tokens=4096, max_output_tokens=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Controller requires exactly one pinned CUDA GPU")
        gpu_config = json.loads((PROJECT_ROOT / "configs" / "gpu.json").read_text())
        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_config["gpu_uuid"]:
            raise RuntimeError("Controller must run through scripts/run_on_ada.py")
        if torch.cuda.get_device_name(0) != gpu_config["gpu_name"]:
            raise RuntimeError("Unexpected CUDA model")
        lock = json.loads(project_path(model_lock).read_text())[MODEL_ID]
        if not lock.get("download_complete"):
            raise RuntimeError("Pinned Qwen download is incomplete")
        self.revision = lock["revision"]
        self.model_path = project_path(lock["path"])
        self.max_context_tokens = min(int(max_context_tokens), 4096)
        self.max_output_tokens = min(int(max_output_tokens), 128)
        if self.max_context_tokens <= self.max_output_tokens or self.max_output_tokens < 1:
            raise ValueError("Invalid controller token limits")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False)
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            local_files_only=True, trust_remote_code=False, attn_implementation="sdpa")
        self.model.eval()
        self.model.requires_grad_(False)
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("Controller parameters left CUDA; CPU offload is prohibited")
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.prompt_sha256 = fingerprint(SYSTEM_PROMPT)

    def select_action(self, state: dict) -> dict:
        state = validate_state(state)
        common = {"model": MODEL_ID, "revision": self.revision,
                  "prompt_sha256": self.prompt_sha256, "state_sha256": fingerprint(state),
                  "max_output_tokens": self.max_output_tokens, "generation": "greedy",
                  "measurement_mode": "live_model_call"}
        if state["remaining_actions"] == 0:
            return {**common, "action": "stop", "candidate_id": None,
                    "reason_code": "budget_exhausted", "valid": True, "validation_error": None,
                    "raw_text": "", "input_tokens": 0, "output_tokens": 0,
                    "latency_seconds": 0.0, "cuda_elapsed_ms": 0.0,
                    "measurement_mode": "deterministic_budget_stop"}
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(state, sort_keys=True, separators=(",", ":"))}]
        start = time.perf_counter()
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens + self.max_output_tokens > self.max_context_tokens:
            return {**common, **_reject("context_budget_exceeded"), "raw_text": "",
                    "input_tokens": input_tokens, "output_tokens": 0,
                    "latency_seconds": time.perf_counter() - start, "cuda_elapsed_ms": 0.0,
                    "measurement_mode": "deterministic_context_stop"}
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        self.torch.cuda.synchronize()
        cuda_start = self.torch.cuda.Event(enable_timing=True)
        cuda_end = self.torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded, max_new_tokens=self.max_output_tokens,
                do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id)
        cuda_end.record()
        self.torch.cuda.synchronize()
        generated = output[0, input_tokens:]
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        action = parse_action(raw_text, state)
        return {**common, **action, "raw_text": raw_text, "input_tokens": input_tokens,
                "output_tokens": int(generated.numel()),
                "latency_seconds": time.perf_counter() - start,
                "cuda_elapsed_ms": float(cuda_start.elapsed_time(cuda_end)),
                "peak_allocated_mib": self.torch.cuda.max_memory_allocated() / 1024**2}

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()
