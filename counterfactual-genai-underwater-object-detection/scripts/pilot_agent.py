#!/usr/bin/env python3
"""Validate strict controller schemas; optionally run frozen Qwen on numeric states."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uac.agent import BoundedQwenController, parse_action, validate_state
from uac.runtime import EventLedger, atomic_json, completed_stage, fingerprint, project_path


def fixture_states():
    # These are protocol fixtures, not acoustic measurements or research results.
    # The measured pilot must pass --states with development evidence.
    result = []
    for index in range(24):
        observed = index % 4
        history, observations = [], {}
        tools = ("fourier", "stft", "wavelet")
        for j in range(min(observed, 2)):
            history.append({"action": tools[j], "candidate_id": "fixture", "valid": True})
            observations[tools[j]] = {"candidate_probability": 0.3 + 0.1 * j,
                                      "numeric_fixture": True}
        result.append({
            "candidates": [{"candidate_id": "fixture", "interval_seconds": [0, 3],
                            "base_probability": [0.1, 0.4, 0.6, 0.9][index % 4],
                            "uncertainty": [0.2, 0.8, 0.8, 0.2][index % 4]}],
            "observations": {"fixture": observations}, "history": history,
            "available_tools": ["fourier", "stft", "wavelet", "verify", "stop"],
            "remaining_actions": 3 - len(history), "remaining_verifications": 1,
        })
    return result


def parser_checks():
    state = fixture_states()[0]
    fence = chr(96) * 3
    cases = [
        ('{"action":"fourier","candidate_id":"fixture","reason_code":"missing_representation"}', True),
        ('{"action":"stop","candidate_id":null,"reason_code":"sufficient_evidence"}', True),
        ('{"action":"verify","candidate_id":"missing","reason_code":"verify_candidate"}', False),
        ('{"action":"fourier","candidate_id":"fixture","reason_code":"missing_representation","probability":0.99}', False),
        (fence + 'json\n{"action":"stop","candidate_id":null,"reason_code":"sufficient_evidence"}\n' + fence, False),
        ('not JSON', False),
        ('{"action":"verify","candidate_id":[],"reason_code":"verify_candidate"}', False),
        ('{"action":"stop","action":"verify","candidate_id":"fixture","reason_code":"verify_candidate"}', False),
        ('{"action":[],"candidate_id":"fixture","reason_code":"verify_candidate"}', False),
    ]
    for raw, valid in cases:
        assert parse_action(raw, state)["valid"] is valid
    assert not parse_action(cases[0][0], fixture_states()[1])["valid"]
    return len(cases) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, help="Project-local JSON list of measured development states")
    parser.add_argument("--output", default="manifests/agent_pilot.json")
    parser.add_argument("--parser-only", action="store_true")
    args = parser.parse_args()
    count = parser_checks()
    if args.parser_only:
        print(json.dumps({"status": "passed", "strict_parser_cases": count, "model_loaded": False}))
        return 0
    states = json.loads(project_path(args.states).read_text()) if args.states else fixture_states()
    for state in states:
        validate_state(state)
    if not states:
        raise ValueError("At least one pilot state is required")
    config = {"version": 1, "states_hash": fingerprint(states),
              "source": str(args.states) if args.states else "protocol_fixtures"}
    output = project_path(args.output)
    partial = output.with_suffix(".partial.json")
    rows = []
    if partial.exists():
        previous = json.loads(partial.read_text())
        if previous["config"] != config:
            raise RuntimeError("Existing partial pilot has different states; choose another output")
        rows = previous["actions"]
    controller = BoundedQwenController()
    load_seconds = controller.load_seconds
    try:
        for index in range(len(rows), len(states)):
            action = controller.select_action(states[index])
            rows.append({"state_index": index, "action": action})
            atomic_json(partial, {"config": config, "actions": rows})
    finally:
        controller.close()
    valid = sum(row["action"]["valid"] for row in rows)
    rate = valid / len(rows)
    report = {
        "status": "passed" if rate >= 0.98 else "failed_schema_gate",
        "schema_valid_count": valid, "calls": len(rows), "schema_valid_rate": rate,
        "required_schema_valid_rate": 0.98, "strict_parser_cases": count,
        "state_origin": "measured_development_evidence" if args.states else "protocol_fixtures",
        "measured_evidence_pilot_complete": bool(args.states),
        "model_load_seconds": load_seconds, "config": config, "actions": rows,
        "limitation": "Formatting/tool validity pilot only; does not establish detection improvement or runtime savings.",
    }
    atomic_json(output, report)
    completed_stage(output.with_suffix(".receipt.json"), stage="agent_schema_pilot",
                    config_hash=fingerprint(config), artifacts=[output],
                    details={"schema_valid_rate": rate, "state_origin": report["state_origin"]})
    EventLedger().append("agent_pilot_complete", output=str(output.relative_to(ROOT)),
                         schema_valid_rate=rate, state_origin=report["state_origin"])
    print(json.dumps({key: value for key, value in report.items() if key != "actions"}, indent=2))
    return 0 if rate >= 0.98 else 2


if __name__ == "__main__":
    raise SystemExit(main())
