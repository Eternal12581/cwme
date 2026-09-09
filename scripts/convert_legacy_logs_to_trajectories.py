#!/usr/bin/env python3
"""
Convert legacy text logs into the JSONL trajectory format used by the
offline HKG and procedural-memory builders.

Only the logger record that contains ``Total Score`` is consumed.  The
``<Action>`` entries in the initial-plan dump are deliberately ignored,
because they are proposed actions and were not necessarily executed.

Usage:
  python scripts/convert_legacy_logs_to_trajectories.py \
      --input log/alf_7_seen/result.log \
      --output artifacts/training_trajectories_from_logs.jsonl

The converter reconstructs ``state_before`` from the previous action's
response within the same episode.  The first action normally has no
pre-action observation in legacy logs; those steps are retained with an
empty ``state_before`` and ``state_before_missing=true`` so downstream
analysis cannot mistake the approximation for a complete trajectory.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


_EPISODE_START_RE = re.compile(
    r"Episode start:\s*task=(?P<task>\S+)\s+"
    r"variation=(?P<variation>\S+)\s+ep=(?P<episode>\d+)/(?P<repeat>\d+)",
    re.IGNORECASE,
)
_EPISODE_FINISH_RE = re.compile(
    r"Finished task=(?P<task>\S+)\s+variation=(?P<variation>\S+)\s+"
    r"ep=(?P<episode>\d+)/(?P<repeat>\d+)\s+score=(?P<score>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"(?:Exectued|Executed)\s+action\s+`(?P<action>.*?)`\s+\|\s+"
    r"received response\s+`(?P<response>.*?)`",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(r"Total Score\s+(?P<score>-?\d+(?:\.\d+)?)", re.IGNORECASE)
_LOG_FILE_NAMES = frozenset({"result.log", "experiment.log"})


def _decode_log_value(raw: str) -> str:
    """Decode either a Python-list representation or a plain log string."""
    value = (raw or "").strip()
    try:
        decoded = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        decoded = value

    if isinstance(decoded, (list, tuple)):
        for item in decoded:
            if str(item).strip():
                return str(item).strip()
        return ""
    if decoded is None:
        return ""
    return str(decoded).strip()


def _clean_response(raw: str) -> str:
    """Remove logger bookkeeping appended inside the response field."""
    response = _decode_log_value(raw)
    response = re.split(r"\s*\|\s*Total Score\s+-?\d+(?:\.\d+)?", response, maxsplit=1, flags=re.IGNORECASE)[0]
    return response.strip()


@dataclass
class _Episode:
    task: str
    variation_id: str
    source_file: str
    source_episode: int
    source_repeat: int
    start_line: int
    steps: list[dict] = field(default_factory=list)
    last_state: str = ""
    last_score: float = 0.0
    final_score: float | None = None
    finish_line: int | None = None

    @property
    def complete(self) -> bool:
        return self.finish_line is not None

    def add_action(self, action: str, response: str, score: float, line_no: int) -> None:
        state_before = self.last_state
        score_before = self.last_score
        self.steps.append(
            {
                "step": len(self.steps),
                "state_before": state_before,
                "action": action,
                "state_after": response,
                "score_before": score_before,
                "score_after": score,
                "valid": True,
                "meaningful_change": bool(
                    state_before and response and state_before != response
                ),
                "state_before_missing": not bool(state_before),
                "source_line": line_no,
            }
        )
        self.last_state = response
        self.last_score = score

    def finish(self, score: float, line_no: int) -> None:
        self.final_score = score
        self.finish_line = line_no

    def to_row(self, episode_id: int) -> dict:
        final_score = self.last_score if self.final_score is None else self.final_score
        return {
            "episode_id": episode_id,
            "task": self.task,
            "task_id": self.task,
            "variation_id": self.variation_id,
            "final_score": final_score,
            "success": final_score >= 100,
            "steps": self.steps,
            "source": {
                "kind": "legacy_text_log",
                "file": self.source_file,
                "start_line": self.start_line,
                "finish_line": self.finish_line,
                "source_episode": self.source_episode,
                "source_repeat": self.source_repeat,
                "complete": self.complete,
                "state_before_missing_steps": sum(
                    1 for step in self.steps if step["state_before_missing"]
                ),
            },
        }


def _iter_log_files(inputs: Iterable[str]) -> list[Path]:
    files: dict[str, Path] = {}
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            files[str(path.resolve())] = path
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.name.lower() in _LOG_FILE_NAMES:
                files[str(candidate.resolve())] = candidate
    return [files[key] for key in sorted(files)]


def parse_legacy_logs(
    inputs: Iterable[str],
    *,
    include_incomplete: bool = False,
    tasks: set[str] | None = None,
    variations: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Parse selected legacy log files and return rows plus quality stats."""
    rows: list[dict] = []
    stats = {
        "input_files": [],
        "episodes_started": 0,
        "episodes_finished": 0,
        "episodes_emitted": 0,
        "episodes_skipped_incomplete": 0,
        "episodes_skipped_without_actions": 0,
        "orphan_action_records": 0,
        "duplicate_or_unexpected_boundaries": 0,
        "executed_action_records": 0,
        "episodes_filtered": 0,
    }
    next_episode_id = 0

    for path in _iter_log_files(inputs):
        source_file = str(path.resolve())
        stats["input_files"].append(source_file)
        current: _Episode | None = None

        def flush() -> None:
            nonlocal current, next_episode_id
            if current is None:
                return
            if not current.steps:
                stats["episodes_skipped_without_actions"] += 1
            elif tasks and current.task not in tasks:
                stats["episodes_filtered"] += 1
            elif variations and current.variation_id not in variations:
                stats["episodes_filtered"] += 1
            elif current.complete or include_incomplete:
                rows.append(current.to_row(next_episode_id))
                next_episode_id += 1
                stats["episodes_emitted"] += 1
            else:
                stats["episodes_skipped_incomplete"] += 1
            current = None

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                start = _EPISODE_START_RE.search(line)
                if start:
                    if current is not None:
                        stats["duplicate_or_unexpected_boundaries"] += 1
                        flush()
                    current = _Episode(
                        task=start.group("task"),
                        variation_id=start.group("variation"),
                        source_file=source_file,
                        source_episode=int(start.group("episode")),
                        source_repeat=int(start.group("repeat")),
                        start_line=line_no,
                    )
                    stats["episodes_started"] += 1
                    continue

                finish = _EPISODE_FINISH_RE.search(line)
                if finish:
                    if current is None:
                        stats["duplicate_or_unexpected_boundaries"] += 1
                    else:
                        current.finish(float(finish.group("score")), line_no)
                        stats["episodes_finished"] += 1
                        flush()
                    continue

                action = _ACTION_RE.search(line)
                if not action:
                    continue
                response_raw = action.group("response")
                score_match = _SCORE_RE.search(response_raw)
                if not score_match:
                    # The unprefixed diagnostic line has the same wording but
                    # no cumulative score; the following logger line is the
                    # authoritative execution record.
                    continue
                stats["executed_action_records"] += 1
                if current is None:
                    stats["orphan_action_records"] += 1
                    continue
                current.add_action(
                    _decode_log_value(action.group("action")),
                    _clean_response(response_raw),
                    float(score_match.group("score")),
                    line_no,
                )
        flush()

    return rows, stats


def _write_jsonl(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert selected legacy result.log files to trajectory JSONL"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more result.log files or directories; selection is explicit",
    )
    parser.add_argument("--output", required=True, help="Output trajectory JSONL path")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Keep episodes without Finished task=...; default skips them",
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        default=[],
        help="Keep only this task name; repeat for multiple tasks",
    )
    parser.add_argument(
        "--variation",
        dest="variations",
        action="append",
        default=[],
        help="Keep only this variation id; repeat for multiple ids",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional JSON report path (default: <output>.report.json)",
    )
    args = parser.parse_args()

    rows, stats = parse_legacy_logs(
        args.input,
        include_incomplete=args.include_incomplete,
        tasks={str(x).strip() for x in args.tasks if str(x).strip()} or None,
        variations={str(x).strip() for x in args.variations if str(x).strip()} or None,
    )
    output = Path(args.output).expanduser()
    _write_jsonl(rows, output)
    report = Path(args.report).expanduser() if args.report else Path(f"{output}.report.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({**stats, "output": str(output.resolve())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not rows:
        raise SystemExit(
            "No complete episodes were converted; inspect the report and "
            "select logs containing Episode start/Finished task boundaries."
        )

    print(f"Converted {stats['episodes_emitted']} episodes from {len(stats['input_files'])} log files")
    print(f"  executed records: {stats['executed_action_records']}")
    print(f"  skipped incomplete: {stats['episodes_skipped_incomplete']}")
    print(f"  skipped without actions: {stats['episodes_skipped_without_actions']}")
    print(f"  filtered by task/variation: {stats['episodes_filtered']}")
    print(f"  orphan action records: {stats['orphan_action_records']}")
    print(f"  trajectories: {output}")
    print(f"  report: {report}")


if __name__ == "__main__":
    main()
