"""Legacy artifact readers retained for inspecting historical offline runs."""

import json
import math
import os
from pathlib import Path

from .io import digest, file_hash, read_json


def append_jsonl(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def records(path, repair_tail=False):
    """Only the locked writer may discard a torn final line after interruption."""
    with open(path, "r+b" if repair_tail else "rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                if repair_tail:
                    stream.seek(offset)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    return
                raise ValueError(
                    f"Incomplete JSONL tail in {path}; resume its producing stage"
                )
            yield json.loads(line)


def read_rollouts(path, repair_tail=False, keep_samples=True):
    spec, prompts, samples, complete = None, {}, {}, False
    for record in records(path, repair_tail):
        if complete:
            raise ValueError("Unexpected records after rollout completion")
        kind = record["type"]
        if spec is None:
            if kind != "metadata":
                raise ValueError("Rollout JSONL must start with metadata")
            spec = record["spec"]
            if spec["format_version"] != 2:
                raise ValueError("Unsupported rollout format")
            expected = set(spec["prompt_ids"])
            if len(expected) != len(spec["prompt_ids"]) or spec["group_size"] < 2:
                raise ValueError("Invalid rollout group specification")
        elif kind == "prompt":
            p = record["prompt"]
            if p["id"] not in expected or p["id"] in prompts or not p["prompt_ids"]:
                raise ValueError("Invalid/duplicate prompt")
            prompts[p["id"]] = p
        elif kind == "completion":
            key = (record["prompt_id"], record["sample_index"])
            sample = record["sample"]
            n = len(sample["completion_ids"])
            if (
                key[0] not in prompts
                or not 0 <= key[1] < spec["group_size"]
                or key in samples
                or not n
                or len(sample["old_logprobs"]) != n
                or len(sample["mask"]) != n
                or not any(sample["mask"])
                or not all(m in (0, 1) for m in sample["mask"])
                or not all(math.isfinite(x) for x in sample["old_logprobs"])
            ):
                raise ValueError("Invalid/duplicate completion")
            samples[key] = sample if keep_samples else None
        elif kind == "complete":
            if (
                len(prompts) != len(expected)
                or len(samples) != len(expected) * spec["group_size"]
            ):
                raise ValueError("Incomplete rollout groups")
            complete = True
        else:
            raise ValueError(f"Unexpected rollout record: {kind}")
    return spec, prompts, samples, complete


def read_judgments(path, repair_tail=False):
    spec, scores = None, {}
    for record in records(path, repair_tail):
        if spec is None:
            if record["type"] != "metadata":
                raise ValueError("Judgment JSONL must start with metadata")
            spec = record["spec"]
        elif record["type"] == "judgment":
            key = (record["prompt_id"], record["sample_index"])
            judge = record["judge"]
            if key in scores or not math.isfinite(judge["reward"]):
                raise ValueError("Invalid/duplicate judgment")
            scores[key] = judge
        elif record["type"] != "error":
            raise ValueError("Unexpected judgment record")
    return spec, scores


def checked_manifest(path):
    """Validate the small Stage 2 manifest and both immutable JSONL sources."""
    path = Path(path)
    manifest = read_json(path)
    if manifest.get("format_version") != 2 or not manifest.get("complete"):
        raise ValueError("Stage 2 manifest is incomplete or unsupported")
    payload = {k: v for k, v in manifest.items() if k != "dataset_id"}
    if manifest["dataset_id"] != digest(payload):
        raise ValueError("Manifest integrity failure")
    for key in ("rollouts", "judgments"):
        item = manifest[key]
        if file_hash(path.parent / item["path"]) != item["sha256"]:
            raise ValueError(f"{key} JSONL changed after Stage 2 export")
    return manifest
