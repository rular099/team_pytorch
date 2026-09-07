#!/usr/bin/env python3
"""Verify and merge deterministic query-geometry event shards.

Every input shard must have a valid completion manifest, matching run identity,
the expected round-robin event-occurrence assignment, and a disjoint part of
the complete pinned validation index space. A resampled physical event ID may
have multiple occurrences; every occurrence keeps all seven validation times.
The merged metrics are recomputed from the globally ordered raw sample arrays;
shard-level summary statistics are never averaged.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import diagnose_query_geometry_sensitivity as querydiag  # noqa: E402


METADATA_ARRAY_NAMES = {
    "provenance_json",
    "metric_definitions_json",
    "resolved_config_json",
}
INVARIANT_ARRAY_NAMES = {"radial_scales", "requested_station_counts"}


def shard_prefix(base: Path, shard_id: int, num_shards: int) -> Path:
    return Path(f"{base}.shard-{shard_id:05d}-of-{num_shards:05d}")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_completion_manifest(
    paths: Mapping[str, Path],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    completion_path = paths["completion"]
    if not completion_path.is_file():
        raise FileNotFoundError(f"Missing completion manifest: {completion_path}")
    completion = _load_json(completion_path)
    if completion.get("status") != "complete":
        raise ValueError(
            f"Completion manifest does not have status=complete: {completion_path}"
        )
    artifacts = completion.get("artifacts")
    required = {"resolved_config", "samples", "summary"}
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError(
            f"Completion manifest artifact set must be {sorted(required)}: "
            f"{completion_path}"
        )
    verified: Dict[str, Dict[str, Any]] = {}
    for name in sorted(required):
        path = paths[name]
        record = artifacts[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed artifact: {path}")
        actual_size = path.stat().st_size
        actual_sha = querydiag.sha256_file(path)
        if record.get("file_size_bytes") != actual_size:
            raise ValueError(f"Completion size mismatch for {path}")
        if record.get("sha256") != actual_sha:
            raise ValueError(f"Completion SHA-256 mismatch for {path}")
        verified[name] = {
            "path": str(path.resolve()),
            "file_size_bytes": int(actual_size),
            "sha256": actual_sha,
        }
    return completion, verified


def _identity_without_shard(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    identity = copy.deepcopy(dict(provenance))
    identity.pop("invocation", None)
    identity.pop("event_sharding", None)
    return identity


def _load_verified_shard(
    prefix: Path,
    *,
    expected_shard_id: int,
    expected_num_shards: int,
) -> Dict[str, Any]:
    paths = querydiag.diagnostic_output_paths(prefix)
    completion, verified_artifacts = verify_completion_manifest(paths)
    summary = _load_json(paths["summary"])
    config = _load_json(paths["resolved_config"])
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Summary has no provenance object: {paths['summary']}")
    provenance_sharding = provenance.get("event_sharding")
    selection_sharding = summary.get("selection", {}).get("event_sharding")
    if not isinstance(provenance_sharding, dict):
        raise ValueError(f"Missing event_sharding provenance: {paths['summary']}")
    if provenance_sharding != selection_sharding:
        raise ValueError(
            f"Summary/provenance event_sharding mismatch: {paths['summary']}"
        )
    expected_fields = {
        "mode": "shard",
        "algorithm": querydiag.EVENT_SHARDING_ALGORITHM,
        "num_event_shards": expected_num_shards,
        "event_shard_id": expected_shard_id,
        "samples_per_event": len(querydiag.PINNED_VALIDATION_TIMES),
    }
    for key, expected in expected_fields.items():
        if provenance_sharding.get(key) != expected:
            raise ValueError(
                f"Shard {expected_shard_id} has invalid {key}: "
                f"{provenance_sharding.get(key)!r} != {expected!r}"
            )

    with np.load(paths["samples"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required_arrays = {
        "event_index",
        "event_ordinal",
        "event_key",
        "provenance_json",
        "metric_definitions_json",
        "resolved_config_json",
        *INVARIANT_ARRAY_NAMES,
    }
    for required_name in sorted(required_arrays):
        if required_name not in arrays:
            raise ValueError(
                f"Shard {expected_shard_id} is missing array {required_name!r}"
            )
    embedded_provenance = json.loads(str(arrays["provenance_json"]))
    if embedded_provenance != provenance:
        raise ValueError(
            f"NPZ/summary provenance mismatch for shard {expected_shard_id}"
        )
    if json.loads(str(arrays["resolved_config_json"])) != config:
        raise ValueError(
            f"NPZ/artifact resolved-config mismatch for shard {expected_shard_id}"
        )
    if json.loads(str(arrays["metric_definitions_json"])) != summary.get(
        "metric_definitions"
    ):
        raise ValueError(
            f"NPZ/summary metric-definition mismatch for shard {expected_shard_id}"
        )

    sample_count = int(arrays["event_index"].shape[0])
    sample_names = set(arrays) - METADATA_ARRAY_NAMES - INVARIANT_ARRAY_NAMES
    for name in sample_names:
        if arrays[name].ndim == 0 or arrays[name].shape[0] != sample_count:
            raise ValueError(
                f"Shard {expected_shard_id} array {name!r} is not sample-aligned"
            )

    samples_per_event = int(provenance_sharding["samples_per_event"])
    event_index = arrays["event_index"].astype(np.int64)
    event_ordinal = arrays["event_ordinal"].astype(np.int64)
    selection = summary.get("selection", {})
    dataset_samples = selection.get("dataset_realtime_samples")
    dataset_event_occurrences = provenance_sharding.get(
        "dataset_event_occurrences"
    )
    if (
        dataset_samples is None
        or dataset_event_occurrences is None
        or int(dataset_event_occurrences) * samples_per_event
        != int(dataset_samples)
    ):
        raise ValueError(
            f"Shard {expected_shard_id} has inconsistent full-dataset counts"
        )
    if summary.get("counts", {}).get("realtime_samples") != sample_count:
        raise ValueError(f"Shard {expected_shard_id} realtime sample count mismatch")
    if selection.get("examined_realtime_samples") != sample_count:
        raise ValueError(f"Shard {expected_shard_id} examined sample count mismatch")
    if np.unique(event_index).size != sample_count:
        raise ValueError(f"Shard {expected_shard_id} contains duplicate event_index")
    if np.any(event_ordinal % expected_num_shards != expected_shard_id):
        raise ValueError(f"Shard {expected_shard_id} violates round-robin assignment")
    if np.any(event_index // samples_per_event != event_ordinal):
        raise ValueError(
            f"Shard {expected_shard_id} event_index/event_ordinal mapping is invalid"
        )
    for ordinal in np.unique(event_ordinal):
        mask = event_ordinal == ordinal
        expected_indices = np.arange(
            int(ordinal) * samples_per_event,
            (int(ordinal) + 1) * samples_per_event,
        )
        if not np.array_equal(np.sort(event_index[mask]), expected_indices):
            raise ValueError(
                f"Shard {expected_shard_id} does not contain the complete event "
                f"block for ordinal {int(ordinal)}"
            )
        if np.unique(arrays["event_key"][mask]).size != 1:
            raise ValueError(
                f"Shard {expected_shard_id} event ordinal {int(ordinal)} has "
                "multiple event keys"
            )
    selected_event_occurrences = int(np.unique(event_ordinal).size)
    if (
        provenance_sharding.get("selected_event_occurrences")
        != selected_event_occurrences
    ):
        raise ValueError(
            f"Shard {expected_shard_id} selected_event_occurrences mismatch"
        )
    selected_unique_event_keys = int(np.unique(arrays["event_key"]).size)
    if (
        provenance_sharding.get("selected_unique_event_keys")
        != selected_unique_event_keys
    ):
        raise ValueError(
            f"Shard {expected_shard_id} selected_unique_event_keys mismatch"
        )
    if selection.get("selected_events") != selected_unique_event_keys:
        raise ValueError(f"Shard {expected_shard_id} selected_events mismatch")
    if (
        selection.get("selected_event_occurrences")
        != selected_event_occurrences
    ):
        raise ValueError(
            f"Shard {expected_shard_id} selection occurrence count mismatch"
        )
    if summary.get("counts", {}).get("events") != selected_unique_event_keys:
        raise ValueError(f"Shard {expected_shard_id} event count mismatch")
    return {
        "prefix": prefix,
        "completion": completion,
        "verified_artifacts": verified_artifacts,
        "summary": summary,
        "config": config,
        "provenance": provenance,
        "arrays": arrays,
        "sample_names": sample_names,
    }


def merge_shards(
    input_prefix_base: Path,
    num_shards: int,
    output_prefix: Path,
    *,
    force: bool = False,
    invocation_argv: Sequence[str] = (),
) -> Dict[str, Path]:
    if num_shards < 2:
        raise ValueError("num_shards must be at least 2")
    output_paths = querydiag.diagnostic_output_paths(output_prefix)
    querydiag.refuse_existing_outputs(output_paths, force=force)
    shards = [
        _load_verified_shard(
            shard_prefix(input_prefix_base, shard_id, num_shards),
            expected_shard_id=shard_id,
            expected_num_shards=num_shards,
        )
        for shard_id in range(num_shards)
    ]
    reference = shards[0]
    reference_identity = _identity_without_shard(reference["provenance"])
    reference_protocol = reference["summary"].get("resolved_validation_generators")
    reference_checkpoint = reference["summary"].get("checkpoint_parameters")
    reference_names = set(reference["arrays"])
    reference_sample_names = reference["sample_names"]
    for shard_id, shard in enumerate(shards[1:], start=1):
        if shard["config"] != reference["config"]:
            raise ValueError(f"Resolved config mismatch in shard {shard_id}")
        if _identity_without_shard(shard["provenance"]) != reference_identity:
            raise ValueError(f"Run provenance mismatch in shard {shard_id}")
        if shard["summary"].get("resolved_validation_generators") != reference_protocol:
            raise ValueError(f"Generator protocol mismatch in shard {shard_id}")
        if shard["summary"].get("checkpoint_parameters") != reference_checkpoint:
            raise ValueError(f"Checkpoint parameter report mismatch in shard {shard_id}")
        if (
            shard["summary"]["selection"].get("dataset_realtime_samples")
            != reference["summary"]["selection"].get("dataset_realtime_samples")
        ):
            raise ValueError(f"Full dataset size mismatch in shard {shard_id}")
        if set(shard["arrays"]) != reference_names:
            raise ValueError(f"NPZ array-name mismatch in shard {shard_id}")
        if shard["sample_names"] != reference_sample_names:
            raise ValueError(f"Sample-array mismatch in shard {shard_id}")
        for name in INVARIANT_ARRAY_NAMES:
            if not np.array_equal(shard["arrays"][name], reference["arrays"][name]):
                raise ValueError(f"Invariant array {name!r} mismatch in shard {shard_id}")

    merged_arrays = {
        name: np.concatenate([shard["arrays"][name] for shard in shards], axis=0)
        for name in sorted(reference_sample_names)
    }
    merged_arrays.update({
        name: reference["arrays"][name].copy()
        for name in INVARIANT_ARRAY_NAMES
    })
    order = np.argsort(merged_arrays["event_index"], kind="stable")
    for name in reference_sample_names:
        merged_arrays[name] = merged_arrays[name][order]

    sharding = reference["provenance"]["event_sharding"]
    dataset_samples = int(reference["summary"]["selection"]["dataset_realtime_samples"])
    dataset_event_occurrences = sharding.get("dataset_event_occurrences")
    samples_per_event = int(sharding["samples_per_event"])
    if (
        dataset_event_occurrences is None
        or int(dataset_event_occurrences) * samples_per_event != dataset_samples
    ):
        raise ValueError("Shard metadata has an inconsistent full dataset size")
    expected_indices = np.arange(dataset_samples, dtype=np.int64)
    if not np.array_equal(merged_arrays["event_index"], expected_indices):
        raise ValueError(
            "Shard event_index union is not an exact, disjoint cover of the full dataset"
        )
    expected_ordinals = expected_indices // samples_per_event
    if not np.array_equal(merged_arrays["event_ordinal"], expected_ordinals):
        raise ValueError("Merged event ordinals do not match the complete index space")

    provenance = copy.deepcopy(reference["provenance"])
    provenance["invocation"] = {
        "argv": [str(value) for value in invocation_argv],
        "cwd": str(Path.cwd().resolve()),
    }
    provenance["event_sharding"] = {
        "mode": "merged",
        "algorithm": querydiag.EVENT_SHARDING_ALGORITHM,
        "num_event_shards": int(num_shards),
        "event_shard_id": None,
        "samples_per_event": samples_per_event,
        "dataset_event_occurrences": int(dataset_event_occurrences),
        "selected_event_occurrences": int(dataset_event_occurrences),
        "selected_unique_event_keys": int(
            np.unique(merged_arrays["event_key"]).size
        ),
    }
    provenance["merge"] = {
        "tool": querydiag.file_provenance(Path(__file__), compute_sha256=True),
        "repository": querydiag.git_provenance(REPO_ROOT),
        "source_shards": [
            {
                "shard_id": shard_id,
                "prefix": str(shard["prefix"].resolve()),
                "completion_manifest": {
                    "path": str(
                        querydiag.diagnostic_output_paths(shard["prefix"])[
                            "completion"
                        ].resolve()
                    ),
                    "sha256": querydiag.sha256_file(
                        querydiag.diagnostic_output_paths(shard["prefix"])[
                            "completion"
                        ]
                    ),
                },
                "verified_artifacts": shard["verified_artifacts"],
            }
            for shard_id, shard in enumerate(shards)
        ],
    }

    radial_scales = merged_arrays["radial_scales"].tolist()
    station_counts = merged_arrays["requested_station_counts"].tolist()
    summary = querydiag._summarize_run(
        merged_arrays,
        radial_scales,
        station_counts,
        float(provenance["equivariance_tolerance"]),
        pair_sample_limit=int(provenance["pair_sample_limit"]),
        seed=int(provenance["diagnostic_seed"]),
    )
    summary["selection"] = copy.deepcopy(reference["summary"]["selection"])
    summary["selection"].update({
        "examined_realtime_samples": dataset_samples,
        "max_events": 0,
        "selected_events": int(np.unique(merged_arrays["event_key"]).size),
        "selected_event_occurrences": int(dataset_event_occurrences),
        "event_sharding": copy.deepcopy(provenance["event_sharding"]),
    })
    summary["resolved_validation_generators"] = copy.deepcopy(reference_protocol)
    summary["checkpoint_parameters"] = copy.deepcopy(reference_checkpoint)
    querydiag.write_outputs(
        output_paths,
        config=reference["config"],
        summary=summary,
        arrays=merged_arrays,
        provenance=provenance,
        force=force,
    )
    return output_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and merge complete query-geometry event shards"
    )
    parser.add_argument("--input-prefix-base", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    invocation_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *[str(value) for value in argv]]
    )
    paths = merge_shards(
        Path(args.input_prefix_base).expanduser(),
        args.num_shards,
        Path(args.output_prefix).expanduser(),
        force=args.force,
        invocation_argv=invocation_argv,
    )
    print("[querydiag-merge] complete")
    for name, path in paths.items():
        print(f"[querydiag-merge] {name}={path.resolve()}")


if __name__ == "__main__":
    main()
