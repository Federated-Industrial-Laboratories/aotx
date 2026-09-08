# SPDX-License-Identifier: Apache-2.0
"""Check fixed reference groups against original calibration bounds.

Inputs: a reference index, complete device captures and fixed bounds.
Outputs: calibration bounds or complete row results with separate group counts.
Errors: ValueError or OSError for incomplete, changed or invalid inputs.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from arch_accuracy_io import (capture_rows, digest, model_sequences, read_json,
                              reference_rows, request, retain_input, verify_inputs, write_json)
from arch_accuracy_metrics import METRICS, assess, calibrate, measure, validate_bounds


def load_bundle(args):
    from arch_accuracy_reference import read_reference_index
    inputs = {}
    retain_input(inputs, "reference_index", args.reference_index)
    bundle = read_reference_index(args.reference_index, args.asset_root)
    inputs.update(bundle["verified_inputs"])
    if args.output.exists():
        raise ValueError("the result file already exists")
    return bundle, inputs


def select_captures(paths, bundle, inputs, calibration_only):
    origins = {"original"} if calibration_only else set(bundle["corpora"])
    expected = {}
    for origin in origins:
        entry = bundle["corpora"][origin]
        for model in entry["corpus"]["models"]:
            expected[(digest(entry["path"]), model["id"])] = (origin, entry)
    selected = []
    seen = set()
    for index, path in enumerate(paths):
        retain_input(inputs, "capture:" + str(index), path)
        value = read_json(path)
        key = (value["corpus_sha256"], value["model_id"])
        if key not in expected or key in seen:
            raise ValueError("a capture is repeated or belongs to another reference group")
        seen.add(key)
        origin, entry = expected[key]
        capture, device = capture_rows(path, entry["corpus"], entry["path"])
        row_file = Path(path).parent / capture["rows_file"]
        retain_input(inputs, "device_rows:" + str(index), row_file, capture["rows_sha256"])
        selected.append((origin, entry, path, capture, device))
    if seen != set(expected):
        raise ValueError("one capture is required for every requested model and reference group")
    if len({capture["executable_sha256"] for _, _, _, capture, _ in selected}) != 1:
        raise ValueError("all captures in one check must use the same executable")
    return selected


def measured(selected, inputs, calibration_only):
    for origin, entry, _, capture, device in selected:
        model, sequences = model_sequences(entry["corpus"], capture["model_id"])
        by_id = {s["id"]: s for s in sequences}
        references = {}
        for index, row in enumerate(capture["rows"]):
            sequence = by_id[row["sequence_id"]]
            if calibration_only and sequence["split"] != "calibration":
                continue
            key = sequence["id"]
            if key not in references:
                references[key] = reference_rows(entry["path"].parent, sequence, model["vocab"])
                path = entry["path"].parent / sequence["reference_rows_file"]
                retain_input(inputs, "reference_rows:" + key, path, sequence["reference_rows_sha256"])
                retain_input(inputs, "reference_tokens:" + key, path.with_name("tokens.txt"),
                             sequence["token_metadata_sha256"])
            reference = references[key][row["reference_index"]]
            yield dict(row, origin=origin, model_id=model["id"], split=sequence["split"],
                       **measure(reference, device[index]))


def calibration_members(bundle):
    corpus = bundle["corpora"]["original"]["corpus"]
    expected = {}
    for model in corpus["models"]:
        by_id = {s["id"]: s for s in corpus["sequences"] if s["model_id"] == model["id"]}
        _, rows = request(corpus, model["id"])
        expected[model["id"]] = [dict(row, model_id=model["id"], origin="original", split="calibration")
                                  for row in rows if by_id[row["sequence_id"]]["split"] == "calibration"]
        if len(expected[model["id"]]) != bundle["calibration_rows"][model["id"]]:
            raise ValueError("the original calibration row count differs")
    return expected


def fit_models(grouped, expected):
    if set(grouped) != set(expected):
        raise ValueError("calibration model membership differs")
    result = {}
    for ident, members in expected.items():
        rows = grouped[ident]
        keys = ("sequence_id", "reference_index", "position", "mode", "model_id", "origin", "split")
        actual = [tuple(row[k] for k in keys) for row in rows]
        wanted = [tuple(row[k] for k in keys) for row in members]
        if len(actual) != len(wanted) or sorted(actual) != sorted(wanted):
            raise ValueError("calibration inputs or membership differ")
        modes = Counter(row["mode"] for row in rows)
        if set(modes) != {"serial", "batch64"}:
            raise ValueError("calibration must cover both modes")
        result[ident] = dict(calibrate(rows), modes=dict(modes))
    return result


def make_bounds(args):
    bundle, inputs = load_bundle(args)
    selected = select_captures(args.captures, bundle, inputs, True)
    expected = calibration_members(bundle)
    grouped = {ident: [] for ident in expected}
    for row in measured(selected, inputs, True):
        grouped[row["model_id"]].append(row)
    models = fit_models(grouped, expected)
    verify_inputs(inputs)
    result = dict(schema="aotx-decoded-f32-bounds-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  reference_index_sha256=inputs["reference_index"]["sha256"],
                  reference_identity=bundle["identity"], models=models, rows=grouped,
                  calibration_captures={str(p): digest(p) for p in args.captures},
                  verified_inputs=inputs)
    write_json(args.output, result)
    print("accuracy: calibration bounds frozen for", len(models), "models,",
          sum(len(rows) for rows in grouped.values()), "rows, 0 failures, 0 skips")
    return 0


def collect_results(stream, bounds, model_ids, totals, row_log):
    rows = []
    counts, clear, different, split_counts = Counter(), Counter(), Counter(), Counter()
    fresh_clear = Counter()
    failed = 0
    for row in stream:
        result = assess(row, bounds[row["model_id"]]["bounds"])
        row.update(result)
        rows.append(row)
        row_log.write(json.dumps(row, allow_nan=False) + "\n")
        row_log.flush()
        key = "/".join((row["model_id"], row["mode"], row["origin"], row["split"]))
        counts[key] += 1
        split_counts[row["split"]] += 1
        clear[key] += result["clear"]
        different[key] += row["reference_top"] != row["device_top"]
        failed += bool(result["failed"])
        if row["origin"] == "fresh" and row["split"] == "validation" and result["clear"]:
            fresh_clear[row["model_id"] + "/" + row["mode"]] += 1
    required = dict(calibration=totals["calibration_runtime_rows"],
                    regression=totals["historical_regression_runtime_rows"],
                    validation=totals["fresh_validation_runtime_rows"])
    if len(rows) != totals["runtime_rows_all_modes"] or dict(split_counts) != required:
        raise ValueError("the complete row or split count differs")
    missing = [ident + "/" + mode for ident in sorted(model_ids) for mode in ("serial", "batch64")
               if fresh_clear[ident + "/" + mode] == 0]
    maxima = {}
    for ident in model_ids:
        selected = [row for row in rows if row["model_id"] == ident]
        if not selected:
            raise ValueError("a model has no measured rows")
        maxima[ident] = {key: max(row[key] for row in selected) for key in METRICS}
    return dict(counts=dict(counts), clear_counts=dict(clear), different_counts=dict(different),
                fresh_clear_counts=dict(fresh_clear), maxima=maxima, missing_clear_modes=missing,
                failed_rows=failed, rows=rows, split_counts=dict(split_counts))


def check_bounds(args):
    bundle, inputs = load_bundle(args)
    retain_input(inputs, "bounds", args.bounds)
    frozen = read_json(args.bounds)
    if (frozen["schema"] != "aotx-decoded-f32-bounds-v1"
            or frozen["reference_index_sha256"] != inputs["reference_index"]["sha256"]
            or frozen["reference_identity"] != bundle["identity"]):
        raise ValueError("calibration reference identity differs")
    expected = calibration_members(bundle)
    if fit_models(frozen["rows"], expected) != frozen["models"]:
        raise ValueError("bounds differ from the original calibration maxima")
    for model in frozen["models"].values():
        validate_bounds(model["bounds"])
    selected = select_captures(args.captures, bundle, inputs, False)
    row_path = args.output.with_suffix(".rows.jsonl")
    with row_path.open("x") as row_log:
        result = collect_results(measured(selected, inputs, False), frozen["models"],
                                 set(expected), bundle["totals"], row_log)
    verify_inputs(inputs)
    result.update(schema="aotx-decoded-f32-validation-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  reference_index_sha256=inputs["reference_index"]["sha256"],
                  reference_identity=bundle["identity"], bounds_sha256=inputs["bounds"]["sha256"],
                  captures={str(p): digest(p) for p in args.captures},
                  row_log_file=row_path.name, row_log_sha256=digest(row_path), verified_inputs=inputs)
    write_json(args.output, result)
    print(f"accuracy: {len(result['rows'])} rows, {result['failed_rows']} failed rows, "
          f"{len(result['missing_clear_modes'])} modes without clear fresh winners, 0 skips")
    for ident, values in result["maxima"].items():
        print("accuracy:", ident, values)
    return int(result["failed_rows"] != 0 or bool(result["missing_clear_modes"]))
