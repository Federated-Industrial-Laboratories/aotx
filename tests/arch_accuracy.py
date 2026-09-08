#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture and check full architecture logits against fixed CPU rows.

Inputs: a corpus, model store, capture program and calibration bounds.
Outputs: immutable capture, calibration and validation files with complete counts.
Exit codes: zero on success, one on failed checks or invalid inputs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from arch_accuracy_io import (capture_rows, checked_capture, checked_text, digest, identifier,
                              integer, model_sequences, read_corpus, read_json, read_manifest,
                              reference_rows, request, retain_input, verify_inputs, write_json)
from arch_accuracy_metrics import METRICS, assess, calibrate, measure, validate_bounds


def now():
    return datetime.now(timezone.utc).isoformat()


def capture(args):
    inputs = {}
    retain_input(inputs, "corpus", args.corpus)
    retain_input(inputs, "executable", args.executable)
    identifier(args.model)
    identifier(args.role)
    checked_text(str(args.output))
    corpus = read_corpus(args.corpus, getattr(args, "asset_root", None))
    reference = corpus.get("_reference")
    if reference:
        inputs.update(reference["verified_inputs"])
    model, _ = model_sequences(corpus, args.model)
    manifest = args.store / "manifest.jsonl"
    retain_input(inputs, "store_manifest", manifest)
    entries = read_manifest(manifest)
    selected = [entry for entry in entries if entry["role"] == args.role]
    if len(selected) != 1 or selected[0]["sha256"] != model["sha256"]:
        raise ValueError("the store role does not select the requested model")
    selected = selected[0]
    stored = retain_input(inputs, "model", args.store / selected["path"], model["sha256"])
    if any(entry.get("bytes", stored["bytes"]) != stored["bytes"] for entry in (selected, model)):
        raise ValueError("the model size differs from its identity")
    data, rows = request(corpus, args.model)
    args.output.mkdir()
    input_path = args.output / "request.bin"
    input_path.write_bytes(data)
    retain_input(inputs, "request", input_path, hashlib.sha256(data).hexdigest())
    binary_path = args.output / "rows.f32"
    command = [str(args.executable.resolve()), str(args.store.resolve()), args.role,
               str(input_path.resolve()), str(binary_path.resolve())]
    identity = dict(schema_version=1, created_utc=now(), corpus_sha256=inputs["corpus"]["sha256"],
                    model_id=args.model, model_sha256=model["sha256"], vocab=model["vocab"],
                    executable_sha256=inputs["executable"]["sha256"], input_sha256=inputs["request"]["sha256"],
                    store_manifest_sha256=inputs["store_manifest"]["sha256"],
                    rows_file="rows.f32", rows=rows, command=command, verified_inputs=inputs)
    if reference:
        identity["reference_basis"] = dict(kind=reference["kind"], origin=reference["origin"],
                                           status=reference["status"],
                                           runtime_original=reference["bases"][args.model]["runtime_original"],
                                           reference_derived=reference["bases"][args.model]["reference_derived"])
    with (args.output / "capture.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ValueError("the device capture failed; see capture.log")
    identity["rows_sha256"] = digest(binary_path)
    identity["completed_utc"] = now()
    checked_capture(identity, args.output, corpus, inputs["corpus"]["sha256"])
    verify_inputs(inputs)
    write_json(args.output / "capture.json", identity)
    print(f"accuracy: captured {len(rows)} complete rows, 0 failures, 0 skips")
    return 0


def measured_rows(args, calibration_only):
    corpus = read_corpus(args.corpus)
    model_ids = {model["id"] for model in corpus["models"]}
    captures = [read_json(path) for path in args.captures]
    if len(captures) != len(model_ids) or {c["model_id"] for c in captures} != model_ids:
        raise ValueError("one capture is required for every model")
    for path in args.captures:
        capture, device = capture_rows(path, corpus, args.corpus)
        model, sequences = model_sequences(corpus, capture["model_id"])
        by_id = {s["id"]: s for s in sequences}
        references = {}
        for index, row in enumerate(capture["rows"]):
            sequence = by_id[row["sequence_id"]]
            if calibration_only and sequence["split"] != "calibration":
                continue
            if sequence["id"] not in references:
                references[sequence["id"]] = reference_rows(args.corpus.parent, sequence, model["vocab"])
            reference = references[sequence["id"]][row["reference_index"]]
            yield dict(row, model_id=model["id"], split=sequence["split"],
                       **measure(reference, device[index]))


def calibration(args):
    corpus = read_corpus(args.corpus, getattr(args, "asset_root", None))
    if corpus.get("_reference"):
        raise ValueError("decoded references require --reference-index")
    if corpus.get("purpose") == "validation_extension":
        raise ValueError("a validation extension cannot set bounds")
    grouped = {model["id"]: [] for model in corpus["models"]}
    for row in measured_rows(args, True):
        grouped[row["model_id"]].append(row)
    models = {}
    for model_id, rows in grouped.items():
        counts = Counter(row["mode"] for row in rows)
        if set(counts) != {"serial", "batch64"}:
            raise ValueError("calibration must cover serial and batch modes")
        models[model_id] = dict(calibrate(rows), modes=dict(counts))
    result = dict(schema_version=1, created_utc=now(), corpus_sha256=digest(args.corpus),
                  calibration_captures={read_json(p)["model_id"]: digest(p) for p in args.captures},
                  models=models, rows=grouped)
    write_json(args.output, result)
    print("accuracy: calibration bounds frozen for", len(models), "models")
    return 0


def check(args):
    corpus = read_corpus(args.corpus, getattr(args, "asset_root", None))
    if corpus.get("_reference"):
        raise ValueError("decoded references require --reference-index")
    frozen = read_json(args.bounds)
    model_ids = {model["id"] for model in corpus["models"]}
    if (integer(frozen["schema_version"]) != 1 or frozen["corpus_sha256"] != digest(args.corpus)
            or set(frozen["models"]) != model_ids):
        raise ValueError("calibration identity differs")
    for model in frozen["models"].values():
        validate_bounds(model["bounds"])
    streams = [measured_rows(args, False)]
    if len(args.coverage_corpus or []) != len(args.coverage_captures or []):
        raise ValueError("coverage requires its corpus and all captures")
    coverage_hashes = []
    seen_ids = {s["id"] for s in corpus["sequences"]}
    for coverage_path, coverage_captures in zip(args.coverage_corpus or [], args.coverage_captures or []):
        coverage = read_corpus(coverage_path)
        original_models = {m["id"]: (m["sha256"], m["vocab"]) for m in corpus["models"]}
        added_models = {m["id"]: (m["sha256"], m["vocab"]) for m in coverage["models"]}
        if (coverage.get("purpose") != "validation_extension"
                or coverage["base_corpus_sha256"] != digest(args.corpus)
                or coverage["base_bounds_sha256"] != digest(args.bounds)
                or any(original_models.get(key) != value for key, value in added_models.items())
                or coverage["reference_revision"] != corpus["reference_revision"]
                or coverage["reference_settings"] != corpus["reference_settings"]
                or seen_ids & {s["id"] for s in coverage["sequences"]}):
            raise ValueError("coverage inputs do not extend the fixed bounds")
        seen_ids.update(s["id"] for s in coverage["sequences"])
        coverage_hashes.append(digest(coverage_path))
        streams.append(measured_rows(SimpleNamespace(corpus=coverage_path, captures=coverage_captures), False))
    rows, counts, clear, different = [], Counter(), Counter(), Counter()
    failed = 0
    for row in itertools.chain.from_iterable(streams):
        result = assess(row, frozen["models"][row["model_id"]]["bounds"])
        row.update(result)
        rows.append(row)
        key = row["model_id"] + "/" + row["mode"] + "/" + row["split"]
        counts[key] += 1
        clear[key] += result["clear"]
        different[key] += row["reference_top"] != row["device_top"]
        failed += bool(result["failed"])
    missing_clear = []
    for model_id in model_ids:
        for mode in ("serial", "batch64"):
            if sum(clear[model_id + "/" + mode + "/" + split]
                   for split in ("validation", "regression")) == 0:
                missing_clear.append(model_id + "/" + mode)
    maxima = {}
    for model_id in model_ids:
        selected = [row for row in rows if row["model_id"] == model_id]
        maxima[model_id] = {key: max(row[key] for row in selected) for key in METRICS}
    result = dict(schema_version=1, created_utc=now(), corpus_sha256=digest(args.corpus),
                  bounds_sha256=digest(args.bounds), captures={str(p): digest(p) for p in args.captures},
                  coverage_corpus_sha256=coverage_hashes,
                  coverage_captures=[{str(p): digest(p) for p in paths}
                                     for paths in (args.coverage_captures or [])],
                  counts=dict(counts), clear_counts=dict(clear), different_counts=dict(different),
                  maxima=maxima, missing_clear_modes=missing_clear, failed_rows=failed, rows=rows)
    write_json(args.output, result)
    print(f"accuracy: {len(rows)} rows, {failed} failed rows, {len(missing_clear)} modes without clear winners, 0 skips")
    for model_id, values in maxima.items():
        print("accuracy:", model_id, values)
    return int(failed != 0 or bool(missing_clear))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("capture", "calibrate", "check"):
        command = commands.add_parser(name)
        if name == "capture":
            command.add_argument("--corpus", type=Path, required=True)
        else:
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--corpus", type=Path)
            source.add_argument("--reference-index", type=Path)
        command.add_argument("--asset-root", type=Path)
        command.add_argument("--output", type=Path, required=True)
        if name == "capture":
            command.add_argument("--model", required=True)
            command.add_argument("--store", type=Path, required=True)
            command.add_argument("--role", default="language")
            command.add_argument("--executable", type=Path, required=True)
        else:
            command.add_argument("--captures", type=Path, nargs="+", required=True)
        if name == "check":
            command.add_argument("--bounds", type=Path, required=True)
            command.add_argument("--coverage-corpus", type=Path, action="append")
            command.add_argument("--coverage-captures", type=Path, nargs="+", action="append")
    args = parser.parse_args()
    try:
        if getattr(args, "reference_index", None) is not None:
            if getattr(args, "coverage_corpus", None) or getattr(args, "coverage_captures", None):
                raise ValueError("a reference index already defines every required group")
            from arch_accuracy_bundle import check_bounds, make_bounds
            return {"calibrate": make_bounds, "check": check_bounds}[args.action](args)
        return {"capture": capture, "calibrate": calibration, "check": check}[args.action](args)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        print("accuracy: FAILED:", str(error) or type(error).__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
