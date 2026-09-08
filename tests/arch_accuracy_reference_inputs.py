# SPDX-License-Identifier: Apache-2.0
"""Check frozen reference membership and the declared input counts.

Inputs: checked input plans, source corpora and token files.
Outputs: checked sequence groups and count records.
Errors: ValueError or OSError for incomplete or changed inputs.
"""
import json
from pathlib import Path
import re

from arch_accuracy_io import checked_text, identifier, integer, validate_corpus

ORIGINS = ("original", "short", "long", "fresh")
MODEL_FIELDS = ("id", "model_file", "sha256", "bytes", "vocab", "architecture")
INPUT_FIELDS = ("id", "model_id", "execution_group", "row_count", "prefill_ids",
                "teacher_ids", "consumed_ids", "evaluated_positions", "prefill_count")


def same(actual, expected, name):
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError(name + " differs from its fixed definition")


def keyed(items, key="id"):
    result = {}
    for item in items:
        name = identifier(item[key])
        if name in result:
            raise ValueError("repeated input identifier")
        result[name] = item
    return result


def totals(sequences, models):
    result = dict(sequences=len(sequences), unique_reference_rows=0, reference_row_bytes=0,
                  runtime_rows_all_modes=0, calibration_runtime_rows=0,
                  historical_regression_runtime_rows=0, fresh_validation_runtime_rows=0)
    calibration = {name: 0 for name in models}
    fields = dict(calibration="calibration_runtime_rows", regression="historical_regression_runtime_rows",
                  validation="fresh_validation_runtime_rows")
    for sequence in sequences:
        rows = integer(sequence["row_count"], 1, 64)
        count = rows * (2 if sequence["execution_group"] == "batch64" else 1)
        result["unique_reference_rows"] += rows
        result["reference_row_bytes"] += rows * integer(models[sequence["model_id"]]["vocab"], 2) * 4
        result["runtime_rows_all_modes"] += count
        result[fields[sequence["split"]]] += count
        if sequence["split"] == "calibration":
            calibration[sequence["model_id"]] += count
    return result, calibration


def token_file(path, vocab):
    result = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields or fields[0] in result:
            raise ValueError("absent or repeated token input")
        identifier(fields[0])
        if any(re.fullmatch(r"[0-9]+", field) is None for field in fields[1:]):
            raise ValueError("invalid token input")
        result[fields[0]] = [integer(int(field), 0, vocab - 1) for field in fields[1:]]
    return result


def check_fresh(plan, frozen, models, files, root):
    rule = plan["fresh_rule"]
    count = integer(rule["per_model"], 1)
    prefill = integer(rule["prefill_count"], 1, 512)
    steps = integer(rule["teacher_count"], 1, 64)
    texts = keyed(plan["holdout_texts"])
    if len(texts) != count:
        raise ValueError("incomplete fixed text set")
    same(set_as_list(frozen["tokenization_files"]), set_as_list(models), "token model set")
    for text in texts.values():
        integer(text["slot"], 0, 63)
        checked_text(text["family"])
        files.file(root, text["file"], text["sha256"])
    for name, model in models.items():
        path = files.file(root, name + "-tokens.txt", frozen["tokenization_files"][name])
        tokens = token_file(path, integer(model["vocab"], 2))
        same(set_as_list(tokens), set_as_list(texts), "token text set")
        fresh = [s for s in frozen["sequences"] if s["origin"] == "fresh" and s["model_id"] == name]
        if len(fresh) != count:
            raise ValueError("incomplete fresh input set")
        expected = {}
        for text_id, text in texts.items():
            values = tokens[text_id]
            if len(values) < prefill + steps:
                raise ValueError("incomplete fixed token input")
            prefix, teacher = values[:prefill], values[prefill:prefill + steps]
            expected[name + "-" + text_id] = dict(
                id=name + "-" + text_id, model_id=name, origin="fresh", split="validation",
                original_split=None, execution_group="batch64", slot=text["slot"], family=text["family"],
                prefill_ids=prefix, teacher_ids=teacher, prefill_count=prefill, row_count=steps,
                consumed_ids=prefix + teacher[:-1], evaluated_positions=list(range(prefill - 1, prefill + steps - 1)),
                text_file=text["file"], text_sha256=text["sha256"])
        same(keyed(fresh), expected, "fresh inputs")
        prior = [tuple(s["prefill_ids"]) for s in frozen["sequences"]
                 if s["model_id"] == name and s["origin"] != "fresh"]
        for sequence in fresh:
            prefix = tuple(sequence["prefill_ids"])
            if any(prefix[:min(len(prefix), len(old))] == old[:min(len(prefix), len(old))] for old in prior):
                raise ValueError("fresh and retained prefixes overlap")
            prior.append(prefix)


def set_as_list(value):
    return sorted(value)


def check_fixed(plan, frozen, sources, files, root):
    models = keyed(plan["models"])
    if not models:
        raise ValueError("absent fixed model set")
    same(set_as_list(plan["runtime_model_locations"]), set_as_list(models), "runtime model set")
    same(set_as_list(frozen["model_identities"]), set_as_list(models), "fixed model set")
    for name, model in models.items():
        expected = dict(path=plan["runtime_model_locations"][name], sha256=model["sha256"], bytes=model["bytes"])
        same(frozen["model_identities"][name], expected, "fixed runtime model")
        files.path(files.asset_root, expected["path"])
    expected = {}
    for origin, entry in sources.items():
        source = entry["corpus"]
        validate_corpus(source, entry["path"])
        for model in source["models"]:
            same({key: model[key] for key in MODEL_FIELDS},
                 {key: models[model["id"]][key] for key in MODEL_FIELDS}, "source model")
        for sequence in source["sequences"]:
            item = {key: sequence[key] for key in INPUT_FIELDS}
            if "slot" in sequence:
                item["slot"] = sequence["slot"]
            item.update(origin=origin, original_split=sequence["split"],
                        split="calibration" if origin == "original" and sequence["split"] == "calibration" else "regression")
            if item["id"] in expected:
                raise ValueError("repeated source sequence")
            expected[item["id"]] = item
    same(keyed(plan["existing_sequences"]), expected, "retained inputs")
    fixed = keyed(frozen["sequences"])
    if any(s["origin"] not in ORIGINS or s["model_id"] not in models for s in fixed.values()):
        raise ValueError("unknown frozen origin or model")
    same({name: s for name, s in fixed.items() if s["origin"] != "fresh"}, expected, "frozen retained inputs")
    if frozen["vocabulary_only"] is not True or integer(frozen["model_inference_calls"]) != 0:
        raise ValueError("fixed text inputs contain model inference")
    check_fresh(plan, frozen, models, files, root)
    groups = {origin: [s for s in fixed.values() if s["origin"] == origin] for origin in ORIGINS}
    if any(not group for group in groups.values()):
        raise ValueError("absent reference origin")
    measured, calibration = totals(list(fixed.values()), models)
    same(plan["totals"], measured, "fixed totals")
    same(plan["original_calibration_runtime_rows_per_model"], calibration, "calibration counts")
    return groups, models, calibration
