# SPDX-License-Identifier: Apache-2.0
"""Read fixed accuracy inputs and form complete device capture requests.

Inputs: a corpus JSON file, reference rows and model store files.
Outputs: checked objects, binary requests and row indices.
Errors: ValueError or OSError for incomplete or changed inputs.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import unicodedata

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def parse_json(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_float=finite_number, parse_constant=finite_number)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as out:
        out.write(text)


def identifier(value):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
        raise ValueError("invalid file name or identifier")
    return value


def integer(value, minimum=0, maximum=0xFFFFFFFF):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid integer or extent")
    return value


def hexadecimal(value, length=64):
    if type(value) is not str or re.fullmatch(r"[0-9a-fA-F]{" + str(length) + "}", value) is None:
        raise ValueError("invalid hexadecimal identity")
    return value


def checked_text(value):
    if (type(value) is not str or not value
            or any(unicodedata.category(c) in ("Cc", "Cf", "Cs", "Zl", "Zp") for c in value)):
        raise ValueError("empty text or control character")
    return value


def relative_path(value):
    checked_text(value)
    if "\\" in value or ":" in value or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("the data path must name a relative file")
    path = Path(value)
    if path.is_absolute():
        raise ValueError("the data path must be relative to its input directory")
    return path


def local_file(root, relative):
    path = relative_path(relative)
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("the data path leaves its input directory")
    return result


def check_hash_fields(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "sha256" or key.endswith("_sha256"):
                hexadecimal(item)
            check_hash_fields(item)
    elif isinstance(value, list):
        for item in value:
            check_hash_fields(item)


def file_identity(path):
    path = Path(checked_text(str(path))).absolute()
    resolved = path.resolve(strict=True)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("the input must be a regular file")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("the input must be a regular file")
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    signature = lambda info: [getattr(info, field) for field in fields]
    if (signature(before) != signature(after) or signature(after) != signature(path.stat())
            or path.resolve(strict=True) != resolved):
        raise ValueError("input changed while its identity was read")
    return dict(path=str(path), resolved=str(resolved), sha256=value,
                bytes=after.st_size, stat=signature(after))


def retain_input(inputs, name, path, expected=None):
    identity = file_identity(path)
    if expected is not None and identity["sha256"] != hexadecimal(expected):
        raise ValueError("input digest differs: " + name)
    inputs[name] = identity
    return identity


def verify_inputs(inputs):
    for name, identity in inputs.items():
        if file_identity(identity["path"]) != identity:
            raise ValueError("input changed during the run: " + name)


def read_manifest(path):
    entries = [parse_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for entry in entries:
        identifier(entry["role"])
        identifier(entry["path"])
        hexadecimal(entry["sha256"])
        if "revision" in entry:
            hexadecimal(entry["revision"], 40)
        if "bytes" in entry:
            integer(entry["bytes"], 1, 0x7FFFFFFFFFFFFFFF)
    return entries


def read_corpus(path, asset_root=None):
    corpus = read_json(path)
    if "schema" in corpus:
        from arch_accuracy_reference import read_decoded_corpus
        return read_decoded_corpus(path, asset_root)
    return validate_corpus(corpus, path)


def validate_corpus(corpus, path, split_rules=None):
    if "_reference" in corpus:
        raise ValueError("reference context must come from checked inputs")
    if integer(corpus["schema_version"]) != 1:
        raise ValueError("unsupported corpus version")
    check_hash_fields(corpus)
    if "purpose" in corpus and corpus["purpose"] != "validation_extension":
        raise ValueError("unsupported corpus purpose")
    extension = corpus.get("purpose") == "validation_extension"
    if extension:
        hexadecimal(corpus["base_corpus_sha256"])
        hexadecimal(corpus["base_bounds_sha256"])
    hexadecimal(corpus["reference_revision"], 40)
    if type(corpus["reference_settings"]) is not dict or not corpus["reference_settings"]:
        raise ValueError("absent reference identity or settings")
    for key, value in corpus["reference_settings"].items():
        if key.startswith("n_"):
            integer(value, 0 if key == "n_gpu_layers" else 1)
    for model in corpus["models"]:
        identifier(model["id"])
        identifier(model["model_file"])
        hexadecimal(model["sha256"])
        integer(model["vocab"], 2)
        if "bytes" in model:
            integer(model["bytes"], 1, 0x7FFFFFFFFFFFFFFF)
        if "template_file" in model:
            local_file(Path(path).parent, model["template_file"])
            hexadecimal(model["template_sha256"])
    for library in corpus.get("reference_libraries", []):
        relative_path(library["file"])
        integer(library["bytes"], 1, 0x7FFFFFFFFFFFFFFF)
        hexadecimal(library["sha256"])
    for run in corpus.get("reference_runs", {}).values():
        local_file(Path(path).parent, run["metadata_file"])
        hexadecimal(run["metadata_sha256"])
    models = {model["id"]: model for model in corpus["models"]}
    if not models or len(models) != len(corpus["models"]):
        raise ValueError("absent or repeated model")
    seen = set()
    for sequence in corpus["sequences"]:
        identifier(sequence["id"])
        identifier(sequence["model_id"])
        if sequence["id"] in seen or sequence["model_id"] not in models:
            raise ValueError("repeated sequence or unknown model")
        seen.add(sequence["id"])
        model = models[sequence["model_id"]]
        count = integer(sequence["row_count"], 1, 64)
        integer(sequence["prefill_count"], 1, 512)
        integer(sequence["row_stride_bytes"], 1, 0x7FFFFFFFFFFFFFFF)
        if sequence["execution_group"] == "batch64" or "slot" in sequence:
            integer(sequence["slot"], 0, 63)
        for key in ("prefill_ids", "teacher_ids", "consumed_ids", "evaluated_positions", "reference_argmax_ids"):
            values = sequence[key]
            if type(values) is not list:
                raise ValueError("the token and position fields must be lists")
            for value in values:
                integer(value, 0, 0xFFFFFFFF if key == "evaluated_positions" else model["vocab"] - 1)
        local_file(Path(path).parent, sequence["reference_rows_file"])
        hexadecimal(sequence["reference_rows_sha256"])
        hexadecimal(sequence["token_metadata_sha256"])
        for key in ("prompt_file", "retained_ids_file"):
            if key in sequence:
                local_file(Path(path).parent, sequence[key])
                hexadecimal(sequence[key.removesuffix("_file") + "_sha256"])
        prefill = sequence["prefill_ids"]
        teacher = sequence["teacher_ids"]
        if (sequence["split"] not in ("calibration", "validation", "regression")
                or sequence["execution_group"] not in ("serial", "batch64")
                or len(teacher) != count or len(sequence["reference_argmax_ids"]) != count
                or not 1 <= len(prefill) <= 512
                or sequence["prefill_count"] != len(prefill)
                or sequence["consumed_ids"] != prefill + teacher[:-1]
                or sequence["evaluated_positions"] != list(range(len(prefill) - 1, len(prefill) + count - 1))):
            raise ValueError("invalid token positions or sequence extent")
        if any(type(token) is not int or not 0 <= token < model["vocab"] for token in prefill + teacher):
            raise ValueError("token outside the vocabulary")
        if sequence["row_stride_bytes"] != model["vocab"] * 4:
            raise ValueError("invalid reference row width")
    for key, value in corpus["totals"].items():
        integer(value, 1)
    if "models" in corpus["totals"] and corpus["totals"]["models"] != len(models):
        raise ValueError("incomplete corpus model count")
    if not seen or len(seen) != corpus["totals"]["sequences"]:
        raise ValueError("incomplete corpus sequence count")
    if sum(s["row_count"] for s in corpus["sequences"]) != corpus["totals"]["rows"]:
        raise ValueError("incomplete corpus row count")
    for model_id in models:
        sequences = [s for s in corpus["sequences"] if s["model_id"] == model_id]
        if ("rows_per_model" in corpus["totals"]
                and sum(s["row_count"] for s in sequences) != corpus["totals"]["rows_per_model"]):
            raise ValueError("incomplete per-model row count")
        splits = ({"validation"} if extension else {"calibration", "validation", "regression"})
        if split_rules is not None:
            splits = split_rules[model_id]
        if {s["split"] for s in sequences} != splits:
            raise ValueError("a model lacks a required split")
        batch = [s for s in sequences if s["execution_group"] == "batch64"]
        if len(batch) != 64 or {s["slot"] for s in batch} != set(range(64)):
            raise ValueError("the batch must contain 64 distinct slots")
        if len({tuple(s["prefill_ids"]) for s in batch}) != 64:
            raise ValueError("the batch must contain 64 distinct prefixes")
    return corpus


def model_sequences(corpus, model_id):
    model = next(model for model in corpus["models"] if model["id"] == model_id)
    sequences = [s for s in corpus["sequences"] if s["model_id"] == model_id]
    return model, sequences


def request(corpus, model_id):
    model, sequences = model_sequences(corpus, model_id)
    groups = [[sequence] for sequence in sequences]
    groups.append(sorted((s for s in sequences if s["execution_group"] == "batch64"),
                         key=lambda s: s["slot"]))
    data = bytearray(b"AOTXAC01" + struct.pack("<II", model["vocab"], len(groups)))
    rows = []
    for group in groups:
        steps = group[0]["row_count"]
        if any(s["row_count"] != steps or s["prefill_count"] != group[0]["prefill_count"] for s in group):
            raise ValueError("the requested batch exceeds its shape")
        data.extend(struct.pack("<II", len(group), steps))
        data.extend(struct.pack("<" + "I" * len(group), *(s["prefill_count"] for s in group)))
        for sequence in group:
            ids = sequence["consumed_ids"]
            data.extend(struct.pack("<" + "I" * len(ids), *ids))
        for step in range(steps):
            for sequence in group:
                rows.append(dict(sequence_id=sequence["id"], reference_index=step,
                                 position=sequence["evaluated_positions"][step],
                                 mode="batch64" if len(group) == 64 else "serial"))
    return bytes(data), rows


def reference_metadata(root, sequence):
    rows = relative_path(sequence["reference_rows_file"])
    path = local_file(root, str(rows.with_name("tokens.txt")))
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != hexadecimal(sequence["token_metadata_sha256"]):
        raise ValueError("reference token metadata digest differs")
    expected = dict(prefill=sequence["prefill_ids"], teacher=sequence["teacher_ids"],
                    argmax=sequence["reference_argmax_ids"])
    for values in expected.values():
        if type(values) is not list or any(type(value) is not int or value < 0 for value in values):
            raise ValueError("invalid reference token list")
    if (len(expected["teacher"]) != sequence["row_count"]
            or len(expected["argmax"]) != sequence["row_count"]):
        raise ValueError("incomplete reference token metadata")
    actual = {}
    for line in data.decode("ascii").splitlines():
        fields = line.split()
        if not fields or fields[0] in actual or any(re.fullmatch(r"[0-9]+", word) is None for word in fields[1:]):
            raise ValueError("invalid reference token metadata")
        actual[fields[0]] = list(map(int, fields[1:]))
    if actual != expected:
        raise ValueError("reference token metadata differs from the corpus")
    return path


def reference_rows(root, sequence, vocab):
    integer(vocab, 2)
    integer(sequence["row_count"], 1, 64)
    if integer(sequence["row_stride_bytes"], 1, 0x7FFFFFFFFFFFFFFF) != vocab * 4:
        raise ValueError("invalid reference row width")
    path = local_file(root, sequence["reference_rows_file"])
    if path.stat().st_size != sequence["row_count"] * vocab * 4:
        raise ValueError("incomplete reference row file")
    if digest(path) != hexadecimal(sequence["reference_rows_sha256"]):
        raise ValueError("reference row digest differs")
    reference_metadata(root, sequence)
    return np.memmap(path, dtype="<f4", mode="r", shape=(sequence["row_count"], vocab))


def checked_capture(capture, root, corpus, corpus_sha256):
    integer(capture["schema_version"])
    integer(capture["vocab"], 2)
    identifier(capture["model_id"])
    check_hash_fields(capture)
    hexadecimal(capture["executable_sha256"])
    for row in capture["rows"]:
        identifier(row["sequence_id"])
        integer(row["reference_index"])
        integer(row["position"])
    model, _ = model_sequences(corpus, capture["model_id"])
    if "_reference" in corpus:
        from arch_accuracy_reference_inputs import same
        context = corpus["_reference"]
        basis = context["bases"][model["id"]]
        expected_basis = dict(kind=context["kind"], origin=context["origin"], status=context["status"],
                              runtime_original=basis["runtime_original"], reference_derived=basis["reference_derived"])
        same(capture.get("reference_basis"), expected_basis, "capture reference basis")
    data, expected = request(corpus, model["id"])
    if (capture["schema_version"] != 1 or capture["corpus_sha256"] != corpus_sha256
            or capture["model_sha256"] != model["sha256"] or capture["vocab"] != model["vocab"]
            or capture["rows"] != expected or capture["input_sha256"] != hashlib.sha256(data).hexdigest()):
        raise ValueError("capture identity, inputs or row count differs")
    rows_path = local_file(root, capture["rows_file"])
    if rows_path.stat().st_size != 16 + len(expected) * model["vocab"] * 4:
        raise ValueError("incomplete device row file")
    with rows_path.open("rb") as stream:
        header = stream.read(16)
    if header != b"AOTXAR01" + struct.pack("<II", model["vocab"], len(expected)):
        raise ValueError("device row header differs")
    if digest(rows_path) != capture["rows_sha256"]:
        raise ValueError("device row digest differs")
    return capture, np.memmap(rows_path, dtype="<f4", mode="r", offset=16,
                             shape=(len(expected), model["vocab"]))


def capture_rows(path, corpus, corpus_path):
    return checked_capture(read_json(path), Path(path).parent, corpus, digest(corpus_path))
