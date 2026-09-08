# SPDX-License-Identifier: Apache-2.0
"""Read explicit decoded-weight references and retain their input identities.

Inputs: reference manifests, frozen inputs and an explicit asset root.
Outputs: checked corpora and reference identity records.
Errors: ValueError or OSError for incomplete or changed inputs.
"""
from pathlib import Path

from arch_accuracy_io import (checked_text, hexadecimal, identifier, integer, local_file,
                              read_json, reference_metadata, retain_input, validate_corpus, verify_inputs)
from arch_accuracy_reference_inputs import MODEL_FIELDS, ORIGINS, check_fixed, keyed, same


class ReferenceFiles:
    def __init__(self, asset_root):
        if asset_root is None:
            raise ValueError("decoded references require an asset root")
        self.asset_root = Path(checked_text(str(asset_root))).resolve(strict=True)
        if not self.asset_root.is_dir():
            raise ValueError("the asset root must be a directory")
        self.inputs = {}
        self.tokens = set()

    def path(self, root, name):
        local_file(root, name)
        return Path(root).absolute() / name

    def file(self, root, name, expected):
        path = self.path(root, name)
        key = "reference:" + str(path)
        if key not in self.inputs:
            retain_input(self.inputs, key, path, expected)
        same(self.inputs[key]["sha256"], hexadecimal(expected), "input digest")
        return path

    def json(self, root, name, expected):
        return read_json(self.file(root, name, expected))


def schema(value, expected):
    if value.get("schema") != expected or "schema_version" in value or "purpose" in value:
        raise ValueError("unsupported reference schema")


def provenance(path, wrapper, files):
    root = path.parent
    plan = files.json(root, "input-plan.json", wrapper["plan_sha256"])
    schema(plan, "aotx-decoded-f32-input-plan-v1")
    frozen = files.json(root, "frozen-inputs.json", wrapper["frozen_inputs_sha256"])
    schema(frozen, "aotx-decoded-f32-fixed-inputs-v1")
    same(frozen["plan_sha256"], wrapper["plan_sha256"], "fixed plan")
    build = files.json(root, "build.json", frozen["build_sha256"])
    same(build["plan_sha256"], wrapper["plan_sha256"], "build plan")
    same(sorted(build["binary_sha256"]), ["reference", "tokenize"], "reference programs")
    for name, expected in build["binary_sha256"].items():
        identifier(name)
        files.file(root, name, expected)
    for name, expected in plan["source_sha256"].items():
        files.file(root, name, expected)
    sources = {}
    for entry in plan["origins"]:
        name = identifier(entry["id"])
        if name not in ORIGINS[:-1] or name in sources:
            raise ValueError("unknown or repeated source origin")
        source_path = files.file(files.asset_root, entry["path"], entry["sha256"])
        sources[name] = dict(path=source_path, corpus=read_json(source_path), identity=entry)
    if set(sources) != set(ORIGINS[:-1]):
        raise ValueError("incomplete source origins")
    libraries = {}
    source = sources["original"]
    for library in source["corpus"]["reference_libraries"]:
        name = identifier(Path(library["file"]).name)
        if name in libraries:
            raise ValueError("repeated CPU library")
        location = files.file(source["path"].parent, library["file"], library["sha256"])
        size = integer(library["bytes"], 1, 0x7FFFFFFFFFFFFFFF)
        same(files.inputs["reference:" + str(location)]["bytes"], size, "CPU library size")
        libraries[name] = dict(path=str(location), sha256=library["sha256"], bytes=size)
    if not libraries:
        raise ValueError("absent CPU libraries")
    same({name: value["sha256"] for name, value in libraries.items()}, plan["library_sha256"], "CPU libraries")
    groups, models, calibration = check_fixed(plan, frozen, sources, files, root)
    return dict(plan=plan, frozen=frozen, build=build, sources=sources, libraries=libraries,
                groups=groups, models=models, calibration_rows=calibration)


def check_basis(model, basis, context, files, wrapper):
    schema(basis, "aotx-decoded-f32-model-basis-v1")
    name = model["id"]
    same(basis["model_id"], name, "basis model")
    for key in ("plan_sha256", "frozen_inputs_sha256"):
        same(basis[key], wrapper[key], "basis " + key)
    plan = context["plan"]
    same(basis["library_sha256"], plan["library_sha256"], "basis libraries")
    same(basis["reference_settings"], plan["reference_settings"], "basis CPU settings")
    original, derived = basis["runtime_original"], basis["reference_derived"]
    same(original, context["frozen"]["model_identities"][name], "basis runtime model")
    for entry in (original, derived):
        files.path(files.asset_root, entry["path"])
        hexadecimal(entry["sha256"])
        integer(entry["bytes"], 1, 0x7FFFFFFFFFFFFFFF)
    if derived["all_tensors_f32"] is not True or derived["exact_decoded_weight_values"] is not True:
        raise ValueError("incomplete decoded weight basis")
    declared = plan["expansion_records"][name]
    same(declared["original_sha256"], original["sha256"], "expansion source")
    same(declared["derived_sha256"], derived["sha256"], "expansion model")
    record = basis["proof"]
    for key in ("tensors", "values"):
        integer(record[key], 1, 0x7FFFFFFFFFFFFFFF)
        same(record[key], declared[key], "expansion " + key)
    same(record["path"], declared["verification_file"], "proof path")
    same(record["sha256"], declared["verification_sha256"], "proof identity")
    same(record["derivation_path"], declared["derivation_file"], "derivation path")
    same(record["derivation_sha256"], declared["derivation_sha256"], "derivation identity")
    proof = files.json(files.asset_root, record["path"], record["sha256"])
    derivation = files.json(files.asset_root, record["derivation_path"], record["derivation_sha256"])
    same(proof["model_sha256"], derived["sha256"], "verified model")
    same(derivation["original_model_sha256"], original["sha256"], "verified source")
    same(derivation["libraries"], plan["library_sha256"], "derivation libraries")
    for value in (proof, derivation):
        if "model_id" in value:
            same(value["model_id"], name, "proof model")
    for key in ("tensors", "values"):
        same(proof[key], record[key], "verified " + key)
    size = integer(proof["f32_tensor_bytes"], 1, 0x7FFFFFFFFFFFFFFF)
    if size != 4 * record["values"] or size > derived["bytes"] or proof["all_values_exact"] is not True:
        raise ValueError("incomplete exact-value proof")
    changes = proof["changed_metadata_fields"]
    if type(changes) is not list or any(value != "general.file_type" for value in changes) or len(changes) > 1:
        raise ValueError("unexpected decoded model metadata changes")
    names, values = set(), 0
    for tensor in proof["detail"]:
        label = checked_text(tensor["name"])
        if label in names:
            raise ValueError("repeated verified tensor")
        names.add(label)
        integer(tensor["source_type"])
        values += integer(tensor["values"], 1, 0x7FFFFFFFFFFFFFFF)
    same(len(names), record["tensors"], "verified tensor count")
    same(values, record["values"], "verified value count")


def check_run(path, model, settings):
    actual = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in actual:
            raise ValueError("invalid CPU run metadata")
        actual[fields[0]] = fields[1]
    expected = dict(backend="cpu", vocab=str(model["vocab"]), reference_basis="decoded_weights_f32",
                    teacher_forcing="explicit_fixed_ids", argmax_separate="true",
                    sampling="raw_argmax_lowest_id_tie")
    aliases = dict(n_threads="threads", n_threads_batch="threads_batch", flash_attention="flash_attn",
                   callback="callback_registered")
    for key in ("n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_threads", "n_threads_batch",
                "flash_attention", "kv_unified", "offload_kqv", "op_offload", "n_gpu_layers",
                "callback", "warmup", "parse_special", "add_special", "cache_k", "cache_v"):
        expected[aliases.get(key, key)] = str(settings[key]).lower()
    same({key: actual.get(key) for key in expected}, expected, "CPU run settings")


def decoded(path, files, expected=None, shared=None):
    path = Path(path).absolute()
    if expected is None:
        expected = retain_input(files.inputs, "reference:" + str(path), path)["sha256"]
    wrapper = files.json(path.parent, path.name, expected)
    schema(wrapper, "aotx-decoded-f32-reference-v1")
    origin = wrapper["origin"]
    if origin not in ORIGINS:
        raise ValueError("unknown reference origin")
    checked_text(wrapper["status"])
    context = shared or provenance(path, wrapper, files)
    same(wrapper["plan_sha256"], context["frozen"]["plan_sha256"], "wrapper plan")
    same(wrapper["frozen_inputs_sha256"], files.inputs["reference:" + str(path.parent / "frozen-inputs.json")]["sha256"], "wrapper inputs")
    source = context["sources"].get(origin)
    same(wrapper["source_corpus"], source["identity"] if source else None, "source corpus")
    corpus = wrapper["corpus"]
    if "purpose" in corpus or "schema" in corpus or "_reference" in corpus:
        raise ValueError("mixed reference basis")
    expected_inputs = keyed(context["groups"][origin])
    sequences = keyed(corpus["sequences"])
    same(sorted(sequences), sorted(expected_inputs), "reference membership")
    for name, sequence in sequences.items():
        fixed = expected_inputs[name]
        same({key: sequence.get(key) for key in fixed}, fixed, "reference input")
    models = keyed(corpus["models"])
    same(sorted(models), sorted({s["model_id"] for s in expected_inputs.values()}), "reference models")
    splits = {name: {s["split"] for s in expected_inputs.values() if s["model_id"] == name} for name in models}
    validate_corpus(corpus, path, splits)
    for sequence in corpus["sequences"]:
        name = str(Path(sequence["reference_rows_file"]).with_name("tokens.txt"))
        files.tokens.add(str(files.file(path.parent, name, sequence["token_metadata_sha256"])))
        reference_metadata(path.parent, sequence)
    for key in ("reference_revision", "reference_build_type", "reference_settings"):
        same(corpus[key], context["plan"][key], key)
    same(corpus["reference_build_sha256"], context["frozen"]["build_sha256"], "reference build")
    same(sorted(wrapper["reference_bases"]), sorted(models), "reference bases")
    same(sorted(wrapper["reference_runs"]), sorted(models), "reference runs")
    bases = {}
    for name, model in models.items():
        same({key: model[key] for key in MODEL_FIELDS},
             {key: context["models"][name][key] for key in MODEL_FIELDS}, "reference model")
        entry = wrapper["reference_bases"][name]
        basis = files.json(path.parent, entry["file"], entry["sha256"])
        same(basis, entry["basis"], "embedded basis")
        check_basis(model, basis, context, files, wrapper)
        bases[name] = basis
        run = wrapper["reference_runs"][name]
        check_run(files.file(path.parent, run["file"], run["sha256"]), model, corpus["reference_settings"])
    copy_inputs = {str(Path(item["path"]).relative_to(path.parent)): item["path"]
                   for item in files.inputs.values()
                   if Path(item["path"]).is_relative_to(path.parent) and item["path"] not in files.tokens}
    corpus["_reference"] = dict(kind="decoded_f32", origin=origin, status=wrapper["status"], bases=bases,
                                asset_root=str(files.asset_root), verified_inputs=dict(files.inputs),
                                plan=context["plan"], frozen=context["frozen"], build=context["build"],
                                libraries=context["libraries"], copy_inputs=copy_inputs)
    return corpus, context


def read_decoded_corpus(path, asset_root=None):
    files = ReferenceFiles(asset_root)
    corpus, _ = decoded(path, files)
    verify_inputs(files.inputs)
    return corpus


def read_reference_index(path, asset_root):
    path = Path(path).absolute()
    files = ReferenceFiles(asset_root)
    identity = retain_input(files.inputs, "reference:" + str(path), path)
    index = read_json(path)
    schema(index, "aotx-decoded-f32-reference-index-v1")
    checked_text(index["status"])
    same(sorted(index["manifests"]), sorted(ORIGINS), "reference origins")
    corpora, context = {}, None
    for origin in ORIGINS:
        entry = index["manifests"][origin]
        child = files.file(path.parent, entry["file"], entry["sha256"])
        if child.parent != path.parent:
            raise ValueError("reference manifests must share one directory")
        corpus, context = decoded(child, files, entry["sha256"], context)
        same(corpus["_reference"]["origin"], origin, "indexed origin")
        same(corpus["_reference"]["status"], index["status"], "indexed status")
        same(entry["sequences"], corpus["totals"]["sequences"], "indexed sequence count")
        same(entry["rows"], corpus["totals"]["rows"], "indexed row count")
        corpora[origin] = dict(path=child, corpus=corpus)
    same(index["totals"], context["plan"]["totals"], "indexed totals")
    verify_inputs(files.inputs)
    provenance_id = dict(kind="decoded_f32", index_sha256=identity["sha256"], status=index["status"],
                         plan_sha256=context["frozen"]["plan_sha256"],
                         frozen_inputs_sha256=files.inputs["reference:" + str(path.parent / "frozen-inputs.json")]["sha256"],
                         manifests={name: entry["sha256"] for name, entry in index["manifests"].items()})
    return dict(corpora=corpora, verified_inputs=files.inputs, totals=context["plan"]["totals"],
                calibration_rows=context["calibration_rows"], identity=provenance_id)
