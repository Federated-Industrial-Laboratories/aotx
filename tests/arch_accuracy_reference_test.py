# SPDX-License-Identifier: Apache-2.0
"""Check explicit reference identity and fixed input failures.

Inputs: small temporary reference files with distinct batch prefixes.
Outputs: unittest counts and failure reports.
Exit codes: zero on success, nonzero on failure.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from arch_accuracy_io import checked_capture, digest, read_corpus, read_json, request, verify_inputs
from arch_accuracy_reference import read_reference_index
from arch_accuracy_reference_inputs import INPUT_FIELDS, ORIGINS, totals

SETTINGS = dict(n_ctx=512, n_batch=512, n_ubatch=512, n_seq_max=1, n_threads=1, n_threads_batch=1,
                flash_attention="auto", kv_unified=False, n_gpu_layers=0, offload_kqv=False,
                op_offload=False, warmup=False, callback=False, parse_special=True,
                add_special=False, cache_k="f16", cache_v="f16")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return digest(path)


class ReferenceFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root / "references"
        self.directory.mkdir()
        self.index_path = self.directory / "reference-index.json"
        self.model = dict(id="sample", model_file="weights.gguf", sha256="a" * 64,
                          bytes=8, vocab=512, architecture="sample")
        self.library = self.root / "sources/original/lib/sample.so"
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"library")
        self.libraries = {self.library.name: digest(self.library)}
        self.sources, fixed = {}, []
        for origin_index, origin in enumerate(ORIGINS[:-1]):
            sequences = []
            for slot in range(64):
                split = ("calibration" if slot < 16 else "validation" if slot < 32 else "regression")
                if origin != "original":
                    split = "validation"
                sequence = dict(id=origin + "-" + str(slot), model_id="sample", split=split,
                                execution_group="batch64", slot=slot, row_count=1, prefill_count=1,
                                prefill_ids=[origin_index * 64 + slot], teacher_ids=[3],
                                consumed_ids=[origin_index * 64 + slot], evaluated_positions=[0],
                                reference_argmax_ids=[4], row_stride_bytes=2048,
                                reference_rows_file="rows/" + origin + "-" + str(slot) + "/rows.f32",
                                reference_rows_sha256="b" * 64, token_metadata_sha256="c" * 64)
                sequences.append(sequence)
                item = {key: sequence[key] for key in INPUT_FIELDS}
                item.update(slot=slot, origin=origin, original_split=split,
                            split="calibration" if origin == "original" and split == "calibration" else "regression")
                fixed.append(item)
            source = self.corpus(sequences)
            if origin == "original":
                source["reference_libraries"] = [dict(file="lib/sample.so", sha256=digest(self.library), bytes=7)]
            else:
                source.update(purpose="validation_extension", base_corpus_sha256="d" * 64, base_bounds_sha256="e" * 64)
            self.sources[origin] = source
        texts, token_lines = [], []
        for slot in range(64):
            name = "text-" + str(slot)
            relative = "texts/" + name + ".txt"
            path = self.directory / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text("fixed text " + str(slot))
            entry = dict(id=name, slot=slot, family="sample", file=relative, sha256=digest(path))
            texts.append(entry)
            token_lines.append(name + " " + str(256 + slot) + " 3\n")
            fixed.append(dict(id="sample-" + name, model_id="sample", origin="fresh", split="validation",
                              original_split=None, execution_group="batch64", slot=slot, family="sample",
                              prefill_ids=[256 + slot], teacher_ids=[3], prefill_count=1, row_count=1,
                              consumed_ids=[256 + slot], evaluated_positions=[0], text_file=relative,
                              text_sha256=entry["sha256"]))
        (self.directory / "sample-tokens.txt").write_text("".join(token_lines))
        (self.directory / "reference").write_bytes(b"old reference executable")
        (self.directory / "tokenize").write_bytes(b"old tokenizer executable")
        (self.directory / "prepare.py").write_bytes(b"fixed preparation source")
        self.proof = dict(tensors=2, values=4, f32_tensor_bytes=16, all_values_exact=True,
                          changed_metadata_fields=["general.file_type"], model_sha256="f" * 64,
                          detail=[dict(name="one.weight", source_type=8, values=2),
                                  dict(name="two.weight", source_type=0, values=2)])
        self.derivation = dict(original_model_sha256=self.model["sha256"], libraries=self.libraries)
        measured, calibration = totals(fixed, {"sample": self.model})
        self.plan = dict(schema="aotx-decoded-f32-input-plan-v1", reference_revision="1" * 40,
                         reference_build_type="Release", reference_settings=SETTINGS,
                         models=[self.model], runtime_model_locations={"sample": "models/weights.gguf"},
                         existing_sequences=copy.deepcopy(fixed[:192]), holdout_texts=texts,
                         fresh_rule=dict(per_model=64, prefill_count=1, teacher_count=1),
                         original_calibration_runtime_rows_per_model=calibration, totals=measured,
                         source_sha256={"prepare.py": digest(self.directory / "prepare.py")},
                         library_sha256=self.libraries)
        self.frozen = dict(schema="aotx-decoded-f32-fixed-inputs-v1", sequences=fixed,
                           vocabulary_only=True, model_inference_calls=0,
                           model_identities={"sample": dict(path="models/weights.gguf", sha256="a" * 64, bytes=8)},
                           tokenization_files={"sample": digest(self.directory / "sample-tokens.txt")})
        self.basis = dict(schema="aotx-decoded-f32-model-basis-v1", model_id="sample",
                          runtime_original=copy.deepcopy(self.frozen["model_identities"]["sample"]),
                          reference_derived=dict(path="models/decoded.gguf", sha256="f" * 64, bytes=32,
                                                 all_tensors_f32=True, exact_decoded_weight_values=True),
                          library_sha256=self.libraries, reference_settings=SETTINGS)
        self.wrappers = {}
        for origin in ORIGINS:
            sequences = [dict(copy.deepcopy(s), reference_argmax_ids=[4], row_stride_bytes=2048,
                              reference_rows_file="rows/sample/" + s["id"] + "/rows.f32",
                              reference_rows_sha256="b" * 64, token_metadata_sha256="c" * 64)
                         for s in fixed if s["origin"] == origin]
            self.wrappers[origin] = dict(schema="aotx-decoded-f32-reference-v1", status="proposed_not_adopted",
                                         origin=origin, corpus=self.corpus(sequences))
            for sequence in sequences:
                path = self.directory / Path(sequence["reference_rows_file"]).with_name("tokens.txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("prefill " + str(sequence["prefill_ids"][0]) + "\nteacher 3\nargmax 4\n")
                sequence["token_metadata_sha256"] = digest(path)
        self.run = """backend cpu
vocab 512
n_ctx 512
n_batch 512
n_ubatch 512
n_seq_max 1
threads 1
threads_batch 1
flash_attn auto
kv_unified false
offload_kqv false
op_offload false
n_gpu_layers 0
callback_registered false
warmup false
sampling raw_argmax_lowest_id_tie
parse_special true
add_special false
reference_basis decoded_weights_f32
cache_k f16
cache_v f16
teacher_forcing explicit_fixed_ids
argmax_separate true
"""
        self.save()

    def corpus(self, sequences):
        return dict(schema_version=1, models=[copy.deepcopy(self.model)], sequences=sequences,
                    reference_revision="1" * 40, reference_build_type="Release", reference_settings=SETTINGS,
                    totals=dict(models=1, sequences=len(sequences), rows=sum(s["row_count"] for s in sequences)))

    def save(self):
        self.plan["origins"] = [dict(id=origin, path="sources/" + origin + "/corpus.json",
                                    sha256=save_json(self.root / "sources" / origin / "corpus.json", source))
                                for origin, source in self.sources.items()]
        proof_hash = save_json(self.root / "proof/weights.json", self.proof)
        derivation_hash = save_json(self.root / "proof/inputs.json", self.derivation)
        self.plan["expansion_records"] = {"sample": dict(original_sha256="a" * 64, derived_sha256="f" * 64,
            tensors=2, values=4, verification_file="proof/weights.json", verification_sha256=proof_hash,
            derivation_file="proof/inputs.json", derivation_sha256=derivation_hash)}
        plan_hash = save_json(self.directory / "input-plan.json", self.plan)
        self.build = dict(plan_sha256=plan_hash, binary_sha256={name: digest(self.directory / name)
                                                             for name in ("reference", "tokenize")})
        build_hash = save_json(self.directory / "build.json", self.build)
        self.frozen.update(plan_sha256=plan_hash, build_sha256=build_hash)
        frozen_hash = save_json(self.directory / "frozen-inputs.json", self.frozen)
        self.basis.update(plan_sha256=plan_hash, frozen_inputs_sha256=frozen_hash,
                          proof=dict(path="proof/weights.json", sha256=proof_hash, derivation_path="proof/inputs.json",
                                     derivation_sha256=derivation_hash, tensors=2, values=4))
        basis_hash = save_json(self.directory / "sample-model.json", self.basis)
        (self.directory / "run.txt").write_text(self.run)
        for origin, wrapper in self.wrappers.items():
            wrapper.update(plan_sha256=plan_hash, frozen_inputs_sha256=frozen_hash,
                           source_corpus=next((x for x in self.plan["origins"] if x["id"] == origin), None),
                           reference_bases={"sample": dict(file="sample-model.json", sha256=basis_hash,
                                                            basis=copy.deepcopy(self.basis))},
                           reference_runs={"sample": dict(file="run.txt", sha256=digest(self.directory / "run.txt"))})
            wrapper["corpus"]["reference_build_sha256"] = build_hash
        self.save_wrappers()

    def save_wrappers(self):
        entries = {}
        for origin, wrapper in self.wrappers.items():
            path = self.directory / (origin + ".json")
            entries[origin] = dict(file=path.name, sha256=save_json(path, wrapper),
                                   sequences=wrapper["corpus"]["totals"]["sequences"],
                                   rows=wrapper["corpus"]["totals"]["rows"])
        save_json(self.index_path, dict(schema="aotx-decoded-f32-reference-index-v1", status="proposed_not_adopted",
                                       manifests=entries, totals=self.plan["totals"]))


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.fixture = ReferenceFixture(directory.name)

    def read(self):
        return read_reference_index(self.fixture.index_path, self.fixture.root)

    def test_four_origins_keep_teacher_ids_and_both_modes(self):
        result = self.read()
        self.assertEqual(result["totals"]["runtime_rows_all_modes"], 512)
        self.assertEqual(result["calibration_rows"], {"sample": 32})
        self.assertEqual(set(result["corpora"]), set(ORIGINS))
        self.assertTrue(all(key.startswith("reference:") for key in result["verified_inputs"]))
        json.dumps(result["identity"], allow_nan=False)
        for origin, entry in result["corpora"].items():
            corpus = entry["corpus"]
            context = corpus["_reference"]
            self.assertEqual((context["kind"], context["origin"]), ("decoded_f32", origin))
            self.assertEqual(context["status"], "proposed_not_adopted")
            self.assertEqual(corpus["sequences"][0]["teacher_ids"], [3])
            self.assertEqual(corpus["sequences"][0]["reference_argmax_ids"], [4])
            _, rows = request(corpus, "sample")
            self.assertEqual({row["mode"] for row in rows}, {"serial", "batch64"})
            self.assertEqual(len(rows), 128)
            self.assertIn("sample-tokens.txt", context["copy_inputs"])
            self.assertIn("sample.so", context["libraries"])
            self.assertFalse(any(name.endswith("/tokens.txt") for name in context["copy_inputs"]))

    def test_model_payloads_are_not_opened_by_reader(self):
        result = self.read()
        self.assertEqual(result["calibration_rows"]["sample"], 32)
        self.assertFalse(any(item["path"].endswith(".gguf") for item in result["verified_inputs"].values()))
        self.assertFalse((self.fixture.root / "models").exists())

    def test_decoded_wrapper_requires_asset_root(self):
        with self.assertRaises(ValueError):
            read_corpus(self.fixture.directory / "original.json")
        self.assertEqual(read_corpus(self.fixture.directory / "original.json", self.fixture.root)["_reference"]["origin"], "original")

    def test_unknown_and_mixed_schemas_fail(self):
        for field, value in (("schema", "aotx-other"), ("purpose", "validation_extension"), ("schema_version", 1)):
            original = copy.deepcopy(self.fixture.wrappers["fresh"])
            self.fixture.wrappers["fresh"][field] = value
            self.fixture.save_wrappers()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read()
            self.fixture.wrappers["fresh"] = original

    def test_omitted_case_and_changed_split_or_slot_fail(self):
        for change in (lambda c: c["sequences"].pop(), lambda c: c["sequences"][0].update(split="validation"),
                       lambda c: c["sequences"][0].update(slot=1), lambda c: c["sequences"][0].update(teacher_ids=[4])):
            original = copy.deepcopy(self.fixture.wrappers["short"])
            change(self.fixture.wrappers["short"]["corpus"])
            self.fixture.save_wrappers()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read()
            self.fixture.wrappers["short"] = original

    def test_origin_omission_and_swap_fail(self):
        for change in (lambda i: i["manifests"].pop("long"),
                       lambda i: i["manifests"].update(fresh=i["manifests"]["short"])):
            self.fixture.save_wrappers()
            value = read_json(self.fixture.index_path)
            change(value)
            save_json(self.fixture.index_path, value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read()

    def test_inline_basis_cannot_replace_hashed_basis(self):
        self.fixture.wrappers["fresh"]["reference_bases"]["sample"]["basis"]["reference_derived"]["sha256"] = "0" * 64
        self.fixture.save_wrappers()
        with self.assertRaises(ValueError):
            self.read()

    def test_actual_proof_fields_are_checked_beyond_success(self):
        for field, value in (("all_values_exact", False), ("tensors", True), ("values", 5),
                             ("f32_tensor_bytes", 17), ("changed_metadata_fields", ["tokenizer.ggml.tokens"]),
                             ("model_sha256", "0" * 64)):
            original = copy.deepcopy(self.fixture.proof)
            self.fixture.proof[field] = value
            self.fixture.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read()
            self.fixture.proof = original

    def test_tensor_detail_cannot_drop_or_repeat_a_tensor(self):
        self.fixture.proof["detail"][1] = copy.deepcopy(self.fixture.proof["detail"][0])
        self.fixture.save()
        with self.assertRaises(ValueError):
            self.read()

    def test_derivation_must_bind_original_and_libraries(self):
        for field, value in (("original_model_sha256", "0" * 64), ("libraries", {"other.so": "0" * 64})):
            original = copy.deepcopy(self.fixture.derivation)
            self.fixture.derivation[field] = value
            self.fixture.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.read()
            self.fixture.derivation = original

    def test_stale_dependency_hash_and_late_library_change_fail(self):
        result = self.read()
        self.fixture.library.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.read()
        with self.assertRaises(ValueError):
            verify_inputs(result["verified_inputs"])

    def test_equal_library_bytes_do_not_permit_symlink_retarget(self):
        first = self.fixture.library.with_name("first.so")
        second = self.fixture.library.with_name("second.so")
        self.fixture.library.rename(first)
        second.write_bytes(first.read_bytes())
        self.fixture.library.symlink_to(first)
        result = self.read()
        self.fixture.library.unlink()
        self.fixture.library.symlink_to(second)
        with self.assertRaisesRegex(ValueError, "input changed"):
            verify_inputs(result["verified_inputs"])

    def test_missing_and_wrong_reference_token_metadata_fail_at_read(self):
        sequence = self.fixture.wrappers["fresh"]["corpus"]["sequences"][0]
        path = self.fixture.directory / Path(sequence["reference_rows_file"]).with_name("tokens.txt")
        path.unlink()
        with self.assertRaises(OSError):
            self.read()
        path.write_text("prefill 256\nteacher 4\nargmax 4\n")
        sequence["token_metadata_sha256"] = digest(path)
        self.fixture.save_wrappers()
        with self.assertRaisesRegex(ValueError, "token metadata differs"):
            self.read()

    def test_dependency_symlink_escape_fails(self):
        outside = self.fixture.root / "outside.json"
        outside.write_text("{}")
        (self.fixture.directory / "linked.json").symlink_to(outside)
        self.fixture.wrappers["fresh"]["reference_bases"]["sample"].update(file="linked.json", sha256=digest(outside))
        self.fixture.save_wrappers()
        with self.assertRaises(ValueError):
            self.read()

    def test_unsafe_asset_paths_fail_before_model_use(self):
        for value in ("", "../weights.gguf", "/weights.gguf", "models\\weights.gguf", "models/weights\n.gguf"):
            self.fixture.basis["reference_derived"]["path"] = value
            self.fixture.save()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read()

    def test_frozen_counts_and_input_types_are_strict(self):
        for mutation in (lambda f: f.plan["totals"].update(runtime_rows_all_modes=511),
                         lambda f: f.plan["original_calibration_runtime_rows_per_model"].update(sample=31),
                         lambda f: f.frozen.update(model_inference_calls=False),
                         lambda f: f.frozen["sequences"][0].update(slot=False)):
            saved = (copy.deepcopy(self.fixture.plan), copy.deepcopy(self.fixture.frozen))
            mutation(self.fixture)
            self.fixture.save()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.read()
            self.fixture.plan, self.fixture.frozen = saved

    def test_fresh_inputs_must_match_tokenized_text(self):
        self.fixture.frozen["sequences"][-1]["teacher_ids"] = [4]
        self.fixture.save()
        with self.assertRaises(ValueError):
            self.read()

    def test_fresh_prefix_overlap_fails_with_matching_token_inputs(self):
        tokens = self.fixture.directory / "sample-tokens.txt"
        tokens.write_text(tokens.read_text().replace("text-0 256 3", "text-0 0 3"))
        self.fixture.frozen["tokenization_files"]["sample"] = digest(tokens)
        self.fixture.frozen["sequences"][192].update(prefill_ids=[0], consumed_ids=[0])
        self.fixture.save()
        with self.assertRaisesRegex(ValueError, "prefixes overlap"):
            self.read()

    def test_run_cannot_claim_other_cache_or_teacher_mode(self):
        self.fixture.run = self.fixture.run.replace("teacher_forcing explicit_fixed_ids", "teacher_forcing greedy")
        self.fixture.save()
        with self.assertRaises(ValueError):
            self.read()

    def test_decoded_capture_requires_exact_basis(self):
        result = self.read()
        for origin in ("original", "fresh"):
            entry = result["corpora"][origin]
            corpus = entry["corpus"]
            data, rows = request(corpus, "sample")
            capture = dict(schema_version=1, vocab=512, model_id="sample", model_sha256="a" * 64,
                           executable_sha256="0" * 64, corpus_sha256=digest(entry["path"]),
                           rows=rows, input_sha256=hashlib.sha256(data).hexdigest())
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "reference basis"):
                checked_capture(capture, self.fixture.directory, corpus, digest(entry["path"]))
            basis = corpus["_reference"]["bases"]["sample"]
            capture["reference_basis"] = dict(kind="decoded_f32", origin=origin, status="proposed_not_adopted",
                                               runtime_original=basis["runtime_original"], reference_derived=basis["runtime_original"])
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, "reference basis"):
                checked_capture(capture, self.fixture.directory, corpus, digest(entry["path"]))


if __name__ == "__main__":
    unittest.main()
