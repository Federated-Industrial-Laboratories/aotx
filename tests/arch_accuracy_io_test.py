# SPDX-License-Identifier: Apache-2.0
"""Check missing data, identity and batch coverage failures in the accuracy reader.

Inputs: temporary synthetic corpus and row files.
Outputs: unittest counts and failure reports.
Exit codes: zero on success, nonzero on failure.
"""
import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import arch_accuracy
import arch_reference

from arch_accuracy_io import (capture_rows, digest, local_file, read_corpus, read_json,
                              reference_rows, request, write_json)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        sequences = []
        for index in range(65):
            metadata = f"prefill {index}\nteacher 1\nargmax 64\n".encode()
            sequences.append(dict(id=str(index), model_id="test", row_count=1,
                                  split="regression" if index == 64 else "calibration" if index < 32 else "validation",
                                  execution_group="serial" if index == 64 else "batch64", slot=index % 64,
                                  prefill_ids=[index], teacher_ids=[1], consumed_ids=[index], prefill_count=1,
                                  evaluated_positions=[0], row_stride_bytes=260,
                                  reference_rows_file=f"reference/{index}/rows.f32",
                                  reference_argmax_ids=[64], token_metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                                  reference_rows_sha256=hashlib.sha256(struct.pack("<65f", *range(1, 66))).hexdigest()))
        self.corpus = dict(schema_version=1, reference_revision="a" * 40, reference_settings={"cpu": True},
                           models=[dict(id="test", model_file="test.gguf", vocab=65, sha256="b" * 64)], sequences=sequences,
                           totals=dict(sequences=65, rows=65))
        self.corpus_path = self.root / "corpus.json"
        write_json(self.corpus_path, self.corpus)

    def rewrite_corpus(self):
        self.corpus_path.write_text(json.dumps(self.corpus))

    def make_capture(self):
        data, rows = request(self.corpus, "test")
        binary = b"AOTXAR01" + struct.pack("<II", 65, len(rows))
        binary += struct.pack("<65f", *range(1, 66)) * len(rows)
        path = self.root / "rows.f32"
        path.write_bytes(binary)
        capture = dict(schema_version=1, corpus_sha256=digest(self.corpus_path),
                       model_id="test", model_sha256="b" * 64, vocab=65, rows=rows,
                       executable_sha256="c" * 64,
                       input_sha256=hashlib.sha256(data).hexdigest(), rows_file=path.name,
                       rows_sha256=digest(path))
        self.capture_path = self.root / "capture.json"
        write_json(self.capture_path, capture)
        return capture

    def test_complete_distinct_batch_and_capture(self):
        self.assertEqual(len(read_corpus(self.corpus_path)["sequences"]), 65)
        capture = self.make_capture()
        checked, rows = capture_rows(self.capture_path, self.corpus, self.corpus_path)
        self.assertEqual(checked, capture)
        self.assertEqual(rows.shape, (129, 65))
        self.assertEqual(sum(row["mode"] == "batch64" for row in capture["rows"]), 64)

    def test_request_preserves_sequence_and_step_order(self):
        sequence = self.corpus["sequences"][-1]
        sequence.update(row_count=2, teacher_ids=[3, 4], consumed_ids=[64, 3], evaluated_positions=[0, 1])
        _, rows = request(self.corpus, "test")
        self.assertEqual(rows[64:66], [dict(sequence_id="64", reference_index=i, position=i, mode="serial")
                                     for i in range(2)])

    def test_duplicate_json_key_fails(self):
        self.corpus_path.write_text('{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            read_json(self.corpus_path)

    def test_missing_reference_file_fails(self):
        with self.assertRaises(OSError):
            reference_rows(self.root, self.corpus["sequences"][0], 65)

    def test_reference_truncation_and_changed_bytes_fail(self):
        path = self.root / self.corpus["sequences"][0]["reference_rows_file"]
        path.parent.mkdir(parents=True)
        for data in (b"", b"\0" * 260):
            path.write_bytes(data)
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                reference_rows(self.root, self.corpus["sequences"][0], 65)

    def test_changed_capture_bytes_fail(self):
        self.make_capture()
        path = self.root / "rows.f32"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaises(ValueError):
            capture_rows(self.capture_path, self.corpus, self.corpus_path)

    def test_truncated_capture_fails_even_with_its_new_hash(self):
        capture = self.make_capture()
        path = self.root / "rows.f32"
        path.write_bytes(path.read_bytes()[:-4])
        capture["rows_sha256"] = digest(path)
        self.capture_path.write_text(json.dumps(capture))
        with self.assertRaises(ValueError):
            capture_rows(self.capture_path, self.corpus, self.corpus_path)

    def test_wrong_row_mapping_fails(self):
        capture = self.make_capture()
        capture["rows"][1] = copy.deepcopy(capture["rows"][0])
        self.capture_path.write_text(json.dumps(capture))
        with self.assertRaises(ValueError):
            capture_rows(self.capture_path, self.corpus, self.corpus_path)

    def test_changed_corpus_identity_fails(self):
        self.make_capture()
        self.corpus_path.write_text(self.corpus_path.read_text() + "\n")
        with self.assertRaises(ValueError):
            capture_rows(self.capture_path, self.corpus, self.corpus_path)

    def test_absent_and_duplicate_batch_slots_fail(self):
        original = copy.deepcopy(self.corpus)
        for mutate in (lambda c: c["sequences"].pop(0),
                       lambda c: c["sequences"][1].update(slot=0)):
            self.corpus = copy.deepcopy(original)
            mutate(self.corpus)
            self.rewrite_corpus()
            with self.assertRaises(ValueError):
                read_corpus(self.corpus_path)

    def test_identical_prefixes_fail(self):
        self.corpus["sequences"][1].update(prefill_ids=[0], consumed_ids=[0])
        self.rewrite_corpus()
        with self.assertRaises(ValueError):
            read_corpus(self.corpus_path)

    def test_wrong_position_and_token_fail(self):
        for field, value in (("evaluated_positions", [1]), ("teacher_ids", [65]), ("consumed_ids", [4])):
            original = copy.deepcopy(self.corpus["sequences"][0])
            self.corpus["sequences"][0][field] = value
            self.rewrite_corpus()
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_corpus(self.corpus_path)
            self.corpus["sequences"][0] = original

    def test_absolute_parent_and_external_symlink_paths_fail(self):
        (self.root / "outside").symlink_to(self.root.parent)
        for path in ("/etc/hostname", "../other", "outside/other"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                local_file(self.root, path)

    def test_existing_output_cannot_be_replaced(self):
        with self.assertRaises(FileExistsError):
            write_json(self.corpus_path, {})

    def test_nonfinite_json_values_fail_in_unused_fields(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e999"):
            self.corpus_path.write_text('{"unused":' + value + '}')
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_json(self.corpus_path)

    def test_unknown_purpose_fails(self):
        for purpose in ("calibration", "", None, True):
            self.corpus["purpose"] = purpose
            self.rewrite_corpus()
            with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                read_corpus(self.corpus_path)

    def test_validation_extension_retains_distinct_batch(self):
        self.corpus.update(purpose="validation_extension", base_corpus_sha256="c" * 64,
                           base_bounds_sha256="d" * 64)
        self.corpus["sequences"].pop()
        self.corpus["totals"] = dict(models=1, sequences=64, rows=64, rows_per_model=64)
        for sequence in self.corpus["sequences"]:
            sequence["split"] = "validation"
        self.rewrite_corpus()
        checked = read_corpus(self.corpus_path)
        _, rows = request(checked, "test")
        self.assertEqual([sum(r["mode"] == mode for r in rows) for mode in ("serial", "batch64")], [64, 64])

    def test_identifier_cannot_form_a_path_or_tsv_field(self):
        for value in ("", ".", "..", "../escape", "/escape", "a/b", "a\\b", "a\tb", "a\nb", "a\x00b", "a\u2028b"):
            for field in ("sequence", "model"):
                corpus = copy.deepcopy(self.corpus)
                if field == "sequence":
                    corpus["sequences"][0]["id"] = value
                else:
                    corpus["models"][0]["id"] = value
                    for sequence in corpus["sequences"]:
                        sequence["model_id"] = value
                self.corpus_path.write_text(json.dumps(corpus))
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    read_corpus(self.corpus_path)

    def test_model_file_must_be_a_basename(self):
        for value in ("", "..", "../weights", "/weights", "dir/weights", "dir\\weights", "a\tweights"):
            self.corpus["models"][0]["model_file"] = value
            self.rewrite_corpus()
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_corpus(self.corpus_path)

    def test_numeric_fields_reject_boolean_and_float_equivalents(self):
        cases = [((), "schema_version", True), ((), "schema_version", 1.0),
                 (("models", 0), "vocab", 65.0), (("models", 0), "bytes", True),
                 (("reference_settings",), "n_ctx", True),
                 (("sequences", 0), "row_count", True), (("sequences", 0), "row_count", 1.0),
                 (("sequences", 0), "prefill_count", True), (("sequences", 0), "slot", False),
                 (("sequences", 0), "row_stride_bytes", 260.0),
                 (("sequences", 0), "consumed_ids", [False]),
                 (("sequences", 0), "evaluated_positions", [False]),
                 (("sequences", 0), "reference_argmax_ids", [64.0]),
                 (("totals",), "sequences", 65.0), (("totals",), "rows", 65.0)]
        for parts, field, value in cases:
            corpus = copy.deepcopy(self.corpus)
            target = corpus
            for part in parts:
                target = target[part]
            target[field] = value
            self.corpus_path.write_text(json.dumps(corpus))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                read_corpus(self.corpus_path)

    def test_hexadecimal_identities_fail_before_file_access(self):
        cases = [((), "reference_revision", "z" * 40),
                 (("models", 0), "sha256", "g" * 64),
                 (("sequences", 0), "reference_rows_sha256", "a" * 63),
                 (("sequences", 0), "token_metadata_sha256", True),
                 ((), "base_bounds_sha256", "-" * 64)]
        for parts, field, value in cases:
            corpus = copy.deepcopy(self.corpus)
            target = corpus
            for part in parts:
                target = target[part]
            target[field] = value
            self.corpus_path.write_text(json.dumps(corpus))
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_corpus(self.corpus_path)

    def test_empty_and_escaping_reference_paths_fail_in_corpus(self):
        (self.root / "outside").symlink_to(self.root.parent)
        for value in ("", ".", "a//b", "a/./b", "a\\b", "a\nb", "../b", "outside/b"):
            self.corpus["sequences"][0]["reference_rows_file"] = value
            self.rewrite_corpus()
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_corpus(self.corpus_path)

    def test_declared_auxiliary_files_require_their_hashes(self):
        changes = [lambda c: c["models"][0].update(template_file="template.jinja"),
                   lambda c: c["sequences"][0].update(prompt_file="prompt.txt"),
                   lambda c: c.update(reference_libraries=[dict(file="lib/test.so", bytes=1)]),
                   lambda c: c.update(reference_runs={"test": dict(metadata_file="run.txt")})]
        for change in changes:
            corpus = copy.deepcopy(self.corpus)
            change(corpus)
            self.corpus_path.write_text(json.dumps(corpus))
            with self.subTest(corpus=corpus.keys()), self.assertRaises((KeyError, ValueError)):
                read_corpus(self.corpus_path)

    def reference_file(self, sequence, root=None):
        path = (root or self.root) / sequence["reference_rows_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(struct.pack("<65f", *range(1, 66)) * sequence["row_count"])
        metadata = "".join(name + " " + " ".join(map(str, sequence[key])) + "\n"
                           for name, key in (("prefill", "prefill_ids"), ("teacher", "teacher_ids"),
                                             ("argmax", "reference_argmax_ids")))
        path.with_name("tokens.txt").write_text(metadata)
        return path

    def test_reference_metadata_matches_tokens_and_full_row_count(self):
        for count in (1, 64):
            sequence = copy.deepcopy(self.corpus["sequences"][0])
            sequence.update(row_count=count, teacher_ids=[1] * count, reference_argmax_ids=[64] * count)
            path = self.reference_file(sequence)
            sequence["reference_rows_sha256"] = digest(path)
            sequence["token_metadata_sha256"] = digest(path.with_name("tokens.txt"))
            with self.subTest(rows=count):
                rows = reference_rows(self.root, sequence, 65)
                self.assertEqual(rows.shape, (count, 65))
                self.assertEqual(rows[-1].argmax(), 64)
                path.with_name("tokens.txt").write_text("prefill 2\nteacher 1\nargmax 64\n")
                sequence["token_metadata_sha256"] = digest(path.with_name("tokens.txt"))
                with self.assertRaises(ValueError):
                    reference_rows(self.root, sequence, 65)

    def test_reference_metadata_changed_bytes_fail(self):
        sequence = self.corpus["sequences"][0]
        path = self.reference_file(sequence)
        path.with_name("tokens.txt").write_text("prefill 2\nteacher 1\nargmax 64\n")
        with self.assertRaises(ValueError):
            reference_rows(self.root, sequence, 65)

    def test_capture_row_numbers_reject_boolean_equivalents_in_both_modes(self):
        capture = self.make_capture()
        for index in (0, 65):
            for field in ("reference_index", "position"):
                changed = copy.deepcopy(capture)
                changed["rows"][index][field] = False
                self.capture_path.write_text(json.dumps(changed))
                with self.subTest(mode=capture["rows"][index]["mode"], field=field), self.assertRaises(ValueError):
                    capture_rows(self.capture_path, self.corpus, self.corpus_path)

    def test_capture_schema_and_identity_are_typed(self):
        capture = self.make_capture()
        for field, value in (("schema_version", True), ("vocab", 65.0), ("executable_sha256", "q" * 64)):
            changed = dict(capture, **{field: value})
            self.capture_path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                capture_rows(self.capture_path, self.corpus, self.corpus_path)

    def test_boolean_bounds_version_fails_before_measurement(self):
        bounds = self.root / "bounds.json"
        write_json(bounds, dict(schema_version=True, corpus_sha256=digest(self.corpus_path),
                   models={"test": {"bounds": dict(linf=0.1, relative_l2=0.1, total_variation=0.1)}}))
        with patch.object(arch_accuracy, "measured_rows") as measure:
            with self.assertRaises(ValueError):
                arch_accuracy.check(SimpleNamespace(corpus=self.corpus_path, bounds=bounds,
                                                     coverage_corpus=[], coverage_captures=[]))
            measure.assert_not_called()

    def capture_arguments(self, name):
        executable = self.root / "program"
        executable.write_bytes(b"fixed executable")
        store = self.root / "store"
        store.mkdir(exist_ok=True)
        model = store / "test.gguf"
        model.write_bytes(b"fixed model")
        self.corpus["models"][0]["sha256"] = digest(model)
        self.rewrite_corpus()
        entry = dict(role="language", path=model.name, sha256=digest(model), revision="d" * 40,
                     bytes=model.stat().st_size)
        (store / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
        return SimpleNamespace(corpus=self.corpus_path, model="test", role="language", store=store,
                               executable=executable, output=self.root / name)

    def capture_program(self, command, **kwargs):
        Path(command[-1]).write_bytes(b"AOTXAR01" + struct.pack("<II", 65, 129)
                                     + struct.pack("<65f", *range(1, 66)) * 129)
        return SimpleNamespace(returncode=0)

    def test_capture_success_records_stable_inputs_and_both_modes(self):
        args = self.capture_arguments("capture-good")
        with patch.object(arch_accuracy.subprocess, "run", side_effect=self.capture_program), redirect_stdout(io.StringIO()):
            self.assertEqual(arch_accuracy.capture(args), 0)
        capture = read_json(args.output / "capture.json")
        self.assertEqual({row["mode"] for row in capture["rows"]}, {"serial", "batch64"})
        self.assertIn("verified_inputs", capture)
        self.assertEqual(set(capture["verified_inputs"]), {"corpus", "executable", "store_manifest", "model", "request"})
        self.assertEqual(capture["store_manifest_sha256"], digest(args.store / "manifest.jsonl"))

    def test_capture_accepts_a_fixed_model_symlink(self):
        args = self.capture_arguments("capture-linked")
        model = args.store / "test.gguf"
        target = self.root / "retained.gguf"
        model.rename(target)
        model.symlink_to(target)
        with patch.object(arch_accuracy.subprocess, "run", side_effect=self.capture_program), redirect_stdout(io.StringIO()):
            self.assertEqual(arch_accuracy.capture(args), 0)

    def test_capture_rejects_changed_inputs_before_success_file(self):
        for field in ("corpus", "executable", "manifest", "model", "request", "same_bytes_replacement"):
            args = self.capture_arguments("capture-" + field)
            def program(command, **kwargs):
                result = self.capture_program(command, **kwargs)
                paths = dict(corpus=args.corpus, executable=args.executable,
                             manifest=args.store / "manifest.jsonl", model=args.store / "test.gguf",
                             request=Path(command[-2]), same_bytes_replacement=args.executable)
                path = paths[field]
                if field == "same_bytes_replacement":
                    other = self.root / "replacement"
                    other.write_bytes(path.read_bytes())
                    other.replace(path)
                else:
                    path.write_bytes(path.read_bytes() + b" ")
                return result
            with self.subTest(field=field), patch.object(arch_accuracy.subprocess, "run", side_effect=program):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    arch_accuracy.capture(args)
                self.assertFalse((args.output / "capture.json").exists())

    def test_capture_rejects_retargeted_model_symlink_with_equal_bytes(self):
        args = self.capture_arguments("capture-retargeted")
        model = args.store / "test.gguf"
        first, second = self.root / "first.gguf", self.root / "second.gguf"
        model.rename(first)
        second.write_bytes(first.read_bytes())
        model.symlink_to(first)
        def program(command, **kwargs):
            result = self.capture_program(command, **kwargs)
            model.unlink()
            model.symlink_to(second)
            return result
        with patch.object(arch_accuracy.subprocess, "run", side_effect=program), self.assertRaisesRegex(ValueError, "input changed"):
            arch_accuracy.capture(args)
        self.assertFalse((args.output / "capture.json").exists())

    def test_invalid_capture_output_has_no_success_file(self):
        args = self.capture_arguments("capture-truncated")
        def program(command, **kwargs):
            Path(command[-1]).write_bytes(b"AOTXAR01")
            return SimpleNamespace(returncode=0)
        with patch.object(arch_accuracy.subprocess, "run", side_effect=program), self.assertRaises(ValueError):
            arch_accuracy.capture(args)
        self.assertFalse((args.output / "capture.json").exists())

    def test_manifest_escape_duplicate_and_invalid_revision_fail_before_run(self):
        for field, value in (("path", "../test.gguf"), ("revision", "g" * 40), ("bytes", True), ("duplicate", None)):
            args = self.capture_arguments("capture-bad-" + field)
            path = args.store / "manifest.jsonl"
            entry = json.loads(path.read_text())
            (self.root / "test.gguf").write_bytes((args.store / "test.gguf").read_bytes())
            if field == "duplicate":
                path.write_text(json.dumps(entry)[:-1] + ',"role":"language"}\n')
            else:
                entry[field] = value
                path.write_text(json.dumps(entry))
            with self.subTest(field=field), patch.object(arch_accuracy.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    arch_accuracy.capture(args)
                run.assert_not_called()

    def reference_arguments(self, name):
        args = self.capture_arguments(name)
        return ["arch_reference.py", "--corpus", str(args.corpus), "--models", str(args.store),
                "--executable", str(args.executable), "--output", str(args.output)], args

    def reference_program(self, command, **kwargs):
        generated = Path(command[-1])
        for sequence in self.corpus["sequences"]:
            self.reference_file(dict(sequence, reference_rows_file=sequence["id"] + "/rows.f32"), generated)
        return SimpleNamespace(returncode=0)

    def reference_run(self, argv, program):
        with patch.object(sys, "argv", argv), patch.object(arch_reference.subprocess, "run", side_effect=program):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                return arch_reference.main()

    def test_reference_regeneration_retains_checked_metadata(self):
        argv, args = self.reference_arguments("reference-good")
        self.assertEqual(self.reference_run(argv, self.reference_program), 0)
        output = read_json(args.output / "generation.json")
        self.assertEqual((output["sequences"], output["rows"]), (65, 65))
        self.assertEqual(digest(args.output / "corpus.json"), digest(args.corpus))
        for sequence in self.corpus["sequences"]:
            self.assertEqual(reference_rows(args.output, sequence, 65).shape, (1, 65))

    def test_reference_changed_inputs_have_no_success_file(self):
        for field in ("corpus", "executable", "model", "jobs", "tokens"):
            argv, args = self.reference_arguments("reference-" + field)
            def program(command, **kwargs):
                result = self.reference_program(command, **kwargs)
                paths = dict(corpus=args.corpus, executable=args.executable, model=args.store / "test.gguf",
                             jobs=Path(command[-2]), tokens=args.output / "jobs/0.ids")
                path = paths[field]
                path.write_bytes(path.read_bytes() + b" ")
                return result
            with self.subTest(field=field):
                self.assertEqual(self.reference_run(argv, program), 1)
                self.assertFalse((args.output / "generation.json").exists())

    def test_reference_rejects_wrong_tokens_even_with_expected_row_bytes(self):
        argv, args = self.reference_arguments("reference-wrong-tokens")
        def program(command, **kwargs):
            result = self.reference_program(command, **kwargs)
            (Path(command[-1]) / "0/tokens.txt").write_text("prefill 9\nteacher 1\nargmax 64\n")
            return result
        self.assertEqual(self.reference_run(argv, program), 1)
        self.assertFalse((args.output / "generation.json").exists())

    def test_reference_identifier_escape_is_rejected_before_jobs(self):
        argv, args = self.reference_arguments("reference-escape")
        self.corpus["sequences"][0]["id"] = "../../outside"
        self.rewrite_corpus()
        with patch.object(sys, "argv", argv), patch.object(arch_reference.subprocess, "run") as run:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(arch_reference.main(), 1)
            run.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_reference_output_directory_cannot_split_tsv_fields(self):
        argv, args = self.reference_arguments("reference\toutput")
        with patch.object(sys, "argv", argv), patch.object(arch_reference.subprocess, "run") as run:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(arch_reference.main(), 1)
            run.assert_not_called()
        self.assertFalse(args.output.exists())

    def test_file_cannot_supply_a_checked_reference_context(self):
        self.corpus["_reference"] = dict(kind="decoded_f32", origin="fresh")
        self.rewrite_corpus()
        with self.assertRaisesRegex(ValueError, "reference context"):
            read_corpus(self.corpus_path)


if __name__ == "__main__":
    unittest.main()
