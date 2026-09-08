# SPDX-License-Identifier: Apache-2.0
"""Check fixed reference generation and its input identities.

Inputs: small temporary model, library, token and row files.
Outputs: unittest counts and failure reports.
Exit codes: zero on success, nonzero on failure.
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import arch_reference
from arch_accuracy_io import digest, file_identity, read_json


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.exe = self.root / "reference"
        self.exe.write_bytes(b"fixed CPU program")
        self.original = self.root / "original.gguf"
        self.derived = self.root / "derived.gguf"
        self.original.write_bytes(b"original model")
        self.derived.write_bytes(b"decoded model")
        library = self.root / "lib/libllama.so.0"
        library.parent.mkdir()
        library.write_bytes(b"pinned CPU library")
        self.library = library
        self.proof = self.root / "proof.json"
        self.proof.write_bytes(b"independent proof")
        self.input_path = self.root / "reference-index.json"
        self.input_path.write_bytes(b"fixed reference index")
        self.metadata = b"prefill 2\nteacher 0 2\nargmax 1 1\n"
        self.payload = struct.pack("<6f", 0, 1, -1, 0, 1, -1)
        self.sequences = []
        bases = dict(test=dict(runtime_original=self.model(self.original), reference_derived=self.model(self.derived)))
        libraries = {library.name: dict(path=str(library), sha256=digest(library), bytes=library.stat().st_size)}
        dependencies = {"proof.json": str(self.proof)}
        retained = {str(path): file_identity(path) for path in (self.proof, self.input_path)}
        corpora = {}
        for origin in ("original", "short", "long", "fresh"):
            path = self.root / (origin + ".json")
            path.write_bytes(origin.encode())
            retained[str(path)] = file_identity(path)
            sequence = dict(id=origin, model_id="test", row_count=2, prefill_ids=[2], teacher_ids=[0, 2],
                            reference_argmax_ids=[1, 1], reference_rows_file="rows/test/" + origin + "/rows.f32",
                            row_stride_bytes=12, reference_rows_sha256=hashlib.sha256(self.payload).hexdigest(),
                            token_metadata_sha256=hashlib.sha256(self.metadata).hexdigest())
            self.sequences.append(sequence)
            context = dict(kind="decoded_f32", origin=origin, bases=bases, libraries=libraries,
                           copy_inputs=dependencies, verified_inputs=retained)
            corpus = dict(models=[dict(id="test", model_file="original.gguf", vocab=3)],
                          sequences=[sequence], _reference=context)
            corpora[origin] = dict(path=path, corpus=corpus)
        self.index = dict(corpora=corpora, verified_inputs=retained, identity={"kind": "decoded_f32"})

    def model(self, path):
        return dict(path=path.name, sha256=digest(path), bytes=path.stat().st_size)

    def program(self, command, **kwargs):
        if command[0] == "ldd":
            return SimpleNamespace(returncode=0, stdout=f"libllama.so.0 => {self.library} (0x01)\n")
        self.assertEqual(command[-1], "--fixed-teacher")
        self.assertEqual(Path(command[1]), self.derived)
        self.assertEqual(kwargs["env"]["LD_LIBRARY_PATH"], str(self.library.parent))
        self.assertEqual(kwargs["env"]["GGML_BACKEND_PATH"], str(self.library.parent))
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        generated = Path(command[-2])
        generated.mkdir()
        (generated / "run.txt").write_bytes(b"actual fixed teacher program metadata")
        for line in Path(command[2]).read_text().splitlines():
            fields = line.split("\t")
            self.assertEqual(Path(fields[4]).read_text(), "prefill 2\ngenerated 0 2\n")
            destination = generated / fields[0]
            destination.mkdir()
            (destination / "rows.f32").write_bytes(self.payload)
            (destination / "tokens.txt").write_bytes(self.metadata)
        return SimpleNamespace(returncode=0)

    def run_generation(self, program=None):
        argv = ["arch_reference.py", "--reference-index", str(self.input_path), "--asset-root", str(self.root),
                "--executable", str(self.exe), "--output", str(self.output)]
        reader = SimpleNamespace(read_reference_index=lambda *args: self.index)
        with patch.dict(sys.modules, {"arch_accuracy_reference": reader}), patch.object(sys, "argv", argv):
            with patch.object(arch_reference.subprocess, "run", side_effect=program or self.program):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    return arch_reference.main()

    def test_fixed_inputs_and_four_manifests_are_preserved(self):
        self.assertEqual(self.run_generation(), 0)
        result = read_json(self.output / "generation.json")
        self.assertEqual((result["sequences"], result["rows"], result["mode"]), (4, 8, "fixed_teacher"))
        self.assertEqual(result["executable_sha256"], digest(self.exe))
        for name in ("reference-index.json", "proof.json", "original.json", "short.json", "long.json", "fresh.json"):
            self.assertEqual((self.output / name).read_bytes(), (self.root / name).read_bytes())
        for sequence in self.sequences:
            row = self.output / sequence["reference_rows_file"]
            self.assertEqual(row.read_bytes(), self.payload)
            self.assertEqual(row.with_name("tokens.txt").read_bytes(), self.metadata)
        self.assertIn("actual_run:test", result["verified_inputs"])

    def test_changed_dependencies_fail_without_success(self):
        for field in ("original", "derived", "library", "proof", "executable", "jobs", "tokens"):
            with self.subTest(field=field):
                self.output = self.root / ("output-" + field)
                changed = []
                def program(command, **kwargs):
                    result = self.program(command, **kwargs)
                    if command[0] != "ldd":
                        paths = dict(original=self.original, derived=self.derived, library=self.library,
                                     proof=self.proof, executable=self.exe, jobs=Path(command[2]),
                                     tokens=self.output / "jobs/original.ids")
                        path = paths[field]
                        changed.append((path, path.read_bytes()))
                        path.write_bytes(path.read_bytes() + b" ")
                    return result
                self.assertEqual(self.run_generation(program), 1)
                self.assertFalse((self.output / "generation.json").exists())
                for path, data in changed:
                    path.write_bytes(data)
                for path in list(self.index["verified_inputs"]):
                    self.index["verified_inputs"][path] = file_identity(path)

    def test_nonfinite_zero_and_false_argmax_fail_even_with_matching_hash(self):
        for values in ((0, float("nan"), -1) * 2, (0, 0, 0) * 2, (4, 1, -1) * 2):
            with self.subTest(values=values):
                self.output = self.root / ("bad-rows-" + str(values[0]) + "-" + str(values[1]))
                self.payload = struct.pack("<6f", *values)
                for sequence in self.sequences:
                    sequence["reference_rows_sha256"] = hashlib.sha256(self.payload).hexdigest()
                self.assertEqual(self.run_generation(), 1)
                self.assertFalse((self.output / "generation.json").exists())

    def test_truncated_rows_fail_even_with_matching_hash(self):
        self.payload = self.payload[:-4]
        for sequence in self.sequences:
            sequence["reference_rows_sha256"] = hashlib.sha256(self.payload).hexdigest()
        self.assertEqual(self.run_generation(), 1)
        self.assertFalse((self.output / "generation.json").exists())

    def test_changed_teacher_metadata_fails_with_expected_rows(self):
        self.metadata = b"prefill 2\nteacher 1 1\nargmax 1 1\n"
        self.assertEqual(self.run_generation(), 1)
        self.assertFalse((self.output / "generation.json").exists())

    def test_linked_library_must_resolve_to_pinned_file(self):
        def program(command, **kwargs):
            self.assertEqual(command[0], "ldd")
            return SimpleNamespace(returncode=0, stdout="libllama.so.0 => /different/libllama.so.0 (0x01)\n")
        self.assertEqual(self.run_generation(program), 1)
        self.assertFalse(self.output.exists())

    def test_legacy_preserves_contained_library_alias_names(self):
        versioned = self.library.with_name("libllama.so.0.1")
        self.library.rename(versioned)
        self.library.symlink_to(versioned.name)
        alias = self.library.with_name("libllama.so")
        alias.symlink_to(versioned.name)
        names = [alias, self.library, versioned]
        corpus = dict(models=[], sequences=[], reference_libraries=[
            dict(file=str(path.relative_to(self.root)), sha256=digest(path), bytes=path.stat().st_size)
            for path in names])
        args = SimpleNamespace(corpus=self.input_path, executable=self.exe, output=self.output,
                               asset_root=None, models=self.root)
        with patch.object(arch_reference, "read_corpus", return_value=corpus):
            with patch.object(arch_reference.subprocess, "run", side_effect=self.program):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(arch_reference.legacy(args), 0)
        result = read_json(self.output / "generation.json")
        for path in names:
            identity = result["verified_inputs"]["library:" + path.name]
            self.assertEqual(identity["path"], str(path))
            self.assertEqual(identity["resolved"], str(versioned))
            self.assertEqual((self.output / path.relative_to(self.root)).read_bytes(), versioned.read_bytes())

    def test_failed_program_has_no_success_record(self):
        def program(command, **kwargs):
            if command[0] == "ldd":
                return self.program(command, **kwargs)
            return SimpleNamespace(returncode=1)
        self.assertEqual(self.run_generation(program), 1)
        self.assertFalse((self.output / "generation.json").exists())


if __name__ == "__main__":
    unittest.main()
