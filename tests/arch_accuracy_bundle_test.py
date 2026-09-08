#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check calibration membership and independent clear-winner coverage.

Inputs: distinct synthetic row records in both execution modes.
Outputs: host test results for valid and invalid result sets.
Exit: zero when all expected outcomes match; nonzero otherwise.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from arch_accuracy_bundle import collect_results, fit_models
from arch_accuracy_io import digest, read_json, request
from arch_accuracy_reference_test import ReferenceFixture, save_json


def fixture():
    rows = []
    for index, (origin, split) in enumerate((("original", "calibration"),
                                           ("short", "regression"), ("fresh", "validation"))):
        for mode in ("serial", "batch64"):
            rows.append(dict(sequence_id="s" + str(index), reference_index=index, position=7 + index,
                             mode=mode, model_id="model", origin=origin, split=split,
                             linf=0.125, relative_l2=0.03125, total_variation=0.0625,
                             reference_gap=1.0, reference_top=2 + index, device_top=2 + index))
    limits = dict(model=dict(bounds=dict(linf=0.25, relative_l2=0.125, total_variation=0.25)))
    totals = dict(runtime_rows_all_modes=6, calibration_runtime_rows=2,
                  historical_regression_runtime_rows=2, fresh_validation_runtime_rows=2)
    return rows, limits, totals


def collect(rows, limits, totals):
    output = io.StringIO()
    result = collect_results(iter(rows), limits, {"model"}, totals, output)
    return result, output.getvalue().splitlines()


class Coverage(unittest.TestCase):
    def test_both_fresh_modes_pass(self):
        result, output = collect(*fixture())
        self.assertEqual(result["missing_clear_modes"], [])
        self.assertEqual(result["failed_rows"], 0)
        self.assertEqual(result["fresh_clear_counts"], {"model/serial": 1, "model/batch64": 1})
        self.assertEqual(len(output), 6)

    def test_historical_clear_rows_cannot_supply_coverage(self):
        rows, limits, totals = fixture()
        for row in rows:
            if row["origin"] == "fresh":
                row["reference_gap"] = 0.5
        result, _ = collect(rows, limits, totals)
        self.assertEqual(result["missing_clear_modes"], ["model/serial", "model/batch64"])
        self.assertEqual(result["failed_rows"], 0)
        self.assertEqual(sum(result["clear_counts"].values()), 4)

    def test_one_clear_mode_cannot_supply_the_other(self):
        for missing_mode in ("serial", "batch64"):
            with self.subTest(mode=missing_mode):
                rows, limits, totals = fixture()
                for row in rows:
                    if row["origin"] == "fresh" and row["mode"] == missing_mode:
                        row["reference_gap"] = 0.25
                result, _ = collect(rows, limits, totals)
                self.assertEqual(result["missing_clear_modes"], ["model/" + missing_mode])

    def test_fresh_label_does_not_replace_validation_split(self):
        rows, limits, totals = fixture()
        rows[4]["split"], rows[2]["split"] = rows[2]["split"], rows[4]["split"]
        result, _ = collect(rows, limits, totals)
        self.assertEqual(result["missing_clear_modes"], ["model/serial"])

    def test_raw_failure_does_not_stop_remaining_rows(self):
        rows, limits, totals = fixture()
        rows[0]["linf"] = 0.5
        result, output = collect(rows, limits, totals)
        self.assertEqual(result["failed_rows"], 1)
        self.assertEqual(result["rows"][0]["failed"], ["linf"])
        self.assertEqual(len(output), 6)

    def test_wrong_clear_winner_fails(self):
        rows, limits, totals = fixture()
        rows[5]["device_top"] = 0
        result, _ = collect(rows, limits, totals)
        self.assertEqual(result["failed_rows"], 1)
        self.assertIn("clear_argmax", result["rows"][5]["failed"])

    def test_missing_row_is_not_a_complete_result(self):
        rows, limits, totals = fixture()
        with self.assertRaisesRegex(ValueError, "row or split count"):
            collect(rows[:-1], limits, totals)

    def test_wrong_split_count_fails(self):
        rows, limits, totals = fixture()
        rows[0]["split"] = "regression"
        with self.assertRaisesRegex(ValueError, "row or split count"):
            collect(rows, limits, totals)


class Membership(unittest.TestCase):
    def fixture(self):
        rows = fixture()[0][:2]
        keys = ("sequence_id", "reference_index", "position", "mode", "model_id", "origin", "split")
        members = [{key: row[key] for key in keys} for row in rows]
        return {"model": rows}, {"model": members}

    def test_original_members_set_bounds(self):
        result = fit_models(*self.fixture())["model"]
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["factor"], 1.5)
        self.assertEqual(result["bounds"], dict(linf=0.1875, relative_l2=0.046875, total_variation=0.09375))

    def test_changed_member_is_rejected(self):
        for key, replacement in (("position", 8), ("origin", "fresh"), ("split", "validation"),
                                 ("sequence_id", "other"), ("mode", "serial")):
            with self.subTest(key=key):
                rows, members = self.fixture()
                rows["model"][1][key] = replacement
                with self.assertRaisesRegex(ValueError, "membership differ"):
                    fit_models(rows, members)

    def test_duplicate_member_is_rejected(self):
        rows, members = self.fixture()
        rows["model"][1] = copy.deepcopy(rows["model"][0])
        with self.assertRaisesRegex(ValueError, "membership differ"):
            fit_models(rows, members)

    def test_missing_model_is_rejected(self):
        rows, members = self.fixture()
        with self.assertRaisesRegex(ValueError, "model membership"):
            fit_models({}, members)


class Commands(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = ReferenceFixture(directory.name)
        self.captures = {}
        self.bounds = self.data.root / "bounds.json"

    def prepare(self, fresh_clear=True):
        for origin, wrapper in self.data.wrappers.items():
            for sequence in wrapper["corpus"]["sequences"]:
                row = np.zeros(512, dtype="<f4")
                row[4] = 5
                row[0] = sequence["prefill_ids"][0] / 512
                if origin == "fresh" and not fresh_clear:
                    row[5] = 4.95
                path = self.data.directory / sequence["reference_rows_file"]
                row.tofile(path)
                sequence["reference_rows_sha256"] = digest(path)
        self.data.save_wrappers()
        for origin, wrapper in self.data.wrappers.items():
            corpus = wrapper["corpus"]
            data, rows = request(corpus, "sample")
            sequences = {s["id"]: s for s in corpus["sequences"]}
            output = self.data.root / ("capture-" + origin)
            output.mkdir()
            with (output / "rows.f32").open("wb") as stream:
                stream.write(b"AOTXAR01" + struct.pack("<II", 512, len(rows)))
                for entry in rows:
                    sequence = sequences[entry["sequence_id"]]
                    row = np.fromfile(self.data.directory / sequence["reference_rows_file"], dtype="<f4")
                    row[7] += (1 + sequence["slot"] % 3) / 64
                    stream.write(row.tobytes())
            basis = self.data.basis
            capture = dict(schema_version=1, corpus_sha256=digest(self.data.directory / (origin + ".json")),
                           model_id="sample", model_sha256=self.data.model["sha256"], vocab=512,
                           executable_sha256="2" * 64, input_sha256=hashlib.sha256(data).hexdigest(),
                           rows=rows, rows_file="rows.f32", rows_sha256=digest(output / "rows.f32"),
                           reference_basis=dict(kind="decoded_f32", origin=origin, status=wrapper["status"],
                                                runtime_original=basis["runtime_original"],
                                                reference_derived=basis["reference_derived"]))
            self.captures[origin] = output / "capture.json"
            save_json(self.captures[origin], capture)

    def command(self, action, captures, output, extra=()):
        command = [sys.executable, str(Path(__file__).with_name("arch_accuracy.py")), action,
                   "--reference-index", str(self.data.index_path), "--asset-root", str(self.data.root),
                   "--captures", *map(str, captures), "--output", str(output), *map(str, extra)]
        return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def calibrate(self):
        result = self.command("calibrate", [self.captures["original"]], self.bounds)
        self.assertEqual(result.returncode, 0, result.stdout)
        return read_json(self.bounds)

    def check(self, captures=None):
        path = self.data.root / "validation.json"
        result = self.command("check", captures or self.captures.values(), path, ["--bounds", self.bounds])
        return result, path

    def test_complete_commands_keep_all_groups(self):
        self.prepare()
        bounds = self.calibrate()
        self.assertEqual(bounds["models"]["sample"]["rows"], 32)
        self.assertEqual(bounds["models"]["sample"]["modes"], dict(serial=16, batch64=16))
        result, path = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)
        report = read_json(path)
        self.assertEqual(len(report["rows"]), 512)
        self.assertEqual(report["failed_rows"], 0)
        self.assertEqual(report["missing_clear_modes"], [])
        self.assertEqual(report["fresh_clear_counts"], {"sample/serial": 64, "sample/batch64": 64})

    def test_complete_commands_require_fresh_clear_rows(self):
        self.prepare(fresh_clear=False)
        self.calibrate()
        result, path = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        report = read_json(path)
        self.assertEqual(report["failed_rows"], 0)
        self.assertEqual(report["missing_clear_modes"], ["sample/serial", "sample/batch64"])
        self.assertEqual(len(report["rows"]), 512)

    def test_validation_capture_cannot_set_bounds(self):
        self.prepare()
        result = self.command("calibrate", [self.captures["fresh"]], self.bounds)
        self.assertEqual(result.returncode, 1)
        self.assertIn("another reference group", result.stdout)
        self.assertFalse(self.bounds.exists())

    def test_missing_capture_has_no_success_file(self):
        self.prepare()
        self.calibrate()
        result, path = self.check([self.captures[key] for key in ("original", "short", "fresh")])
        self.assertEqual(result.returncode, 1)
        self.assertIn("one capture is required", result.stdout)
        self.assertFalse(path.exists())

    def test_mixed_executables_are_rejected(self):
        self.prepare()
        self.calibrate()
        path = self.captures["fresh"]
        data = read_json(path)
        data["executable_sha256"] = "3" * 64
        save_json(path, data)
        result, path = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("same executable", result.stdout)
        self.assertFalse(path.exists())

    def test_raised_bound_is_rejected(self):
        self.prepare()
        data = self.calibrate()
        data["models"]["sample"]["bounds"]["linf"] *= 2
        save_json(self.bounds, data)
        result, path = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("original calibration maxima", result.stdout)
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
