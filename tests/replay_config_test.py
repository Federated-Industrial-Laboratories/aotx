#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check affect replay selection from the configured build option.

Inputs: the replay affect script. Outputs: case counts and failures.
Exit: zero when the configured option selects the expected forms, otherwise nonzero.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else Path(__file__).with_name('replay_affect.sh')
DRIVER = '''set -u
build="$1"
journal="$build/journal"
source "$2"
affect_slots() { printf '64\\n'; }
affect_form() { printf 'form %s\\n' "$1"; }
scenario_affect
'''


class ReplayConfig(unittest.TestCase):
    def check_mode(self, setting, executable, expected):
        with tempfile.TemporaryDirectory() as name:
            build = Path(name)
            if setting is not None:
                (build / 'CMakeCache.txt').write_text(setting)
            if executable:
                program = build / 'aotx_affect_device_test'
                program.write_text('#!/bin/sh\nexit 0\n')
                program.chmod(0o700)
            result = subprocess.run(['bash', '-c', DRIVER, 'replay-config', str(build), str(SCRIPT)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            self.assertEqual(result.stdout, 'form 1\nform 64\n' if expected == 0 else '')
            if expected == 1:
                self.assertIn('build does not state affect support', result.stderr)

    def test_on_without_test_program(self):
        self.check_mode('AOTX_AFFECT:BOOL=ON\n', False, 0)

    def test_on_with_test_program(self):
        self.check_mode('AOTX_AFFECT:BOOL=ON\n', True, 0)

    def test_off_without_test_program(self):
        self.check_mode('AOTX_AFFECT:BOOL=OFF\n', False, 2)

    def test_off_with_stale_test_program(self):
        self.check_mode('AOTX_AFFECT:BOOL=OFF\n', True, 2)

    def test_missing_cache(self):
        self.check_mode(None, True, 1)

    def test_missing_option(self):
        self.check_mode('AOTX_ARCH:STRING=86\n', False, 1)

    def test_duplicate_option(self):
        self.check_mode('AOTX_AFFECT:BOOL=ON\nAOTX_AFFECT:BOOL=OFF\n', True, 1)

    def test_true_values(self):
        for value in ['1', 'TRUE', 'YES', 'Y', 'on']:
            with self.subTest(value=value):
                self.check_mode('AOTX_AFFECT:BOOL=' + value + '\n', False, 0)

    def test_false_values(self):
        for value in ['', '0', 'FALSE', 'NO', 'N', 'IGNORE', 'NOTFOUND', 'x-NOTFOUND', 'off']:
            with self.subTest(value=value):
                self.check_mode('AOTX_AFFECT:BOOL=' + value + '\n', True, 2)

    def check_selector(self, selected, expected):
        with tempfile.TemporaryDirectory() as name:
            build = Path(name)
            (build / 'CMakeCache.txt').write_text('AOTX_AFFECT:BOOL=OFF\n')
            (build / 'manifest.jsonl').write_text('{"name":"language"}\n')
            result = subprocess.run(['bash', str(SCRIPT.with_name('replay_test.sh')),
                                     str(build), str(build / 'journal'), str(build)],
                                    env={**os.environ, 'AOTX_REPLAY_SCENARIO': selected},
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            if expected == 77:
                self.assertIn('scenarios applied 0, skipped 1: ' + selected, result.stdout)
                self.assertNotIn('scenarios applied 1', result.stdout)
            else:
                self.assertIn('unknown scenario', result.stderr)

    def test_unknown_selector(self):
        self.check_selector('unknown', 2)

    def test_off_selector(self):
        self.check_selector('affect', 77)

    def test_missing_model_selector(self):
        self.check_selector('model', 77)

    def test_missing_module_selector(self):
        self.check_selector('module', 77)


if __name__ == '__main__':
    unittest.main()
