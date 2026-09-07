#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check separate process results and complete restore reports.
Inputs: the architecture process driver beside this file.
Outputs: case counts and failures. Exit: 0 on success, 1 on failure.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import arch_process


class LoadLineTests(unittest.TestCase):
    expected = 'layers: 3 linear_delta 1 attention_gated'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='aotx-process-check-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def check_lines(self, lines):
        active = []
        for index, text in enumerate(lines):
            path = self.root / ('boot-' + str(index) + '.log')
            if text is not None:
                path.write_text(text)
            active.append(SimpleNamespace(log_path=path))
        with contextlib.redirect_stdout(io.StringIO()):
            return arch_process.check_load_lines(active, self.expected)

    def test_no_process_is_not_a_pass(self):
        self.assertFalse(self.check_lines([]))

    def test_missing_file_is_not_a_pass(self):
        self.assertFalse(self.check_lines([None]))

    def test_missing_line_is_not_a_pass(self):
        self.assertFalse(self.check_lines(['boot: ready\n']))

    def test_partial_line_is_not_a_match(self):
        self.assertFalse(self.check_lines([self.expected + ' extra\n']))

    def test_each_process_line_is_checked(self):
        for index in range(3):
            with self.subTest(process=index):
                lines = [self.expected + '\n'] * 3
                lines[index] = 'layers: 4 attention_gated\n'
                self.assertFalse(self.check_lines(lines))

    def test_all_three_processes_match(self):
        self.assertTrue(self.check_lines([self.expected + '\n'] * 3))

    def run_reply_failure(self, line):
        build = self.root / 'build'
        build.mkdir()
        (build / 'CMakeCache.txt').write_text('')
        (self.root / 'modules/roles').mkdir(parents=True)
        store = self.root / 'store'
        store.mkdir()
        (store / 'model.gguf').write_bytes(b'')
        (store / 'manifest.jsonl').write_text(json.dumps({
            'name': 'sample', 'role': 'language', 'path': 'model.gguf'}) + '\n')

        class FailedReply:
            def __init__(self, args, root, journal):
                root.mkdir()
                self.log_path = root / 'boot.log'
                self.log_path.write_text(line)

            def ready(self):
                pass

            def turn(self, index):
                raise AssertionError('wrap text in reply')

            def close(self):
                pass

        argv = ['arch_process.py', '--build', str(build), '--store', str(store),
                '--name', 'sample', '--load-line', self.expected,
                '--out', str(self.root / 'output')]
        output = io.StringIO()
        with patch.object(arch_process, 'Run', FailedReply), patch('sys.argv', argv), \
                contextlib.redirect_stdout(output):
            status = arch_process.main()
        self.assertIn('process check failed: wrap text in reply', output.getvalue())
        self.assertIn('1 processes observed', output.getvalue())
        return status

    def test_reply_failure_keeps_a_matching_line_result(self):
        self.assertEqual(self.run_reply_failure(self.expected + '\n'), 1)

    def test_reply_failure_keeps_a_wrong_line_failure(self):
        self.assertEqual(self.run_reply_failure('layers: 4 attention_gated\n'), 3)


class RestoreReportTests(unittest.TestCase):
    expected = {'replayed': '21', 'state_hash': '00000000000012ab'}
    line = 'restore: applied 21 hash 12ab decode_refused 0 pages 0 paced 7 rejected 0'

    def parse(self, text=None, expected=None):
        return arch_process.check_restore_log(self.line if text is None else text,
                                              self.expected if expected is None else expected)

    def test_decode_refusals_fail_with_no_inbound_rejection(self):
        with self.assertRaises(AssertionError):
            self.parse(self.line.replace('decode_refused 0', 'decode_refused 4'))

    def test_leading_zero_hashes_are_equal(self):
        self.assertEqual(self.parse()['state_hash'], '12ab')

    def test_an_expected_open_refusal_can_be_replayed(self):
        text = self.line.replace('decode_refused 0', 'decode_refused 1')
        report = arch_process.check_restore_log(text, self.expected, expected_decode_refused=1)
        self.assertEqual(report['decode_refused'], 1)

    def test_an_expected_refusal_must_be_reproduced(self):
        with self.assertRaises(AssertionError):
            arch_process.check_restore_log(self.line, self.expected, expected_decode_refused=1)

    def test_zero_applied_records_are_not_a_pass(self):
        with self.assertRaises(AssertionError):
            self.parse(self.line.replace('applied 21', 'applied 0'),
                       dict(self.expected, replayed='0'))

    def test_applied_count_must_match(self):
        with self.assertRaises(AssertionError):
            self.parse(self.line.replace('applied 21', 'applied 20'))

    def test_hash_must_match(self):
        with self.assertRaises(AssertionError):
            self.parse(self.line.replace('hash 12ab', 'hash 12ac'))

    def test_record_rejections_fail(self):
        with self.assertRaises(AssertionError):
            self.parse(self.line.replace('rejected 0', 'rejected 1'))

    def test_missing_or_incomplete_report_fails(self):
        for text in ('', self.line.replace(' rejected 0', ''),
                     self.line.replace('decode_refused', 'refused')):
            with self.subTest(text=text), self.assertRaises(AssertionError):
                self.parse(text)

    def test_duplicate_report_fails(self):
        with self.assertRaises(AssertionError):
            self.parse(self.line + '\n' + self.line)


if __name__ == '__main__':
    unittest.main()
