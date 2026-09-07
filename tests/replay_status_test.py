#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check replay wait bounds and execution failure classification without a GPU.

Inputs: the replay driver path, or the adjacent driver by default.
Outputs: N=1/N=64 case counts and failed assertions.
Exit: zero when every control passes, otherwise nonzero.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else Path(__file__).with_name('replay_test.sh')
MARKER = '# ---- the scenarios ----\n'
SCENARIOS = ['lines', 'settings', 'say', 'auth', 'answered', 'late', 'wide',
             'session', 'affect', 'model', 'module']
TOTAL = 0


class ReplayStatus(unittest.TestCase):
    def run_case(self, number, target, mode, selected=True, missing=None):
        global TOTAL
        with tempfile.TemporaryDirectory(prefix='aotx-replay-status-') as directory:
            root = Path(directory)
            build = root / 'build'
            build.mkdir()
            models = root / 'models'
            models.mkdir()
            if missing != 'manifest':
                names = [] if missing == 'language' else ['language']
                if missing != 'language-q4':
                    names.append('language-q4')
                (models / 'manifest.jsonl').write_text(''.join('{"name":"' + name + '"}\n' for name in names))
            (build / 'CMakeCache.txt').write_text('AOTX_AFFECT:BOOL=OFF\n')
            for program in ['aotx_boot', 'aotx_journal']:
                file = build / program
                file.write_text('#!/bin/sh\nexit 0\n')
                file.chmod(0o700)
            restore = build / 'aotx_restore'
            restore.write_text('''#!/bin/sh
if [ -e "$2/restored" ]; then id=bbbb; else id=aaaa; fi
: >"$2/restored"
printf 'restore boot=%s last_tick=1 replayed=1 state_hash=abc\n' "$id"
''')
            restore.chmod(0o700)
            for source in SCRIPT.parent.glob('replay_*.sh'):
                if source.name != 'replay_test.sh':
                    (root / source.name).symlink_to(source)
            source = SCRIPT.read_text()
            self.assertEqual(source.count(MARKER), 1, 'the driver has one scenario dispatcher')
            definitions, dispatcher = source.split(MARKER)
            stubs = '\n'.join('scenario_' + name + '() { return 0; }'
                              for name in SCENARIOS if name != target)
            controls = '''
# Exercise the real expiry path with one empty probe and no wall-clock delay.
seq() { printf '%s\n' "$*" >>"$journal-seq"; printf '1\n'; }
sleep() { :; }
kill() { :; }
feed_auth() { :; }
feed_auth_answer() { :; }
'''
            if mode == 'empty_restore':
                controls += '''
wait_request() {
    mkdir -p "$auth_journal/manifest"
    printf '{"agent":1,"tool":"fs_read","request":%s,"output_hash":"abc"}\n' "$case_id" \
        >"$auth_journal/manifest/aaaa.jsonl"
    return 0
}
'''
            if mode == 'bounds':
                controls += '''
for wait_name in wait_first_request wait_ticking wait_prompt wait_sampled wait_turn wait_request wait_late_request; do
    : >"$journal-seq"
    "$wait_name" "$journal-missing"
    result=$?
    printf 'bound %s status %s seq %s\n' "$wait_name" "$result" "$(cat "$journal-seq")"
done
exit 0
'''
            driver = root / 'replay_test.sh'
            driver.write_text(definitions + '\ncase_id=' + str(number) + '\n' + stubs +
                              controls + MARKER + dispatcher)
            environment = {**os.environ, 'AOTX_REPLAY_SCENARIO': target if selected else ''}
            result = subprocess.run(['bash', str(driver), str(build), str(root / ('journal-' + str(number))), str(models)],
                                    capture_output=True, text=True, env=environment, timeout=10)
            TOTAL += 1
            return result

    def test_expiry_is_failure(self):
        for count in [1, 64]:
            before = TOTAL
            for index in range(count):
                for selected in [True, False]:
                    with self.subTest(count=count, index=index, selected=selected):
                        result = self.run_case(count * 1000 + index, 'auth', 'timeout', selected)
                        output = result.stdout + result.stderr
                        self.assertEqual(result.returncode, 1, output)
                        self.assertIn('auth gave no request', output)
                        self.assertIn('in 360 seconds', output)
                        self.assertIn('scenarios applied ' + ('1' if selected else '11') + ', skipped 0', output)
                        self.assertNotIn('auth (the restored run made no turn)', output)
            print(f'replay expiry N={count}: {TOTAL - before} controls', flush=True)

    def test_empty_restore_is_failure(self):
        for count in [1, 64]:
            before = TOTAL
            for index in range(count):
                for selected in [True, False]:
                    with self.subTest(count=count, index=index, selected=selected):
                        number = count * 1000 + index
                        result = self.run_case(number, 'auth', 'empty_restore', selected)
                        output = result.stdout + result.stderr
                        self.assertEqual(result.returncode, 1, output)
                        self.assertIn('request ' + str(number) + ' waiting', output)
                        self.assertIn('run made no turn in 4000 ticks', output)
                        self.assertIn('scenarios applied ' + ('1' if selected else '11') + ', skipped 0', output)
            print(f'replay empty restore N={count}: {TOTAL - before} controls', flush=True)

    def test_wait_bounds(self):
        result = self.run_case(1, 'auth', 'bounds')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ['wait_first_request', 'wait_ticking', 'wait_prompt', 'wait_turn',
                     'wait_request', 'wait_late_request']:
            self.assertIn('bound ' + name + ' status 1 seq 1 3600\n', result.stdout)
        self.assertIn('bound wait_sampled status 1 seq 1 720\n', result.stdout)

    def test_missing_inputs_are_skips(self):
        for target in ['late', 'module', 'affect']:
            for selected in [True, False]:
                with self.subTest(target=target, selected=selected):
                    result = self.run_case(1, target, 'missing', selected)
                    self.assertEqual(result.returncode, 77 if selected else 0, result.stdout + result.stderr)
                    self.assertIn('scenarios applied ' + ('0' if selected else '10') + ', skipped 1', result.stdout)
                    self.assertIn(target, result.stdout)
        for missing in ['manifest', 'language']:
            with self.subTest(missing=missing):
                result = self.run_case(1, '', 'missing', False, missing)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('scenarios applied 3, skipped 8', result.stdout)
                for scenario in ['say', 'auth', 'answered', 'late', 'wide', 'session', 'affect', 'model']:
                    self.assertIn(scenario, result.stdout)
        result = self.run_case(1, '', 'missing', False, 'language-q4')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('scenarios applied 10, skipped 1', result.stdout)
        self.assertIn('model (no model is named language-q4)', result.stdout)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReplayStatus)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f'replay status: {TOTAL} controls, {len(outcome.failures)} failures, {len(outcome.errors)} errors', flush=True)
    sys.exit(0 if outcome.wasSuccessful() and TOTAL == 270 else 1)
