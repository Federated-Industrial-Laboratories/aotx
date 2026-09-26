#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check visible runtime failure when an essential disk child ends during a run or shutdown.
# Inputs: Build path, new output path and optional case, count or window mode. Outputs: Logs and results. Exit: 0 pass, 1 failure.
import argparse
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time
from live_boot_test import children_of, children_exited, wait

CASES = {'writer': 'aotx_drain', 'feeder': 'aotx_feed', 'service': 'aotx_service',
         'shutdown': 'aotx_drain', 'clean': None}
LABELS = {'aotx_drain': 'disk writer', 'aotx_feed': 'feeder', 'aotx_service': 'service'}


def workload(child, path, count):
    # Slot zero is the console agent. The other slots hold distinct worker agents.
    lines = ['spawn worker ' + str(min(8, count - i)) for i in range(1, count, 8)]
    child.stdin.write(('\n'.join(lines + ['agents']) + '\n').encode()); child.stdin.flush()

    def admitted():
        text = path.read_text()
        found = [(int(i), role) for i, role in re.findall(r'^  (\d+) (conductor|worker) idle ', text, re.M)]
        expected = [(i, 'worker' if i else 'conductor') for i in range(count)]
        return found if found == expected else None
    return wait(admitted, child, 10)


def summary(build, journal, log_path):
    result = subprocess.run([str(build / 'aotx_restore'), '--journal', str(journal), '--summary'],
                            capture_output=True, text=True, timeout=15)
    text = result.stdout + result.stderr
    log_path.write_text(text)
    match = re.search(r'state_hash=([0-9a-f]+)', text)
    return result.returncode == 0 and match is not None, int(match[1], 16) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('build', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--window', action='store_true')
    parser.add_argument('--count', type=int, choices=(1, 64))
    parser.add_argument('--case', choices=tuple(CASES))
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = output / 'settings'; settings.write_text('derive.list = console,bus\n')
    grants = bytearray(192); grants[:8] = b'AOTXAPI1'
    struct.pack_into('<I', grants, 8, 1); struct.pack_into('<Q', grants, 32, 1)
    struct.pack_into('<I', grants, 76, 1); struct.pack_into('<I', grants, 88, 64)
    struct.pack_into('<16sQIIIIIIQ8x', grants, 128, bytes.fromhex('01'*16), 1, 8, 0, 1, 8, 1, 0, 0)
    grant_path = output / 'grants'; grant_path.write_bytes(grants); grant_path.chmod(0o600)
    results = []
    counts = (args.count,) if args.count else (1, 64)
    cases = (args.case,) if args.case else tuple(CASES)
    try:
        for count in counts:
            for case in cases:
                name = CASES[case]; label = str(count) + '-' + case
                journal = output / label
                argv = ['stdbuf', '-oL', '-eL', str(build / 'aotx_boot'), '--journal', str(journal),
                        '--settings', str(settings), '--service-grants', str(grant_path), '--ticks', '0']
                if args.window: argv.append('--window')
                log_path = output / (label + '.log')
                with log_path.open('w') as log:
                    child = subprocess.Popen(argv, cwd=output, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
                    owned = {}
                    try:
                        phase = journal / 'phase'
                        wait(lambda: phase.exists() and phase.read_text().startswith('running '), child, 120)
                        owned = children_of(child.pid)
                        targets = [pid for pid in owned if (Path('/proc') / str(pid) / 'comm').read_text().strip() == name]
                        if name and len(targets) != 1: raise AssertionError('the run has exactly one selected essential child')
                        admitted = workload(child, log_path, count)
                        if case == 'shutdown':
                            os.kill(targets[0], signal.SIGSTOP)
                            child.stdin.write(b'quit\n'); child.stdin.flush()
                            others = {pid: identity for pid, identity in owned.items() if pid != targets[0]}
                            wait(lambda: children_exited(others), child, 10)
                        start = time.monotonic()
                        if case == 'clean': child.stdin.write(b'quit\n'); child.stdin.flush()
                        else: os.kill(targets[0], signal.SIGKILL)
                        status = child.wait(timeout=15); elapsed = time.monotonic() - start
                        text = log_path.read_text()
                        readable, saved_hash = summary(build, journal, output / (label + '-restore.log'))
                        final_hash = re.search(r'ticks \d+ .* hash ([0-9a-f]+)', text)
                        reason = ('boot: the ' + LABELS[name] + ' ended during ' +
                                  ('shutdown' if case == 'shutdown' else 'the run')) if name else ''
                        checks = {
                            'exact_admitted_batch': len(admitted) == count,
                            'expected_exit': status == (0 if case == 'clean' else 1),
                            'visible_reason': reason in text if name else 'ended during' not in text,
                            'owned_children_stopped': children_exited(owned),
                            'phase_closed': phase.read_text().startswith('closed '),
                            'bounded_exit': elapsed < 15,
                            'saved_prefix_reads': readable,
                        }
                        if case == 'clean':
                            checks['exact_saved_stop'] = bool(final_hash) and int(final_hash[1], 16) == saved_hash
                        results.append(dict(argv=argv, case=case, count=count, agents=admitted,
                                            exit=status, seconds=elapsed, checks=checks))
                        if not all(checks.values()): raise AssertionError('runtime failure checks failed')
                    finally:
                        if child.poll() is None: child.kill(); child.wait(timeout=15)
                        if child.stdin: child.stdin.close()
                        for pid, identity in owned.items():
                            if not children_exited({pid: identity}):
                                try: os.kill(pid, signal.SIGKILL)
                                except ProcessLookupError: pass
    finally:
        (output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
    if len(results) != len(counts) * len(cases): raise AssertionError('the requested case count is incomplete')
    print(json.dumps({'cases': len(results), 'checks': sum(len(r['checks']) for r in results), 'failures': 0}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
