#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive real boot/feed/journal turns for architecture checks 5 and 6.
Inputs: existing build, store, model name, expected descriptor load line, new output path.
Outputs: terminal logs, commands, summaries, authoritative tokens and comparison JSON.
Exit bits: 1 restore/say failure, 2 console load-line failure, 4 setup failure.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

command_log = None
QUESTIONS = [
    'Remember this: my favorite color is blue. Reply with just the color.',
    'What is my favorite color? Reply with just the color.',
    'Name my favorite color and explain in one sentence how you know it.',
]

def record(text):
    with command_log.open('a') as log:
        log.write(text + '\n')

def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))

def wait(predicate, child=None, seconds=360):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        if child is not None and child.poll() is not None:
            raise RuntimeError('boot exited before the required state: ' + str(child.returncode))
        time.sleep(0.1)
    raise TimeoutError('required durable state did not arrive')

def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result

class Run:
    def __init__(self, args, root, journal, restore=False):
        self.args, self.root, self.journal = args, root, journal
        root.mkdir(parents=True, exist_ok=False)
        self.log_path = root / 'boot.log'
        self.log = self.log_path.open('w')
        argv = ['stdbuf', '-oL', '-eL', args.build / 'aotx_boot', '--models', args.selected_store,
                '--roles', args.role, '--settings', args.settings, '--modules', args.modules,
                '--journal', journal, '--ticks', '0']
        if restore and not args.omit_restore:
            argv.append('--restore')
        record(shlex.join(map(str, argv)))
        self.started_ns = time.time_ns()
        self.child = subprocess.Popen(list(map(str, argv)), cwd=args.build.parent,
                                      stdin=subprocess.PIPE, stdout=self.log,
                                      stderr=subprocess.STDOUT, text=True)
        record('boot pid=' + str(self.child.pid))
        self.boot = None

    def ready(self):
        phase = self.journal / 'phase'
        def running():
            current = re.search(r'^boot: id ([0-9a-f]+)', self.log_path.read_text(), re.M)
            return (current and phase.exists() and phase.stat().st_mtime_ns >= self.started_ns
                    and phase.read_text().startswith('running '))
        wait(running, self.child)

    def summary(self, label):
        text = self.command(['aotx_restore', '--journal', self.journal, '--summary'], label)
        summary = fields(next(line for line in text.splitlines() if line.startswith('restore boot=')))
        assert int(summary['last_tick']) > 0 and int(summary['replayed']) > 0
        return summary

    def command(self, args, label):
        argv = [self.args.build / args[0]] + list(args[1:])
        record(shlex.join(map(str, argv)))
        done = subprocess.run(list(map(str, argv)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, timeout=60)
        (self.root / (label + '.txt')).write_text(done.stdout)
        record('exit=' + str(done.returncode) + '; output=' + str(self.root / (label + '.txt')))
        assert done.returncode == 0, str(argv[0]) + ' failed'
        return done.stdout

    def send(self, text):
        record('pid=' + str(self.child.pid) + ' stdin: ' + text)
        self.child.stdin.write(text + '\n')
        self.child.stdin.flush()

    def turns(self):
        if self.boot:
            return rows(self.journal / 'manifest' / (self.boot + '.jsonl'))
        for path in (self.journal / 'manifest').glob('*.jsonl'):
            found = rows(path)
            if found:
                self.boot = path.stem
                return found
        return []

    def turn(self, index):
        self.send('say ' + QUESTIONS[index - 1])
        wait(lambda: any(row.get('agent') == 0 and row.get('turn') == index for row in self.turns()), self.child)
        reply = wait(lambda: next((row for row in rows(self.journal / self.boot / 'transcript/0.jsonl')
                                  if row.get('kind') == 'reply' and row.get('turn') == index), None), self.child)
        assert reply['text'].strip(), 'empty reply'
        assert not any(marker in reply['text'] for marker in ('<|', '|>', '<think>')), 'wrap text in reply'
        print('User: ' + QUESTIONS[index - 1], flush=True)
        print('Assistant: ' + reply['text'], flush=True)
        return reply

    def tokens(self, label, summary):
        text = self.command(['aotx_journal', 'tokens', self.journal, '--boot', summary['boot']], label)
        result = [fields(line) for line in text.splitlines() if line.startswith('slot=')]
        return [row for row in result if int(row['tick']) <= int(summary['last_tick'])]

    def stop(self, killed=False):
        if self.child.poll() is not None:
            raise RuntimeError('boot stopped before the requested stop')
        children_file = Path('/proc') / str(self.child.pid) / 'task' / str(self.child.pid) / 'children'
        descendants = children_file.read_text().split() if children_file.exists() else []
        if killed:
            record('kill -9 ' + str(self.child.pid))
            self.child.kill()
            self.child.stdin.close()
        else:
            self.send('quit')
        status = self.child.wait(timeout=60)
        record('boot pid=' + str(self.child.pid) + ' exit=' + str(status))
        assert status == (-signal.SIGKILL if killed else 0)
        def drained():
            for pid in descendants:
                stat = Path('/proc') / pid / 'stat'
                try:
                    state = stat.read_text().rsplit(')', 1)[1].strip().split()[0]
                    if state != 'Z':
                        return False
                except FileNotFoundError:
                    pass
            return True
        wait(drained, seconds=60)
        self.log.close()

    def close(self):
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=30)
        self.log.close()

def sampled_turns(tokens, replayed):
    groups = []
    for row in tokens:
        if row['slot'] != '0' or int(row['replayed']) != replayed:
            continue
        if row['position'] == '0':
            groups.append([])
        assert groups, 'token run has no sequence start'
        if row['sampled'] == '1':
            groups[-1].append(int(row['token']))
    return groups

def compare(left, right, label):
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            raise AssertionError(label + ' first difference at ' + str(index) + ': ' + repr(a) + ' != ' + repr(b))

def check_restore_log(text, expected, expected_decode_refused=0):
    pattern = (r'^restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+) '
               r'pages (\d+) paced (\d+) rejected (\d+)$')
    matches = re.findall(pattern, text, re.M)
    assert len(matches) == 1, 'the process has no unique complete restore report'
    applied, state_hash, decode_refused, pages, paced, rejected = matches[0]
    assert int(applied) == int(expected['replayed']) > 0, 'the applied record count differs'
    assert int(state_hash, 16) == int(expected['state_hash'], 16), 'the applied restore hash differs'
    assert int(rejected) == 0, 'restore rejected inbound records'
    assert int(decode_refused) == expected_decode_refused, 'restore decode refusals differ from the expected count'
    return {'applied': int(applied), 'state_hash': state_hash, 'decode_refused': int(decode_refused),
            'pages': int(pages), 'paced': int(paced), 'rejected': int(rejected)}

def check_load_lines(active, expected):
    good = bool(active)
    for run in active:
        try:
            matches = [line for line in run.log_path.read_text().splitlines() if line == expected]
        except OSError:
            matches = []
        good &= bool(matches)
        print('console load line ' + str(run.log_path) + ': ' + (matches[0] if matches else 'MISSING'), flush=True)
    print('console load check: ' + str(len(active)) + ' processes observed', flush=True)
    return good

def main():
    global command_log
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--commands', type=Path, default=os.environ.get('AOTX_TEST_COMMANDS'))
    parser.add_argument('--name', required=True)
    parser.add_argument('--load-line', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--omit-restore', action='store_true', help='negative control: omit the restore option')
    args = parser.parse_args()
    args.build, args.store = args.build.resolve(), args.store.resolve()
    root = args.out.resolve() / args.name
    root.mkdir(parents=True, exist_ok=False)
    command_log = args.commands.resolve() if args.commands else root / 'commands.txt'
    entry = next(row for row in map(json.loads, (args.store / 'manifest.jsonl').read_text().splitlines()) if row['name'] == args.name)
    model_path = (args.store / entry['path']).resolve()
    args.role = entry['role']
    args.selected_store = root / 'store'
    args.selected_store.mkdir()
    (args.selected_store / model_path.name).symlink_to(model_path)
    entry['path'] = model_path.name
    (args.selected_store / 'manifest.jsonl').write_text(json.dumps(entry, separators=(',', ':')) + '\n')
    (root / 'inputs.json').write_text(json.dumps({'entry': entry, 'prompts': QUESTIONS, 'load_line': args.load_line}, indent=2) + '\n')
    args.settings = root / 'settings'
    settings = 'sample.temperature = 0\nsample.seed = 7\ndecode.reply_limit = 32\nderive.list = transcript,tokens,pages\n'
    cache = (args.build / 'CMakeCache.txt').read_text()
    if re.search(r'^AOTX_AFFECT:BOOL=ON$', cache, re.M):
        settings += 'affect.on = 0\nquality.on = 0\n'
    args.settings.write_text(settings)
    args.modules = root / 'modules'
    shutil.copytree(args.build.parent / 'modules/roles', args.modules)
    active = []
    failure = None
    try:
        baseline = Run(args, root / 'baseline', root / 'journal-baseline')
        active.append(baseline)
        baseline.ready()
        replies = [baseline.turn(i) for i in (1, 2, 3)]
        baseline.stop()
        base_summary = baseline.summary('summary')
        base_tokens = baseline.tokens('tokens', base_summary)
        base_turns = baseline.turns()
        baseline.command(['aotx_journal', 'manifest', baseline.journal], 'manifest-check')
        before = Run(args, root / 'before', root / 'journal-restored')
        active.append(before)
        before.ready()
        before.turn(1)
        before.turn(2)
        committed = before.summary('before-kill-summary')
        historical = before.tokens('before-kill-tokens', committed)
        assert len(sampled_turns(historical, 0)) == 2, 'two turns are not durable'
        assert all(sampled_turns(historical, 0)), 'a durable turn has no sampled tokens'
        before.stop(killed=True)
        old = before.summary('summary')
        old_turns = before.turns()
        assert len(old_turns) == 2
        old_tokens = before.tokens('tokens', old)
        after = Run(args, root / 'after', before.journal, restore=True)
        active.append(after)
        after.ready()
        check_restore_log(after.log_path.read_text(), old)
        current = after.summary('after-replay-summary')
        assert current['boot'] != old['boot'] and current.get('restore_of') == old['boot']
        assert current.get('restore_hash') == old['state_hash']
        after.boot = current['boot']
        wait(lambda: len(after.turns()) == 2, after.child)
        compare(old_turns, after.turns(), 'historical manifests')
        replies.append(after.turn(3))
        after.stop()
        new = after.summary('summary')
        new_tokens = after.tokens('tokens', new)
        after.command(['aotx_journal', 'manifest', after.journal], 'manifest-check')
        replayed = [row for row in new_tokens if row['replayed'] == '1']
        keys = lambda data: [(row['slot'], row['position'], row['token'], row['flags']) for row in data]
        compare(keys(old_tokens), keys(replayed), 'replayed token prefix')
        base_samples = sampled_turns(base_tokens, 0)
        new_samples = sampled_turns(new_tokens, 0)
        assert len(base_samples) == 3 and len(new_samples) == 1 and base_samples[2] and new_samples[0]
        compare(base_samples[2], new_samples[0], 'new third-turn sampled IDs')
        fields_to_compare = ('agent', 'turn', 'input_hash', 'output_hash', 'tokens', 'finish', 'tool', 'request')
        compare([base_turns[2][key] for key in fields_to_compare],
                [after.turns()[2][key] for key in fields_to_compare], 'third-turn completion fields')
        (root / 'comparison.json').write_text(json.dumps({'baseline': base_samples[2], 'restored': new_samples[0],
                                                        'completion': after.turns()[2], 'replies': replies}, indent=2) + '\n')
    except Exception as error:
        failure = str(error)
        (root / 'first-difference.txt').write_text(failure + '\n')
        print('process check failed: ' + failure, flush=True)
    finally:
        for run in active:
            run.close()
    # A failed reply does not change a layer line already read from a process.
    load_ok = check_load_lines(active, args.load_line)
    return (1 if failure else 0) | (0 if load_ok else 2)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print('process setup failed: ' + str(error), file=sys.stderr)
        sys.exit(4)
