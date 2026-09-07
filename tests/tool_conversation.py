#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive four operator inputs through the attach socket of a real boot.
Inputs: build, store, model name, embedding store, and a new output directory.
Outputs: prompts, replies, tool results, elapsed seconds, and the journal.
Exit: 0 for four completed inputs, 1 for a failed input or an unanswered tool.
"""
import argparse
import array
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import time

import arch_process as process
import tool_restore

QUESTIONS = [
    'Hello!',
    'What is the capital of France?',
    'Please save this in memory: my garden gate code is 4827.',
    'What is my garden gate code? Please recall it from memory.',
]


class Conversation(process.Run):
    def ready(self):
        super().ready()
        self.boot = re.search(r'^boot: id ([0-9a-f]+)', self.log_path.read_text(), re.M)[1]
        self.console = self.journal / self.boot / 'console.log'
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(30)
        directory = os.open(self.journal, os.O_RDONLY | os.O_DIRECTORY)
        try:
            process.wait(lambda: (self.journal / 'aotx.sock').exists(), self.child)
            self.connection.connect('/proc/self/fd/' + str(directory) + '/aotx.sock')
        finally:
            os.close(directory)
        payload, controls, _, _ = self.connection.recvmsg(1, socket.CMSG_SPACE(4))
        assert payload == b'M', 'the attach socket gave no mirror'
        for level, kind, value in controls:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                descriptors = array.array('i')
                descriptors.frombytes(value[:len(value) - len(value) % descriptors.itemsize])
                for descriptor in descriptors:
                    os.close(descriptor)
        self.refused_requests = set()
        self.await_idle(0)

    def send(self, text):
        process.record('attach line: ' + text)
        body = text.encode()
        self.connection.sendall(b'L' + struct.pack('<I', len(body)) + body)

    def await_idle(self, least_turn):
        deadline = time.monotonic() + self.args.seconds
        while time.monotonic() < deadline:
            before = len(self.console.read_text()) if self.console.exists() else 0
            self.send('agents')
            def state():
                text = self.console.read_text()[before:] if self.console.exists() else ''
                return re.search(r'^\s*0 conductor (\w+) \S+ (\S+) (\S+) (\d+) \d+', text, re.M)
            row = process.wait(state, self.child, seconds=30)
            if row[1] == 'idle' and int(row[4]) >= least_turn:
                return int(row[4])
            if row[1] == 'tool' and row[3].isdigit():
                request = int(row[3])
                events = process.rows(self.journal / self.boot / 'transcript/0.jsonl')
                waiting = any(event.get('kind') == 'call' and event.get('request') == request
                              and event.get('status') == 'waiting' for event in events)
                if waiting and request not in self.refused_requests:
                    self.send('refuse ' + str(request))
                    self.refused_requests.add(request)
            time.sleep(0.5)
        raise TimeoutError('the agent did not become idle before the input deadline')

    def exchange(self, question):
        before = max((int(row['turn']) for row in self.turns()), default=0)
        console_start = len(self.console.read_text()) if self.console.exists() else 0
        started = time.monotonic()
        self.send('say ' + question)
        last = self.await_idle(before + 1)
        elapsed = time.monotonic() - started
        transcript = self.journal / self.boot / 'transcript/0.jsonl'
        process.wait(lambda: any(row.get('kind') == 'reply' and row.get('turn') == last
                                for row in process.rows(transcript)), self.child, seconds=30)
        records = [row for row in process.rows(transcript)
                   if before < row.get('turn', 0) <= last and row.get('kind') != 'part']
        calls = [row for row in records if row.get('kind') == 'call']
        results = [row for row in records if row.get('kind') == 'result']
        unanswered = [row for row in calls if not any(answer['request'] == row['request']
                                                     for answer in results)]
        value = {'input': question, 'seconds': elapsed, 'first_turn': before + 1,
                 'last_turn': last, 'records': records, 'unanswered': unanswered,
                 'console': self.console.read_text()[console_start:]}
        print(json.dumps(value, ensure_ascii=True), flush=True)
        return value

    def close(self):
        if hasattr(self, 'connection'):
            self.connection.close()
        super().close()


def select(store, name=None, role=None):
    entries = process.rows(store / 'manifest.jsonl')
    chosen = [entry for entry in entries
              if (name is None or entry['name'] == name)
              and (role is None or entry['role'] == role)]
    assert len(chosen) == 1, 'the model selection is not unique'
    entry = chosen[0].copy()
    model = (store / entry['path']).resolve(strict=True)
    return entry, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--embedding-store', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commands', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=360)
    parser.add_argument('--questions', type=Path)
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument('--restore-after-write', action='store_true',
                        help='kill after the memory write, restore, then check actual recall')
    restore.add_argument('--restore-after-refusal', action='store_true',
                        help='restore a refused request, raise the page limit to 64, then converse')
    args = parser.parse_args()
    args.build = args.build.resolve()
    args.store = args.store.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    process.command_log = args.commands.resolve()
    entry, model = select(args.store, name=args.name)
    assert entry['role'] in ('language', 'language-q4'), 'the selected role cannot hold a conversation'
    chosen = [(entry, model)]
    if args.embedding_store:
        chosen.append(select(args.embedding_store.resolve(), role='embedding'))
    args.selected_store = args.out / 'store'
    args.selected_store.mkdir()
    manifest = []
    for row, path in chosen:
        row['path'] = path.name
        (args.selected_store / path.name).symlink_to(path)
        manifest.append(row)
    (args.selected_store / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in manifest))
    args.role = ','.join(row['role'] for row in manifest)
    args.settings = args.out / 'settings'
    settings = ('sample.temperature = 0\nsample.seed = 7\ndecode.reply_limit = 256\n'
                'derive.list = console,bus,requests,transcript,tokens,pages\n')
    if re.search(r'^AOTX_AFFECT:BOOL=ON$', (args.build / 'CMakeCache.txt').read_text(), re.M):
        settings += 'affect.on = 0\nquality.on = 0\n'
    args.settings.write_text(settings)
    args.modules = args.out / 'modules'
    shutil.copytree(args.build.parent / 'modules/roles', args.modules)
    questions = json.loads(args.questions.read_text()) if args.questions else QUESTIONS
    assert not args.restore_after_write or args.questions is None, 'restore uses the four default inputs'
    assert len(questions) >= 4 and all(isinstance(text, str) and text for text in questions)
    (args.out / 'inputs.json').write_text(json.dumps({'models': manifest, 'questions': questions}, indent=2) + '\n')
    run = Conversation(args, args.out / 'boot', args.out / 'journal')
    report = {'name': args.name, 'models': manifest, 'turns': []}
    try:
        run.ready()
        if args.restore_after_refusal:
            run = tool_restore.restart_refusal(args, run, report)
        for index, question in enumerate(questions):
            report['turns'].append(run.exchange(question))
            (args.out / 'conversation.json').write_text(json.dumps(report, indent=2) + '\n')
            if args.restore_after_write and index == 2:
                run = tool_restore.restart(args, run, report)
        if args.restore_after_write:
            tool_restore.check_recall(report)
        run.stop()
        report['boot_exit'] = run.child.returncode
    except Exception as error:
        report['error'] = str(error)
        print('conversation failed: ' + str(error), flush=True)
    finally:
        run.close()
        (args.out / 'conversation.json').write_text(json.dumps(report, indent=2) + '\n')
    return int('error' in report or any(turn['unanswered'] for turn in report['turns']))


if __name__ == '__main__':
    raise SystemExit(main())
