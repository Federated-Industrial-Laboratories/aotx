#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check caller deadlines with distinct saved operations and delayed responses.
# Inputs: None. Outputs: Check counts. Exit: 0 pass, 1 failed assertion.
import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import patch
import shared_runtime_client as module


class Checks:
    def __init__(self):
        self.count = 0

    def check(self, value, label, **details):
        self.count += 1
        if not value:
            raise AssertionError(label)


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    async def sleep(self, seconds):
        self.value += seconds


class Response:
    def __init__(self, status, value):
        self.status, self.value = status, value

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        return self.value


class HTTP:
    def __init__(self, clock, identity, mode):
        self.clock, self.identity, self.mode = clock, identity, mode
        self.calls = 0
        self.output = ('Saved response ' + identity).encode()

    def request(self, method, path, **fields):
        self.calls += 1
        self.clock.value += 300
        if self.mode == 'busy' or self.mode == 'retry' and self.calls < 4:
            return Response(429, {})
        done = self.calls >= 4 and self.mode != 'pending'
        raw = self.output if done else b''
        value = dict(id=self.identity, offset='0', next_offset=str(len(raw)),
                     output=dict(base64=base64.b64encode(raw).decode()), output_bytes=str(len(raw)),
                     state='completed' if done else 'running', saved_terminal=done,
                     saved_admission=True, save=dict(generation='1', boot='1', commit_sha256='01' * 32))
        return Response(200, value)


async def case(checks, identity, mode, seconds, passes, terminal=True):
    clock = Clock(); http = HTTP(clock, identity, mode)
    client = module.aotx_shared_client(checks, http, 'http://local', 'key', identity, seconds)
    with patch.object(module, 'time', SimpleNamespace(monotonic=clock.monotonic)), \
         patch.object(module, 'asyncio', SimpleNamespace(sleep=clock.sleep)):
        try:
            result = await client.terminal(identity) if terminal else await client.request('GET', '/value')
        except (AssertionError, TimeoutError):
            checks.check(not passes, 'a bounded pending operation stops')
        else:
            checks.check(passes, 'a delayed operation completes only within its allowance')
            if terminal:
                checks.check(base64.b64decode(result['exact_output']) == http.output,
                             'the completed operation retains its distinct response')
    checks.check(http.calls == (4 if passes else 3), 'the caller allowance controls the exact attempt count')


async def main():
    checks = Checks()
    for count in (1, 64):
        before = checks.count
        for i in range(count):
            identity = str(count) + '-' + str(i)
            for mode, seconds, passes, terminal in (
                    ('done', 1500, True, True), ('done', 900, False, True),
                    ('pending', 900, False, True), ('retry', 1500, True, False),
                    ('retry', 900, False, False), ('busy', 900, False, True)):
                await case(checks, identity, mode, seconds, passes, terminal)
        print(f'Shared deadlines N={count}: {checks.count - before} checks')
    default = module.aotx_shared_client(checks, None, '', '', '')
    checks.check(default.work_seconds == 900, 'existing callers retain their default allowance')
    for value in (0, -1, float('nan'), float('inf')):
        try:
            module.aotx_shared_client(checks, None, '', '', '', value)
        except ValueError:
            checks.check(True, 'invalid caller allowances are refused')
        else:
            raise AssertionError('An invalid caller allowance was accepted.')
    print(f'Shared deadlines: {checks.count} checks, 0 failures')


if __name__ == '__main__':
    asyncio.run(main())
