#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check review controls and complete-file recovery with fixed prepared evidence.
# Inputs: Build, source, model store, new output and batch. Outputs: Commands and checks.
# Exit: 0 pass, 1 failed check or cleanup, 2 bad arguments.
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import sys
import time
from aiohttp import ClientSession
from live_boot_test import wait
from text_boot_test import setup
from runtime_boot_test import RuntimeTest, RuntimeRun, durable
from gateway_runtime_test import aotx_http_run, aotx_http

sys.dont_write_bytecode = True


def review_status(run):
    start = len(run.console()); run.send('policy status')
    pattern = r'policy: review (on|off) active (\d+) pending (\d+) completed (\d+) interrupted (\d+) status (\d+) control revision (\d+)'
    def received():
        match = re.search(pattern, run.console()[start:])
        if match:
            return dict(enabled=match[1] == 'on', **dict(zip(('active', 'pending', 'completed', 'interrupted', 'status', 'revision'),
                                                         map(int, match.groups()[1:]))))
        return None
    return wait(received, run.child, 60)


async def controls(test, run, config, keys, count, recovered, previous=None):
    gateway = aotx_http_run(test, config, 'after-http' if recovered else 'before-http')
    cfg = json.loads(config.read_text()); url = 'http://127.0.0.1:'+str(cfg['port'])+'/aotx/v1/policy'
    headers = [{'Authorization': 'Bearer '+key} for key in keys]
    try:
        async with ClientSession() as client:
            end = time.monotonic()+30
            while True:
                try:
                    async with client.get(url, headers=headers[0]) as response:
                        if response.status == 200: state = await response.json(); break
                except OSError: pass
                if gateway.child.poll() is not None or time.monotonic() >= end: raise TimeoutError('Policy HTTP startup failed.')
                await asyncio.sleep(0.1)
            test.check(state['abi'] == 3 and state['completed'] == (count if recovered else 0),
                       'native policy reports the expected complete source batch', state=state)
            if recovered:
                test.check(state['control_revision'] == previous['control_revision'] and state['source_frontier'] == previous['source_frontier'],
                           'copied-file activation restores the exact control revision and source frontier')
                await aotx_http(test, client, 'POST', url, headers[0], 410,
                    json=dict(action='review_off', epoch=previous['epoch'], control_revision=previous['control_revision']))
            else:
                test.check(not state['review_enabled'], 'complete runtime starts with review disabled')
                run.send('policy review on')
                state = await asyncio.to_thread(wait, lambda: (s if (s := review_status(run))['completed'] == count else None), run.child, 120)
                test.check(state['enabled'] and not state['active'] and not state['status'], 'local console completes the fixed evidence batch', state=state)
                state, _ = await aotx_http(test, client, 'GET', url, headers[0])
            await asyncio.to_thread(durable, run)
            body = dict(action='pause', epoch=state['epoch'], control_revision=state['control_revision'])
            await aotx_http(test, client, 'POST', url, headers[1], 403, json=body)
            await aotx_http(test, client, 'POST', url, headers[2], 403, json=body)
            paused, _ = await aotx_http(test, client, 'POST', url, headers[0], json=body)
            test.check(paused['state'] == 'paused', 'native operator control pauses the same resident policy')
            await asyncio.to_thread(durable, run)
            await aotx_http(test, client, 'POST', url, headers[0], 409, json=body)
            resumed, _ = await aotx_http(test, client, 'POST', url, headers[0],
                json=dict(action='resume', epoch=paused['epoch'], control_revision=paused['control_revision']))
            test.check(resumed['completed'] == count and not resumed['active_rows'], 'native resume preserves completed work')
            telemetry, _ = await aotx_http(test, client, 'GET', url, headers[1])
            test.check('sources' not in telemetry and telemetry['completed'] == count, 'read-only status has aggregate counters without source text')
            await aotx_http(test, client, 'GET', url, headers[2], 403)
            return resumed
    finally:
        gateway.close()


def main():
    p = argparse.ArgumentParser()
    for name in ('build', 'source', 'store', 'output'): p.add_argument(name, type=Path)
    p.add_argument('count', type=int, choices=(1, 64)); args = p.parse_args()
    test = RuntimeTest(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve())
    for name in ('aotx_boot', 'aotx_runtime_offline', 'aotx_drain', 'aotx_feed', 'aotx_restore',
                 'aotx_service', 'aotx_reflection_runtime_fixture', 'aotx_recall_cli_fixture',
                 'aotx_policy_pack', 'aotx_ccir_pack'):
        test.check((test.build/name).is_file(), name+' is built before packaging')
    setup(test)
    entries = [json.loads(line) for line in (test.store/'manifest.jsonl').read_text().splitlines() if line.strip()]
    model = next(row['sha256'] for row in entries if row['role'] == 'language')
    checkpoint, memory = test.output/'inputs/checkpoint', test.output/'inputs/memory.aotxccir'
    policy, runtime = test.output/'policy.bin', test.output/'state.aotxccir'
    (test.output/'provenance').write_text('Supplied bounded task review rules.\n')
    test.command([test.build/'aotx_reflection_runtime_fixture', checkpoint, args.count, model], 'prepared-evidence')
    test.command([test.build/'aotx_recall_cli_fixture', checkpoint, '-', memory, args.count], 'memory-file')
    test.command([test.build/'aotx_policy_pack', '--output', policy, '--mode', 'rules', '--abi', 3,
                  '--provenance', test.output/'provenance', '--license', test.source/'LICENSE'], 'review-policy')
    command = [test.build/'aotx_ccir_pack', '--memory', memory, '--models', test.store, '--modules', test.output/'modules',
               '--settings', test.output/'settings', '--roles', 'language,embedding', '--policy', policy, '--output', runtime]
    if 'AOTX_AFFECT:BOOL=ON' in (test.build/'CMakeCache.txt').read_text():
        command += ['--phrases', test.source/'tests/fixtures/quality/refusal-phrases.txt']
    test.command(command, 'complete-runtime')
    keys = ['review-test-'+hashlib.sha256(str(i).encode()).hexdigest() for i in range(3)]
    with socket.socket() as port: port.bind(('127.0.0.1', 0)); selected = port.getsockname()[1]
    cfg = {'socket': str(test.output/'before-journal/service.sock'), 'port': selected, 'revision': '1',
           'models': {'text': {'role': 'language', 'published_at': 1}}, 'principals': [
               {'id': '%032x' % (i+1), 'token_sha256': [hashlib.sha256(key.encode()).hexdigest()], 'models': ['text'],
                'actions': [actions], 'requests': 1, 'tokens': 32, 'pages': 0}
               for i, (key, actions) in enumerate(zip(keys, ('policy_manage', 'telemetry', 'infer')))]}
    config = test.output/'gateway.json'; config.write_text(json.dumps(cfg)); config.chmod(0o600)
    grants = test.output/'grants'
    test.command(['env', 'PYTHONPATH='+str(test.source), sys.executable, '-m', 'gateway', 'grants',
                  '--config', config, '--output', grants], 'operator-grants')
    run = None
    try:
        run = RuntimeRun(test, 'before', runtime, ('--service-grants', grants)); run.ready(seconds=900)
        before = asyncio.run(controls(test, run, config, keys, args.count, False))
        durable(run); run.stop(); run = None
        copied = test.output/'copied.aotxccir'; shutil.copy2(runtime, copied)
        # Complete-file activation has no source policy, settings or fixture dependency.
        for path in (policy, checkpoint, memory, test.output/'settings'): path.unlink()
        shutil.rmtree(test.output/'modules')
        cfg['socket'] = str(test.output/'after-journal/service.sock'); config.write_text(json.dumps(cfg))
        run = RuntimeRun(test, 'after', copied, ('--service-grants', grants)); run.ready(seconds=900)
        after = asyncio.run(controls(test, run, config, keys, args.count, True, before))
        durable(run); run.stop(); run = None
        test.check(before['completed'] == after['completed'] == args.count and not after['active_rows'],
                   'continued native use creates no repeated completed review')
        (test.output/'result.json').write_text(json.dumps(dict(before=before, after=after, count=args.count), indent=2)+'\n')
    finally:
        if run and not run.log.closed: run.close()
        test.flush_checks()
    return 0


if __name__ == '__main__': raise SystemExit(main())
