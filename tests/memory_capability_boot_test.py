#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check model memory availability, shared source retention and complete file recovery through HTTP.
# Inputs: Build, source, model store, new output path and batch count. Outputs: Commands and checks.
# Exit: 0 pass, 1 failed check or cleanup, 2 bad arguments.
import argparse
import asyncio
import base64
import json
from pathlib import Path
import re
import shutil
import sys
import time
from aiohttp import ClientSession, ClientTimeout
from gateway_runtime_test import aotx_http_run, aotx_ready, aotx_http
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections
from shared_runtime_client import aotx_shared_client
from shared_runtime_cases import aotx_shared_memory_object
from shared_runtime_test import aotx_configuration
from text_boot_test import setup


def aotx_prepare(test, wrapper_mismatch):
    f = setup(test)
    inputs = test.output / 'inputs'
    checkpoint, memory = inputs / 'checkpoint', inputs / 'memory.aotxccir'
    checkpoint.write_bytes(f.image([], 0, 10))
    test.command([test.build / 'aotx_recall_cli_fixture', checkpoint, '-', memory, 1], 'empty-memory')
    store = test.output / 'selected-models'; store.mkdir()
    entries = [json.loads(line) for line in (test.store / 'manifest.jsonl').read_text().splitlines()]
    for entry in entries:
        original = (test.store / entry['path']).resolve()
        local = store / original.name; local.symlink_to(original); entry['path'] = local.name
        if wrapper_mismatch and entry['role'] == 'language':
            # Keep the header end separate from the following text tokens.
            entry['wrap']['system_head'] = '\n' + entry['wrap']['system_head']
    (store / 'manifest.jsonl').write_text(''.join(json.dumps(entry) + '\n' for entry in entries))
    runtime = test.output / 'state.aotxccir'
    test.command([test.build / 'aotx_ccir_pack', '--memory', memory, '--models', store,
        '--modules', test.output / 'modules', '--settings', test.output / 'settings',
        '--roles', 'language,embedding', '--output', runtime,
        '--phrases', test.source / 'tests/fixtures/quality/refusal-phrases.txt'], 'pack-runtime')
    test.command([test.build / 'aotx_ccir_pack', '--runtime', runtime, '--shared'], 'enable-shared')
    shutil.rmtree(store); shutil.rmtree(inputs); shutil.rmtree(test.output / 'modules')
    (test.output / 'settings').unlink()
    return f, runtime


async def aotx_surface(test, cfg, path, keys, count, qualified, label, saved=None):
    run = test.active[-1]
    test.check(re.search(r'wrap: .+ spans=pass ends=pass prefill=pass .+ usable=yes',
        run.path.read_text()) is not None, 'runtime has a usable language wrapper')
    gateway = aotx_http_run(test, path, label)
    url = 'http://127.0.0.1:' + str(cfg['port'])
    try:
        async with ClientSession(timeout=ClientTimeout(total=960)) as http:
            clients = [aotx_shared_client(test, http, url, keys[i], '%032x' % (i+1)) for i in range(count)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            capabilities, _ = await aotx_http(test, http, 'GET', url + '/aotx/v1/capabilities', clients[0].headers)
            models, _ = await aotx_http(test, http, 'GET', url + '/v1/models', clients[0].headers)
            for rows in (capabilities['models'], models['data']):
                text = [row for row in rows if row['id'] == 'text']
                test.check(len(text) == 1 and text[0]['automatic_memory'] is qualified,
                    'both discovery routes expose exact automatic memory availability', rows=rows)
            for client in clients:
                person = await client.discover()
                if not person['registered']: await client.done('/participant')
            if saved is None:
                created = await asyncio.gather(*(client.create() for client in clients))
                saved = [{'space': space, 'conversation': conversation} for space, conversation in created]
            else:
                for i, entry in enumerate(saved):
                    test.check(await clients[i].inventory(entry['space']) == entry['inventory'],
                        'copied file restores exact scoped memory metadata')
                    receipt = await clients[i].request('GET', '/operations/' + entry['receipt']['id'])
                    test.check(receipt['output']['base64'] == entry['receipt']['exact_output'] and receipt['saved_terminal'],
                        'copied file restores each exact saved result')
            wave = 2 if 'inventory' in saved[0] else 1
            async def one(i):
                entry, client = saved[i], clients[i]
                source = 'The private label is amber%04d.' % i
                text = source if wave == 1 else 'What is the private label?'
                receipt, _ = await client.answer(entry['conversation'], text)
                inventory = await client.inventory(entry['space'])
                owned = [row for row in inventory if row['owner'] == entry['space'].split('-')[-1]]
                test.check(bool(owned) and all(row['scope'] == 'private' for row in owned),
                    'every input retains memory in its private space', objects=owned)
                assertions = [row for row in owned if row['kind'] == 2]
                test.check(bool(assertions) == qualified,
                    'only a qualified pair publishes semantic assertions', objects=owned)
                sources = [row for row in owned if row['kind'] == 1]
                bodies = [await aotx_shared_memory_object(test, client, entry['space'], row['id']) for row in sources]
                test.check(any(text.encode() in body and row['actor'] == client.participant for row, body in bodies),
                    'ordinary input retains its exact source text and actor')
                if qualified and wave == 2:
                    output = base64.b64decode(receipt['exact_output']).decode()
                    test.check('amber%04d' % i in output, 'qualified conversation recalls its saved private label', output=output)
                if count > 1:
                    await clients[(i+1) % count].request('GET', '/conversations/' + entry['conversation'], (404,))
                return dict(entry, inventory=inventory, receipt=receipt)
            return await asyncio.gather(*(one(i) for i in range(count)))
    finally:
        gateway.close()


def main():
    parser = argparse.ArgumentParser()
    for name in ('build', 'source', 'store', 'output'): parser.add_argument(name, type=Path)
    parser.add_argument('count', type=int, choices=(1, 64))
    parser.add_argument('--qualified', action='store_true')
    parser.add_argument('--wrapper-mismatch', action='store_true')
    parser.add_argument('--ready-seconds', type=int, default=900)
    args = parser.parse_args()
    if args.qualified and args.wrapper_mismatch: parser.error('A changed wrapper cannot be qualified.')
    if len(str(args.output.resolve() / 'after-journal/service.sock').encode()) >= 108:
        parser.error('The output path exceeds the local socket path limit.')
    test = RuntimeTest(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve(), snapshot_every=64)
    start, status = time.monotonic(), 0
    try:
        f, runtime = aotx_prepare(test, args.wrapper_mismatch)
        cfg, path, grants, keys = aotx_configuration(test, args.count, 'before', 1)
        run = RuntimeRun(test, 'before', runtime, extra=('--service-grants', grants)); run.ready(args.ready_seconds)
        saved = asyncio.run(aotx_surface(test, cfg, path, keys, args.count, args.qualified, 'before'))
        durable(run); run.stop(killed=True)
        before = sections(test, runtime)
        copied = test.output / 'copy.aotxccir'
        test.command([test.build / 'aotx_ccir', 'compact', runtime, copied], 'copy-runtime')
        runtime.unlink(); shutil.rmtree(run.journal)
        cfg, path, grants, keys = aotx_configuration(test, args.count, 'after', 2)
        run = RuntimeRun(test, 'after', copied, extra=('--service-grants', grants)); run.ready(args.ready_seconds)
        match = re.search(r'restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)', run.path.read_text())
        test.check(match and int(match[1]) == f.get(before[7], 40) and int(match[2], 16) == f.get(before[7], 32)
            and not int(match[3]) and not int(match[4]), 'copied file restores its exact record count and state hash')
        asyncio.run(aotx_surface(test, cfg, path, keys, args.count, args.qualified, 'after', saved))
        durable(run); run.stop()
    except Exception as error:
        status = 1; (test.output / 'failure.txt').write_text(type(error).__name__ + ': ' + str(error) + '\n')
    finally:
        for run in reversed(test.active):
            if not run.log.closed:
                try: run.close()
                except Exception as error: status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    status |= any(not row['passed'] for row in test.checks)
    result = dict(exit=status, checks=len(test.checks), failed=sum(not row['passed'] for row in test.checks),
        seconds=time.monotonic()-start, qualified=args.qualified, wrapper_mismatch=args.wrapper_mismatch)
    (test.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return status


if __name__ == '__main__': sys.exit(main())
