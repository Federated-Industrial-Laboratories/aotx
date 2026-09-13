#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check continuing shared conversations, exact retries and complete file recovery through the native gateway.
# Inputs: Build, source, model store, new output directory and batch count. Outputs: Commands and checks. Exit: 0 pass, 1 failure.
import argparse
import asyncio
import base64
import hashlib
import json
import importlib.util
import traceback
from pathlib import Path
import re
import shutil
import socket
import sys
import time
from aiohttp import ClientSession, ClientTimeout
from audio_runtime_test import prepare
from gateway_runtime_test import aotx_http_run, aotx_ready, aotx_http
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections
from shared_runtime_client import aotx_shared_client, PREFIX
from shared_runtime_cases import (aotx_shared_control_cases, aotx_shared_publication_cases,
    aotx_shared_stream_case, aotx_shared_media_cases, aotx_shared_retirement_case,
    aotx_shared_verify_retirement, aotx_shared_verify_results)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway.config import aotx_load_config
from gateway.__main__ import aotx_write_grants


def aotx_configuration(test, count, label, revision):
    with socket.socket() as port:
        port.bind(('127.0.0.1', 0)); number = port.getsockname()[1]
    keys = ['aotx-shared-' + hashlib.sha256((str(revision) + ':' + str(i)).encode()).hexdigest() for i in range(max(count + 1, 3))]
    cfg = {'socket': str(test.output / (label + '-journal/service.sock')), 'port': number, 'revision': str(revision),
        'models': {'text': {'role': 'language', 'published_at': 1}, 'audio': {'role': 'language-audio', 'published_at': 1}},
        'limits': {'body_readers': 64}, 'principals': [
            {'id': '%032x' % (i+1), 'token_sha256': [hashlib.sha256(key.encode()).hexdigest()],
                'models': ['text', 'audio'], 'actions': ['infer', 'upload', 'telemetry', 'shared_read', 'shared_write', 'shared_manage'],
                'requests': 4, 'tokens': 512, 'pages': 0} for i, key in enumerate(keys)]}
    path = test.output / (label + '-gateway.json'); path.write_text(json.dumps(cfg)); path.chmod(0o600)
    grants = test.output / (label + '-grants'); aotx_write_grants(aotx_load_config(path), grants)
    test.record(grants=str(grants), revision=revision, principals=len(keys))
    return cfg, path, grants, keys


async def aotx_scopes(test, clients, spaces, conversations):
    a, b, outsider = clients[:3]
    await b.request('GET', '/spaces/' + spaces[0], (404,))
    await b.request('GET', '/conversations/' + conversations[0], (404,))
    separate, separate_conversation = await a.create()
    test.check(all(r['scope'] == 'instance' for r in await a.inventory(separate)),
        'same participant private spaces expose only shared instance memory before input')
    room, discussion = await a.create('room')
    await b.request('GET', '/spaces/' + room, (404,))
    await a.done('/spaces/' + room + '/members', participant=b.participant, permissions=['read', 'write'])
    row = await b.request('GET', '/spaces/' + room)
    test.check(row['scope'] == 'room' and row['permissions'] == ['read', 'write'], 'explicit room membership')
    first, _ = await a.answer(discussion, 'The room label is copper. Remember this label.')
    second, _ = await b.answer(discussion, 'What is the room label in memory?')
    viewed = await a.request('GET', '/operations/' + second['id'])
    test.check(viewed['actor'] == b.participant and viewed['sequence'] is None and viewed['next_sequence'] is None,
        'shared output preserves sender and hides foreign private counters')
    events = await a.request('GET', '/conversations/' + discussion + '/events')
    test.check([r['id'] for r in events['items']] == [first['id'], second['id']] and
        [r['input_order'] for r in events['items']] == ['1', '2'], 'shared conversation has one local input order')
    await outsider.request('GET', '/operations/' + first['id'], (404,))
    await a.done('/spaces/' + room + '/members', participant=b.participant, permissions=[])
    await b.request('GET', '/operations/' + first['id'], (404,))
    await b.request('GET', '/conversations/' + discussion + '/events', (404,))
    instance, instance_conversation = await a.create('instance')
    visible = await outsider.request('GET', '/spaces/' + instance)
    test.check(visible['scope'] == 'instance', 'instance space is visible to a currently admitted participant')
    return {'room': room, 'discussion': discussion, 'separate': separate, 'instance': instance}


async def aotx_initial(test, cfg, path, keys, count, fixtures, media_retry_seconds=900):
    gateway = aotx_http_run(test, path, 'initial')
    url = 'http://127.0.0.1:' + str(cfg['port'])
    try:
        async with ClientSession(timeout=ClientTimeout(total=960)) as http:
            clients = [aotx_shared_client(test, http, url, key, '%032x' % (i+1)) for i, key in enumerate(keys)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            caps, _ = await aotx_http(test, http, 'GET', url + '/aotx/v1/capabilities', clients[0].headers)
            test.check(caps['features']['continuing_ccir'] and caps['features']['persistent_requests'], 'actual shared runtime capability')
            for client in clients:
                person = await client.discover()
                if not person['registered']: await client.done('/participant')
            created = await asyncio.gather(*(clients[i].create() for i in range(count)))
            spaces, conversations = map(list, zip(*created))
            retained = []
            for wave in (1, 2):
                async def one(i):
                    text = ('The private member label is label%04d. Remember this label.' % i if wave == 1
                        else 'State the private member label from memory.')
                    final, body = await clients[i].answer(conversations[i], text)
                    again = await clients[i].request('POST', '/conversations/' + conversations[i] + '/inputs', json=body)
                    test.check(again['id'] == final['id'] and again['output']['base64'] == final['exact_output'],
                        'exact canonical retry returns the original saved result')
                    changed = dict(body, text=text + ' changed')
                    await clients[i].request('POST', '/conversations/' + conversations[i] + '/inputs', (409,), json=changed)
                    return dict(actor=i, conversation=conversations[i], receipt=final, body=body)
                values = await asyncio.gather(*(one(i) for i in range(count)))
                retained.extend(values)
                test.check(len({v['receipt']['id'] for v in values}) == count, 'distinct shared input batch', count=count, wave=wave)
            for i in range(count):
                items = await clients[i].inventory(spaces[i])
                owned = [r for r in items if r['owner'] == spaces[i].split('-')[-1] and r['scope'] == 'private']
                test.check(bool(owned) and all(r in owned or r['scope'] == 'instance' for r in items),
                    'private memory remains in its persistent space', participant=i, objects=items)
                events = await clients[i].request('GET', '/conversations/' + conversations[i] + '/events')
                test.check([r['input_order'] for r in events['items']] == ['1', '2'], 'conversation survives temporary slot reuse')
            scopes = await aotx_scopes(test, clients, spaces, conversations)
            inventory = await clients[0].inventory(spaces[0])
            events = await clients[0].request('GET', '/conversations/' + conversations[0] + '/events')
            ordinary, _ = await aotx_http(test, http, 'POST', url + '/v1/chat/completions', clients[0].headers,
                json={'model': 'text', 'temperature': 0, 'max_tokens': 16, 'messages': [
                    {'role': 'user', 'content': 'The temporary label is violet.'},
                    {'role': 'assistant', 'content': 'I have the temporary label.'},
                    {'role': 'user', 'content': 'State the temporary label from these messages.'}]})
            test.check(bool(ordinary['choices'][0]['message']['content'].strip()) and ordinary['usage']['completion_tokens'] > 0,
                'ordinary supplied-history inference remains available beside shared conversations', response=ordinary)
            after = await clients[0].request('GET', '/conversations/' + conversations[0] + '/events')
            test.check(after['items'] == events['items'] and await clients[0].inventory(spaces[0]) == inventory,
                'ordinary inference does not append shared input or retained memory')
            saved = {'spaces': spaces, 'conversations': conversations, 'operations': retained, 'scopes': scopes}
            saved['advanced'] = {
                'control': await aotx_shared_control_cases(test, clients[0], clients[1]),
                'publication': await aotx_shared_publication_cases(test, *clients[:3]),
                'stream': await aotx_shared_stream_case(test, clients[0]),
                'media': await aotx_shared_media_cases(test, clients[:count], clients[count], fixtures, retry_seconds=media_retry_seconds),
                'retirement': await aotx_shared_retirement_case(test, clients[count]),
            }
            before = await clients[0].inventory(spaces[0])
            pending, body = await clients[0].input(conversations[0],
                'Write the integers from 1 to 10000, separated by commas.', max_output_tokens=512)
            end = time.monotonic() + 900
            while time.monotonic() < end:
                current = await clients[0].request('GET', '/operations/' + pending['id'], audit=False)
                if current['state'] == 'running' and current['saved_admission']: break
                if current['state'] in ('completed', 'failed', 'cancelled'): raise AssertionError('The interruption input completed before the selected cut.')
                await asyncio.sleep(0.01)
            else: raise TimeoutError('The saved interruption input did not start.')
            saved['interrupted'] = {'receipt': current, 'body': body, 'conversation': conversations[0], 'memory': before}
            (test.output / 'retained.json').write_text(json.dumps(saved, indent=2) + '\n')
            return saved
    finally:
        gateway.close()


async def aotx_recovered(test, cfg, path, keys, old_keys, retained, label):
    gateway = aotx_http_run(test, path, label)
    url = 'http://127.0.0.1:' + str(cfg['port'])
    try:
        async with ClientSession(timeout=ClientTimeout(total=960)) as http:
            clients = [aotx_shared_client(test, http, url, key, '%032x' % (i+1)) for i, key in enumerate(keys)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            await aotx_http(test, http, 'GET', url + '/v1/models', {'Authorization': 'Bearer ' + old_keys[0]}, 401)
            for client in clients: await client.discover()
            await aotx_shared_verify_retirement(test, clients[len(retained['spaces'])], retained['advanced']['retirement'])
            await aotx_shared_verify_results(test, clients, retained['advanced'])
            for entry in retained['operations']:
                client, old = clients[entry['actor']], entry['receipt']
                actual = await client.terminal(old['id'])
                test.check(all(actual[k] == old[k] for k in ('id', 'actor', 'sequence', 'input_order', 'status', 'state', 'exact_output', 'usage', 'finish')),
                    'file-only recovery preserves exact shared result', receipt=actual)
                retry = await client.request('POST', '/conversations/' + entry['conversation'] + '/inputs', json=entry['body'])
                test.check(retry['id'] == old['id'], 'recovered canonical retry cannot repeat input')
            pending = retained['interrupted']
            stopped = await clients[0].terminal(pending['receipt']['id'])
            test.check(stopped['state'] == 'interrupted' and stopped['status'] == 598 and stopped['gap'],
                'unfinished saved input becomes an explicit interrupted result')
            retry = await clients[0].request('POST', '/conversations/' + pending['conversation'] + '/inputs', json=pending['body'])
            test.check(retry['id'] == stopped['id'] and retry['state'] == 'interrupted', 'interrupted retry cannot run inference again')
            if label == 'recovered-2':
                actual_memory = await clients[0].inventory(retained['spaces'][0])
                test.check(actual_memory == pending['memory'], 'unsaved interrupted input cannot add recovered memory')
            for i, conversation in enumerate(retained['conversations']):
                await clients[i].answer(conversation, 'State the private member label from memory.')
            await clients[1].request('GET', '/spaces/' + retained['spaces'][0], (404,))
            await clients[1].request('GET', '/spaces/' + retained['scopes']['room'], (404,))
    finally:
        gateway.close()


def aotx_main():
    parser = argparse.ArgumentParser()
    for name in ('build', 'source', 'store', 'output'): parser.add_argument(name, type=Path)
    parser.add_argument('count', type=int, choices=(1, 64))
    parser.add_argument('--runtime', type=Path, help='Use an existing prepared shared test runtime in place.')
    parser.add_argument('--ready-seconds', type=int, default=900, help='Set the file activation and replay time limit.')
    parser.add_argument('--media-retry-seconds', type=int, default=900, help='Set the caller time limit for media pressure retries.')
    parser.add_argument('--media-fixtures', type=Path, required=True, help='Read the qualified image and audio input manifest.')
    args = parser.parse_args()
    if args.ready_seconds < 1: parser.error('The readiness time limit must be positive.')
    if args.media_retry_seconds < 1: parser.error('The media retry time limit must be positive.')
    if len(str(args.output.resolve() / 'recovered-3-journal/service.sock').encode()) >= 108:
        parser.error('The output path exceeds the local socket path limit.')
    test = RuntimeTest(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve(), snapshot_every=64)
    start, status = time.monotonic(), 0
    try:
        if args.runtime:
            runtime = args.runtime.resolve()
            spec = importlib.util.spec_from_file_location('shared_bytes', test.source / 'tests/recall_cli_test.py')
            f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
            test.check(bool(f.get(sections(test, runtime)[5], 20, 4) & 8), 'prepared runtime requires shared semantics')
        else:
            f, runtime = prepare(test, args.count)
            test.command([test.build / 'aotx_ccir_pack', '--runtime', runtime, '--shared'], 'enable-shared')
        cfg, path, grants, keys = aotx_configuration(test, args.count, 'initial', 1)
        run = RuntimeRun(test, 'initial', runtime, extra=('--service-grants', grants)); run.ready(args.ready_seconds)
        retained = asyncio.run(aotx_initial(test, cfg, path, keys, args.count, args.media_fixtures, args.media_retry_seconds))
        run.stop(killed=True)
        for revision in (2, 3):
            saved = sections(test, runtime)
            shutil.rmtree(run.journal)
            label = 'recovered-' + str(revision)
            old_keys = keys
            cfg, path, grants, keys = aotx_configuration(test, args.count, label, revision)
            run = RuntimeRun(test, label, runtime, extra=('--service-grants', grants)); run.ready(args.ready_seconds)
            match = re.search(r'restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)', run.path.read_text())
            test.check(match and int(match[1]) == f.get(saved[7], 40) and int(match[2], 16) == f.get(saved[7], 32)
                and not int(match[3]) and not int(match[4]), 'shared complete file restores its exact record count and hash')
            asyncio.run(aotx_recovered(test, cfg, path, keys, old_keys, retained, label))
            durable(run); run.stop()
    except Exception as error:
        status = 1; (test.output / 'failure.txt').write_text(type(error).__name__ + ': ' + str(error) + '\n')
        traceback.print_exc()
        if hasattr(error, 'body'): test.record(error=error.body())
    finally:
        for run in reversed(test.active):
            if not run.log.closed:
                try: run.close()
                except Exception as error: status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    result = dict(status=status, checks=len(test.checks), seconds=time.monotonic() - start)
    (test.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result), flush=True)
    return status


if __name__ == '__main__': sys.exit(aotx_main())
