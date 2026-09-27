#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check saved prompt and affect setting recovery through the real shared gateway.
# Inputs: Build, source, prepared shared file, new short output path. Outputs: Checks and logs. Exit: 0 pass, 1 failure.
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import traceback
from aiohttp import ClientSession, ClientTimeout
from runtime_boot_test import RuntimeTest, RuntimeRun, durable
from gateway_runtime_test import aotx_http_run, aotx_ready, aotx_http
from shared_runtime_client import aotx_shared_client
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway.config import aotx_load_config
from gateway.__main__ import aotx_write_grants


def config(test, label, revision):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    keys = ['controls-fixture-' + hashlib.sha256(str(i).encode()).hexdigest() for i in range(2)]
    cfg = {'socket': str(test.output / (label + '-journal/service.sock')), 'port': port, 'revision': str(revision),
        'models': {'text': {'role': 'language', 'published_at': 0}}, 'principals': [
            {'id': '%032x' % (i + 101), 'token_sha256': [hashlib.sha256(key.encode()).hexdigest()],
             'models': ['text'], 'actions': ['infer', 'telemetry', 'shared_read', 'shared_write', 'shared_manage'] +
             (['affect_manage'] if i == 0 else []), 'requests': 4, 'tokens': 16, 'pages': 0} for i, key in enumerate(keys)]}
    path = test.output / (label + '-gateway.json'); path.write_text(json.dumps(cfg)); path.chmod(0o600)
    grants = test.output / (label + '-grants'); aotx_write_grants(aotx_load_config(path), grants)
    return cfg, path, grants, keys


async def exercise(test, cfg, path, keys, label, retained):
    gateway = aotx_http_run(test, path, label); url = 'http://127.0.0.1:' + str(cfg['port'])
    try:
        async with ClientSession(timeout=ClientTimeout(total=60)) as http:
            clients = [aotx_shared_client(test, http, url, key, '%032x' % (i+101), work_seconds=60) for i, key in enumerate(keys)]
            await aotx_ready(test, gateway, http, url, clients[0].headers)
            async def affect(method='GET', body=None, status=200, index=0):
                return (await aotx_http(test, http, method, url+'/aotx/v1/affect/settings', clients[index].headers,
                    expected=status, **({'json': body} if body is not None else {})))[0]
            owner, other = clients
            for c in clients:
                person = await c.discover()
                if not person['registered']: await c.done('/participant')
            caps = await owner.request('GET', '/capabilities')
            test.check(caps['features']['conversation_prompt'] and caps['limits']['system_prompt_bytes']==2048, 'device advertises exact prompt contract')
            if retained is None:
                space, _ = await owner.done('/spaces', scope='private'); space = 'spc-'+owner.lineage+'-'+space['resource']
                prompts = []
                for prompt in (None, '', 'Use only the selected reference. €', 'é'*1024):
                    fields = {} if prompt is None else {'system_prompt': prompt}
                    row, body = await owner.done('/spaces/'+space+'/conversations', **fields)
                    conversation = 'con-'+owner.lineage+'-'+row['resource']
                    retry = await owner.request('POST', '/spaces/'+space+'/conversations', json=body)
                    test.check(retry['id']==row['id'], 'prompt creation exact retry keeps its saved identity')
                    prompts.append([conversation, prompt])
                before = await affect()
                change = dict(schema='aotx.affect.settings.mutation.v1', epoch=before['epoch'], revision=before['revision'],
                    key='affect.decay_fast', value=2500, scale=10000)
                after = await affect('POST', change)
                test.check(int(after['revision'])==int(before['revision'])+1, 'one setting changes the exact device revision')
                await affect('POST', change, 409)
                retained = dict(prompts=prompts, revision=after['revision'], epoch=after['epoch'], change=change)
            else:
                after = await affect()
                test.check(after['epoch']!=retained['epoch'] and after['revision']==retained['revision'], 'recovery changes epoch and retains setting revision')
                await affect('POST', retained['change'], 410)
            test.check(next(s for s in after['settings'] if s['key']=='affect.decay_fast')['value']==2500, 'setting readback survives the complete file')
            telemetry = await affect(index=1); test.check(not telemetry['writable'], 'telemetry grant remains read-only')
            await affect('POST', dict(retained['change'], epoch=after['epoch'], revision=after['revision']), 403, 1)
            for conversation, prompt in retained['prompts']:
                row = await owner.request('GET', '/conversations/'+conversation+'/prompt')
                test.check(row['prompt']==dict(schema='aotx.conversation.prompt.v1', mode='runtime' if prompt is None else 'explicit',
                    system_prompt=prompt, bytes=0 if prompt is None else len(prompt.encode()), mutable=False), 'exact prompt mode and bytes survive gateway and storage')
                await other.request('GET', '/conversations/'+conversation+'/prompt', (404,))
            await owner.done('/save')
            return retained
    finally: gateway.close()


def main():
    build, source, prepared, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
    test = RuntimeTest(build, source, source, output); status = 0
    try:
        runtime = output/'runtime.aotxccir'; shutil.copy2(prepared, runtime)
        retained = None
        for revision, label in enumerate(('initial', 'recovered'), 1):
            cfg, path, grants, keys = config(test, label, revision)
            run = RuntimeRun(test, label, runtime, extra=('--service-grants', grants)); run.ready(180)
            retained = asyncio.run(exercise(test, cfg, path, keys, label, retained))
            (output/'retained.json').write_text(json.dumps(retained, indent=2))
            durable(run); run.stop()
            test.command([build/'aotx_ccir', 'verify', runtime], 'verify-saved-file')
            if revision == 1:
                copied = output/'recovered.aotxccir'; shutil.copy2(runtime, copied); runtime = copied
        (output/'retained.json').write_text(json.dumps(retained, indent=2))
    except Exception:
        status = 1; (output/'failure.txt').write_text(traceback.format_exc()); traceback.print_exc()
    finally:
        for run in reversed(test.active):
            if not run.log.closed: run.close()
        test.flush_checks()
    (output/'result.json').write_text(json.dumps(dict(status=status, checks=len(test.checks)), indent=2))
    return status


if __name__ == '__main__': raise SystemExit(main())
