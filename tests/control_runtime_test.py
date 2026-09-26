#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check exact control selection, authored identity and copied runtime recovery through real consumers.
# Inputs: Build, source, model store, identity text and new output path. Outputs: Commands and checks. Exit: 0 pass, 1 failure.
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from aiohttp import ClientSession, ClientTimeout
from runtime_boot_test import RuntimeTest, RuntimeRun, durable
from text_boot_test import setup
from shared_runtime_test import aotx_configuration
from shared_runtime_client import aotx_shared_client
from gateway_runtime_test import aotx_http_run, aotx_ready
from live_boot_test import wait


def prepare(test, identity):
    f = setup(test)
    (test.output/'modules/conductor/overlay.txt').write_bytes(identity.read_bytes())
    manifest = test.output/'modules/conductor/module.manifest'
    manifest.write_text(manifest.read_text().replace('version: 1', 'version: 2'))
    with (test.output/'settings').open('a') as out:
        out.write('decode.reply_limit = 96\n')
    inputs = test.output/'inputs'
    checkpoint, memory = inputs/'checkpoint', inputs/'memory.aotxccir'
    checkpoint.write_bytes(f.image([], 0, 10))
    test.command([test.build/'aotx_recall_cli_fixture', checkpoint, '-', memory, 1], 'empty-memory')
    store = test.output/'selected-models'; store.mkdir()
    for p in test.store.iterdir():
        if p.is_file():
            if p.suffix == '.gguf': (store/p.name).symlink_to(p.resolve())
            else: shutil.copy2(p, store/p.name)
    runtime = test.output/'state.aotxccir'
    test.command([test.build/'aotx_ccir_pack', '--memory', memory, '--models', store,
        '--modules', test.output/'modules', '--settings', test.output/'settings', '--roles', 'language,embedding',
        '--output', runtime, '--phrases', test.source/'tests/fixtures/quality/refusal-phrases.txt'], 'pack-runtime')
    test.command([test.build/'aotx_ccir_pack', '--runtime', runtime, '--shared'], 'enable-shared')
    shutil.rmtree(store); shutil.rmtree(inputs); shutil.rmtree(test.output/'modules'); (test.output/'settings').unlink()
    return runtime


def local_control(test, run, value):
    previous = len(run.console())
    run.send('agent 0 decode.steer0 '+value)
    wanted = 'agent: decode.steer0 '+value+' changes at the next turn'
    wait(lambda: wanted in run.console()[previous:], run.child, 30)
    test.check(True, 'local control command is accepted')


def local_identity(test, run, name, dose, expected, label):
    local_control(test, run, name+':'+str(dose/10000))
    run.send('set decode.reply_limit 96')
    previous = len(run.events(0))
    run.send('say What is your name and project? Answer briefly, without a question.')
    row = wait(lambda: next((r for r in run.events(0)[previous:] if r.get('kind') == 'reply'), None), run.child, 180)
    text = row.get('text', '')
    test.check(all(word.lower() in text.lower() for word in expected), 'local identity facts reach the selected consumer', text=text)
    local_control(test, run, 'absent'); durable(run)
    (test.output/(label+'-local.json')).write_text(json.dumps(row, indent=2)+'\n')


async def surface(test, run, cfg, keys, identity, expected, saved=None):
    url = 'http://127.0.0.1:'+str(cfg['port'])
    async with ClientSession(timeout=ClientTimeout(total=240)) as http:
        clients = [aotx_shared_client(test, http, url, key, '%032x'%(i+1), work_seconds=180) for i, key in enumerate(keys[:2])]
        for c in clients:
            person = await c.discover()
            if not person['registered']: await c.done('/participant')
        a, b = clients
        async with http.get(url+'/aotx/v1/capabilities', headers=a.headers) as response:
            caps = await response.json(); test.check(response.status == 200, 'capability HTTP status')
        controls = [c for m in caps['models'] if m['id'] == 'text' for c in m['controls'] if c['available']]
        test.check(caps['features']['control_selection'] and len(controls) == 1, 'one exact accepted control is selectable')
        selected = controls[0]
        test.check(selected['accepted_doses'] == [5000] and selected['positions'] == 'response', 'fixed response-only dose is reported')
        control = {'schema': 'aotx.control.selection.v1', 'kind': 'residual_vector',
            'qualification_sha256': selected['qualification_sha256'], 'dose': 5000}
        question = 'What is your name and project? Answer briefly, without a question.'
        body = {'model': 'text', 'max_tokens': 96, 'temperature': 0,
            'messages': [{'role': 'system', 'content': identity.read_text()}, {'role': 'user', 'content': question}], 'control': control}
        async with http.post(url+'/aotx/v1/requests', headers=a.headers, json=body) as response:
            native = await response.json(); test.check(response.status == 202, 'native request admits the exact control')
        end = time.monotonic()+180
        while time.monotonic() < end:
            async with http.get(url+'/aotx/v1/requests/'+native['id'], headers=a.headers) as response:
                value = await response.json()
            if value['state'] in ('completed', 'failed', 'cancelled'): break
            await asyncio.sleep(0.1)
        test.check(value['state'] == 'completed', 'native selected request completes', response=value)
        text = base64.b64decode(value['output']['bytes']).decode()
        test.check(all(w.lower() in text.lower() for w in expected), 'native authored facts remain in the selected reply', text=text)
        before = saved is None
        if before:
            _, conversation = await a.create(); _, private = await b.create()
        else:
            conversation, private = saved['conversation'], saved['private']
            test.check(control == saved['control'], 'copied runtime retains exact qualification and dose')
            for entry in saved['receipts']:
                r = await a.request('GET', '/operations/'+entry['id'])
                test.check(r['saved_terminal'] and r['output']['base64'] == entry['exact_output'], 'copied runtime retains the exact saved reply')
        await b.request('GET', '/conversations/'+conversation, (404,))
        receipts = []
        for choice in (None, control):
            extra = {'control': choice} if choice else {}
            receipt, _ = await a.answer(conversation, question, max_output_tokens=96, **extra)
            text = base64.b64decode(receipt['exact_output']).decode()
            test.check(all(w.lower() in text.lower() for w in expected), 'shared identity facts survive control selection and recovery', text=text)
            receipts.append(receipt)
        for patch in ({'dose': 7500}, {'qualification_sha256': 'fe'*32}):
            await a.request('POST', '/conversations/'+conversation+'/inputs', (503,),
                json=a.command(text=question, model='text', max_output_tokens=96, control={**control, **patch}))
        run.send('set affect.on 1'); durable(run)
        await a.request('POST', '/conversations/'+conversation+'/inputs', (503,),
            json=a.command(text=question, model='text', max_output_tokens=96, control=control))
        affect, _ = await b.answer(private, 'Reply with the word ready.', max_output_tokens=16)
        state = (await b.request('GET', '/conversations/'+private+'/affect'))['affect']
        test.check(state['enabled'] and state['probes_at_last_turn'] == {'valence':'unavailable', 'arousal':'unavailable'},
            'event state remains separate from unavailable fitted probes')
        run.send('set affect.on 0'); durable(run)
        local_identity(test, run, selected['name'], 5000, expected, 'before' if before else 'after')
        result = {'control': control, 'conversation': conversation, 'private': private, 'receipts': receipts,
            'capabilities': caps, 'native': value, 'affect': state, 'url': url, 'keys': keys}
        return result


def main():
    p = argparse.ArgumentParser(description='Check qualified controls and complete recovery.')
    for name in ('build', 'source', 'store', 'identity', 'output'): p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--expected', required=True, nargs='+')
    p.add_argument('--observe', action='store_true')
    p.add_argument('--runtime', type=Path)
    args = p.parse_args(); test = RuntimeTest(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve())
    gateway = None; result = 1
    try:
        if args.runtime:
            runtime = test.output/'state.aotxccir'; shutil.copyfile(args.runtime, runtime)
            test.record(source_runtime=str(args.runtime.resolve()), copied_runtime=str(runtime))
        else: runtime = prepare(test, args.identity.resolve())
        saved = None
        for revision, label in enumerate(('before', 'after'), 1):
            cfg, path, grants, keys = aotx_configuration(test, 2, label, revision)
            cfg['origins'] = ['http://127.0.0.1:'+str(cfg['port'])]
            path.write_text(json.dumps(cfg)); path.chmod(0o600)
            run = RuntimeRun(test, label, runtime, extra=('--service-grants', grants)); run.ready(300)
            gateway = aotx_http_run(test, path, label)
            async def ready():
                async with ClientSession() as http: await aotx_ready(test, gateway, http, 'http://127.0.0.1:'+str(cfg['port']), {'Authorization':'Bearer '+keys[0]})
            asyncio.run(ready())
            saved = asyncio.run(surface(test, run, cfg, keys, args.identity.resolve(), args.expected, saved))
            (test.output/(label+'-surface.json')).write_text(json.dumps(saved, indent=2)+'\n')
            durable(run)
            if args.observe:
                marker = test.output/(label+'-continue')
                wait(lambda: marker.exists(), run.child, 600)
            gateway.close(); gateway = None; run.stop()
            if label == 'before':
                copied = test.output/'copy.aotxccir'
                test.command([test.build/'aotx_ccir', 'compact', runtime, copied], 'copy-runtime')
                runtime.unlink(); shutil.rmtree(run.journal); runtime = copied
        result = 0
    finally:
        if gateway: gateway.close()
        for run in reversed(test.active):
            if not run.log.closed: run.close()
        test.flush_checks()
        result |= any(not c['passed'] for c in test.checks)
        (test.output/'result.json').write_text(json.dumps({'exit': result, 'checks': len(test.checks),
            'failed': sum(not c['passed'] for c in test.checks)}, indent=2)+'\n')
    return result


if __name__ == '__main__': raise SystemExit(main())
