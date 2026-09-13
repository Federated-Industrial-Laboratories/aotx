#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check real scoped HTTP inference, media and client recovery against a GPU runtime.
# Inputs: Build, source, model store, new output path and batch count.
# Outputs: Commands, checks and response records. Exit: 0 pass, 1 failure.
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from aiohttp import ClientSession, ClientTimeout
from openai import AsyncOpenAI
from live_boot_test import Test, Run

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class aotx_http_run:
    def __init__(self, test, config, label):
        self.test = test
        self.path = test.output/(label+'-gateway.log')
        self.log = self.path.open('w')
        argv = [sys.executable, '-m', 'gateway', 'serve', '--config', str(config)]
        test.record(command=argv, cwd=str(test.source), output=str(self.path))
        self.child = subprocess.Popen(argv, cwd=test.source, stdout=self.log, stderr=subprocess.STDOUT)
        test.record(pid=self.child.pid)

    def close(self):
        if self.child.poll() is None: self.child.terminate()
        status = self.child.wait(timeout=30)
        self.test.record(pid=self.child.pid, exit=status)
        self.log.close()
        self.test.check(status == 0, 'gateway stop status', status=status)


async def aotx_ready(test, gateway, client, url, headers):
    end = time.monotonic()+30
    while time.monotonic() < end:
        if gateway.child.poll() is not None: raise RuntimeError('The gateway exited before readiness.')
        try:
            async with client.get(url+'/v1/models', headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    test.check(bool(result['data']), 'HTTP model discovery', result=result)
                    return
        except OSError: pass
        await asyncio.sleep(0.1)
    raise TimeoutError('The gateway did not become ready.')


async def aotx_http(test, client, method, path, headers, expected=200, **kw):
    async with client.request(method, path, headers=headers, **kw) as response:
        raw = await response.read()
        test.check(response.status == expected, method+' HTTP status', status=response.status,
            expected=expected, body=raw.decode(errors='replace'))
        return json.loads(raw) if raw else None, dict(response.headers)


async def aotx_terminal(test, client, url, headers, handle):
    end = time.monotonic()+300
    while time.monotonic() < end:
        async with client.get(url+'/aotx/v1/requests/'+handle, headers=headers) as response:
            if response.status == 429:
                value = await response.json()
                test.check(response.headers.get('Retry-After') == '1' and
                    value['error']['code'] == 'transport_limit', 'status reads report bounded transport pressure')
                await asyncio.sleep(1)
                continue
            if response.status != 200:
                detail = await response.text()
                test.record(status_read_error=response.status, handle=handle, body=detail)
                raise RuntimeError('The owned status request failed: '+str(response.status)+' '+detail)
            value = await response.json()
        if value['state'] in ('completed', 'failed', 'cancelled'): return value
        await asyncio.sleep(0.05)
    raise TimeoutError('The device request did not reach a terminal state.')


async def aotx_exercise(test, args, cfg, config_path, keys):
    url = 'http://127.0.0.1:'+str(cfg['port'])
    headers = [{'Authorization': 'Bearer '+key} for key in keys]
    gateway = aotx_http_run(test, config_path, 'first')
    try:
        async with ClientSession(timeout=ClientTimeout(total=360)) as client:
            await aotx_ready(test, gateway, client, url, headers[0])
            capabilities, _ = await aotx_http(test, client, 'GET', url+'/aotx/v1/capabilities', headers[0])
            test.check(not capabilities['features']['continuing_ccir'], 'ordinary profile reports its persistence contract')
            test.check(capabilities['limits']['kv_pages_per_request'] > 32, 'automatic page grant uses the device profile')
            await aotx_http(test, client, 'GET', url+'/v1/models', {'Authorization': 'Bearer invalid'}, 401)
            body = {'model': 'text', 'messages': [{'role': 'system', 'content': 'Give a short direct answer.'},
                {'role': 'user', 'content': 'What color is a clear daytime sky?'}], 'temperature': 0, 'max_tokens': 16}
            async def one(i):
                color = ('blue', 'green', 'red', 'yellow', 'orange', 'purple', 'white', 'black')[i%8]
                distinct = dict(body, messages=[{'role': 'system', 'content': 'Give a short direct answer.'},
                    {'role': 'user', 'content': 'For request '+str(i)+', the check color is '+color+'.'},
                    {'role': 'assistant', 'content': 'The check color is set.'},
                    {'role': 'user', 'content': 'What is the check color?'}])
                value, h = await aotx_http(test, client, 'POST', url+'/v1/chat/completions', headers[i], json=distinct)
                text = value['choices'][0]['message']['content']
                test.check(color in text.lower() and value['usage']['completion_tokens'] > 0,
                    'actual GPU completion', principal=i, text=text, usage=value['usage'], finish=value['choices'][0]['finish_reason'])
                return h['X-Request-ID']
            start = time.monotonic()
            handles = await asyncio.gather(*(one(i) for i in range(args.count)))
            test.check(len(set(handles)) == args.count, 'distinct concurrent request identities', seconds=time.monotonic()-start)
            foreign = headers[1] if len(headers) > 1 else headers[-1]
            await aotx_http(test, client, 'GET', url+'/aotx/v1/requests/'+handles[0], foreign, 404)
            await aotx_http(test, client, 'POST', url+'/aotx/v1/requests/'+handles[0]+'/cancel', foreign, 404, json={})
            await aotx_http(test, client, 'GET', url+'/aotx/v1/requests/'+handles[0]+'?cursor=9999999', headers[0], 409)
            async def sdk_one(i):
                color = ('blue', 'green', 'red', 'yellow', 'orange', 'purple', 'white', 'black')[i%8]
                distinct = dict(body, messages=[{'role': 'user', 'content':
                    'For client '+str(i)+', repeat this label: case '+str(i)+' '+color+'.'}])
                async with AsyncOpenAI(api_key=keys[i], base_url=url+'/v1', max_retries=0, timeout=300) as sdk:
                    response = await sdk.chat.completions.create(**distinct)
                    handle = 'req-%016x-%s' % (int(capabilities['runtime_epoch']), response.id.removeprefix('chatcmpl-'))
                    native = await aotx_terminal(test, client, url, headers[i], handle)
                    text = base64.b64decode(native['output']['bytes']).decode()
                    test.check(response.choices[0].message.content == text and bool(text.strip()) and
                        all(getattr(response.usage, k) == v for k, v in native['usage'].items()),
                        'Python SDK exact owned JSON output and usage', principal=i, response=response.model_dump(), native=native)
                    stream = await sdk.chat.completions.create(**distinct, stream=True, stream_options={'include_usage': True})
                    chunks = [c.model_dump() async for c in stream]
                    identity = chunks[0]['id']
                    handle = 'req-%016x-%s' % (int(capabilities['runtime_epoch']), identity.removeprefix('chatcmpl-'))
                    native = await aotx_terminal(test, client, url, headers[i], handle)
                    content = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks if c['choices'])
                    test.check(all(c['id'] == identity for c in chunks) and chunks[0]['choices'][0]['delta']['role'] == 'assistant'
                        and chunks[-1]['choices'] == [] and content.encode() == base64.b64decode(native['output']['bytes'])
                        and all(chunks[-1]['usage'][k] == v for k, v in native['usage'].items())
                        and chunks[-2]['choices'][0]['finish_reason'] == native['finish_reason'],
                        'Python SDK exact owned stream and usage', principal=i, chunks=chunks, native=native)
                    return response.id, identity
            for n in ([1] if args.count == 1 else [1, 64]):
                ids = await asyncio.gather(*(sdk_one(i) for i in range(n)))
                test.check(len({identity for pair in ids for identity in pair}) == n*2,
                    'Python SDK distinct caller batch', count=n)
            request, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/requests', headers[0], 202, json=body)
            retained = request['id']
            gateway.close(); gateway = aotx_http_run(test, config_path, 'second')
            await aotx_ready(test, gateway, client, url, headers[0])
            value = await aotx_terminal(test, client, url, headers[0], retained)
            test.check(value['state'] == 'completed' and value['usage']['completion_tokens'] > 0,
                'gateway restart preserves live device handles', result=value)
            async with client.get(url+'/aotx/v1/requests/'+retained+'/events', headers=headers[0]) as response:
                raw = await response.text(); test.check(response.status == 200, 'native event stream status')
            emissions = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ')]
            parts = [e for e in emissions if e['schema'] == 'aotx.emission.v1']
            test.check(b''.join(base64.b64decode(e['bytes']) for e in parts) == base64.b64decode(value['output']['bytes']),
                'native event bytes match the retained output')
            cancel_body = dict(body, max_tokens=256, messages=[{'role': 'user',
                'content': 'Write the integers from 1 to 1000 in order. Separate each integer with a comma.'}])
            request, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/requests', headers[0], 202, json=cancel_body)
            receipt, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/requests/'+request['id']+'/cancel', headers[0], json={})
            stopped = await aotx_terminal(test, client, url, headers[0], request['id'])
            expected = 'cancelled' if receipt['cancel_requested'] else 'completed'
            test.check(stopped['state'] == expected, 'exact request cancellation agrees with its receipt',
                receipt=receipt, result=stopped)
            if args.fixtures:
                fixtures = json.loads(Path(args.fixtures).read_text())
                for fixture in fixtures:
                    data = Path(fixture['path']).read_bytes()
                    upload, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/media',
                        dict(headers[0], **{'Content-Type': fixture['type']}), 201, data=data)
                    test.check(upload['sha256'] == hashlib.sha256(data).hexdigest(), 'uploaded source digest', source=fixture['path'])
                    await aotx_http(test, client, 'GET', url+'/aotx/v1/media/'+upload['id'], foreign, 404)
                    mb = {'model': fixture['model'], 'messages': [{'role': 'user', 'content': [
                        {'type': 'media', 'media_id': upload['id'], 'modality': fixture['modality']},
                        {'type': 'text', 'text': fixture['prompt']}]}], 'temperature': 0, 'max_tokens': 64}
                    request, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/requests', headers[0], 202, json=mb)
                    media_result = await aotx_terminal(test, client, url, headers[0], request['id'])
                    text = base64.b64decode(media_result['output']['bytes']).decode()
                    test.check(media_result['state'] == 'completed' and bool(text.strip()),
                        'actual media-conditioned response', source=fixture['path'], text=text, result=media_result)
                    await aotx_http(test, client, 'DELETE', url+'/aotx/v1/media/'+upload['id'], headers[0], 204)
                if args.count == 64:
                    async def media_one(i):
                        fixture = fixtures[i%len(fixtures)]
                        data = Path(fixture['path']).read_bytes()
                        upload, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/media',
                            dict(headers[i], **{'Content-Type': fixture['type']}), 201, data=data)
                        return fixture, upload
                    batch = await asyncio.gather(*(media_one(i) for i in range(args.count)))
                    async def ask_one(i):
                        fixture, upload = batch[i]
                        mb = {'model': fixture['model'], 'messages': [{'role': 'user', 'content': [
                            {'type': 'media', 'media_id': upload['id'], 'modality': fixture['modality']},
                            {'type': 'text', 'text': fixture['prompt']}]}], 'temperature': 0, 'max_tokens': 32}
                        admitted, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/requests', headers[i], 202, json=mb)
                        result = await aotx_terminal(test, client, url, headers[i], admitted['id'])
                        text = base64.b64decode(result['output']['bytes']).decode()
                        test.check(result['state'] == 'completed' and bool(text.strip()), 'concurrent media response',
                            principal=i, source=fixture['path'], text=text, result=result)
                    await asyncio.gather(*(ask_one(i) for i in range(args.count)))
                    for i, (_, upload) in enumerate(batch):
                        await aotx_http(test, client, 'DELETE', url+'/aotx/v1/media/'+upload['id'], headers[i], 204)
            if args.js_module:
                js_config = test.output/'js-client.json'
                js_config.write_text(json.dumps({'baseURL': url+'/v1', 'keys': keys, 'model': 'text',
                    'batches': [1] if args.count == 1 else [1, 64], 'runtime_epoch': capabilities['runtime_epoch']})); js_config.chmod(0o600)
                argv = [args.node, str(test.source/'tests/gateway_sdk_test.mjs'), args.js_module,
                    str(js_config), str(test.output/'js-results.json')]
                test.record(command=argv)
                js = await asyncio.to_thread(subprocess.run, argv, capture_output=True, text=True, timeout=300)
                test.check(js.returncode == 0, 'JavaScript SDK real client checks', status=js.returncode, output=js.stdout+js.stderr)
            return {'last_handle': retained, 'capabilities': capabilities}
    finally:
        if not gateway.log.closed: gateway.close()


def aotx_main():
    parser = argparse.ArgumentParser()
    for name in ('build', 'source', 'store', 'output'): parser.add_argument(name, type=Path)
    parser.add_argument('count', type=int, choices=(1, 64))
    parser.add_argument('--roles', default='language')
    parser.add_argument('--fixtures')
    parser.add_argument('--js-module')
    parser.add_argument('--node', default='node')
    parser.add_argument('--boundaries', action='store_true')
    args = parser.parse_args()
    test = Test(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve())
    shutil.copytree(test.source/'modules/roles', test.output/'modules')
    (test.output/'settings').write_text('sample.temperature = 0\nsample.seed = 7\n')
    keys = ['aotx-test-'+hashlib.sha256(str(i).encode()).hexdigest() for i in range(max(2, args.count))]
    with socket.socket() as port:
        port.bind(('127.0.0.1', 0)); selected = port.getsockname()[1]
    cfg = {'socket': str(test.output/'journal/service.sock'), 'port': selected, 'revision': '1',
        'limits': {'body_readers': 64},
        'models': {'text': {'role': 'language', 'published_at': 1}}, 'principals': [
            {'id': '%032x' % (i+1), 'token_sha256': [hashlib.sha256(key.encode()).hexdigest()], 'models': ['text'],
                'actions': ['infer', 'upload', 'fetch', 'telemetry'], 'requests': 4, 'tokens': 512, 'pages': 0}
            for i, key in enumerate(keys)]}
    if 'language-audio' in args.roles:
        cfg['models']['audio'] = {'role': 'language-audio', 'published_at': 1}
        for principal in cfg['principals']: principal['models'].append('audio')
    config_path = test.output/'gateway.json'; config_path.write_text(json.dumps(cfg)); config_path.chmod(0o600)
    grants = test.output/'grants'
    argv = [sys.executable, '-m', 'gateway', 'grants', '--config', str(config_path), '--output', str(grants)]
    test.record(command=argv, cwd=str(test.source))
    grant_result = subprocess.run(argv, cwd=test.source, capture_output=True, text=True, timeout=30)
    test.check(grant_result.returncode == 0, 'operator grant file command', status=grant_result.returncode,
        output=grant_result.stdout+grant_result.stderr)
    boot = None
    try:
        boot = Run(test, 'http', extra=('--service-grants', grants), roles=args.roles)
        boot.ready(); test.check(True, 'runtime is ready')
        result = asyncio.run(aotx_exercise(test, args, cfg, config_path, keys))
        if args.boundaries:
            from gateway_boundary_test import aotx_boundaries
            asyncio.run(aotx_boundaries(test, boot, cfg, config_path, keys, args.count))
        (test.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        boot.stop()
    finally:
        if boot and not boot.log.closed: boot.close()
        test.flush_checks()


if __name__ == '__main__': aotx_main()
