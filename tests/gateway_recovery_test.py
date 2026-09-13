#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check HTTPS media admission and complete file recovery with fresh deployment credentials.
# Inputs: Build, source, model store, media fixtures, new output path and batch count.
# Outputs: Commands, source identities and recovery checks. Exit: 0 pass, 1 failure.
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import ssl
import subprocess
import sys
from aiohttp import ClientSession, ClientTimeout, web
from audio_runtime_test import prepare
from gateway_runtime_test import aotx_http_run, aotx_ready, aotx_http, aotx_terminal
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections, replay_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway.config import aotx_load_config
from gateway.__main__ import aotx_write_grants


def aotx_configuration(test, count, label, revision):
    with socket.socket() as port:
        port.bind(('127.0.0.1', 0)); number = port.getsockname()[1]
    keys = ['aotx-recovery-'+hashlib.sha256((str(revision)+':'+str(i)).encode()).hexdigest() for i in range(count)]
    cfg = {'socket': str(test.output/(label+'-journal/service.sock')), 'port': number, 'revision': str(revision),
        'models': {'text': {'role': 'language', 'published_at': 1}, 'audio': {'role': 'language-audio', 'published_at': 1}},
        'limits': {'body_readers': 64, 'fetches': 64, 'upload_bytes': 1048576, 'body_credit': 134217728},
        'principals': [{'id': '%032x' % (i+1), 'token_sha256': [hashlib.sha256(key.encode()).hexdigest()],
            'models': ['text', 'audio'], 'actions': ['infer', 'upload', 'fetch', 'telemetry'], 'requests': 4,
            'tokens': 256, 'pages': 0} for i, key in enumerate(keys)]}
    path = test.output/(label+'-gateway.json'); path.write_text(json.dumps(cfg)); path.chmod(0o600)
    grants = test.output/(label+'-grants'); aotx_write_grants(aotx_load_config(path), grants)
    test.record(grants=str(grants), revision=revision, principals=count)
    return cfg, path, grants, keys


async def aotx_sources(test, fixtures):
    cert, key = test.output/'source-cert.pem', test.output/'source-key.pem'
    argv = ['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', str(key), '-out', str(cert),
        '-days', '1', '-subj', '/CN=localhost', '-addext', 'subjectAltName=IP:127.0.0.1,DNS:localhost']
    test.record(command=argv)
    made = await asyncio.to_thread(subprocess.run, argv, capture_output=True, text=True, timeout=30)
    test.check(made.returncode == 0, 'local source certificate command', status=made.returncode)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.load_cert_chain(cert, key)
    data = [Path(f['path']).read_bytes() for f in fixtures]
    hits = []
    async def source(request):
        if not re.fullmatch('/[0-9]+', request.path): return web.Response(status=404)
        index = int(request.path[1:])
        if index >= len(data): return web.Response(status=404)
        hits.append({'index': index, 'address': request.transport.get_extra_info('sockname')[0]})
        return web.Response(body=data[index], content_type=fixtures[index]['type'])
    runner = web.ServerRunner(web.Server(source), shutdown_timeout=1); await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0, ssl_context=context); await site.start()
    origin = 'https://127.0.0.1:'+str(runner.addresses[0][1])
    return runner, origin, cert, hits


async def aotx_questions(test, client, url, headers, sources, label):
    async def ask(source):
        h = headers[source['principal']]
        body = {'model': source['model'], 'messages': [{'role': 'user', 'content': [
            {'type': 'media', 'media_id': source['id'], 'modality': source['modality']},
            {'type': 'text', 'text': source['prompt']}]}], 'temperature': 0, 'max_tokens': 32}
        admitted, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/requests', h, 202, json=body)
        result = await aotx_terminal(test, client, url, h, admitted['id'])
        text = base64.b64decode(result['output']['bytes']).decode()
        test.check(result['state'] == 'completed' and bool(text.strip()), label+' file media response', result=result, text=text)
        return admitted['id']
    return await asyncio.gather(*(ask(s) for s in sources))


async def aotx_initial(test, cfg, path, keys, fixtures, count):
    runner, origin, cert, hits = await aotx_sources(test, fixtures)
    cfg['urls'] = {'public': False, 'ca_file': str(cert),
        'private': [{'origin': origin, 'networks': ['127.0.0.1/32']}]}
    path.write_text(json.dumps(cfg))
    gateway = aotx_http_run(test, path, 'initial')
    headers = [{'Authorization': 'Bearer '+key} for key in keys]
    url = 'http://127.0.0.1:'+str(cfg['port'])
    try:
        async with ClientSession(timeout=ClientTimeout(total=360)) as client:
            await aotx_ready(test, gateway, client, url, headers[0])
            await aotx_http(test, client, 'POST', url+'/aotx/v1/media/import', headers[0], 403,
                json={'url': 'https://127.0.0.1:1/bytes'})
            sources = []
            async def upload(i):
                which, owner = i%len(fixtures), i%count
                f = fixtures[which]
                value, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/media/import', headers[owner], 201,
                    json={'url': origin+'/'+str(which)})
                test.check(value['sha256'] == hashlib.sha256(Path(f['path']).read_bytes()).hexdigest(), 'HTTPS source digest')
                return dict(f, id=value['id'], sha256=value['sha256'], principal=owner)
            if count == 1:
                for i in range(len(fixtures)): sources.append(await upload(i))
            else: sources = await asyncio.gather(*(upload(i) for i in range(count)))
            test.check(len(hits) == len(sources) and all(h['address'] == '127.0.0.1' for h in hits),
                'only approved HTTPS sources were contacted', hits=hits)
            # Serial requests use the same principal quota; parallel requests use distinct principals.
            if count == 1:
                handles = []
                for source in sources: handles.extend(await aotx_questions(test, client, url, headers, [source], 'initial'))
            else: handles = await aotx_questions(test, client, url, headers, sources, 'initial')
            return sources, handles
    finally:
        gateway.close(); await runner.cleanup()


async def aotx_recovered(test, cfg, path, keys, old_keys, sources, handles, count):
    gateway = aotx_http_run(test, path, 'recovered')
    url = 'http://127.0.0.1:'+str(cfg['port'])
    headers = [{'Authorization': 'Bearer '+key} for key in keys]
    try:
        async with ClientSession(timeout=ClientTimeout(total=360)) as client:
            await aotx_ready(test, gateway, client, url, headers[0])
            await aotx_http(test, client, 'GET', url+'/v1/models', {'Authorization': 'Bearer '+old_keys[0]}, 401)
            await aotx_http(test, client, 'GET', url+'/aotx/v1/requests/'+handles[0], headers[0], 410)
            for owner in range(count):
                inventory, _ = await aotx_http(test, client, 'GET', url+'/aotx/v1/media', headers[owner])
                expected = {s['id']: s['sha256'] for s in sources if s['principal'] == owner}
                actual = {s['id']: s['sha256'] for s in inventory['data']}
                test.check(actual == expected and inventory['next_cursor'] is None,
                    'recovered inventory preserves exact scoped sources', principal=owner, actual=actual)
            if count > 1:
                await aotx_http(test, client, 'GET', url+'/aotx/v1/media/'+sources[0]['id'], headers[1], 404)
            if count == 1:
                for source in sources: await aotx_questions(test, client, url, headers, [source], 'recovered')
            else: await aotx_questions(test, client, url, headers, sources, 'recovered')
    finally: gateway.close()


def aotx_canonical(test, f, saved, sources):
    test.check(not replay_records(f, saved[7], 14), 'ordinary service tokens are absent from the replay state')
    records = replay_records(f, saved[7], 35)
    stored, contiguous = {}, True
    for p in records:
        op, identity = f.get(p, 4, 4), 'media-'+p[8:24].hex()
        if op == 1:
            stored[identity] = {'data': bytearray(), 'principal': p[80:96].hex(), 'digest': p[96:128].hex()}
        elif op == 2:
            row = stored[identity]
            contiguous &= f.get(p, 32) == len(row['data'])
            row['data'].extend(p[40:])
    test.check(contiguous, 'canonical media chunks are contiguous')
    for source in sources:
        row = stored[source['id']]
        test.check(row['principal'] == '%032x' % (source['principal']+1) and
            hashlib.sha256(row['data']).hexdigest() == source['sha256'] == row['digest'],
            'complete file retains exact source bytes and owner')
    return records


def aotx_main():
    parser = argparse.ArgumentParser()
    for name in ('build', 'source', 'store', 'fixtures', 'output'): parser.add_argument(name, type=Path)
    parser.add_argument('count', type=int, choices=(1, 64))
    args = parser.parse_args()
    test = RuntimeTest(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve(), snapshot_every=64)
    fixtures = json.loads(args.fixtures.read_text())
    f, runtime = prepare(test, args.count)
    cfg, path, grants, old_keys = aotx_configuration(test, args.count, 'initial', 1)
    run = None
    try:
        run = RuntimeRun(test, 'initial', runtime, extra=('--service-grants', grants)); run.ready(900)
        sources, handles = asyncio.run(aotx_initial(test, cfg, path, old_keys, fixtures, args.count))
        durable(run); run.stop()
        saved = sections(test, runtime); canonical = aotx_canonical(test, f, saved, sources)
        shutil.rmtree(run.journal)
        cfg, path, grants, keys = aotx_configuration(test, args.count, 'recovered', 2)
        cfg['urls'] = {'public': False}; path.write_text(json.dumps(cfg))
        run = RuntimeRun(test, 'recovered', runtime, extra=('--service-grants', grants)); run.ready(900)
        match = re.search(r'restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)', run.path.read_text())
        test.check(match and int(match[1]) == f.get(saved[7], 40) and int(match[2], 16) == f.get(saved[7], 32)
            and not int(match[3]) and not int(match[4]), 'cold file restore has the exact count and hash')
        test.check('network: IPv4 and IPv6 sockets are disabled' in run.path.read_text(), 'core restore requires no IP network')
        asyncio.run(aotx_recovered(test, cfg, path, keys, old_keys, sources, handles, args.count))
        durable(run); run.stop()
        final = sections(test, runtime)
        test.check(replay_records(f, final[7], 35) == canonical, 'fresh inference preserves canonical source records')
        test.check(not replay_records(f, final[7], 14), 'fresh service output remains outside replayed operator turns')
        (test.output/'sources.json').write_text(json.dumps(sources, indent=2)+'\n')
    finally:
        if run and not run.log.closed: run.close()
        test.flush_checks()


if __name__ == '__main__': aotx_main()
