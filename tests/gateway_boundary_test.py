#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check disconnections and operator grant replacement through a running service.
# Inputs: The live gateway test state. Outputs: Scoped checks. Exit: An assertion fails the caller.
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
from aiohttp import ClientSession, ClientTimeout
from gateway_runtime_test import aotx_http_run, aotx_ready, aotx_http, aotx_terminal
from live_boot_test import children_of, wait
from gateway.config import aotx_load_config
from gateway.__main__ import aotx_write_grants
from gateway.errors import aotx_error
from gateway.wire import INFO, READ, aotx_packet, aotx_wire


async def aotx_revoke(test, boot, config, path, grants):
    children = children_of(boot.child.pid)
    brokers = [pid for pid in children if Path('/proc', str(pid), 'exe').resolve() == test.build/'aotx_service']
    test.check(len(brokers) == 1, 'one owned service broker')
    aotx_write_grants(config, grants)
    test.record(signal='SIGHUP', pid=brokers[0], revision=config.revision)
    os.kill(brokers[0], signal.SIGHUP)
    wire = aotx_wire(config.socket)
    principal = aotx_load_config(path).principals[0]
    try:
        async with asyncio.timeout(10):
            while True:
                try: await wire.call(principal, INFO)
                except aotx_error as error:
                    if error.status == 403: break
                    if error.status != 503: raise
                await asyncio.sleep(0.02)
        test.check(True, 'device refuses the previous grant revision')
    finally: await wire.close()


async def aotx_boundaries(test, boot, cfg, path, keys, count):
    config = aotx_load_config(path)
    headers = [{'Authorization': 'Bearer '+key} for key in keys]
    url = 'http://127.0.0.1:'+str(cfg['port'])
    gateway = aotx_http_run(test, path, 'boundary')
    try:
        async with ClientSession(timeout=ClientTimeout(total=300)) as client:
            await aotx_ready(test, gateway, client, url, headers[0])
            body = {'model': 'text', 'messages': [{'role': 'user',
                'content': 'Write the integers from 1 to 1000 in order, separated by commas.'}],
                'temperature': 0, 'max_tokens': 64, 'stream': True}
            async def disconnect(i):
                response = await client.post(url+'/v1/chat/completions', headers=headers[i], json=body)
                test.check(response.status == 200, 'stream admitted before client disconnect', principal=i)
                handle = response.headers['X-Request-ID']
                response.close()
                result = await aotx_terminal(test, client, url, headers[i], handle)
                test.check(result['state'] == 'completed' and not result['cancel_requested'],
                    'client disconnect preserves admitted inference', principal=i, result=result)
            pending = asyncio.gather(*(disconnect(i) for i in range(count)))
            boot.send('say Reply with the word blue.')
            await pending
            operator = await asyncio.to_thread(wait,
                lambda: next((row for row in boot.events(0) if row.get('kind') == 'reply'), None), boot.child, 180)
            test.check(bool(operator['text'].strip()), 'operator conversation remains available during service work', reply=operator)

            # Abandoned mailbox replies must not become replies to replacement connections.
            gateway.close()
            loop = asyncio.get_running_loop()
            abandoned = []
            try:
                for i in range(count):
                    stream = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET); stream.setblocking(False)
                    abandoned.append(stream); await loop.sock_connect(stream, config.socket)
                    await loop.sock_sendall(stream, aotx_packet(config.principals[i], READ,
                        identity=hashlib.sha256(str(i).encode()).digest()[:16], epoch=1))
            finally:
                for stream in abandoned: stream.close()
            wire = aotx_wire(config.socket)
            try:
                replies = []
                for start in range(0, count, 32):
                    replies.extend(await asyncio.gather(*(wire.call(config.principals[i], INFO)
                        for i in range(start, min(count, start+32)))))
                test.check(len(replies) == count and all(r.status == 200 and r.identity == bytes(16) for r in replies),
                    'replacement connections receive only their own mailbox replies')
            finally: await wire.close()
            gateway = aotx_http_run(test, path, 'grants')
            await aotx_ready(test, gateway, client, url, headers[0])

            request, _ = await aotx_http(test, client, 'POST', url+'/aotx/v1/requests', headers[0], 202,
                json=dict(body, stream=False, max_tokens=256))
            grants = test.output/'grants'
            await aotx_revoke(test, boot, replace(config, revision=2, principals=()), path, grants)
            for i in range(count):
                async with client.get(url+'/v1/models', headers=headers[i]) as response:
                    test.check(response.status in (403, 503), 'revoked gateway fails closed', principal=i,
                        status=response.status, body=await response.json())
            async with asyncio.timeout(10):
                while True:
                    async with client.get(url+'/v1/models', headers=headers[0]) as response:
                        if response.status == 403: break
                        test.check(response.status == 503, 'closed old transport cannot bypass revocation')
                    await asyncio.sleep(0.02)
            await aotx_http(test, client, 'GET', url+'/aotx/v1/requests/'+request['id'], headers[0], 403)

            restored = replace(config, revision=3, principals=tuple(replace(p, revision=3) for p in config.principals))
            await aotx_revoke(test, boot, restored, path, grants)
            gateway.close()
            cfg['revision'], cfg['host'] = '3', '::1'
            new_keys = ['aotx-rotated-'+hashlib.sha256(key.encode()).hexdigest() for key in keys]
            for p, key in zip(cfg['principals'], new_keys): p['token_sha256'] = [hashlib.sha256(key.encode()).hexdigest()]
            path.write_text(json.dumps(cfg))
            gateway = aotx_http_run(test, path, 'ipv6')
            url = 'http://[::1]:'+str(cfg['port'])
            new_headers = [{'Authorization': 'Bearer '+key} for key in new_keys]
            await aotx_ready(test, gateway, client, url, new_headers[0])
            await aotx_http(test, client, 'GET', url+'/v1/models', headers[0], 401)
            await aotx_http(test, client, 'GET', url+'/aotx/v1/requests/'+request['id'], new_headers[0], 404)
            await aotx_http(test, client, 'POST', url+'/aotx/v1/requests/'+request['id']+'/cancel', new_headers[0], 404, json={})
            async def fresh(i):
                value, _ = await aotx_http(test, client, 'POST', url+'/v1/chat/completions', new_headers[i],
                    json=dict(body, stream=False, max_tokens=16, messages=[{'role': 'user',
                        'content': 'Request '+str(i)+': what color is a clear daytime sky?'}]))
                test.check('blue' in value['choices'][0]['message']['content'].lower(),
                    'fresh IPv6 inference after grant and key replacement', principal=i)
            await asyncio.gather(*(fresh(i) for i in range(count)))
    finally:
        if not gateway.log.closed: gateway.close()
