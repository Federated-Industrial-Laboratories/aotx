#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check HTTP validation, output framing and transport bounds with injected device replies.
# Inputs: The gateway Python environment. Outputs: Test results. Exit: 0 pass, 1 failure.
import asyncio
import base64
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import socket
import sys
import unittest
from aiohttp import ClientSession, web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway.config import DEFAULTS, aotx_config, aotx_principal
from gateway.errors import aotx_error
from gateway.fetch import aotx_public, aotx_url, aotx_resolver
from gateway.json_wire import aotx_json
from gateway.server import aotx_server, aotx_state
from gateway.wire import INFO, METRICS, SUBMIT, READ, CANCEL, MEDIA, MEDIA_READ, aotx_reply
import ipaddress

TOKEN = 'aotx-test-credential-'+'a'*32
BODY = {'model': 'text', 'messages': [{'role': 'user', 'content': 'Return the word blue.'}], 'max_tokens': 4}


class aotx_device:
    def __init__(self):
        self.calls, self.jobs = [], {}
        self.output = 'A\u00e9\U0001f426\n'.encode()
        self.fail = False
        self.admission_error = False
        self.chunk = 1
        self.memory = 0
        self.model_role = 2
        self.control = None
        self.selection = 0

    async def call(self, principal, op, **kw):
        self.calls.append((principal.id, op, kw))
        epoch, identity, cursor = 17, kw.get('identity', bytes(16)), kw.get('cursor', 0)
        phase, finish, code, data, total = 0, 0, 0, b'', 0
        if op in (INFO, METRICS):
            data = bytearray(232)
            struct.pack_into('<16I', data, 0, 1, 64, 6144, 2048, 128, max(65536, len(self.output)),
                128, 256, 640, 64, 15, 1, 128, 0, 0, 16)
            struct.pack_into('<Q', data, 64, 33554432)
            struct.pack_into('<I', data, 156, self.memory)
            struct.pack_into('<II32s', data, 192, self.model_role, 1, b'm'*32)
            if self.control is not None:
                struct.pack_into('<I', data, 0, 2)
                struct.pack_into('<III', data, 160, len(self.control)//160, 160, self.selection)
                data.extend(self.control)
        elif op == SUBMIT:
            self.jobs[identity] = self.output
            if self.admission_error: raise aotx_error(503, 'The device connection is unavailable.', 'device_connection')
            phase = 1
        elif op in (READ, CANCEL):
            if kw['epoch'] != epoch: raise aotx_error(410, 'The runtime epoch has ended.', 'device_refused')
            if identity not in self.jobs: raise aotx_error(404, 'The resource is unavailable.', 'device_refused')
            raw = self.jobs[identity]; total = len(raw)
            if cursor > total: raise aotx_error(409, 'The output cursor is outside the result.', 'device_refused')
            data = raw[cursor:cursor+self.chunk]
            phase = 4 if cursor+len(data) == total else 3
            finish = 1 if phase == 4 else 0
            if self.fail and phase == 4: phase, finish, code = 5, 0, 503
            if op == CANCEL: phase, code, finish = 6, 409, 0
        return aotx_reply(202 if op == SUBMIT else 200, phase, epoch, identity, cursor,
            2, total, 7, 3, finish, code, int(op == CANCEL), bytes(data))


class aotx_http_tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        principal = aotx_principal(b'p'*16, 1, (hashlib.sha256(TOKEN.encode()).digest(),),
            15, ('text',), 0, 256, 64, 16, 33554432)
        config = aotx_config('/tmp/unused-service.sock', '127.0.0.1', 8081, ('https://client.example',),
            {'text': {'role': 'language', 'published_at': 1}}, (principal,), 1,
            dict(DEFAULTS, header_seconds=1, operation_seconds=5), {}, None, None)
        self.state = aotx_state(config)
        self.device = aotx_device()
        self.state.wire = self.device
        self.server = aotx_server(self.state)
        self.runner = web.ServerRunner(self.server, shutdown_timeout=1)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await self.site.start()
        host, self.port = self.runner.addresses[0]
        self.url = 'http://%s:%d' % (host, self.port)
        self.client = ClientSession(headers={'Authorization': 'Bearer '+TOKEN})

    async def asyncTearDown(self):
        await self.client.close()
        await self.runner.cleanup()
        self.assertEqual(self.state.budget.bytes, 0)
        self.assertFalse(any(self.state.budget.counts.values()))

    async def test_media_pressure_and_fixed_limits(self):
        original = self.device.call
        records, cause = {}, 0
        async def media(principal, op, **kw):
            reply = await original(principal, op, **kw)
            identity = kw.get('identity')
            if op == MEDIA:
                frame = kw['payload']; action = struct.unpack_from('<I', frame, 4)[0]
                if action == 1:
                    records[identity] = {'actor': principal.id, 'bytes': struct.unpack_from('<Q', frame, 24)[0],
                        'digest': frame[80:112], 'data': bytearray(), 'cancelled': False}
                row = records[identity]
                self.assertEqual(row['actor'], principal.id)
                if action == 2: row['data'].extend(frame[64:])
                if action == 3:
                    self.assertEqual(len(row['data']), row['bytes'])
                    self.assertEqual(hashlib.sha256(row['data']).digest(), row['digest'])
                if action == 4: row['cancelled'] = True
            if op == MEDIA_READ:
                row = records[identity]; self.assertEqual(row['actor'], principal.id)
                data = bytearray(64); data[:32] = row['digest']
                struct.pack_into('<QIIIII', data, 32, row['bytes'], 7 if cause else 6, cause, 1, 0, 4)
                reply = replace(reply, data=bytes(data))
            return reply
        self.device.call = media
        template = self.state.config.principals[0]
        keys = ['aotx-media-'+hashlib.sha256(str(i).encode()).hexdigest() for i in range(64)]
        principals = tuple(replace(template, id=(i+1).to_bytes(16, 'little'),
            hashes=(hashlib.sha256(key.encode()).digest(),)) for i, key in enumerate(keys))
        self.state.config = replace(self.state.config, principals=principals,
            limits=dict(self.state.config.limits, body_readers=64))
        for count in (1, 64):
            for cause, status, code in ((12, 429, 'media_pressure'), (2, 413, 'media_limit'),
                    (1, 400, 'media_refused'), (0, 201, None)):
                with self.subTest(count=count, cause=cause):
                    records.clear()
                    async def upload(i):
                        data = ('image-source-%d-%d-%d' % (count, cause, i)).encode()
                        async with self.client.post(self.url+'/aotx/v1/media', data=data,
                                headers={'Authorization': 'Bearer '+keys[i], 'Content-Type': 'image/jpeg'}) as response:
                            return response.status, response.headers.get('Retry-After'), await response.json()
                    results = await asyncio.gather(*(upload(i) for i in range(count)))
                    self.assertEqual(len(records), count)
                    self.assertEqual(len({r['actor'] for r in records.values()}), count)
                    self.assertTrue(all(r['cancelled'] == bool(cause) for r in records.values()))
                    for actual, delay, value in results:
                        self.assertEqual(actual, status)
                        if code: self.assertEqual(value['error']['code'], code)
                        else: self.assertEqual(value['phase'], 6)
                        if status == 429:
                            self.assertEqual(delay, '1')
                            self.assertEqual(value['error']['type'], 'rate_limit_error')

    async def test_json_and_batches(self):
        original = self.device.call
        usage = {}
        async def distinct(principal, op, **kw):
            reply = await original(principal, op, **kw)
            if op == SUBMIT:
                raw = kw['payload']
                self.assertEqual(struct.unpack_from('<5I', raw), (1, 1, 1, 0, len(raw)-20))
                self.device.jobs[kw['identity']] = principal.id.hex().encode()+b':'+raw[20:]
                usage[kw['identity']] = (len(raw), 10+int.from_bytes(principal.id, 'little'))
            if op == READ:
                prompt, sampled = usage[kw['identity']]
                return replace(reply, prompt=prompt, sampled=sampled)
            return reply
        self.device.call = distinct
        self.device.chunk = 7
        template = self.state.config.principals[0]
        keys = ['aotx-test-credential-'+hashlib.sha256(str(i).encode()).hexdigest() for i in range(64)]
        principals = tuple(replace(template, id=(i+1).to_bytes(16, 'little'),
            hashes=(hashlib.sha256(key.encode()).digest(),)) for i, key in enumerate(keys))
        self.state.config = replace(self.state.config, principals=principals)
        for n in (1, 64):
            async def one(i):
                text = 'case-'+str(i)+':'+'x'*i+'\u00e9\U0001f426'
                body = dict(BODY, messages=[{'role': 'user', 'content': text}])
                headers = {'Authorization': 'Bearer '+keys[i]}
                async with self.client.post(self.url+'/v1/chat/completions', headers=headers, json=body) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
                    result = await response.json()
                    self.assertEqual(result['choices'][0]['message']['content'], principals[i].id.hex()+':'+text)
                    prompt, sampled = 20+len(text.encode()), 11+i
                    self.assertEqual(result['usage'], {'prompt_tokens': prompt, 'completion_tokens': sampled,
                        'total_tokens': prompt+sampled})
                    return result['id']
            ids = await asyncio.gather(*(one(i) for i in range(n)))
            self.assertEqual(len(set(ids)), n)

    async def test_full_roles_and_frozen_sampling(self):
        body = dict(BODY, messages=[{'role': role, 'content': text} for role, text in
            [('system', 'Use one word.'), ('user', 'The color is amber.'), ('assistant', 'Amber.'), ('user', 'Repeat the color.')]],
            temperature=0.25, top_p=0.75)
        async with self.client.post(self.url+'/v1/chat/completions', json=body) as response:
            self.assertEqual(response.status, 200); await response.read()
        call = next(kw for _, op, kw in self.device.calls if op == SUBMIT)
        self.assertEqual((call['temperature'], call['top_p'], call['limit']), (0.25, 0.75, 4))
        p, at = call['payload'], 4
        self.assertEqual(struct.unpack_from('<I', p)[0], 4)
        for role, text in [(0, 'Use one word.'), (1, 'The color is amber.'), (2, 'Amber.'), (1, 'Repeat the color.')]:
            self.assertEqual(struct.unpack_from('<III', p, at), (role, 1, 0))
            size = struct.unpack_from('<I', p, at+12)[0]
            self.assertEqual(p[at+16:at+16+size], text.encode()); at += 16+size
        self.assertEqual(at, len(p))

    async def test_sse_utf8_usage_and_failure(self):
        for fail in (False, True):
            self.device.fail = fail
            body = dict(BODY, stream=True, stream_options={'include_usage': True})
            async with self.client.post(self.url+'/v1/chat/completions', json=body) as response:
                self.assertEqual(response.status, 200)
                raw = await response.text()
            lines = [v[6:] for v in raw.splitlines() if v.startswith('data: ')]
            done = lines[-1] == '[DONE]'
            values = [json.loads(v) for v in lines if v != '[DONE]']
            self.assertEqual(done, not fail)
            self.assertEqual(values[0]['choices'][0]['delta']['role'], 'assistant')
            content = ''.join(v['choices'][0]['delta'].get('content', '') for v in values if v.get('choices'))
            self.assertEqual(content, 'A\u00e9\U0001f426\n')
            if fail: self.assertEqual(values[-1]['error']['code'], 'request_failed')
            else:
                self.assertEqual(values[-2]['choices'][0]['finish_reason'], 'stop')
                self.assertEqual(values[-1]['choices'], [])
                self.assertEqual(values[-1]['usage']['total_tokens'], 10)

    async def test_native_control_selection(self):
        control = {'schema': 'aotx.control.selection.v1', 'kind': 'residual_vector',
            'qualification_sha256': '17'*32, 'dose': 5000}
        body = {**BODY, 'control': control}
        async with self.client.post(self.url+'/aotx/v1/requests', json=body) as response:
            self.assertEqual(response.status, 503)
        self.device.control = bytearray(160)
        struct.pack_into('<6I', self.device.control, 0, 2, 1, 0, 1, 1, 0)
        struct.pack_into('<Q', self.device.control, 24, 1)
        self.device.control[32:34] = b'v\0'
        self.device.selection = 1
        for n in (1, 64):
            for i in range(n):
                selected = {**control, 'qualification_sha256': (i+1).to_bytes(32, 'little').hex()}
                async with self.client.post(self.url+'/aotx/v1/requests', json={**BODY, 'control': selected}) as response:
                    self.assertEqual(response.status, 202)
                sent = [v[2] for v in self.device.calls if v[1] == SUBMIT][-1]['control']
                self.assertEqual(sent, struct.pack('<IIiI', 1, 1, 5000, 0)+(i+1).to_bytes(32, 'little'))
        before = sum(v[1] == SUBMIT for v in self.device.calls)
        for patch in ({'dose': 0}, {'dose': True}, {'dose': 0.5}, {'schema': 'unknown'},
                {'qualification_sha256': '00'*32}, {'extra': 1}):
            async with self.client.post(self.url+'/aotx/v1/requests', json={**BODY, 'control': {**control, **patch}}) as response:
                self.assertEqual(response.status, 400)
        async with self.client.post(self.url+'/v1/chat/completions', json=body) as response:
            self.assertEqual(response.status, 400)
        self.assertEqual(sum(v[1] == SUBMIT for v in self.device.calls), before)

    async def test_native_cursor_and_cancel(self):
        async with self.client.post(self.url+'/aotx/v1/requests', json=BODY) as response:
            self.assertEqual(response.status, 202); handle = (await response.json())['id']
        path = self.url+'/aotx/v1/requests/'+handle
        async with self.client.get(path+'/events', headers={'Last-Event-ID': handle+':2'}) as response:
            self.assertEqual(response.status, 200); raw = await response.text()
        values = [json.loads(v[6:]) for v in raw.splitlines() if v.startswith('data: ')]
        emissions = [v for v in values if v['schema'] == 'aotx.emission.v1']
        self.assertEqual(b''.join(base64.b64decode(v['bytes']) for v in emissions), self.device.output[2:])
        self.assertEqual(emissions[0]['offset'], '2')
        async with self.client.get(path+'?cursor=999') as response: self.assertEqual(response.status, 409)
        async with self.client.post(path+'/cancel', json={}) as response:
            result = await response.json(); self.assertEqual(response.status, 200)
            self.assertTrue(result['cancel_requested']); self.assertEqual(result['state'], 'cancelled')

    async def test_terminal_event_follows_all_output_pages(self):
        original = self.device.call
        self.device.output = b'x'*70000
        self.device.chunk = 65408
        phase = 4
        async def completed(principal, op, **kw):
            reply = await original(principal, op, **kw)
            if op == SUBMIT: self.device.jobs[kw['identity']] = b'x'*69984+principal.id
            if op == READ: return replace(reply, phase=phase, finish=1 if phase == 4 else 0,
                code=503 if phase == 5 else 409 if phase == 6 else 0, cancelling=int(phase == 6))
            return reply
        self.device.call = completed
        template = self.state.config.principals[0]
        keys = ['aotx-test-credential-'+hashlib.sha256(str(i).encode()).hexdigest() for i in range(64)]
        principals = tuple(replace(template, id=(i+1).to_bytes(16, 'little'),
            hashes=(hashlib.sha256(key.encode()).digest(),)) for i, key in enumerate(keys))
        self.state.config = replace(self.state.config, principals=principals)
        for phase in (4, 5, 6):
            for n in (1, 64):
                async def one(i):
                    headers = {'Authorization': 'Bearer '+keys[i]}
                    async with self.client.post(self.url+'/aotx/v1/requests', headers=headers, json=BODY) as response:
                        self.assertEqual(response.status, 202); handle = (await response.json())['id']
                    path = self.url+'/aotx/v1/requests/'+handle+'/events'
                    async with self.client.get(path, headers=headers) as response:
                        self.assertEqual(response.status, 200); raw = await response.text()
                    rows = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ')]
                    emissions = [row for row in rows if row['schema'] == 'aotx.emission.v1']
                    self.assertEqual(b''.join(base64.b64decode(row['bytes']) for row in emissions),
                        b'x'*69984+principals[i].id)
                    state = {4: 'completed', 5: 'failed', 6: 'cancelled'}[phase]
                    terminal = [row for row in rows if row.get('state') == state]
                    self.assertEqual(len(terminal), 1)
                    self.assertIs(rows[-1], terminal[0])
                    self.assertTrue(all(row.get('request_id', row.get('id')) == handle for row in rows))
                    async with self.client.get(path+'?cursor=70000', headers=headers) as response:
                        self.assertEqual(response.status, 200); raw = await response.text()
                    tail = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ')]
                    self.assertEqual(len(tail), 1); self.assertEqual(tail[0]['state'], state)
                await asyncio.gather(*(one(i) for i in range(n)))

    async def test_invalid_envelopes_have_no_admission(self):
        invalid = [dict(BODY, tools=[]), dict(BODY, temperature=True), dict(BODY, top_p=0),
            dict(BODY, max_tokens=0), dict(BODY, n=True), dict(BODY, seed=1), dict(BODY, model='missing'),
            dict(BODY, messages=[{'role': 'tool', 'content': 'x'}]), dict(BODY, stream_options={'include_usage': True}),
            dict(BODY, store=True), dict(BODY, messages=[{'role': 'user', 'content': 'x'*6145}])]
        for body in invalid:
            async with self.client.post(self.url+'/v1/chat/completions', json=body) as response:
                self.assertGreaterEqual(response.status, 400); self.assertIn('error', await response.json())
        for raw in (b'{"model":"text","model":"text"}', b'{"n":NaN}', b'{"n":1e999}', b'{"n":"\\ud800"}'):
            async with self.client.post(self.url+'/v1/chat/completions', data=raw, headers={'Content-Type': 'application/json'}) as response:
                self.assertEqual(response.status, 400)
        self.assertFalse(any(op == SUBMIT for _, op, _ in self.device.calls))

    async def test_auth_cors_and_capabilities(self):
        async with self.client.get(self.url+'/v1/models', headers={'Authorization': 'Bearer invalid'}) as response:
            self.assertEqual(response.status, 401); self.assertEqual(response.headers['WWW-Authenticate'], 'Bearer')
        self.assertFalse(self.device.calls)
        async with self.client.get(self.url+'/aotx/v1/capabilities', headers={'Origin': 'https://client.example'}) as response:
            value = await response.json(); self.assertEqual(response.status, 200)
            self.assertFalse(value['features']['continuing_ccir'])
            self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'https://client.example')
            exposed = {value.strip().lower() for value in response.headers['Access-Control-Expose-Headers'].split(',')}
            self.assertTrue({'x-request-id', 'x-aotx-media-ids', 'retry-after'} <= exposed)
        async with self.client.get(self.url+'/v1/models', headers={'Origin': 'https://foreign.example'}) as response:
            self.assertEqual(response.status, 403)
        async with self.client.options(self.url+'/v1/models', headers={'Origin': 'https://client.example',
            'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'authorization'}) as response:
            self.assertEqual(response.status, 204)

    async def test_model_memory_capability(self):
        for mask, enabled in ((0, False), (1 << 2, True), (1 << 3, False)):
            self.device.memory = mask
            async with self.client.get(self.url+'/v1/models') as response:
                value = await response.json()
                self.assertEqual(response.status, 200)
                self.assertIs(value['data'][0]['automatic_memory'], enabled)
            async with self.client.get(self.url+'/aotx/v1/capabilities') as response:
                value = await response.json()
                self.assertIs(value['models'][0]['automatic_memory'], enabled)
                self.assertEqual(value['models'][0]['input'], ['text'])
        self.device.model_role = 0xffffffff
        async with self.client.get(self.url+'/v1/models') as response:
            self.assertEqual((await response.json())['data'], [])

    async def test_control_capabilities(self):
        for count in (1, 64):
            for batch in range(count):
                controls = bytearray()
                expected = []
                for i in range(1 if count == 1 else 16):
                    control = bytearray(160)
                    struct.pack_into('<6IQ', control, 0, 2, 1, 1, 1, 1, 2, 1 << ((batch+i) % 32))
                    name = ('trait-'+str(i)).encode()
                    control[32:32+len(name)] = name
                    control[64:96] = hashlib.sha256(bytes((batch, i))).digest()
                    struct.pack_into('<2i', control, 96, 5000+batch+i, 10000+batch+i)
                    controls.extend(control)
                    expected.append((name.decode(), [5000+batch+i, 10000+batch+i],
                        [(batch+i) % 32], control[64:96].hex()))
                self.device.control = controls
                async with self.client.get(self.url+'/aotx/v1/capabilities') as response:
                    value = await response.json(); self.assertEqual(response.status, 200)
                    rows = value['models'][0]['controls']; self.assertEqual(len(rows), len(expected))
                    for row, (name, doses, layers, digest) in zip(rows, expected):
                        self.assertEqual(row['name'], name); self.assertTrue(row['available'])
                        self.assertEqual(row['accepted_doses'], doses); self.assertEqual(row['layers'], layers)
                        self.assertEqual(row['qualification_sha256'], digest); self.assertFalse(row['combinations'])
                last = len(controls)-160
                for offset, fmt, value in ((0, '<I', 3), (4, '<I', 4), (8, '<I', 2), (12, '<I', 2),
                        (20, '<I', 17), (100, '<i', 5000+batch+len(expected)-1), (100, '<i', 40001), (104, '<i', 1)):
                    self.device.control = bytearray(controls)
                    struct.pack_into(fmt, self.device.control, last+offset, value)
                    async with self.client.get(self.url+'/aotx/v1/capabilities') as response:
                        self.assertEqual(response.status, 503)
                if len(expected) > 1:
                    self.device.control = bytearray(controls)
                    self.device.control[last+32:last+64] = controls[32:64]
                    async with self.client.get(self.url+'/aotx/v1/capabilities') as response:
                        self.assertEqual(response.status, 503)
                self.device.control = bytearray(controls)
                struct.pack_into('<I', self.device.control, last+8, 0)
                struct.pack_into('<I', self.device.control, last+20, 0)
                self.device.control[last+96:] = bytes(64)
                async with self.client.get(self.url+'/aotx/v1/capabilities') as response:
                    value = await response.json(); self.assertEqual(response.status, 200)
                    rows = value['models'][0]['controls']; self.assertEqual(len(rows), len(expected))
                    self.assertFalse(rows[-1]['available']); self.assertEqual(rows[-1]['accepted_doses'], [])

    async def test_header_timeout_and_expect(self):
        reader, writer = await asyncio.open_connection('127.0.0.1', self.port)
        writer.write(b'GET /v1/models HTTP/1.1\r\nHost: test\r\n'); await writer.drain()
        self.assertEqual(await asyncio.wait_for(reader.read(), 3), b'')
        writer.close(); await writer.wait_closed()
        async with self.client.post(self.url+'/v1/chat/completions', json=BODY, expect100=True) as response:
            self.assertEqual(response.status, 200); await response.read()

    async def test_capacity_recovers(self):
        async with self.state.budget.claim('body_readers', self.state.config.limits['body_credit']):
            async with self.client.post(self.url+'/v1/chat/completions', json=BODY) as response:
                self.assertEqual(response.status, 429)
        async with self.client.post(self.url+'/v1/chat/completions', json=BODY) as response:
            self.assertEqual(response.status, 200); await response.read()

    async def test_uncertain_admission_retains_its_identity(self):
        self.device.admission_error = True
        async with self.client.post(self.url+'/v1/chat/completions', json=BODY) as response:
            self.assertEqual(response.status, 503)
            handle = response.headers['X-Request-ID']
        self.device.admission_error = False
        async with self.client.get(self.url+'/aotx/v1/requests/'+handle) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())['id'], handle)

    async def test_slow_reader_releases_transport_capacity(self):
        self.device.output = b'x'*(16*1024*1024)
        self.device.chunk = 65408
        self.state.config.limits['write_seconds'] = 1
        async with self.client.post(self.url+'/aotx/v1/requests', json=BODY) as response:
            self.assertEqual(response.status, 202); handle = (await response.json())['id']
        stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        stream.setblocking(False)
        loop = asyncio.get_running_loop()
        await loop.sock_connect(stream, ('127.0.0.1', self.port))
        try:
            packet = ('GET /aotx/v1/requests/'+handle+'/events HTTP/1.1\r\nHost: test\r\n'
                'Authorization: Bearer '+TOKEN+'\r\n\r\n').encode()
            await loop.sock_sendall(stream, packet)
            async with asyncio.timeout(3):
                while not self.state.budget.counts.get('work'): await asyncio.sleep(0.01)
            async with asyncio.timeout(5):
                while self.state.budget.counts.get('work'): await asyncio.sleep(0.05)
            reads = [kw for _, op, kw in self.device.calls if op == READ]
            self.assertTrue(reads)
            self.assertLess(max(kw['cursor'] for kw in reads), len(self.device.output)-65408)
            self.assertFalse(any(op == CANCEL for _, op, _ in self.device.calls))
        finally: stream.close()
        async with self.client.get(self.url+'/aotx/v1/requests/'+handle) as response:
            self.assertEqual(response.status, 200)

    async def test_partial_connections_are_bounded(self):
        self.state.config.limits['connections'] = 2
        pairs = [await asyncio.open_connection('127.0.0.1', self.port) for _ in range(3)]
        try:
            for _, writer in pairs:
                writer.write(b'GET /v1/models HTTP/1.1\r\n'); await writer.drain()
            self.assertEqual(await asyncio.wait_for(pairs[-1][0].read(), 0.5), b'')
            self.assertLessEqual(len(self.server.transports), 2)
            for reader, _ in pairs[:2]: self.assertEqual(await asyncio.wait_for(reader.read(), 3), b'')
        finally:
            for _, writer in pairs: writer.close()
            for _, writer in pairs: await writer.wait_closed()
        async with self.client.get(self.url+'/v1/models') as response:
            self.assertEqual(response.status, 200)

    async def test_slow_json_reader_releases_connection(self):
        self.device.output = b'x'*(16*1024*1024)
        self.device.chunk = 65408
        self.state.config.limits['write_seconds'] = 1
        stream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        stream.setblocking(False)
        loop = asyncio.get_running_loop()
        await loop.sock_connect(stream, ('127.0.0.1', self.port))
        try:
            data = json.dumps(BODY).encode()
            header = ('POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n'
                'Authorization: Bearer '+TOKEN+'\r\nContent-Type: application/json\r\n'
                'Content-Length: '+str(len(data))+'\r\n\r\n').encode()
            await loop.sock_sendall(stream, header+data)
            async with asyncio.timeout(3):
                while not self.device.jobs: await asyncio.sleep(0.01)
            async with asyncio.timeout(5):
                while self.server.transports: await asyncio.sleep(0.05)
            self.assertEqual(len(self.device.jobs), 1)
            self.assertFalse(any(op == CANCEL for _, op, _ in self.device.calls))
        finally: stream.close()


class aotx_fetch_tests(unittest.IsolatedAsyncioTestCase):
    async def test_addresses_and_pinning(self):
        for value in ('127.0.0.1', '10.0.0.1', '100.64.0.1', '192.0.2.3', '224.1.2.3', '::1',
            '::ffff:127.0.0.1', '2001:db8::1', '2002:0808:0808::1', 'fc00::1', 'fe80::1'):
            self.assertFalse(aotx_public(ipaddress.ip_address(value)), value)
        for value in ('8.8.8.8', '2606:4700:4700::1111'):
            self.assertTrue(aotx_public(ipaddress.ip_address(value)), value)
        for value in ('http://example.com', 'file:///x', 'https://user:pass@example.com',
            'https://127.1', 'https://2130706433', 'https://0177.0.0.1', 'https://example.com/#part',
            'https://example.com./x', 'https://[fe80::1%25eth0]/'):
            with self.assertRaises(aotx_error): aotx_url(value)
        resolver = aotx_resolver('feed.example', 443, [ipaddress.ip_address('8.8.8.8')])
        self.assertEqual((await resolver.resolve('feed.example', 443))[0]['host'], '8.8.8.8')
        with self.assertRaises(OSError): await resolver.resolve('feed.example', 444)
        with self.assertRaises(OSError): await resolver.resolve('foreign.example', 443)


if __name__ == '__main__': unittest.main(verbosity=2)
