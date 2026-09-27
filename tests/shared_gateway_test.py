# SPDX-License-Identifier: Apache-2.0
# Check canonical shared bytes and exact response offsets; inputs are distinct fixtures, output is test counts, failure exits nonzero.
import base64
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import struct
import json
from unittest.mock import patch
import unittest
from types import SimpleNamespace
from gateway.errors import aotx_error
from gateway.shared_output import aotx_shared_decode
from gateway.shared_wire import (INPUT, MEMBER, OPERATION, REGISTER, SPACE, AFFECT, CONVERSATION, PROMPT, CAPABILITIES, aotx_shared_counter,
    aotx_shared_handle, aotx_shared_mutation, aotx_shared_parse, aotx_shared_read_frame)


class aotx_shared_tests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(config=SimpleNamespace(models={'test': {'role': 'language'}}))
        self.principal = SimpleNamespace(models=('test',), tokens=512, pages=64)
        self.lineage = bytes.fromhex('ab'*16)

    def body(self, i):
        return {'schema': 'aotx.shared.mutation.v1', 'lineage': self.lineage.hex(),
            'operation_key': (i+1).to_bytes(16, 'little').hex(), 'sequence': str(2**53+i+1)}

    def test_distinct_canonical_batches(self):
        for n in (1, 64):
            payloads = []
            for i in range(n):
                body = {**self.body(i), 'text': 'item %d \u00e9 \U0001f642' % i, 'model': 'test', 'temperature': 0.5}
                payload = aotx_shared_mutation(self.state, self.principal, body, INPUT, target=(i+70).to_bytes(16, 'little'))
                self.assertEqual(payload, aotx_shared_mutation(self.state, self.principal, dict(reversed(list(body.items()))), INPUT,
                    target=(i+70).to_bytes(16, 'little')))
                self.assertEqual(struct.unpack_from('<Q', payload, 16)[0], 2**53+i+1)
                self.assertEqual(payload[192:], body['text'].encode())
                self.assertEqual(struct.unpack_from('<I', payload, 136)[0], len(body['text'].encode()))
                changed = aotx_shared_mutation(self.state, self.principal, {**body, 'text': body['text']+'x'}, INPUT,
                    target=(i+70).to_bytes(16, 'little'))
                self.assertNotEqual(payload, changed); payloads.append(payload)
            self.assertEqual(len(set(payloads)), n)

    def test_conversation_prompt_bytes_and_bounds(self):
        for n in (1, 64):
            for i in range(n):
                text = 'Neutral %d \u00e9' % i
                base = self.body(i)
                old = aotx_shared_mutation(self.state, self.principal, base, CONVERSATION)
                explicit = aotx_shared_mutation(self.state, self.principal, dict(base, system_prompt=text), CONVERSATION)
                blank = aotx_shared_mutation(self.state, self.principal, dict(base, system_prompt=''), CONVERSATION)
                self.assertEqual(old[144:192], bytes(48))
                self.assertEqual(struct.unpack_from('<II', explicit, 144), (1, 1))
                self.assertEqual(explicit[192:], text.encode())
                self.assertNotEqual(blank, old)
        for text in (None, True, '\0', '\ud800', '\u00e9'*1025):
            with self.assertRaises(aotx_error): aotx_shared_mutation(self.state, self.principal,
                dict(self.body(0), system_prompt=text), CONVERSATION)
        value = aotx_shared_mutation(self.state, self.principal, dict(self.body(0), system_prompt='\u00e9'*1024), CONVERSATION)
        self.assertEqual(len(value), 192+2048)
        with self.assertRaises(aotx_error): aotx_shared_mutation(self.state, self.principal,
            dict(self.body(0), system_prompt='x'), REGISTER)

    def test_prompt_read_and_old_capability_decode(self):
        def reply(kind, data):
            p = bytearray(320); p[:8] = b'AOTXSHR1'; p[16:32] = self.lineage
            p[32:48] = b'a'*16; p[48:64] = b'b'*16
            struct.pack_into('<I', p, 8, kind); struct.pack_into('<II', p, 192, 1, len(data))
            return SimpleNamespace(data=bytes(p)+data)
        for mode, text in ((0, None), (1, ''), (1, 'Prompt \u00e9')):
            raw = (text or '').encode(); data = struct.pack('<4I', 1, mode, len(raw), 0)+raw
            got = aotx_shared_decode(reply(PROMPT, data))
            self.assertEqual(got['prompt']['system_prompt'], text)
            self.assertEqual(got['prompt']['mode'], 'explicit' if mode else 'runtime')
            for bad in (struct.pack('<4I', 2, mode, len(raw), 0)+raw, data+b'x', struct.pack('<4I',1,1,1,0)+b'\0'):
                with self.assertRaises(aotx_error): aotx_shared_decode(reply(PROMPT,bad))
        for version in (0, 1):
            data = struct.pack('<10I', *range(1, 11)) + (struct.pack('<II',1,2048) if version else b'')
            got = aotx_shared_decode(reply(CAPABILITIES,data))
            self.assertEqual(got['features']['conversation_prompt'], bool(version))
            self.assertEqual(got['limits']['system_prompt_bytes'], 2048 if version else 0)

    def test_selected_control_bytes(self):
        from gateway.controls import aotx_control_selection
        from gateway.wire import aotx_packet, SUBMIT
        for n in (1, 64):
            for i in range(n):
                body = {**self.body(i), 'text': 'control %d' % i, 'model': 'test'}
                old = aotx_shared_mutation(self.state, self.principal, body, INPUT)
                control = {'schema': 'aotx.control.selection.v1', 'kind': 'residual_vector',
                    'qualification_sha256': (i+1).to_bytes(32, 'little').hex(), 'dose': 5000 if i % 2 else -10000}
                raw = aotx_control_selection(control)
                current = aotx_shared_mutation(self.state, self.principal, {**body, 'control': control}, INPUT)
                self.assertEqual(old[144:192], bytes(48))
                self.assertEqual(current[:144]+current[192:], old[:144]+old[192:])
                self.assertEqual(current[144:192], raw)
                principal = SimpleNamespace(id=(i+1).to_bytes(16, 'little'), revision=1)
                packet = aotx_packet(principal, SUBMIT, payload=b'text', control=raw)
                self.assertEqual(struct.unpack_from('<II', packet, 88), (52, 48))
                self.assertEqual(packet[128:], b'text'+raw)
                for patch in ({'dose': 0}, {'dose': 50001}, {'dose': True}, {'dose': 0.5},
                        {'schema': 'other'}, {'kind': 'probe'}, {'qualification_sha256': '00'*32}, {'extra': 1}):
                    with self.assertRaises(aotx_error): aotx_control_selection({**control, **patch})
                with self.assertRaises(aotx_error): aotx_shared_mutation(self.state, self.principal,
                    {**self.body(i), 'control': control}, REGISTER)

    def test_strict_envelopes(self):
        for value in ('01', '-1', '1.0', '18446744073709551616', 1, True):
            with self.assertRaises(aotx_error): aotx_shared_counter(value, 'sequence')
        for fields in ({'extra': 1}, {'schema': 'other'}, {'operation_key': '00'*16}, {'sequence': '0'}):
            with self.assertRaises(aotx_error): aotx_shared_mutation(self.state, self.principal, {**self.body(0), **fields}, REGISTER)
        for text in ('\x00', '\ud800', 'x'*2049):
            with self.assertRaises(aotx_error): aotx_shared_mutation(self.state, self.principal,
                {**self.body(0), 'text': text, 'model': 'test'}, INPUT)
        with self.assertRaises(aotx_error): aotx_shared_mutation(self.state, self.principal,
            {**self.body(0), 'participant': '11'*16, 'permissions': ['read', 'read']}, MEMBER)

    def test_scope_and_persistent_identity(self):
        for i in range(64):
            actor, key = (i+1).to_bytes(16, 'little'), (i+700).to_bytes(16, 'little')
            for kind in ('op', 'spc', 'con'):
                handle = aotx_shared_handle(kind, self.lineage, key)
                self.assertEqual(aotx_shared_parse(handle, kind), (self.lineage, key, bytes(16)))
            body = self.body(i)
            payload = aotx_shared_mutation(self.state, self.principal, body, SPACE)
            self.assertEqual(payload[56:72], bytes.fromhex(body['operation_key']))
            self.assertEqual(struct.unpack_from('<I', payload, 12)[0], 0)
        with self.assertRaises(aotx_error): aotx_shared_parse('op-'+'00'*16+'-'+'11'*16+'-'+'22'*16, 'op')

    def test_exact_output_and_saved_fields(self):
        for n in (1, 64):
            for i in range(n):
                raw = ('answer %d \u00e9' % i).encode()
                p = bytearray(320+len(raw)); p[:8] = b'AOTXSHR1'; p[16:32] = self.lineage
                p[64:80] = (i+1).to_bytes(16, 'little'); p[240:256] = (i+4).to_bytes(16, 'little'); p[296:312] = (i+900).to_bytes(16, 'little')
                struct.pack_into('<II', p, 8, OPERATION, 4); struct.pack_into('<Q', p, 80, 2**53+i)
                struct.pack_into('<II', p, 168, 200, 7); struct.pack_into('<I', p, 176, len(raw)+5)
                struct.pack_into('<Q', p, 208, 5); p[320:] = raw
                value = aotx_shared_decode(SimpleNamespace(data=bytes(p)))
                self.assertEqual(base64.b64decode(value['output']['base64']), raw)
                self.assertEqual(value['next_offset'], str(5+len(raw)))
                self.assertEqual(value['sequence'], str(2**53+i)); self.assertTrue(value['saved_terminal'])
                p[172] = 1; value = aotx_shared_decode(SimpleNamespace(data=bytes(p)))
                self.assertTrue(value['device_committed']); self.assertFalse(value['saved_admission'])
                p[320:] = b'\xc3' + raw[1:]
                value = aotx_shared_decode(SimpleNamespace(data=bytes(p)))
                self.assertIsNone(value['output']['text']); self.assertEqual(value['output']['bytes'], str(len(raw)))

    def test_scoped_affect_state_and_unavailable_probes(self):
        for n in (1, 64):
            for i in range(n):
                p = bytearray(416); p[:8] = b'AOTXSHR1'; p[16:32] = self.lineage
                p[32:48] = (i+1).to_bytes(16, 'little'); p[48:64] = (i+90).to_bytes(16, 'little')
                struct.pack_into('<I', p, 8, AFFECT); struct.pack_into('<II', p, 192, 1, 96)
                struct.pack_into('<IIQ', p, 320, 1, 1, 2**53+i)
                struct.pack_into('<4h4hHHIfIII', p, 336, i, -i, 0, 0, i+1, -i-1, 0, 0,
                    65535, 2, 0, 0.0, 1, i % 4, 2)
                p[376:408] = b'a'*32
                value = aotx_shared_decode(SimpleNamespace(data=bytes(p)))['affect']
                self.assertEqual(value['revision'], str(2**53+i))
                self.assertEqual(value['fast_q15'], [i, -i, 0, 0])
                self.assertEqual(value['probes_at_last_turn']['valence'], 'available' if i % 4 & 1 else 'unavailable')
                for offset, raw in ((320, b'\x02'), (324, b'\x02'), (354, b'\x03'),
                        (360, struct.pack('<f', float('nan'))), (368, b'\x04'), (408, b'\x01')):
                    bad = bytearray(p); bad[offset:offset+len(raw)] = raw
                    with self.assertRaises(aotx_error): aotx_shared_decode(SimpleNamespace(data=bytes(bad)))
                wire = aotx_shared_read_frame(AFFECT, self.lineage, p[32:48])
                self.assertEqual(struct.unpack_from('<I', wire, 8)[0], AFFECT)

    def test_read_bounds(self):
        p = aotx_shared_read_frame(OPERATION, self.lineage, b'a'*16, b'b'*16, byte=2**53+4)
        self.assertEqual(len(p), 96); self.assertEqual(struct.unpack_from('<Q', p, 72)[0], 2**53+4)
        for size in (0, 257):
            with self.assertRaises(aotx_error): aotx_shared_read_frame(OPERATION, limit=size)
        with self.assertRaises(aotx_error): aotx_shared_decode(SimpleNamespace(data=b'AOTXSHR1'))


class aotx_shared_stream_tests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_resume_follows_unsaved_inputs(self):
        from gateway.shared import aotx_shared_stream
        from gateway.shared_wire import EVENTS
        calls, writes = [], []
        class response:
            def force_close(self): pass
            async def prepare(self, request): return None
            async def write(self, data): writes.append(data)
            async def write_eof(self): return None
        async def read(state, principal, kind, **args):
            calls.append(args['cursor'])
            items = [] if len(calls) > 3 else [{'input_order': '1', 'saved_terminal': len(calls) == 3,
                'state': 'running' if len(calls) == 1 else 'completed'}]
            return {'items': items, 'next_cursor': '2'}
        async def pause(seconds): return None
        state = SimpleNamespace(config=SimpleNamespace(limits={'event_bytes': 4096, 'write_seconds': 1, 'operation_seconds': 30}))
        server, request = SimpleNamespace(state=state), SimpleNamespace(path='/aotx/v1/shared/conversations/con-test/events')
        with patch('gateway.shared.web.StreamResponse', return_value=response()), patch('gateway.shared.aotx_shared_read', side_effect=read), \
            patch('gateway.shared.asyncio.sleep', new=pause), patch('gateway.shared.time.monotonic', side_effect=lambda: 31 if len(calls) > 3 else 0):
            await aotx_shared_stream(server, request, None, {}, EVENTS, b'a'*16, b'b'*16, bytes(16), 0, 0)
        self.assertEqual(calls, [0, 1, 1, 2])
        self.assertEqual([data.split(b'\n', 1)[0].rsplit(b':', 1)[-1] for data in writes], [b'1', b'1', b'2', b'2'])
        values = [json.loads(data.split(b'data: ', 1)[1]) for data in writes]
        self.assertEqual([value['items'][0]['saved_terminal'] for value in values[:3]], [False, False, True])

    async def test_saved_terminal_drains_every_reply_chunk(self):
        from gateway.shared import aotx_shared_stream
        from gateway.shared_output import aotx_shared_bytes
        from gateway.shared_wire import OPERATION
        for event_limit in (262144, 4096):
            raw = (b'answer: ' + bytes(range(1, 128))) * 520
            calls, writes = [], []
            class response:
                def force_close(self): pass
                async def prepare(self, request): return None
                async def write(self, data): writes.append(data)
                async def write_eof(self): return None
            async def read(state, principal, kind, **args):
                byte = args['byte']; calls.append(byte)
                data = raw[byte:byte+60000]
                return {'schema': 'aotx.shared.resource.v1', 'state': 'completed', 'saved_terminal': True,
                    'offset': str(byte), 'next_offset': str(byte+len(data)), 'output_bytes': str(len(raw)),
                    'output': aotx_shared_bytes(data)}
            async def pause(seconds): return None
            state = SimpleNamespace(config=SimpleNamespace(limits={'event_bytes': event_limit, 'write_seconds': 1, 'operation_seconds': 30}))
            server, request = SimpleNamespace(state=state), SimpleNamespace(path='/aotx/v1/shared/operations/op-test/events')
            with patch('gateway.shared.web.StreamResponse', return_value=response()), patch('gateway.shared.aotx_shared_read', side_effect=read), \
                patch('gateway.shared.asyncio.sleep', new=pause):
                await aotx_shared_stream(server, request, None, {}, OPERATION, b'a'*16, b'b'*16, bytes(16), 0, 0)
            values = [json.loads(data.split(b'data: ', 1)[1]) for data in writes if b'data: ' in data]
            self.assertGreater(len(calls), 1)
            self.assertEqual(b''.join(base64.b64decode(value['output']['base64']) for value in values), raw)
            self.assertEqual(int(values[-1]['next_offset']), len(raw))
            self.assertTrue(all(len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()) <= event_limit for value in values))


if __name__ == '__main__': unittest.main()
