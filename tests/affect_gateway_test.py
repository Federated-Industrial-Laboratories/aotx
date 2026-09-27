#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check affect setting grants, scaled values and strict operator envelopes.
# Inputs: Gateway Python environment. Outputs: Test results. Exit: 0 pass, 1 failure.
from dataclasses import replace
import struct
import unittest
import gateway_protocol_test as protocol
from gateway.affect import KEYS, aotx_affect_decode
from gateway.errors import aotx_error
from gateway.wire import aotx_reply


class AffectRoutes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await protocol.aotx_http_tests.asyncSetUp(self)
        self.calls = []
        self.device_status = 200
        async def affect(principal, op, **kwargs):
            self.calls.append((principal, op, kwargs))
            if self.device_status != 200: raise aotx_error(self.device_status, 'The device refused the request.', 'device_refused')
            p = bytearray(32+13*64); struct.pack_into('<IIQII',p,0,1,13,2**53+7,int(bool(principal.actions&256)),0)
            for i,key in enumerate(KEYS):
                p[32+i*64:64+i*64] = key.encode().ljust(32,b'\0')
                struct.pack_into('<qqqI',p,64+i*64,i,0,40000,10000)
            return aotx_reply(200,0,71,bytes(16),0,0,0,0,0,0,0,0,bytes(p))
        self.device.call = affect

    async def asyncTearDown(self):
        await protocol.aotx_http_tests.asyncTearDown(self)

    def grant(self, actions):
        config = self.state.config
        self.state.config = replace(config,principals=(replace(config.principals[0],actions=actions),))

    def body(self):
        return dict(schema='aotx.affect.settings.mutation.v1',epoch='71',revision=str(2**53+7),key='affect.decay_fast',value=3210,scale=10000)

    async def test_read_and_write_grant_isolation(self):
        for actions in (1,8,16,32,64,128,256):
            self.grant(actions); before=len(self.calls)
            async with self.client.get(self.url+'/aotx/v1/affect/settings') as r:
                self.assertEqual(r.status,200 if actions in (8,256) else 403)
                if r.status==200:
                    value=await r.json(); self.assertEqual(value['revision'],str(2**53+7))
                    self.assertEqual(len(value['settings']),13); self.assertFalse(value['paths']['ordinary_http'])
                    self.assertEqual(value['writable'],actions==256)
            async with self.client.post(self.url+'/aotx/v1/affect/settings',json=self.body()) as r:
                self.assertEqual(r.status,200 if actions==256 else 403)
            self.assertEqual(len(self.calls)-before,int(actions in (8,256))+int(actions==256))

    async def test_distinct_exact_mutation_batches(self):
        self.grant(256)
        for n in (1,64):
            for i in range(n):
                body=dict(self.body(),value=i-20,revision=str(2**53+i))
                async with self.client.post(self.url+'/aotx/v1/affect/settings',json=body) as r: self.assertEqual(r.status,200)
                _,op,args=self.calls[-1]; self.assertEqual(op,13)
                self.assertEqual(args,dict(epoch=71,payload=struct.pack('<IIQqII64s',1,17,2**53+i,i-20,10000,0,b'affect.decay_fast')))

    async def test_invalid_envelopes_and_conflicts(self):
        self.grant(256); body=self.body()
        for patch in ({'extra':1},{'epoch':71},{'revision':'01'},{'epoch':'0'},{'revision':str(2**64)},
                {'value':True},{'value':1.5},{'scale':99},{'key':'sample.temperature'},{'schema':'future'}):
            async with self.client.post(self.url+'/aotx/v1/affect/settings',json=dict(body,**patch)) as r: self.assertEqual(r.status,400)
        self.assertFalse(self.calls)
        for status in (409,410,429,501):
            self.device_status=status
            async with self.client.post(self.url+'/aotx/v1/affect/settings',json=body) as r: self.assertEqual(r.status,status)
        async with self.client.get(self.url+'/aotx/v1/affect/settings?secret=x') as r: self.assertEqual(r.status,400)
        async with self.client.post(self.url+'/aotx/v1/affect/settings',data='{"schema":"x","schema":"y"}',headers={'Content-Type':'application/json'}) as r: self.assertEqual(r.status,400)


if __name__=='__main__': unittest.main(verbosity=2)
