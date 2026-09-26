#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check policy HTTP routes, explicit grants and exact native control translation.
# Inputs: Gateway Python environment. Outputs: Test results. Exit: 0 pass, 1 failure.
from dataclasses import replace
import struct
import unittest
import gateway_protocol_test as protocol
from gateway.errors import aotx_error
from gateway.wire import aotx_reply


class PolicyRoutes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await protocol.aotx_http_tests.asyncSetUp(self)
        self.calls = []
        self.device_status = 200
        async def policy(principal, op, **kwargs):
            self.calls.append((principal, op, kwargs))
            if self.device_status != 200:
                raise aotx_error(self.device_status, 'The device refused the request.', 'device_refused')
            p = bytearray(160)
            struct.pack_into('<8I', p, 0, 1, 3, 2, 1, 1, 0, 0, 0)
            struct.pack_into('<11Q', p, 32, 42, 81, 64, 3, 0, 4, 6, 7000, 6000, 0, 0)
            return aotx_reply(200, 1, 71, bytes(16), 0, 0, 0, 0, 0, 0, 0, 0, bytes(p))
        self.device.call = policy

    async def asyncTearDown(self):
        await protocol.aotx_http_tests.asyncTearDown(self)

    def grant(self, actions):
        config = self.state.config
        self.state.config = replace(config, principals=(replace(config.principals[0], actions=actions),))

    async def test_read_grants(self):
        for actions, status in ((1, 403), (64, 403), (8, 200), (128, 200)):
            self.grant(actions)
            before = len(self.calls)
            async with self.client.get(self.url+'/aotx/v1/policy') as r:
                self.assertEqual(r.status, status)
                if status == 200:
                    p = await r.json()
                    self.assertEqual((p['schema'], p['control_revision'], p['completed']), ('aotx.policy.v1', 42, 64))
                    self.assertNotIn('sources', p)
            self.assertEqual(len(self.calls)-before, int(status == 200))

    async def test_operator_and_translation(self):
        body = dict(action='review_on', epoch=71, control_revision=41)
        for actions in (1, 8, 64):
            self.grant(actions)
            async with self.client.post(self.url+'/aotx/v1/policy', json=body) as r:
                self.assertEqual(r.status, 403)
        self.assertFalse(self.calls)
        self.grant(128)
        for action, value in (('pause', 1), ('resume', 2), ('stop', 3), ('review_on', 4), ('review_off', 5)):
            async with self.client.post(self.url+'/aotx/v1/policy', json=dict(body, action=action)) as r:
                self.assertEqual(r.status, 200)
            _, op, args = self.calls[-1]
            self.assertEqual((op, args), (12, dict(epoch=71, payload=struct.pack('<IIQ', 1, value, 41))))

    async def test_rejected_envelopes_and_device_conflicts(self):
        self.grant(128)
        body = dict(action='pause', epoch=71, control_revision=41)
        for invalid in (dict(body, extra=1), dict(body, epoch=True), dict(body, control_revision=-1),
                        dict(body, epoch=2**64), dict(body, action='run'), {'action': 'pause'}):
            async with self.client.post(self.url+'/aotx/v1/policy', json=invalid) as r:
                self.assertEqual(r.status, 400)
        self.assertFalse(self.calls)
        for status in (409, 410, 429):
            self.device_status = status
            async with self.client.post(self.url+'/aotx/v1/policy', json=body) as r:
                self.assertEqual(r.status, status)
        async with self.client.get(self.url+'/aotx/v1/policy?source=private') as r:
            self.assertEqual(r.status, 400)


if __name__ == '__main__': unittest.main(verbosity=2)
