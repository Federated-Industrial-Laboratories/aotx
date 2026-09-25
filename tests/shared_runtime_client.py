# SPDX-License-Identifier: Apache-2.0
# Check native shared resources against a running device; inputs are distinct clients, outputs are exact receipts and checks.
import asyncio
import base64
import hashlib
import math
import time

PREFIX = '/aotx/v1/shared'
TERMINAL = ('completed', 'failed', 'cancelled', 'interrupted')


class aotx_shared_client:
    def __init__(self, test, client, url, key, participant, work_seconds=900):
        if not math.isfinite(work_seconds) or work_seconds <= 0:
            raise ValueError('The work deadline must be positive and finite.')
        self.test, self.client, self.url = test, client, url
        self.work_seconds = work_seconds
        self.headers = {'Authorization': 'Bearer ' + key}
        self.participant = participant
        self.sequence, self.lineage = 1, None
        self.pending = None

    async def request(self, method, path, expected=(200,), audit=True, deadline=None, **kw):
        end = time.monotonic() + self.work_seconds
        if deadline is not None:
            end = min(end, deadline)
        while True:
            async with self.client.request(method, self.url + PREFIX + path, headers=self.headers, **kw) as response:
                value = await response.json()
                status = response.status
            if status == 429 and status not in expected and time.monotonic() < end:
                await asyncio.sleep(0.1)
                continue
            if audit or status not in expected:
                self.test.check(status in expected, 'native shared HTTP status', method=method, path=path,
                    status=status, expected=expected, response=value)
            return value

    async def discover(self):
        value = await self.request('GET', '/participant')
        self.lineage, self.sequence = value['lineage'], int(value['next_sequence'])
        self.test.check(value['participant'] == self.participant, 'authenticated participant identity')
        return value

    def command(self, **fields):
        key = hashlib.sha256((self.participant + ':' + str(self.sequence)).encode()).hexdigest()[:32]
        return dict(schema='aotx.shared.mutation.v1', lineage=self.lineage,
            operation_key=key, sequence=str(self.sequence), **fields)

    async def mutate(self, path, **fields):
        body = self.command(**fields)
        self.pending = (path, body)
        value = await self.request('POST', path, (202, 200), json=body)
        self.test.check(value['accepted'] and value['sequence'] == str(self.sequence),
            'accepted operation retains its exact sequence', receipt=value)
        self.sequence += 1
        self.pending = None
        return value, body

    async def terminal(self, identity, saved=True):
        end = time.monotonic() + self.work_seconds
        data, offset, polls = bytearray(), 0, 0
        while time.monotonic() < end:
            value = await self.request('GET', '/operations/' + identity + '?offset=' + str(offset), audit=False, deadline=end)
            part = base64.b64decode(value['output']['base64'], validate=True)
            valid = value['id'] == identity and int(value['offset']) == offset and int(value['next_offset']) == offset + len(part)
            if not valid: self.test.check(False, 'exact operation byte cursor', receipt=value)
            polls += 1
            data.extend(part); offset += len(part)
            if value['state'] in TERMINAL and (not saved or value['saved_terminal']) and offset == int(value['output_bytes']):
                self.test.check(True, 'exact operation byte cursors', polls=polls, bytes=offset)
                value['exact_output'] = base64.b64encode(data).decode()
                self.test.check(not saved or value['saved_admission'] and int(value['save']['generation']) > 0 and
                    int(value['save']['boot']) > 0 and any(bytes.fromhex(value['save']['commit_sha256'])),
                    'saved receipt carries an actual complete file cut', receipt=value)
                return value
            await asyncio.sleep(0.1)
        raise TimeoutError('The shared operation did not reach its required terminal state.')

    async def done(self, path, **fields):
        value, body = await self.mutate(path, **fields)
        final = await self.terminal(value['id'])
        self.test.check(final['state'] == 'completed' and final['status'] == 200,
            'shared mutation completes and is saved', receipt=final)
        return final, body

    async def create(self, scope='private'):
        space, _ = await self.done('/spaces', scope=scope)
        identity = 'spc-' + self.lineage + '-' + space['resource']
        conversation, _ = await self.done('/spaces/' + identity + '/conversations')
        return identity, 'con-' + self.lineage + '-' + conversation['resource']

    async def input(self, conversation, text, **fields):
        return await self.mutate('/conversations/' + conversation + '/inputs',
            text=text, model=fields.pop('model', 'text'), max_output_tokens=fields.pop('max_output_tokens', 32), **fields)

    async def answer(self, conversation, text, **fields):
        receipt, body = await self.input(conversation, text, **fields)
        final = await self.terminal(receipt['id'])
        self.test.check(final['state'] == 'completed' and final['status'] == 200 and
            bool(base64.b64decode(final['exact_output']).strip()) and final['usage']['output_tokens'] > 0,
            'shared input receives a real model result', receipt=final)
        return final, body

    async def inventory(self, space):
        items, cursor = [], '0'
        while True:
            value = await self.request('GET', '/spaces/' + space + '/memory?cursor=' + cursor)
            items.extend(value['items']); cursor = value['next_cursor']
            if cursor == '0': return items
