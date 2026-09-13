# SPDX-License-Identifier: Apache-2.0
# Check shared HTTP boundaries; inputs are registered live clients and media fixtures, outputs are bounded evidence, failures raise.
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import time
from shared_runtime_client import PREFIX, TERMINAL


async def aotx_shared_observe(client, identity, accept, *, seconds=900):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = await client.request('GET', '/operations/' + identity, audit=False)
        if accept(value): return value
        await asyncio.sleep(0.05)
    raise TimeoutError('The operation did not reach the required state.')


async def aotx_shared_observe_phase(client, identity, phase):
    return await aotx_shared_observe(client, identity, lambda r: r['state'] == phase or r['state'] in TERMINAL or
        (phase == 'queued' and r['state'] == 'running'))


async def aotx_shared_commit(client, identity):
    return await aotx_shared_observe(client, identity, lambda r: r['device_committed'])


async def aotx_shared_control_cases(test, owner, member, *, attempts=4):
    test.check(1 <= attempts <= 8 and owner.participant != member.participant, 'bounded control case inputs')
    private, _ = await owner.create()
    room, _ = await owner.create('room')
    results = []
    for action in ('cancel', 'revoke'):
        for phase in ('queued', 'running'):
            covered = None
            for attempt in range(attempts):
                actor = owner if action == 'cancel' else member
                space = private if action == 'cancel' else room
                if action == 'revoke':
                    await owner.done('/spaces/' + room + '/members', participant=member.participant, permissions=['read', 'write'])
                conversation, _ = await owner.done('/spaces/' + space + '/conversations')
                conversation = 'con-' + owner.lineage + '-' + conversation['resource']
                receipt, body = await actor.input(conversation,
                    'List the integers from one to three hundred. Put each integer on a separate line. Case ' +
                    action + ' ' + phase + ' ' + str(attempt) + '.', max_output_tokens=512)
                observed = await aotx_shared_observe_phase(actor, receipt['id'], phase)
                if observed['state'] != phase:
                    test.record(shared_control_race=dict(action=action, wanted=phase, observed=observed['state'], attempt=attempt))
                    await owner.terminal(receipt['id'])
                    continue
                if action == 'cancel':
                    wrong = actor.command(target_sequence=str(int(body['sequence']) + 1))
                    await actor.request('POST', '/operations/' + receipt['id'] + '/cancel', (404,), json=wrong)
                    control, control_body = await actor.mutate('/operations/' + receipt['id'] + '/cancel', target_sequence=body['sequence'])
                else:
                    control, control_body = await owner.mutate('/spaces/' + room + '/members', participant=member.participant, permissions=[])
                committed = await aotx_shared_commit(owner, control['id'])
                after = await owner.request('GET', '/operations/' + receipt['id'], audit=False)
                final = await owner.terminal(receipt['id'])
                if int(final['terminal_source']) < int(committed['admission_source']) or (phase == 'queued' and
                    after['state'] != 'queued' and after['saved_admission']):
                    test.record(shared_control_race=dict(action=action, wanted=phase, observed='control_boundary_not_observed', attempt=attempt))
                    await owner.terminal(control['id'])
                    continue
                if action == 'cancel':
                    test.check(final['state'] == 'cancelled' and final['status'] == 409, 'exact cancellation stops an active operation', phase=phase)
                else:
                    await member.request('GET', '/operations/' + receipt['id'], (404,))
                    await member.request('GET', '/conversations/' + conversation + '/events', (404,))
                    await member.request('GET', '/spaces/' + room + '/memory', (404,))
                    test.check(final['state'] == 'failed' and final['status'] in (403, 503), 'member revocation stops active publication', phase=phase)
                test.check(final['output_bytes'] == after['output_bytes'], 'committed control preserves only the recorded output prefix', action=action, phase=phase)
                retry_owner = actor if action == 'cancel' else owner
                retry_path = '/operations/' + receipt['id'] + '/cancel' if action == 'cancel' else '/spaces/' + room + '/members'
                retry = await retry_owner.request('POST', retry_path, json=control_body)
                test.check(retry['id'] == control['id'], 'exact control retry returns its original receipt')
                await owner.terminal(control['id'])
                events = await owner.request('GET', '/conversations/' + conversation + '/events')
                test.check(len(events['items']) == 1 and events['items'][0]['id'] == receipt['id'], 'control retries do not add conversation input')
                covered = dict(action=action, phase=phase, operation=final, control=control['id'], conversation=conversation)
                break
            test.check(covered is not None, 'required control state was observed', action=action, phase=phase, attempts=attempts)
            results.append(covered)
    return results


async def aotx_shared_memory_object(test, client, space, identity):
    data, offset, metadata = bytearray(), 0, None
    while True:
        value = await client.request('GET', '/spaces/' + space + '/memory/' + identity + '?offset=' + str(offset), audit=False)
        test.check(len(value['items']) == 1 and value['items'][0]['id'] == identity, 'memory detail has one exact object')
        row = value['items'][0]
        if metadata is None: metadata = row
        test.check(row == metadata, 'memory object version stays fixed across byte reads')
        part = base64.b64decode(value['payload']['base64'], validate=True)
        test.check(int(value['offset']) == offset and int(value['next_offset']) == offset + len(part), 'memory payload has exact byte offsets')
        data.extend(part); offset += len(part)
        if offset == int(row['bytes']): return row, bytes(data)
        test.check(bool(part) and offset < int(row['bytes']), 'memory detail makes bounded forward progress')


async def aotx_shared_publication_cases(test, owner, member, outsider):
    test.check(len({c.participant for c in (owner, member, outsider)}) == 3, 'publication uses three distinct participants')
    private, conversation = await owner.create()
    source_text = 'The private publication label is native-' + owner.participant[-8:] + '-' + str(owner.sequence) + '.'
    await owner.answer(conversation, source_text)
    inventory = await owner.inventory(private)
    owned = [r for r in inventory if r['scope'] == 'private' and r['owner'] == private.split('-')[-1]]
    working = []
    for row in owned:
        if row['kind'] != 7: continue
        metadata, payload = await aotx_shared_memory_object(test, owner, private, row['id'])
        if payload.startswith(b'AOTXMEM1') and source_text.encode() in payload: working.append((metadata, payload))
    test.check(len(working) == 1, 'native retained working source is identified by exact input bytes')
    w, wbytes = working[0]
    source = [r for r in owned if r['kind'] == 1 and r['id'] == w['source']]
    vector = [r for r in owned if r['kind'] == 9 and r['source'] == w['source']]
    test.check(len(source) == len(vector) == 1, 'native publication graph has one source and one embedding')
    original = {}
    for row in (source[0], vector[0], w):
        original[row['kind']] = await aotx_shared_memory_object(test, owner, private, row['id'])
    test.check(original[1][1].startswith(b'AOTXMEM1') and original[9][1].startswith(b'AOTXVEC2'), 'native publication source formats')
    results = []
    for scope in ('room', 'instance'):
        destination, _ = await owner.create(scope)
        if scope == 'room':
            await owner.done('/spaces/' + destination + '/members', participant=member.participant, permissions=['read', 'write'])
        before = {r['id'] for r in await owner.inventory(destination)}
        test.check(not set(r[0]['id'] for r in original.values()) & before, 'broader space does not expose prior private object IDs')
        wrong = owner.command(source_version=str(int(w['version']) + 1))
        await owner.request('POST', '/spaces/' + destination + '/publish/' + w['id'], (409,), json=wrong)
        forbidden = member.command(source_version=w['version'])
        await member.request('POST', '/spaces/' + destination + '/publish/' + w['id'], (404,), json=forbidden)
        receipt, body = await owner.done('/spaces/' + destination + '/publish/' + w['id'], source_version=w['version'])
        retry = await owner.request('POST', '/spaces/' + destination + '/publish/' + w['id'], json=body)
        test.check(retry['id'] == receipt['id'], 'publication retries do not clone the source again')
        added = [r for r in await member.inventory(destination) if r['id'] not in before and r['owner'] == destination.split('-')[-1]]
        test.check(len(added) == 3 and {r['kind'] for r in added} == {1, 7, 9}, 'publication adds one complete native graph')
        copied = {r['kind']: await aotx_shared_memory_object(test, member, destination, r['id']) for r in added}
        for kind, (row, payload) in copied.items():
            test.check(row['scope'] == scope and row['owner'] == destination.split('-')[-1] and row['actor'] == original[kind][0]['actor'],
                'published objects preserve source actor and exact destination scope', kind=kind)
            test.check(row['id'] != original[kind][0]['id'], 'publication assigns distinct object identities', kind=kind)
            if kind in (1, 7): test.check(payload == original[kind][1], 'published text payload is byte exact', kind=kind)
        test.check(copied[7][0]['source'] == copied[1][0]['id'] == copied[9][0]['source'], 'published source references remain inside the copied graph')
        vector_bytes = copied[9][1]
        test.check(vector_bytes[:88] == original[9][1][:88] and vector_bytes[112:] == original[9][1][112:] and
            vector_bytes[88:104].hex() == copied[1][0]['id'] and int.from_bytes(vector_bytes[104:112], 'little') == int(copied[1][0]['version']),
            'published embedding preserves numerical bytes and changes only its source reference')
        for row, payload in original.values():
            current = await aotx_shared_memory_object(test, owner, private, row['id'])
            test.check(current == (row, payload), 'publication leaves the private source graph unchanged')
            await member.request('GET', '/spaces/' + private + '/memory/' + row['id'], (404,))
        if scope == 'room': await outsider.request('GET', '/spaces/' + destination + '/memory', (404,))
        else:
            visible = await outsider.inventory(destination)
            test.check({r['id'] for r in added} <= {r['id'] for r in visible}, 'instance publication reaches admitted participants')
        results.append(dict(scope=scope, space=destination, receipt=receipt['id'], objects=added))
    return dict(private=private, conversation=conversation, source=w['id'], publications=results)


async def aotx_shared_sse(test, client, identity, *, last=None, stop_after_bytes=False, byte=0):
    path = PREFIX + '/operations/' + identity + '/events'
    headers = dict(client.headers)
    if last: headers['Last-Event-ID'] = last
    url = client.url + path + ('' if last or not byte else '?offset=' + str(byte))
    output, offset, last_id, final = bytearray(), byte, None, None
    async with asyncio.timeout(900):
        async with client.client.get(url, headers=headers) as response:
            test.check(response.status == 200 and response.content_type == 'text/event-stream', 'authenticated shared event stream opens')
            event_id, event_kind, data = None, None, []
            while True:
                line = await response.content.readline()
                if not line: break
                test.check(len(line) <= 131072, 'event line has a bounded byte length')
                line = line.rstrip(b'\r\n')
                if line.startswith(b'id: '): event_id = line[4:].decode('ascii')
                elif line.startswith(b'event: '): event_kind = line[7:].decode('ascii')
                elif line.startswith(b'data: '): data.append(line[6:])
                elif not line and data:
                    value = json.loads(b'\n'.join(data)); data = []
                    test.check(event_kind == 'state', 'event stream carries device state', response=value)
                    part = base64.b64decode(value['output']['base64'], validate=True)
                    test.check(value['id'] == identity and int(value['offset']) == offset and
                        int(value['next_offset']) == offset + len(part), 'event output uses exact contiguous byte offsets')
                    offset += len(part); output.extend(part); last_id = event_id; final = value
                    test.check(last_id == path + ':' + str(offset), 'event ID names its exact resume cursor')
                    if stop_after_bytes and part: break
                    if value['state'] in TERMINAL and value['saved_terminal'] and offset == int(value['output_bytes']): break
            test.check(final is not None and last_id is not None, 'event stream returns an observable device state')
    return bytes(output), last_id, final


async def aotx_shared_conversation_sse(test, client, conversation, identity, *, last=None, saved=False):
    path = PREFIX + '/conversations/' + conversation + '/events'
    headers = dict(client.headers)
    if last: headers['Last-Event-ID'] = last
    states = set()
    async with asyncio.timeout(900):
        async with client.client.get(client.url + path + '?stream=true', headers=headers) as response:
            test.check(response.status == 200 and response.content_type == 'text/event-stream', 'authenticated conversation stream opens')
            event_id, event_kind, data = None, None, []
            while True:
                line = await response.content.readline()
                if not line: break
                test.check(len(line) <= 131072, 'conversation event line has a bounded byte length')
                line = line.rstrip(b'\r\n')
                if line.startswith(b'id: '): event_id = line[4:].decode('ascii')
                elif line.startswith(b'event: '): event_kind = line[7:].decode('ascii')
                elif line.startswith(b'data: '): data.append(line[6:])
                elif not line and data:
                    value = json.loads(b'\n'.join(data)); data = []
                    test.check(event_kind == 'state' and not value.get('gap'), 'conversation stream has complete device events', response=value)
                    pending = [int(r['input_order']) for r in value['items'] if not r['saved_terminal']]
                    cursor = min(pending) if pending else int(value['next_cursor'])
                    test.check(event_id == path + ':' + str(cursor), 'conversation resume cursor preserves every unsaved terminal transition')
                    matches = [r for r in value['items'] if r['id'] == identity]
                    test.check(len(matches) <= 1, 'conversation state has one row per input')
                    if matches:
                        row = matches[0]; states.add((row['state'], row['saved_terminal']))
                        if not saved or row['saved_terminal']:
                            return dict(row=row, last=event_id, states=sorted(states))
    raise AssertionError('The conversation stream lost its required input state.')


async def aotx_shared_stream_case(test, client, conversation=None, *, attempts=4):
    test.check(1 <= attempts <= 8, 'bounded stream case attempts')
    if conversation is None: _, conversation = await client.create()
    pending = None
    for attempt in range(attempts):
        receipt, body = await client.input(conversation, 'Write a numbered list of fifty common colors. Case ' + str(client.sequence) + '.', max_output_tokens=256)
        pending = await aotx_shared_conversation_sse(test, client, conversation, receipt['id'])
        if not pending['row']['saved_terminal']: break
        test.record(shared_stream_race=dict(attempt=attempt, state='saved_before_first_event'))
        await client.terminal(receipt['id'])
    test.check(pending is not None and not pending['row']['saved_terminal'], 'conversation reconnect starts from an observed unsaved input')
    follow = asyncio.create_task(aotx_shared_conversation_sse(test, client, conversation, receipt['id'], last=pending['last'], saved=True))
    try:
        first, last, partial = await aotx_shared_sse(test, client, receipt['id'], stop_after_bytes=True)
        final = await client.terminal(receipt['id'])
        followed = await follow
    finally:
        if not follow.done(): follow.cancel()
        await asyncio.gather(follow, return_exceptions=True)
    rest, _, resumed = await aotx_shared_sse(test, client, receipt['id'], last=last, byte=len(first))
    exact = base64.b64decode(final['exact_output'], validate=True)
    test.check(first + rest == exact and resumed['saved_terminal'], 'event reconnect returns every exact byte once')
    test.check(all(followed['row'][k] == final[k] for k in ('id', 'state', 'status', 'output_bytes', 'saved_admission', 'saved_terminal', 'input_order')),
        'conversation reconnect observes the exact saved terminal operation')
    if len(exact) > 1:
        cut = max(1, len(exact)//2)
        suffix, _, ended = await aotx_shared_sse(test, client, receipt['id'], byte=cut)
        test.check(exact[:cut] + suffix == exact and ended['saved_terminal'], 'explicit byte cursor can resume inside a prior event span')
    test.check(bool(exact) and final['state'] == 'completed', 'event stream uses a real generated result')
    return dict(conversation=conversation, receipt=final, body=body, first_offset=str(len(first)), first_state=partial['state'],
        conversation_first=pending, conversation_terminal=followed)


async def aotx_shared_verify_results(test, clients, saved):
    by_actor = {client.participant: client for client in clients}
    entries = [(saved['stream']['conversation'], saved['stream']['receipt'], saved['stream'].get('body'))]
    for group in saved['media']:
        for source in group['sources']:
            entries.extend((group['conversation'], source[name], source.get(body))
                for name, body in (('receipt', 'body'), ('reused', 'reused_body')))
    for conversation, old, body in entries:
        client = by_actor[old['actor']]
        actual = await client.terminal(old['id'])
        test.check(all(actual[k] == old[k] for k in ('id', 'actor', 'sequence', 'input_order', 'status', 'state', 'exact_output', 'usage', 'finish')),
            'file-only recovery preserves exact media and streamed results', receipt=actual)
        if body is not None:
            retry = await client.request('POST', '/conversations/' + conversation + '/inputs', json=body)
            test.check(retry['id'] == old['id'] and retry['output']['base64'] == old['exact_output'],
                'recovered media and stream retries return their exact saved results')


async def aotx_shared_retirement_case(test, client):
    space, conversation = await client.create()
    first, first_body = await client.answer(conversation, 'The receipt label is early-' + str(client.sequence) + '.')
    second, second_body = await client.answer(conversation, 'The receipt label is later-' + str(client.sequence) + '.')
    floor = int(first['sequence']) + 1
    retired, _ = await client.done('/retire', retry_floor=str(floor))
    person = await client.request('GET', '/participant')
    test.check(int(person['retry_floor']) == floor, 'saved receipt retirement advances the exact participant floor')
    await client.request('POST', '/conversations/' + conversation + '/inputs', (410,), json=first_body)
    await client.request('POST', '/conversations/' + conversation + '/inputs', (410,), json=dict(first_body, text=first_body['text'] + ' changed'))
    await client.request('GET', '/operations/' + first['id'], (404, 410))
    later = await client.terminal(second['id'])
    test.check(later['exact_output'] == second['exact_output'], 'retirement preserves the later exact receipt')
    events = await client.request('GET', '/conversations/' + conversation + '/events?cursor=' + first['input_order'])
    test.check(events['gap'] and int(events['event_floor']) == int(first['input_order']) + 1 and
        [r['id'] for r in events['items']] == [second['id']], 'retired event cursor returns an explicit scoped gap')
    body = client.command(); body['operation_key'] = first_body['operation_key']
    reused = await client.request('POST', '/save', (202, 200), json=body)
    test.check(reused['sequence'] == str(client.sequence) and reused['id'] != first['id'], 'retired key receives a distinct persistent operation ID')
    client.sequence += 1
    await client.terminal(reused['id']); await client.request('GET', '/operations/' + first['id'], (404, 410))
    return dict(participant=client.participant, space=space, conversation=conversation, retry_floor=str(floor),
        retired=first, retired_body=first_body, retained=second, retained_body=second_body, reused=reused['id'], retirement=retired['id'])


async def aotx_shared_verify_retirement(test, client, saved):
    person = await client.discover()
    test.check(client.participant == saved['participant'] and int(person['retry_floor']) >= int(saved['retry_floor']), 'recovery preserves the participant retry floor')
    await client.request('POST', '/conversations/' + saved['conversation'] + '/inputs', (410,), json=saved['retired_body'])
    await client.request('GET', '/operations/' + saved['retired']['id'], (404, 410))
    retained = await client.terminal(saved['retained']['id'])
    test.check(retained['exact_output'] == saved['retained']['exact_output'] and retained['usage'] == saved['retained']['usage'],
        'recovery preserves the unretired result and exact usage')


async def aotx_shared_media_http(client, method, path, expected, *, retry_seconds=900, **kwargs):
    headers = {**client.headers, **kwargs.pop('headers', {})}
    end = time.monotonic() + retry_seconds
    while True:
        async with client.client.request(method, client.url + '/aotx/v1/media' + path, headers=headers, **kwargs) as response:
            raw = await response.read(); status = response.status
            value = json.loads(raw) if raw else None
        if status == 429 and status not in expected and time.monotonic() < end:
            await asyncio.sleep(1); continue
        client.test.check(status in expected, 'scoped media HTTP status', method=method, status=status, expected=expected, response=value)
        return status, value


async def aotx_shared_media_cases(test, clients, outsider, fixtures, *, attempts=4, retry_seconds=900):
    test.check(1 <= attempts <= 8, 'bounded media case attempts')
    test.check(retry_seconds > 0, 'positive media retry time limit')
    test.record(media_retry_seconds=retry_seconds)
    async def media_http(client, method, path, expected, **kwargs):
        return await aotx_shared_media_http(client, method, path, expected, retry_seconds=retry_seconds, **kwargs)
    if isinstance(fixtures, (str, Path)): fixtures = json.loads(Path(fixtures).read_text())
    selected = []
    for modality in ('image', 'audio'):
        found = [f for f in fixtures if f.get('modality') == modality]
        test.check(bool(found), 'qualified media fixture exists', modality=modality)
        selected.append(found[0])
    test.check(len(clients) in (1, 64) and outsider.participant not in {c.participant for c in clients}, 'media cases use distinct bounded principals')
    data = [Path(f['path']).read_bytes() for f in selected]
    test.record(shared_media_sources=[dict(path=f['path'], sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)) for f, raw in zip(selected, data)])
    async def actor_case(index, client):
        space, conversation = await client.create(); results = []
        for f, raw in zip(selected, data):
            covered = None
            for attempt in range(attempts):
                _, uploaded = await media_http(client, 'POST', '', (201,), data=raw, headers={'Content-Type': f['type']})
                test.check(uploaded['sha256'] == hashlib.sha256(raw).hexdigest() and uploaded['phase'] == 6, 'qualified upload retains exact source identity')
                await media_http(outsider, 'GET', '/' + uploaded['id'], (404,))
                media = [{'type': f['modality'], 'sha256': uploaded['sha256']}]
                receipt, first_body = await client.input(conversation, f['prompt'] + ' Case ' + str(index) + '-' + str(attempt) + '.',
                    model=f['model'], media=media, max_output_tokens=64)
                observed = await aotx_shared_observe_phase(client, receipt['id'], 'queued')
                if observed['state'] != 'queued':
                    test.record(shared_media_race=dict(principal=index, modality=f['modality'], state=observed['state'], attempt=attempt))
                    await client.terminal(receipt['id']); await media_http(client, 'DELETE', '/' + uploaded['id'], (204,)); continue
                status, _ = await media_http(client, 'DELETE', '/' + uploaded['id'], (204, 409))
                boundary = await client.request('GET', '/operations/' + receipt['id'], audit=False)
                final = await client.terminal(receipt['id'])
                if status == 409 and boundary['state'] != 'queued':
                    test.record(shared_media_race=dict(principal=index, modality=f['modality'], state='queue_boundary_not_observed', attempt=attempt))
                    await media_http(client, 'DELETE', '/' + uploaded['id'], (204,)); continue
                if status == 204:
                    test.check(final['state'] == 'completed', 'a media removal race follows completed execution')
                    test.record(shared_media_race=dict(principal=index, modality=f['modality'], state='terminal_before_delete', attempt=attempt)); continue
                test.check(final['state'] == 'completed' and bool(base64.b64decode(final['exact_output']).strip()), 'queued private media stays leased through its real model result')
                _, available = await media_http(client, 'GET', '/' + uploaded['id'], (200,))
                test.check(available['sha256'] == uploaded['sha256'], 'terminal media source remains available for explicit reuse')
                reused, reused_body = await client.answer(conversation, f['prompt'] + ' Reuse case ' + str(index) + '.', model=f['model'], media=media)
                await media_http(client, 'DELETE', '/' + uploaded['id'], (204,))
                status, removed = await media_http(client, 'GET', '/' + uploaded['id'], (200, 404))
                if status == 200:
                    test.check(removed['phase'] == 7 and removed['status'] == 4 and removed['sha256'] == uploaded['sha256'],
                        'removed source retains only its exact canceled descriptor until reuse')
                memory = await client.inventory(space)
                for old, canonical in ((final, first_body), (reused, reused_body)):
                    retry = await client.request('POST', '/conversations/' + conversation + '/inputs', json=canonical)
                    test.check(retry['id'] == old['id'] and retry['output']['base64'] == old['exact_output'],
                        'removed media source does not invalidate an exact completed retry')
                body = client.command(text='Use the removed source.', model=f['model'], media=media, max_output_tokens=32)
                await client.request('POST', '/conversations/' + conversation + '/inputs', (404,), json=body)
                person = await client.request('GET', '/participant')
                test.check(int(person['next_sequence']) == client.sequence and await client.inventory(space) == memory,
                    'removed source admission cannot advance the participant or retain input memory')
                covered = dict(principal=client.participant, modality=f['modality'], sha256=uploaded['sha256'], source=uploaded['id'],
                    receipt=final, body=first_body, reused=reused, reused_body=reused_body, queued_state=observed['state'])
                break
            test.check(covered is not None, 'queued source lease is observed within the bounded attempts', principal=index, modality=f['modality'])
            results.append(covered)
        return dict(space=space, conversation=conversation, sources=results)
    return await asyncio.gather(*(actor_case(i, client) for i, client in enumerate(clients)))
