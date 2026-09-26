# SPDX-License-Identifier: Apache-2.0
# Serve native shared resources; inputs are authenticated HTTP requests, outputs are device-backed JSON or bounded events.
import asyncio
import base64
import re
import time
from aiohttp import web
from .config import aotx_hex
from .errors import aotx_bad, aotx_error
from .json_wire import aotx_encode
from .shared_output import aotx_shared_decode, aotx_shared_bytes
from .shared_wire import (MUTATE, READ, REGISTER, SPACE, MEMBER, CONVERSATION, INPUT, CANCEL, RETIRE, PUBLISH, SAVE,
    CAPABILITIES, PARTICIPANT, SPACES, SPACE_READ, MEMBERS, CONVERSATIONS, CONVERSATION_READ, OPERATION,
    EVENTS, MEMORY, SAVE_READ, AFFECT, aotx_shared_counter, aotx_shared_parse, aotx_shared_read_frame, aotx_shared_mutation)

PREFIX = '/aotx/v1/shared'


async def aotx_shared_read(state, principal, kind, *, lineage=bytes(16), target=bytes(16), parent=bytes(16), cursor=0, byte=0, limit=64):
    payload = aotx_shared_read_frame(kind, lineage, target, parent, cursor, byte, limit)
    reply = await state.wire.call(principal, READ, payload=payload)
    value = aotx_shared_decode(reply)
    if kind == MEMORY:
        value['offset'] = str(byte)
        if any(parent): value['object'] = parent.hex()
    return value


async def aotx_shared_mutate(server, request, principal, operation, *, target=None, space=None, lineage=None):
    async with server.state.budget.claim('work'):
        async with server.body(request) as body:
            if lineage is not None and body.get('lineage') != lineage.hex(): raise aotx_bad('lineage')
            payload = aotx_shared_mutation(server.state, principal, body, operation, target=target, space=space)
        reply = await server.state.wire.call(principal, MUTATE, payload=payload)
        return aotx_shared_decode(reply), reply.status


def aotx_shared_event_span(value, limit):
    if len(aotx_encode(value)) <= limit: return value
    if 'output' not in value: raise aotx_error(413, 'The event exceeds its byte limit.', 'event_limit')
    raw = base64.b64decode(value['output']['base64'])
    empty = {**value, 'output': aotx_shared_bytes(b''), 'next_offset': value['offset']}
    take = min(len(raw), max(0, (limit-len(aotx_encode(empty)))//8))
    if not take: raise aotx_error(413, 'The event exceeds its byte limit.', 'event_limit')
    result = {**value, 'output': aotx_shared_bytes(raw[:take]), 'next_offset': str(int(value['offset'])+take)}
    if len(aotx_encode(result)) > limit: raise aotx_error(413, 'The event exceeds its byte limit.', 'event_limit')
    return result


async def aotx_shared_stream(server, request, principal, headers, kind, lineage, target, parent, cursor, byte):
    state = server.state
    current = await aotx_shared_read(state, principal, kind, lineage=lineage, target=target, parent=parent, cursor=cursor, byte=byte)
    response = web.StreamResponse(status=200, headers={**headers, 'Content-Type': 'text/event-stream', 'X-Accel-Buffering': 'no'})
    response.force_close()
    await asyncio.wait_for(response.prepare(request), state.config.limits['write_seconds'])
    deadline = time.monotonic() + state.config.limits['operation_seconds']
    prior, heartbeat = None, time.monotonic()
    try:
        while True:
            current = aotx_shared_event_span(current, state.config.limits['event_bytes'])
            encoded = aotx_encode(current)
            if len(encoded) > state.config.limits['event_bytes']: raise aotx_error(413, 'The event exceeds its byte limit.', 'event_limit')
            if kind == EVENTS:
                pending = [int(item['input_order']) for item in current['items'] if not item['saved_terminal']]
                cursor = min(pending) if pending else int(current['next_cursor'])
            if encoded != prior:
                number = current['next_offset'] if kind == OPERATION else str(cursor)
                event = b'id: ' + request.path.encode('ascii') + b':' + number.encode('ascii') + b'\nevent: state\ndata: ' + encoded + b'\n\n'
                await asyncio.wait_for(response.write(event), state.config.limits['write_seconds'])
                prior, heartbeat = encoded, time.monotonic()
            if kind == OPERATION:
                byte = int(current['next_offset'])
                if current['state'] in ('completed', 'failed', 'cancelled', 'interrupted') and current['saved_terminal'] and byte == int(current['output_bytes']): break
            if time.monotonic() >= deadline: break
            if time.monotonic() - heartbeat >= 15:
                await asyncio.wait_for(response.write(b': keep-alive\n\n'), state.config.limits['write_seconds'])
                heartbeat = time.monotonic()
            await asyncio.sleep(0.1)
            current = await aotx_shared_read(state, principal, kind, lineage=lineage, target=target, parent=parent, cursor=cursor, byte=byte)
        await asyncio.wait_for(response.write_eof(), state.config.limits['write_seconds'])
    except aotx_error as error:
        event = b'event: error\ndata: ' + aotx_encode(error.body()) + b'\n\n'
        await asyncio.wait_for(response.write(event), state.config.limits['write_seconds'])
        await asyncio.wait_for(response.write_eof(), state.config.limits['write_seconds'])
    return response


async def aotx_shared_route(server, request, principal, headers):
    path, method, state = request.path, request.method, server.state
    if path != PREFIX and not path.startswith(PREFIX + '/'): return None
    parts = path[len(PREFIX):].strip('/').split('/')
    allowed = {'cursor', 'offset', 'limit', 'stream'} if method == 'GET' else set()
    if set(request.query) - allowed or any(len(request.query.getall(k)) != 1 for k in request.query): raise aotx_bad('query')
    cursor = aotx_shared_counter(request.query.get('cursor', '0'), 'cursor')
    byte = aotx_shared_counter(request.query.get('offset', '0'), 'offset')
    limit = aotx_shared_counter(request.query.get('limit', '64'), 'limit', 1)
    if limit > 256 or request.query.get('stream', 'false') not in ('true', 'false'): raise aotx_bad('query')
    stream = request.query.get('stream') == 'true'
    kind = operation = None
    lineage, target, parent, space = bytes(16), bytes(16), bytes(16), None
    if parts == ['capabilities'] and method == 'GET': kind = CAPABILITIES
    elif parts == ['participant']:
        if method == 'GET': kind = PARTICIPANT
        elif method == 'POST': operation = REGISTER
    elif parts == ['spaces']:
        if method == 'GET': kind = SPACES
        elif method == 'POST': operation = SPACE
    elif parts == ['save']:
        if method == 'GET': kind = SAVE_READ
        elif method == 'POST': operation = SAVE
    elif parts == ['retire'] and method == 'POST': operation = RETIRE
    elif len(parts) >= 2 and parts[0] == 'spaces':
        lineage, target, _ = aotx_shared_parse(parts[1], 'spc')
        if len(parts) == 2 and method == 'GET': kind = SPACE_READ
        elif parts[2:] == ['members']:
            if method == 'GET': kind = MEMBERS
            elif method == 'POST': operation = MEMBER
        elif parts[2:] == ['conversations']:
            if method == 'GET': kind = CONVERSATIONS
            elif method == 'POST': operation, space, target = CONVERSATION, target, None
        elif parts[2:] == ['memory'] and method == 'GET': kind = MEMORY
        elif len(parts) == 4 and parts[2] == 'memory' and method == 'GET':
            kind, parent = MEMORY, aotx_hex(parts[3], 16, 'object')
        elif len(parts) == 4 and parts[2] == 'publish' and method == 'POST':
            operation, space, target = PUBLISH, target, aotx_hex(parts[3], 16, 'object')
    elif len(parts) >= 2 and parts[0] == 'conversations':
        lineage, target, _ = aotx_shared_parse(parts[1], 'con')
        if len(parts) == 2 and method == 'GET': kind = CONVERSATION_READ
        elif parts[2:] == ['inputs'] and method == 'POST': operation = INPUT
        elif parts[2:] == ['events'] and method == 'GET': kind = EVENTS
        elif parts[2:] == ['affect'] and method == 'GET': kind = AFFECT
    elif len(parts) >= 2 and parts[0] == 'operations':
        lineage, target, parent = aotx_shared_parse(parts[1], 'op')
        if len(parts) == 2 and method == 'GET': kind = OPERATION
        elif parts[2:] == ['events'] and method == 'GET': kind, stream = OPERATION, True
        elif parts[2:] == ['cancel'] and method == 'POST':
            operation = CANCEL
    if operation is not None:
        value, status = await aotx_shared_mutate(server, request, principal, operation,
            target=target, space=space, lineage=lineage if any(lineage) else None)
        headers['X-Request-ID'] = value['id']
        return server.response(value, headers, status)
    if kind is None: raise aotx_error(404, 'The shared route is unavailable.', 'route_not_found')
    if not any(lineage) and kind not in (CAPABILITIES, PARTICIPANT):
        current = await aotx_shared_read(state, principal, PARTICIPANT)
        lineage = bytes.fromhex(current['lineage'])
    if kind == MEMORY and limit > 64: raise aotx_bad('limit')
    if stream and kind not in (EVENTS, OPERATION): raise aotx_bad('stream')
    if byte and kind not in (OPERATION, MEMORY): raise aotx_bad('offset')
    if cursor and kind not in (SPACES, MEMBERS, CONVERSATIONS, EVENTS, MEMORY): raise aotx_bad('cursor')
    if stream:
        last = request.headers.get('Last-Event-ID')
        if last:
            if 'cursor' in request.query or 'offset' in request.query or not last.startswith(path + ':'): raise aotx_bad('Last-Event-ID')
            value = aotx_shared_counter(last[len(path)+1:], 'Last-Event-ID')
            if kind == OPERATION: byte = value
            else: cursor = value
        async with state.budget.claim('work'):
            return await aotx_shared_stream(server, request, principal, headers, kind, lineage, target, parent, cursor, byte)
    result = await aotx_shared_read(state, principal, kind, lineage=lineage, target=target, parent=parent, cursor=cursor, byte=byte, limit=limit)
    return server.response(result, headers)
