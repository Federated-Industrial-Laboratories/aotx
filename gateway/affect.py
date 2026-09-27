# SPDX-License-Identifier: Apache-2.0
# Translate exact runtime affect settings and revision-bound operator writes.
import struct
from .errors import aotx_bad, aotx_error
from .json_wire import aotx_fields, aotx_integer
from .shared_wire import aotx_shared_counter

KEYS = ('affect.on', 'quality.on', 'affect.probe_gain', 'affect.decay_fast', 'affect.decay_slow',
    'affect.gain_fast', 'affect.gain_slow', 'affect.cap_valence', 'affect.cap_arousal',
    'affect.temperature_gain', 'affect.voice_gain', 'affect.steer_gain', 'affect.budget')


def aotx_affect_decode(reply):
    p = reply.data
    invalid = lambda: aotx_error(503, 'The affect settings response is invalid.', 'invalid_device_response')
    if len(p) != 32 + 13 * 64: raise invalid()
    schema, count, revision, writable, pending = struct.unpack_from('<IIQII', p)
    if schema != 1 or count != 13 or writable > 1 or pending > 64 or any(p[24:32]): raise invalid()
    rows = []
    for i, key in enumerate(KEYS):
        row = p[32 + i * 64:96 + i * 64]
        if row[:32] != key.encode().ljust(32, b'\0') or any(row[60:]): raise invalid()
        value, low, high, scale = struct.unpack_from('<qqqI', row, 32)
        if scale not in (1, 10000) or not -10000 <= low <= value <= high <= 40000: raise invalid()
        rows.append({'key': key, 'value': value, 'minimum': low, 'maximum': high, 'scale': scale})
    return {'schema': 'aotx.affect.settings.v1', 'epoch': str(reply.epoch), 'revision': str(revision),
        'scope': 'runtime', 'effect': 'next_sequence', 'writable': bool(writable), 'pending_local': pending,
        'paths': {'native': True, 'shared': True, 'ordinary_http': False}, 'settings': rows}


async def aotx_affect_route(server, request, principal, headers):
    if request.path != '/aotx/v1/affect/settings': return None
    if request.query: raise aotx_bad('query')
    if request.method not in ('GET', 'POST'): raise aotx_error(405, 'The method is not supported.', 'method')
    write = request.method == 'POST'
    if not principal.actions & (256 if write else 264):
        raise aotx_error(403, 'The affect operation is not permitted.', 'affect_permission')
    payload, epoch = b'', 0
    if write:
        async with server.body(request) as value:
            fields = {'schema', 'epoch', 'revision', 'key', 'value', 'scale'}
            aotx_fields(value, fields, fields, 'affect')
            if value['schema'] != 'aotx.affect.settings.mutation.v1': raise aotx_bad('schema')
            epoch = aotx_shared_counter(value['epoch'], 'epoch', 1)
            revision = aotx_shared_counter(value['revision'], 'revision')
            key = value['key']
            if not isinstance(key, str) or key not in KEYS: raise aotx_bad('key')
            number = aotx_integer(value['value'], -10000, 40000, 'value')
            scale = aotx_integer(value['scale'], 1, 10000, 'scale')
            if scale not in (1, 10000): raise aotx_bad('scale')
            payload = struct.pack('<IIQqII64s', 1, len(key), revision, number, scale, 0, key.encode())
    reply = await server.state.wire.call(principal, 13, epoch=epoch, payload=payload)
    return server.response(aotx_affect_decode(reply), headers)
