# SPDX-License-Identifier: Apache-2.0
# Transfer immutable media bytes to the scoped device source store.
import asyncio
import base64
import binascii
import hashlib
import re
import struct
import time
import uuid
from dataclasses import replace
from .config import aotx_hex
from .errors import aotx_error, aotx_bad
from .wire import MEDIA, MEDIA_READ, MEDIA_LIST

FORMATS = {'image/jpeg': 1, 'audio/wav': 3, 'audio/x-wav': 3}


def aotx_base64(value, limit, param):
    if not isinstance(value, str) or len(value) > ((limit+2)//3)*4: raise aotx_error(413, 'The media object is too large.', 'media_limit', param)
    try:
        decoded = base64.b64decode(value, validate=True)
        if len(decoded) > limit or base64.b64encode(decoded).decode('ascii') != value: raise ValueError()
        return decoded
    except (ValueError, binascii.Error, UnicodeError): raise aotx_bad(param, 'The base64 input is invalid.') from None


def aotx_media_id(value):
    if not isinstance(value, str) or not re.fullmatch('media-[0-9a-f]{32}', value): raise aotx_bad('media_id')
    return aotx_hex(value[6:], 16, 'media_id')


def aotx_media_result(identity, reply):
    if len(reply.data) != 64: raise aotx_error(503, 'The media response is invalid.', 'invalid_device_response')
    source_bytes, phase, status, format_id, samples, rows = struct.unpack_from('<QIIIII', reply.data, 32)
    return {'id': 'media-'+identity.hex(), 'sha256': reply.data[:32].hex(), 'bytes': str(source_bytes),
        'phase': phase, 'status': status, 'format': format_id, 'samples': samples, 'rows': rows}


async def aotx_media_status(state, principal, identity):
    return aotx_media_result(identity, await state.wire.call(principal, MEDIA_READ, identity=identity))


async def aotx_media_list(state, principal, cursor):
    reply = await state.wire.call(principal, MEDIA_LIST, cursor=cursor)
    if len(reply.data) % 80: raise aotx_error(503, 'The media list is invalid.', 'invalid_device_response')
    rows = [aotx_media_result(reply.data[at:at+16], replace(reply, data=reply.data[at+16:at+80]))
        for at in range(0, len(reply.data), 80)]
    return {'schema': 'aotx.media-list.v1', 'runtime_epoch': str(reply.epoch), 'data': rows,
        'next_cursor': str(reply.cursor) if reply.cursor else None, 'consistency': 'current'}


async def aotx_media_frame(state, principal, identity, op, size=0, offset=0, data=b''):
    frame = struct.pack('<II16sQQIII12x', 1, op, identity, size, offset, len(data), 0, 0) + data
    return await state.wire.call(principal, MEDIA, identity=identity, payload=frame)


async def aotx_media_upload(state, principal, data, content_type):
    if not principal.actions & 2: raise aotx_error(403, 'The grant does not permit uploads.', 'media_forbidden')
    if content_type not in FORMATS: raise aotx_error(415, 'The media type is not supported.', 'media_type')
    if not data or len(data) > min(state.config.limits['upload_bytes'], principal.media_bytes):
        raise aotx_error(413, 'The media byte count is outside its limit.', 'media_limit')
    identity = uuid.uuid4().bytes
    digest = hashlib.sha256(data).digest()
    started = False
    try:
        await aotx_media_frame(state, principal, identity, 1, len(data),
            data=struct.pack('<IIII32s', FORMATS[content_type], 0, 0, 0, digest))
        started = True
        for at in range(0, len(data), 8192):
            await aotx_media_frame(state, principal, identity, 2, len(data), at, data[at:at+8192])
        await aotx_media_frame(state, principal, identity, 3, len(data), len(data))
        deadline = time.monotonic()+state.config.limits['operation_seconds']
        while time.monotonic() < deadline:
            result = await aotx_media_status(state, principal, identity)
            if result['phase'] == 6:
                if result['sha256'] != digest.hex(): raise aotx_error(503, 'The source digest does not match.', 'source_digest')
                return result
            if result['phase'] == 7: raise aotx_error(400, 'The device refused the media source.', 'media_refused')
            await asyncio.sleep(0.05)
        raise aotx_error(504, 'The media preparation deadline expired.', 'media_timeout')
    except BaseException:
        if started:
            try: await aotx_media_frame(state, principal, identity, 4)
            except (Exception, asyncio.CancelledError): pass
        raise
