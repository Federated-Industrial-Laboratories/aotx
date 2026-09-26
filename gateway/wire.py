# SPDX-License-Identifier: Apache-2.0
# Exchange bounded packets with the scoped local broker without operator descriptors.
import asyncio
from dataclasses import dataclass
import os
import socket
import stat
import struct
import time
from .errors import aotx_error

HEAD, FRAME = 128, 65536
GRANTS, INFO, SUBMIT, READ, CANCEL, MEDIA, MEDIA_READ, METRICS, MEDIA_LIST = range(1, 10)
FREE, QUEUED, PREPARE, RUNNING, DONE, FAILED, CANCELLED = range(7)


@dataclass(frozen=True)
class aotx_reply:
    status: int
    phase: int
    epoch: int
    identity: bytes
    cursor: int
    role: int
    total: int
    prompt: int
    sampled: int
    finish: int
    code: int
    cancelling: int
    data: bytes


def aotx_packet(principal, op, *, epoch=0, identity=bytes(16), cursor=0, role=0, limit=0,
                temperature=0.0, top_p=0.0, payload=b'', control=b''):
    if len(control) not in (0, 48) or (control and op != SUBMIT):
        raise aotx_error(400, 'The control selection is invalid.', 'control_selection')
    payload = bytes(payload) + control
    if len(payload) > FRAME-HEAD or len(identity) != 16: raise aotx_error(413, 'The service frame is too large.', 'frame_limit')
    f = bytearray(HEAD + len(payload))
    f[:8] = b'AOTXAPI1'
    struct.pack_into('<I', f, 8, op)
    f[16:32] = principal.id
    struct.pack_into('<QQ', f, 32, principal.revision, epoch)
    f[48:64] = identity
    struct.pack_into('<QIIffI', f, 64, cursor, role, limit, temperature, top_p, len(payload))
    struct.pack_into('<I', f, 92, len(control))
    f[HEAD:] = payload
    return bytes(f)


def aotx_unpack(data, principal, identity):
    if len(data) < HEAD or len(data) > FRAME or data[:8] != b'AOTXAPI1' or data[16:32] != principal.id or data[48:64] != identity:
        raise aotx_error(503, 'The device response is invalid.', 'invalid_device_response')
    size = struct.unpack_from('<I', data, 88)[0]
    if len(data) != HEAD+size or struct.unpack_from('<Q', data, 32)[0] != principal.revision:
        raise aotx_error(503, 'The device response is invalid.', 'invalid_device_response')
    status, phase = struct.unpack_from('<II', data, 8)
    epoch = struct.unpack_from('<Q', data, 40)[0]
    cursor, role, total, prompt, sampled = struct.unpack_from('<QIIII', data, 64)
    finish, code, cancelling = struct.unpack_from('<III', data, 92)
    return aotx_reply(status, phase, epoch, identity, cursor, role, total, prompt, sampled, finish, code, cancelling, data[HEAD:])


class aotx_wire:
    def __init__(self, path, limit=64):
        self.path, self.limit = path, limit
        self.pool = asyncio.LifoQueue()
        self.count = 0
        self.closed = False

    async def _take(self):
        while not self.pool.empty():
            stream, returned = self.pool.get_nowait()
            if time.monotonic()-returned < 10: return stream
            stream.close(); self.count -= 1
        if self.closed: raise aotx_error(503, 'The service is closed.', 'service_closed')
        if self.count >= self.limit: raise aotx_error(429, 'The service transport is full.', 'transport_limit')
        self.count += 1
        stream = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        stream.setblocking(False)
        try:
            st = os.stat(self.path, follow_symlinks=False)
            if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.geteuid(): raise OSError()
            await asyncio.wait_for(asyncio.get_running_loop().sock_connect(stream, self.path), 5)
            return stream
        except BaseException:
            stream.close(); self.count -= 1; raise

    async def call(self, principal, op, **kwargs):
        packet = aotx_packet(principal, op, **kwargs)
        stream = None
        try:
            stream = await self._take()
            loop = asyncio.get_running_loop()
            async with asyncio.timeout(15):
                await loop.sock_sendall(stream, packet)
                data = await loop.sock_recv(stream, FRAME+1)
            reply = aotx_unpack(data, principal, kwargs.get('identity', bytes(16)))
            if self.closed: stream.close(); self.count -= 1
            else: self.pool.put_nowait((stream, time.monotonic()))
            stream = None
            if reply.status >= 400:
                messages = {400: 'The device refused the input.', 403: 'The device grant does not permit this operation.',
                    404: 'The resource is unavailable.', 409: 'The request conflicts with current state.',
                    410: 'The runtime epoch has ended.', 413: 'The input exceeds device capacity.',
                    429: 'The device has no admission capacity.', 503: 'The device capability is unavailable.'}
                raise aotx_error(reply.status, messages.get(reply.status, 'The device refused the operation.'), 'device_refused')
            return reply
        except (OSError, TimeoutError):
            raise aotx_error(503, 'The device connection is unavailable.', 'device_connection') from None
        finally:
            if stream is not None: stream.close(); self.count -= 1

    async def close(self):
        self.closed = True
        while not self.pool.empty():
            stream, _ = self.pool.get_nowait(); stream.close(); self.count -= 1
