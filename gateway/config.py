# SPDX-License-Identifier: Apache-2.0
# Read operator transport settings and the non-cognitive principal grant map.
from dataclasses import dataclass
import hashlib
import hmac
import os
import re
import stat
from .errors import aotx_error, aotx_bad
from .json_wire import aotx_fields, aotx_integer, aotx_json

ROLES = {'language': 2, 'language-q4': 3, 'language-audio': 4}
DEFAULTS = {'connections': 128, 'body_readers': 8, 'json_bytes': 8388608,
    'upload_bytes': 33554432, 'body_credit': 67108864, 'work': 64,
    'fetches': 4, 'write_seconds': 10, 'body_seconds': 60, 'header_seconds': 10,
    'operation_seconds': 300, 'event_bytes': 65536, 'wire_connections': 64}


@dataclass(frozen=True)
class aotx_principal:
    id: bytes
    revision: int
    hashes: tuple[bytes, ...]
    actions: int
    models: tuple[str, ...]
    pages: int
    tokens: int
    requests: int
    media: int
    media_bytes: int


@dataclass(frozen=True)
class aotx_config:
    socket: str
    host: str
    port: int
    origins: tuple[str, ...]
    models: dict
    principals: tuple[aotx_principal, ...]
    revision: int
    limits: dict
    urls: dict
    certificate: str | None
    key: str | None

    def authenticate(self, header):
        if not isinstance(header, str) or not header.startswith('Bearer ') or not 32 <= len(header[7:]) <= 256:
            raise aotx_error(401, 'A valid bearer credential is required.', 'invalid_api_key')
        digest = hashlib.sha256(header[7:].encode('utf-8')).digest()
        found = None
        for principal in self.principals:
            for expected in principal.hashes:
                if hmac.compare_digest(digest, expected): found = principal
        if found is None: raise aotx_error(401, 'A valid bearer credential is required.', 'invalid_api_key')
        return found


def aotx_hex(value, size, param):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{%d}' % (size * 2), value):
        raise aotx_bad(param)
    return bytes.fromhex(value)


def aotx_load_config(path):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o022 or st.st_size > 1048576:
            raise aotx_bad('config')
        with os.fdopen(fd, 'rb', closefd=False) as source: data = source.read(1048577)
    finally: os.close(fd)
    c = aotx_json(data)
    aotx_fields(c, {'socket', 'host', 'port', 'origins', 'models', 'principals', 'revision', 'limits',
        'urls', 'certificate', 'key'}, {'socket', 'models', 'principals', 'revision'}, 'config')
    revision = c['revision']
    if not isinstance(revision, str) or not re.fullmatch('[1-9][0-9]{0,19}', revision) or int(revision) > 2**64-1:
        raise aotx_bad('revision')
    revision = int(revision)
    if not isinstance(c['socket'], str) or '\x00' in c['socket'] or not 1 <= len(c['socket'].encode()) < 108: raise aotx_bad('socket')
    models = c['models']
    if not isinstance(models, dict) or not 1 <= len(models) <= 32: raise aotx_bad('models')
    for alias, model in models.items():
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._/-]{0,127}', alias): raise aotx_bad('models')
        aotx_fields(model, {'role', 'published_at'}, {'role', 'published_at'}, 'models')
        if not isinstance(model['role'], str) or model['role'] not in ROLES: raise aotx_bad('models')
        aotx_integer(model['published_at'], 0, 2**53-1, 'published_at')
    limits = dict(DEFAULTS)
    aotx_fields(c.get('limits', {}), DEFAULTS, param='limits')
    for key, value in c.get('limits', {}).items(): limits[key] = aotx_integer(value, 1, 2**31-1, 'limits')
    if limits['event_bytes'] < 1024: raise aotx_bad('event_bytes')
    principals, ids, hashes = [], set(), set()
    if not isinstance(c['principals'], list) or len(c['principals']) > 1022: raise aotx_bad('principals')
    for p in c['principals']:
        aotx_fields(p, {'id', 'token_sha256', 'models', 'actions', 'pages', 'tokens', 'requests', 'media', 'media_bytes'},
            {'id', 'token_sha256', 'models', 'actions'}, 'principals')
        identity = aotx_hex(p['id'], 16, 'principal.id')
        if not any(identity) or identity in ids: raise aotx_bad('principal.id')
        ids.add(identity)
        if not isinstance(p['token_sha256'], list) or not 1 <= len(p['token_sha256']) <= 4: raise aotx_bad('token_sha256')
        keys = tuple(aotx_hex(v, 32, 'token_sha256') for v in p['token_sha256'])
        if any(v in hashes for v in keys) or len(set(keys)) != len(keys): raise aotx_bad('token_sha256')
        hashes.update(keys)
        if not isinstance(p['models'], list) or any(not isinstance(v, str) or v not in models for v in p['models']):
            raise aotx_bad('principal.models')
        actions = p['actions']
        if not isinstance(actions, list) or not actions or any(v not in ('infer', 'upload', 'fetch', 'telemetry') for v in actions):
            raise aotx_bad('actions')
        bits = sum(bit for name, bit in [('infer', 1), ('upload', 2), ('fetch', 4), ('telemetry', 8)] if name in actions)
        if bits & 4 and not bits & 2: raise aotx_bad('actions')
        principals.append(aotx_principal(identity, revision, keys, bits, tuple(p['models']),
            aotx_integer(p.get('pages', 0), 0, 2**31-1, 'pages'),
            aotx_integer(p.get('tokens', 256), 1, 2**31-1, 'tokens'),
            aotx_integer(p.get('requests', 2), 1, 2**31-1, 'requests'),
            aotx_integer(p.get('media', 16), 0, 2**31-1, 'media'),
            aotx_integer(p.get('media_bytes', 33554432), 0, 2**63-1, 'media_bytes')))
    origins = c.get('origins', [])
    if not isinstance(origins, list) or any(not isinstance(o, str) or not re.fullmatch(r'https?://[^/\s]+', o) for o in origins):
        raise aotx_bad('origins')
    host = c.get('host', '127.0.0.1')
    if not isinstance(host, str) or not 1 <= len(host) <= 253: raise aotx_bad('host')
    certificate, key = c.get('certificate'), c.get('key')
    if bool(certificate) != bool(key) or any(v is not None and (not isinstance(v, str) or not v) for v in (certificate, key)):
        raise aotx_bad('certificate')
    if host not in ('127.0.0.1', '::1', 'localhost') and not certificate: raise aotx_bad('host', 'This bind address requires TLS.')
    urls = c.get('urls', {})
    aotx_fields(urls, {'public', 'private', 'ca_file'}, param='urls')
    if type(urls.get('public', True)) is not bool or not isinstance(urls.get('private', []), list): raise aotx_bad('urls')
    return aotx_config(c['socket'], host, aotx_integer(c.get('port', 8081), 1, 65535, 'port'),
        tuple(origins), models, tuple(principals), revision, limits, urls, certificate, key)
