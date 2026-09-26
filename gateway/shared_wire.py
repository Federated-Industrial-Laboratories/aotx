# SPDX-License-Identifier: Apache-2.0
# Encode bounded shared envelopes; inputs are validated JSON values, output is canonical bytes, errors are explicit.
import re
import struct
from .config import ROLES, aotx_hex
from .controls import aotx_control_selection
from .errors import aotx_bad, aotx_error
from .json_wire import aotx_fields, aotx_integer, aotx_number

MUTATE, READ = 10, 11
COMMAND_HEAD, READ_HEAD, REPLY_HEAD = 192, 96, 320
REGISTER, SPACE, MEMBER, CONVERSATION, INPUT, CANCEL, RETIRE, PUBLISH, SAVE = range(1, 10)
CAPABILITIES, PARTICIPANT, SPACES, SPACE_READ, MEMBERS, CONVERSATIONS, CONVERSATION_READ, OPERATION, EVENTS, MEMORY, SAVE_READ, AFFECT = range(1, 13)
SCOPES = {'private': 0, 'room': 1, 'instance': 2}
RIGHTS = {'read': 1, 'write': 2, 'manage': 4}


def aotx_shared_counter(value, param, minimum=0):
    if not isinstance(value, str) or not re.fullmatch('0|[1-9][0-9]{0,19}', value): raise aotx_bad(param)
    number = int(value)
    if not minimum <= number <= 2**64-1: raise aotx_bad(param)
    return number


def aotx_shared_handle(kind, lineage, identity):
    return '-'.join((kind, lineage.hex(), identity.hex()))


def aotx_shared_parse(value, kind):
    if not isinstance(value, str) or not re.fullmatch(kind + '-[0-9a-f]{32}-[0-9a-f]{32}', value): raise aotx_bad('id')
    parts = tuple(bytes.fromhex(v) for v in value.split('-')[1:])
    if any(not any(v) for v in parts): raise aotx_bad('id')
    return parts[0], parts[1], bytes(16)


def aotx_shared_read_frame(kind, lineage=bytes(16), target=bytes(16), parent=bytes(16), cursor=0, byte=0, limit=64):
    if kind not in range(1, 13) or any(len(v) != 16 for v in (lineage, target, parent)):
        raise aotx_bad('read')
    aotx_integer(cursor, 0, 2**64-1, 'cursor'); aotx_integer(byte, 0, 2**64-1, 'offset')
    aotx_integer(limit, 1, 256, 'limit')
    p = bytearray(READ_HEAD); p[:8] = b'AOTXSHR1'
    struct.pack_into('<I', p, 8, kind); p[16:32] = lineage; p[32:48] = target; p[48:64] = parent
    struct.pack_into('<QQI', p, 64, cursor, byte, limit)
    return bytes(p)


def aotx_shared_mutation(state, principal, body, operation, *, target=None, space=None):
    common = {'schema', 'lineage', 'operation_key', 'sequence'}
    fields = {REGISTER: set(), SPACE: {'id', 'scope'}, MEMBER: {'participant', 'permissions'},
        CONVERSATION: {'id'}, INPUT: {'text', 'model', 'max_output_tokens', 'pages', 'temperature', 'top_p', 'media', 'control'},
        CANCEL: {'target_sequence'}, RETIRE: {'retry_floor'}, PUBLISH: {'source_version'}, SAVE: set()}
    required = {MEMBER: {'participant', 'permissions'}, INPUT: {'text', 'model'}, CANCEL: {'target_sequence'},
        RETIRE: {'retry_floor'}, PUBLISH: {'source_version'}}
    aotx_fields(body, common | fields[operation], common | required.get(operation, set()))
    if body['schema'] != 'aotx.shared.mutation.v1': raise aotx_bad('schema')
    lineage = aotx_hex(body['lineage'], 16, 'lineage')
    key = aotx_hex(body['operation_key'], 16, 'operation_key')
    if not any(lineage) or not any(key): raise aotx_bad('operation_key')
    sequence = aotx_shared_counter(body['sequence'], 'sequence', 1)
    p = bytearray(COMMAND_HEAD); p[:8] = b'AOTXSHR1'
    struct.pack_into('<I', p, 8, operation); struct.pack_into('<Q', p, 16, sequence)
    p[24:40] = key; p[40:56] = lineage
    if target is not None: p[56:72] = target
    if space is not None: p[72:88] = space
    tail = b''
    if operation in (SPACE, CONVERSATION):
        identity = aotx_hex(body.get('id', key.hex()), 16, 'id')
        if not any(identity): raise aotx_bad('id')
        p[56:72] = identity
    if operation == SPACE:
        scope = body.get('scope', 'private')
        if not isinstance(scope, str) or scope not in SCOPES: raise aotx_bad('scope')
        struct.pack_into('<I', p, 12, SCOPES[scope])
    if operation == MEMBER:
        member = aotx_hex(body['participant'], 16, 'participant')
        if not any(member): raise aotx_bad('participant')
        permissions = body['permissions']
        if not isinstance(permissions, list) or len(permissions) > 3 or any(not isinstance(v, str) or v not in RIGHTS for v in permissions):
            raise aotx_bad('permissions')
        if len(set(permissions)) != len(permissions): raise aotx_bad('permissions')
        p[88:104] = member; struct.pack_into('<I', p, 116, sum(RIGHTS[v] for v in permissions))
    if operation == INPUT:
        if 'control' in body: p[144:192] = aotx_control_selection(body['control'])
        text = body['text']
        if not isinstance(text, str) or '\x00' in text: raise aotx_bad('text')
        try: tail = text.encode('utf-8')
        except UnicodeError: raise aotx_bad('text') from None
        if len(tail) > 2048: raise aotx_error(413, 'The input text exceeds 2048 bytes.', 'input_limit')
        model = body['model']
        if not isinstance(model, str) or model not in principal.models or model not in state.config.models: raise aotx_bad('model')
        role = ROLES[state.config.models[model]['role']]
        limit = aotx_integer(body.get('max_output_tokens', 256), 1, principal.tokens, 'max_output_tokens')
        pages = aotx_integer(body.get('pages', 0), 0, principal.pages, 'pages')
        temperature = aotx_number(body.get('temperature', 0.0), 0, 2, 'temperature')
        top_p = aotx_number(body.get('top_p', 1.0), 0.000001, 1, 'top_p')
        media = body.get('media', [])
        if not isinstance(media, list) or len(media) > 8 or (not text and not media): raise aotx_bad('media')
        struct.pack_into('<III', p, 104, role, limit, pages)
        struct.pack_into('<ff', p, 120, temperature, top_p); struct.pack_into('<II', p, 136, len(tail), len(media))
        for value in media:
            aotx_fields(value, {'type', 'sha256'}, {'type', 'sha256'}, 'media')
            if value['type'] not in ('image', 'audio'): raise aotx_bad('media.type')
            digest = aotx_hex(value['sha256'], 32, 'media.sha256')
            if not any(digest): raise aotx_bad('media.sha256')
            tail += struct.pack('<II', 1 if value['type'] == 'image' else 2, 0) + digest
        if len(text.encode('utf-8')) + len(media)*73 > 2048: raise aotx_error(413, 'The text and media links exceed 2048 bytes.', 'input_limit')
    number = {CANCEL: 'target_sequence', RETIRE: 'retry_floor', PUBLISH: 'source_version'}.get(operation)
    if number: struct.pack_into('<Q', p, 128, aotx_shared_counter(body[number], number, 1))
    return bytes(p) + tail
