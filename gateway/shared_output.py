# SPDX-License-Identifier: Apache-2.0
# Decode shared device replies; inputs are bounded bytes, output is scoped JSON, malformed replies fail closed.
import base64
import math
import struct
from .errors import aotx_error
from .shared_wire import (REPLY_HEAD, CAPABILITIES, PARTICIPANT, SPACES, SPACE_READ, MEMBERS,
    CONVERSATIONS, CONVERSATION_READ, OPERATION, EVENTS, MEMORY, SAVE_READ, AFFECT, aotx_shared_handle)

PHASES = ('free', 'accepted', 'queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted')
SCOPES = ('private', 'room', 'instance')


def aotx_shared_invalid():
    return aotx_error(503, 'The shared device response is invalid.', 'invalid_device_response')


def aotx_shared_bytes(data):
    try: text = data.decode('utf-8')
    except UnicodeError: text = None
    return {'text': text, 'base64': base64.b64encode(data).decode('ascii'), 'bytes': str(len(data))}


def aotx_shared_decode(reply):
    p = reply.data
    if len(p) < REPLY_HEAD or p[:8] != b'AOTXSHR1' or any(p[312:320]): raise aotx_shared_invalid()
    u32 = lambda at: struct.unpack_from('<I', p, at)[0]
    u64 = lambda at: struct.unpack_from('<Q', p, at)[0]
    kind, phase, flags, scope = u32(8), u32(12), u32(172), u32(228)
    if kind not in range(1, 13) or phase >= len(PHASES) or flags & ~15 or scope >= len(SCOPES): raise aotx_shared_invalid()
    lineage, target, space, actor, key = p[16:32], p[32:48], p[48:64], p[64:80], p[240:256]
    if not any(lineage): raise aotx_shared_invalid()
    if kind == OPERATION and (not any(actor) or not any(p[296:312])): raise aotx_shared_invalid()
    count, row, cursor, offset = u32(192), u32(196), u64(200), u64(208)
    data = p[REPLY_HEAD:]
    result = {'schema': 'aotx.shared.resource.v1', 'lineage': lineage.hex(),
        'save': {'source': str(u64(136)), 'generation': str(u64(144)), 'incarnation': p[152:168].hex(),
            'boot': str(u64(256)), 'commit_sha256': p[264:296].hex(), 'pending_bytes': str(u64(216)), 'error': u32(224)}}
    if kind in (CAPABILITIES, PARTICIPANT):
        result.update({'participant': actor.hex(), 'registered': bool(flags & 1),
            'next_sequence': str(u64(88)), 'retry_floor': str(u64(96))})
    if kind == CAPABILITIES:
        if count != 1 or row != 40 or len(data) != 40: raise aotx_shared_invalid()
        names = ('participants', 'spaces', 'conversations', 'members', 'receipts', 'command_bytes',
            'result_bytes', 'input_bytes', 'media_references', 'records_per_tick')
        result['limits'] = dict(zip(names, struct.unpack('<10I', data)))
        result['scopes'] = list(SCOPES); result['persistence'] = 'complete_runtime'
    elif kind in (SPACES, MEMBERS, CONVERSATIONS, EVENTS):
        expected = {SPACES: 64, MEMBERS: 32, CONVERSATIONS: 64, EVENTS: 96}[kind]
        if row != expected or count > 256 or len(data) != count * row: raise aotx_shared_invalid()
        items = []
        for at in range(0, len(data), row):
            r = data[at:at+row]
            if kind == SPACES:
                scope_value, rights = struct.unpack_from('<II', r, 32)
                if scope_value >= 3 or rights & ~7 or any(r[40:]): raise aotx_shared_invalid()
                item = {'id': aotx_shared_handle('spc', lineage, r[:16]), 'owner': r[16:32].hex(),
                    'scope': SCOPES[scope_value], 'permissions': aotx_shared_rights(rights)}
            elif kind == MEMBERS:
                rights = struct.unpack_from('<I', r, 16)[0]
                if rights & ~7 or any(r[20:]): raise aotx_shared_invalid()
                item = {'participant': r[:16].hex(), 'permissions': aotx_shared_rights(rights)}
            elif kind == CONVERSATIONS:
                order, floor, busy = struct.unpack_from('<QQI', r, 32)
                if busy > 1 or any(r[52:]): raise aotx_shared_invalid()
                item = {'id': aotx_shared_handle('con', lineage, r[:16]),
                    'space': aotx_shared_handle('spc', lineage, r[16:32]), 'next_order': str(order),
                    'event_floor': str(floor), 'busy': bool(busy)}
            else:
                sequence, order, state, status, size, tokens, finish, saved = struct.unpack_from('<QQIIIIII', r, 32)
                if state >= len(PHASES) or saved & ~3 or any(r[92:]): raise aotx_shared_invalid()
                item = {'id': aotx_shared_handle('op', lineage, r[:16]), 'actor': r[16:32].hex(),
                    'sequence': str(sequence) if sequence else None, 'input_order': str(order), 'state': PHASES[state],
                    'status': status, 'output_bytes': str(size), 'output_tokens': tokens, 'finish': finish,
                    'saved_admission': bool(saved & 1), 'saved_terminal': bool(saved & 2),
                    'admission_source': str(struct.unpack_from('<Q', r, 72)[0]),
                    'terminal_source': str(struct.unpack_from('<Q', r, 80)[0]), 'gap': bool(struct.unpack_from('<I', r, 88)[0])}
            items.append(item)
        result.update({'items': items, 'next_cursor': str(cursor)})
        if kind == EVENTS: result.update({'gap': bool(flags & 8), 'event_floor': str(u64(112)), 'next_order': str(u64(104))})
    elif kind == OPERATION:
        if count or row or offset > u32(176) or len(data) > u32(176)-offset: raise aotx_shared_invalid()
        result.update({'id': aotx_shared_handle('op', lineage, p[296:312]), 'actor': actor.hex(),
            'operation_key': key.hex(), 'sequence': str(u64(80)) if u64(80) else None, 'next_sequence': str(u64(88)) if u64(88) else None,
            'retry_floor': str(u64(96)) if u64(96) else None, 'state': PHASES[phase], 'status': u32(168),
            'operation': u32(236), 'resource': target.hex(), 'input_order': str(u64(104)),
            'accepted': True, 'device_committed': bool(flags & 1), 'saved_admission': bool(flags & 2),
            'saved_terminal': bool(flags & 4), 'gap': bool(flags & 8),
            'admission_source': str(u64(120)), 'terminal_source': str(u64(128)),
            'output_bytes': str(u32(176)), 'offset': str(offset), 'next_offset': str(offset+len(data)),
            'output': aotx_shared_bytes(data), 'usage': {'input_tokens': u32(180), 'output_tokens': u32(184)},
            'finish': u32(188)})
        if any(space): result['space'] = aotx_shared_handle('spc', lineage, space)
    elif kind == MEMORY:
        if row != 128 or count > 256 or len(data) < count * row: raise aotx_shared_invalid()
        items = []
        for at in range(0, count * row, row):
            r = data[at:at+row]
            version, object_kind, object_scope = struct.unpack_from('<QII', r, 16)
            if object_scope >= 3 or any(r[104:128]): raise aotx_shared_invalid()
            items.append({'id': r[:16].hex(), 'version': str(version), 'kind': object_kind,
                'scope': SCOPES[object_scope], 'owner': r[32:48].hex(), 'room': r[48:64].hex(),
                'bytes': str(struct.unpack_from('<Q', r, 64)[0]), 'source': r[72:88].hex(), 'actor': r[88:104].hex()})
        tail = data[count*row:]
        if tail and count != 1: raise aotx_shared_invalid()
        result.update({'items': items, 'next_cursor': str(cursor), 'next_offset': str(offset),
            'payload': aotx_shared_bytes(tail)})
    elif kind == AFFECT:
        if count != 1 or row != 96 or len(data) != 96: raise aotx_shared_invalid()
        version, enabled, revision = struct.unpack_from('<IIQ', data)
        fast, slow = struct.unpack_from('<4h', data, 16), struct.unpack_from('<4h', data, 24)
        scale, axes, actuators, spent, reason, available, role = struct.unpack_from('<HHIfIII', data, 32)
        if (version != 1 or enabled > 1 or axes != 2 or actuators & ~15 or reason & ~0x7fff or
                available & ~3 or not math.isfinite(spent) or spent < 0 or any(data[88:]) or
                any(fast[2:]) or any(slow[2:])): raise aotx_shared_invalid()
        result['affect'] = {'schema': 'aotx.affect.scope.v1', 'enabled': bool(enabled), 'revision': str(revision),
            'fast_q15': list(fast), 'slow_q15': list(slow), 'scale_q16': scale, 'event_mask': reason,
            'actuator_flags': actuators, 'budget_spent': spent, 'model_role': role,
            'model_sha256': data[56:88].hex() if revision else None,
            'probes_at_last_turn': {name: 'available' if available & (1 << i) else 'unavailable'
                for i, name in enumerate(('valence', 'arousal'))}}
    elif data or count or row: raise aotx_shared_invalid()
    if kind in (SPACE_READ, MEMBERS, CONVERSATIONS, CONVERSATION_READ, EVENTS, MEMORY, AFFECT):
        result.update({'space': aotx_shared_handle('spc', lineage, space), 'scope': SCOPES[scope],
            'permissions': aotx_shared_rights(u32(232))})
    if kind == SPACE_READ: result['id'] = aotx_shared_handle('spc', lineage, target)
    if kind in (CONVERSATION_READ, AFFECT):
        result.update({'id': aotx_shared_handle('con', lineage, target), 'next_order': str(u64(104)),
            'event_floor': str(u64(112)), 'state': PHASES[phase]})
    return result


def aotx_shared_rights(value):
    if value & ~7: raise aotx_shared_invalid()
    return [name for name, bit in (('read', 1), ('write', 2), ('manage', 4)) if value & bit]
