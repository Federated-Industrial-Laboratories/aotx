# SPDX-License-Identifier: Apache-2.0
# Translate current device grants and capacities into public service resources.
import struct
from .config import ROLES
from .errors import aotx_error
from .wire import INFO, METRICS


def aotx_controls(p, at, roles):
    count, stride = struct.unpack_from('<II', p, 160)
    if stride != 160 or count > 16 or len(p) != at + count * stride:
        raise aotx_error(503, 'The control capability response is invalid.', 'invalid_device_response')
    for offset in range(at, len(p), stride):
        role, kind, accepted, positions, hook, doses, layers = struct.unpack_from('<6IQ', p, offset)
        raw = p[offset+32:offset+64]
        name, separator, rest = raw.partition(b'\0')
        if (role not in roles or kind != 1 or accepted > 1 or positions > 1 or hook != 1 or
                doses > 16 or bool(doses) != bool(accepted) or not layers or not separator or any(rest) or
                not name or any(c not in b'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name)):
            raise aotx_error(503, 'The control capability response is invalid.', 'invalid_device_response')
        values = struct.unpack_from('<16i', p, offset+96)
        if (any(v == 0 or abs(v) > 40000 for v in values[:doses]) or
                len(set(values[:doses])) != doses or any(values[doses:])):
            raise aotx_error(503, 'The control dose response is invalid.', 'invalid_device_response')
        controls = roles[role]['controls']
        if any(row['name'] == name.decode('ascii') for row in controls):
            raise aotx_error(503, 'The control name is repeated.', 'invalid_device_response')
        controls.append({'schema': 'aotx.control.v1', 'name': name.decode('ascii'), 'kind': 'residual_vector',
            'available': bool(accepted), 'positions': 'response' if positions else 'all', 'hook': hook,
            'layers': [i for i in range(64) if layers & (1 << i)], 'dose_scale': 10000,
            'accepted_doses': list(values[:doses]), 'combinations': False,
            'qualification_sha256': p[offset+64:offset+96].hex() if any(p[offset+64:offset+96]) else None})


async def aotx_information(state, principal, telemetry=False):
    reply = await state.wire.call(principal, METRICS if telemetry else INFO)
    p = reply.data
    if len(p) < 192: raise aotx_error(503, 'The capability response is invalid.', 'invalid_device_response')
    v = struct.unpack_from('<16I', p)
    model_end = 192+40*v[11]
    if v[0] not in (1, 2) or model_end > len(p) or (v[0] == 1 and len(p) != model_end):
        raise aotx_error(503, 'The capability response is invalid.', 'invalid_device_response')
    roles = {}
    memory = struct.unpack_from('<I', p, 156)[0]
    for at in range(192, model_end, 40):
        role, modalities = struct.unpack_from('<II', p, at)
        roles[role] = {'role': role, 'sha256': p[at+8:at+40].hex(), 'controls': [],
            'automatic_memory': bool(role < 32 and memory & (1 << role)),
            'input': [name for bit, name in ((1, 'text'), (2, 'image'), (4, 'audio')) if modalities & bit]}
    if v[0] == 2: aotx_controls(p, model_end, roles)
    selection = struct.unpack_from('<I', p, 168)[0] if v[0] == 2 else 0
    if selection > 1: raise aotx_error(503, 'The control selection version is invalid.', 'invalid_device_response')
    models = {}
    for alias in principal.models:
        model = state.config.models[alias]
        if ROLES[model['role']] in roles:
            models[alias] = dict(roles[ROLES[model['role']]], published_at=model['published_at'])
    return {'epoch': reply.epoch, 'lineage': p[96:112].hex() if any(p[96:112]) else None,
        'models': models, 'control_selection': bool(selection), 'actions': v[10], 'shared': bool(struct.unpack_from('<I', p, 152)[0]), 'limits': {
            'execution_slots': v[1]-1, 'wrapped_prompt_bytes': v[2], 'sequence_tokens': v[3],
            'request_entries': v[4], 'output_bytes': v[5], 'service_channels': v[6]-1,
            'output_tokens': v[7], 'kv_pages_per_request': v[8], 'active_requests': v[9],
            'media_objects': v[12], 'private_media_objects': v[15],
            'private_media_bytes': str(struct.unpack_from('<Q', p, 64)[0]),
            **dict(zip(('image_pixels', 'image_dimension', 'image_patches', 'image_feature_rows',
                'audio_source_frames', 'audio_feature_rows', 'device_request_seconds', 'device_upload_seconds',
                'principal_entries', 'media_references'), struct.unpack_from('<10I', p, 112)))},
        'sample': {'service_allocation_bytes': str(struct.unpack_from('<Q', p, 72)[0]),
            'tick': str(struct.unpack_from('<Q', p, 80)[0]),
            'clock_nanoseconds': str(struct.unpack_from('<Q', p, 88)[0])}}


def aotx_models(info):
    return {'object': 'list', 'data': [{'id': alias, 'object': 'model',
        'created': m['published_at'], 'owned_by': 'operator',
        'automatic_memory': m['automatic_memory']} for alias, m in info['models'].items()]}


def aotx_capabilities(state, info):
    return {'schema': 'aotx.capabilities.v1', 'runtime_epoch': str(info['epoch']),
        'lineage': info['lineage'], 'mode': 'inference',
        'models': [{'id': alias, **m} for alias, m in info['models'].items()],
        'limits': dict(info['limits'], **state.config.limits),
        'features': {'control_selection': info.get('control_selection', False), 'chat_completions': bool(info['actions'] & 1), 'streaming': bool(info['actions'] & 1),
            'private_media': bool(info['actions'] & 2), 'https_import': bool(info['actions'] & 4),
            'telemetry': bool(info['actions'] & 8), 'continuing_ccir': info.get('shared', False),
            'persistent_requests': info.get('shared', False),
            'embeddings': False, 'responses': False, 'tools': False, 'expression': False, 'presence': False},
        'retention': {'requests': 'bounded_device_results', 'restart': 'expire_requests',
            'media': 'explicit_removal', 'shared_operations': 'saved_until_recorded_retirement' if info.get('shared') else None},
        'shared_api': '/aotx/v1/shared' if info.get('shared') else None, 'media_types': ['image/jpeg', 'audio/wav']}
