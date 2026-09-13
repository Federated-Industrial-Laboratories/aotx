# SPDX-License-Identifier: Apache-2.0
# Translate current device grants and capacities into public service resources.
import struct
from .config import ROLES
from .errors import aotx_error
from .wire import INFO, METRICS


async def aotx_information(state, principal, telemetry=False):
    reply = await state.wire.call(principal, METRICS if telemetry else INFO)
    p = reply.data
    if len(p) < 192: raise aotx_error(503, 'The capability response is invalid.', 'invalid_device_response')
    v = struct.unpack_from('<16I', p)
    if v[0] != 1 or len(p) != 192+40*v[11]:
        raise aotx_error(503, 'The capability response is invalid.', 'invalid_device_response')
    roles = {}
    for at in range(192, len(p), 40):
        role, modalities = struct.unpack_from('<II', p, at)
        roles[role] = {'role': role, 'sha256': p[at+8:at+40].hex(),
            'input': [name for bit, name in ((1, 'text'), (2, 'image'), (4, 'audio')) if modalities & bit]}
    models = {}
    for alias in principal.models:
        model = state.config.models[alias]
        if ROLES[model['role']] in roles:
            models[alias] = dict(roles[ROLES[model['role']]], published_at=model['published_at'])
    return {'epoch': reply.epoch, 'lineage': p[96:112].hex() if any(p[96:112]) else None,
        'models': models, 'actions': v[10], 'shared': bool(struct.unpack_from('<I', p, 152)[0]), 'limits': {
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
        'created': m['published_at'], 'owned_by': 'operator'} for alias, m in info['models'].items()]}


def aotx_capabilities(state, info):
    return {'schema': 'aotx.capabilities.v1', 'runtime_epoch': str(info['epoch']),
        'lineage': info['lineage'], 'mode': 'inference',
        'models': [{'id': alias, **m} for alias, m in info['models'].items()],
        'limits': dict(info['limits'], **state.config.limits),
        'features': {'chat_completions': bool(info['actions'] & 1), 'streaming': bool(info['actions'] & 1),
            'private_media': bool(info['actions'] & 2), 'https_import': bool(info['actions'] & 4),
            'telemetry': bool(info['actions'] & 8), 'continuing_ccir': info.get('shared', False),
            'persistent_requests': info.get('shared', False),
            'embeddings': False, 'responses': False, 'tools': False, 'expression': False, 'presence': False},
        'retention': {'requests': 'bounded_device_results', 'restart': 'expire_requests',
            'media': 'explicit_removal', 'shared_operations': 'saved_until_recorded_retirement' if info.get('shared') else None},
        'shared_api': '/aotx/v1/shared' if info.get('shared') else None, 'media_types': ['image/jpeg', 'audio/wav']}
