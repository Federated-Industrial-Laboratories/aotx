# SPDX-License-Identifier: Apache-2.0
# Translate bounded policy status and operator requests to the device contract.
import struct
from .errors import aotx_bad, aotx_error
from .json_wire import aotx_fields

ACTIONS = {'pause': 1, 'resume': 2, 'stop': 3, 'review_on': 4, 'review_off': 5}
STATES = ('off', 'quiet', 'active', 'paused', 'stopped', 'error', 'recording')
REASONS = ('quiet', 'foreground', 'paused', 'capacity', 'active', 'disabled')


def aotx_policy_decode(reply):
    p = reply.data
    if len(p) != 160:
        raise aotx_error(503, 'The policy response is invalid.', 'invalid_device_response')
    w = struct.unpack_from('<8I', p)
    v = struct.unpack_from('<11Q', p, 32)
    reason = struct.unpack_from('<I', p, 120)[0]
    if w[0] != 1 or w[3] >= len(STATES) or w[4] > 1 or reason >= len(REASONS) or any(p[124:]):
        raise aotx_error(503, 'The policy response is invalid.', 'invalid_device_response')
    result = {'schema': 'aotx.policy.v1', 'epoch': reply.epoch, 'abi': w[1], 'mode': w[2],
        'state': STATES[w[3]], 'review_enabled': bool(w[4]), 'pending': w[5], 'active_rows': w[6],
        'status': w[7], 'reason': REASONS[reason]}
    result.update(zip(('control_revision', 'source_frontier', 'completed', 'interrupted', 'refused',
        'decision', 'saved_generation', 'maximum_ns', 'last_ns', 'written_bytes', 'result_bytes'), v))
    return result


async def aotx_policy_route(server, request, principal, headers):
    if request.path != '/aotx/v1/policy': return None
    if request.query: raise aotx_bad('query')
    if request.method not in ('GET', 'POST'):
        raise aotx_error(405, 'The method is not supported.', 'method')
    mutation = request.method == 'POST'
    if not principal.actions & (128 if mutation else 136):
        raise aotx_error(403, 'The policy operation is not permitted.', 'policy_permission')
    payload, epoch = b'', 0
    if mutation:
        async with server.body(request) as value:
            aotx_fields(value, {'action', 'epoch', 'control_revision'}, {'action', 'epoch', 'control_revision'}, 'policy')
            action = value['action']
            if not isinstance(action, str) or action not in ACTIONS: raise aotx_bad('action')
            for key in ('epoch', 'control_revision'):
                if type(value[key]) is not int or not 0 <= value[key] < 2**64: raise aotx_bad(key)
            epoch = value['epoch']
            payload = struct.pack('<IIQ', 1, ACTIONS[action], value['control_revision'])
    reply = await server.state.wire.call(principal, 12, epoch=epoch, payload=payload)
    return server.response(aotx_policy_decode(reply), headers)
