# SPDX-License-Identifier: Apache-2.0
# Validate message envelopes and transfer typed spans to device admission.
from dataclasses import dataclass
import re
import struct
import uuid
from .capabilities import aotx_information
from .controls import aotx_control_selection
from .errors import aotx_bad, aotx_error
from .json_wire import aotx_fields, aotx_integer, aotx_number
from .media import aotx_base64, aotx_media_id, aotx_media_status, aotx_media_upload
from .wire import SUBMIT, FRAME, HEAD


@dataclass(frozen=True)
class aotx_submission:
    identity: bytes
    epoch: int
    model: str
    stream: bool
    usage: bool
    media: tuple[str, ...]
    output_bytes: int

    @property
    def handle(self): return aotx_handle(self.epoch, self.identity)


def aotx_handle(epoch, identity): return 'req-%016x-%s' % (epoch, identity.hex())


def aotx_parse_handle(value):
    if not isinstance(value, str) or not re.fullmatch('req-[0-9a-f]{16}-[0-9a-f]{32}', value):
        raise aotx_bad('request_id')
    epoch, identity = value[4:20], value[21:]
    if not int(epoch, 16) or not int(identity, 16): raise aotx_bad('request_id')
    return int(epoch, 16), bytes.fromhex(identity)


def aotx_part(part, native, role):
    if not isinstance(part, dict): raise aotx_bad('messages.content')
    kind = part.get('type')
    if kind == 'text':
        aotx_fields(part, {'type', 'text'}, {'type', 'text'}, 'messages.content')
        if not isinstance(part['text'], str): raise aotx_bad('messages.content.text')
        return 0, 'text', part['text']
    if role != 1: raise aotx_bad('messages.content', 'Media requires the user role.')
    if kind == 'image_url':
        aotx_fields(part, {'type', 'image_url'}, {'type', 'image_url'}, 'messages.content')
        image = aotx_fields(part['image_url'], {'url', 'detail'}, {'url'}, 'image_url')
        if image.get('detail', 'auto') != 'auto' or not isinstance(image['url'], str): raise aotx_bad('image_url')
        if image['url'].startswith('data:'):
            prefix = 'data:image/jpeg;base64,'
            if not image['url'].startswith(prefix): raise aotx_bad('image_url', 'Only JPEG data URLs are supported.')
            return 1, 'base64', image['url'][len(prefix):]
        return 1, 'url', image['url']
    if kind == 'input_audio':
        aotx_fields(part, {'type', 'input_audio'}, {'type', 'input_audio'}, 'messages.content')
        audio = aotx_fields(part['input_audio'], {'data', 'format'}, {'data', 'format'}, 'input_audio')
        if audio['format'] != 'wav' or not isinstance(audio['data'], str): raise aotx_bad('input_audio')
        return 2, 'base64', audio['data']
    if kind == 'media' and native:
        aotx_fields(part, {'type', 'media_id', 'modality'}, {'type', 'media_id', 'modality'}, 'messages.content')
        if part['modality'] not in ('image', 'audio'): raise aotx_bad('modality')
        return 1 if part['modality'] == 'image' else 2, 'media', aotx_media_id(part['media_id'])
    raise aotx_bad('messages.content.type')


def aotx_envelope(body, info, native):
    allowed = {'model', 'messages', 'temperature', 'top_p', 'max_tokens', 'max_completion_tokens',
        'stream', 'stream_options', 'n', 'modalities', 'store'}
    if native: allowed.add('control')
    aotx_fields(body, allowed, {'model', 'messages'})
    if 'control' in body: aotx_control_selection(body['control'])
    model = body['model']
    if not isinstance(model, str) or model not in info['models']: raise aotx_error(404, 'The model is unavailable.', 'model_not_found', 'model')
    if type(body.get('stream', False)) is not bool or (native and body.get('stream', False)): raise aotx_bad('stream')
    if 'stream_options' in body:
        aotx_fields(body['stream_options'], {'include_usage'}, {'include_usage'}, 'stream_options')
        if not body.get('stream') or type(body['stream_options']['include_usage']) is not bool: raise aotx_bad('stream_options')
    if type(body.get('n', 1)) is not int or body.get('n', 1) != 1: raise aotx_bad('n')
    if body.get('modalities', ['text']) != ['text']: raise aotx_bad('modalities')
    if body.get('store', False) is not False: raise aotx_bad('store')
    if 'max_tokens' in body and 'max_completion_tokens' in body: raise aotx_bad('max_completion_tokens')
    tokens = aotx_integer(body.get('max_completion_tokens', body.get('max_tokens', min(256, info['limits']['output_tokens']))),
        1, info['limits']['output_tokens'], 'max_completion_tokens')
    temperature = aotx_number(body.get('temperature', 1.0), 0, 2, 'temperature')
    top_p = aotx_number(body.get('top_p', 1.0), 0, 1, 'top_p')
    if not top_p: raise aotx_bad('top_p')
    messages = body['messages']
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64: raise aotx_bad('messages')
    prepared, text_bytes, media_count = [], 0, 0
    for message in messages:
        aotx_fields(message, {'role', 'content'}, {'role', 'content'}, 'messages')
        if message['role'] not in ('system', 'user', 'assistant'): raise aotx_bad('messages.role')
        role = ('system', 'user', 'assistant').index(message['role'])
        content = message['content']
        if isinstance(content, str): parts = [(0, 'text', content)]
        elif isinstance(content, list) and 1 <= len(content) <= 32:
            parts = [aotx_part(p, native, role) for p in content]
        else: raise aotx_bad('messages.content')
        for kind, source, value in parts:
            if kind and ('image' if kind == 1 else 'audio') not in info['models'][model]['input']:
                raise aotx_bad('messages.content', 'The model does not accept this media type.')
            if source == 'text': text_bytes += len(value.encode('utf-8'))
            else: media_count += 1
        prepared.append((role, parts))
    if text_bytes + media_count*72 > info['limits']['wrapped_prompt_bytes']:
        raise aotx_error(413, 'The message content exceeds the prompt capacity.', 'context_limit')
    return prepared, tokens, temperature, top_p


async def aotx_submit(state, principal, body, native=False):
    if not principal.actions & 1: raise aotx_error(403, 'The grant does not permit inference.', 'inference_forbidden')
    info = await aotx_information(state, principal)
    messages, tokens, temperature, top_p = aotx_envelope(body, info, native)
    control = aotx_control_selection(body['control']) if 'control' in body else b''
    if control and not info.get('control_selection'):
        raise aotx_error(503, 'Control selection is unavailable.', 'control_selection')
    payload = bytearray(struct.pack('<I', len(messages)))
    media = []
    for role, parts in messages:
        payload.extend(struct.pack('<II', role, len(parts)))
        for kind, source, value in parts:
            if source == 'text': data = value.encode('utf-8')
            else:
                mime = 'image/jpeg' if kind == 1 else 'audio/wav'
                if source == 'media': result = await aotx_media_status(state, principal, value)
                elif source == 'url':
                    async with state.fetcher.get(principal, value) as (raw, content_type):
                        if content_type != mime: raise aotx_error(415, 'The source media type does not match.', 'media_type')
                        result = await aotx_media_upload(state, principal, raw, mime)
                    del raw
                else:
                    limit = min(state.config.limits['upload_bytes'], principal.media_bytes)
                    async with state.budget.claim('body_buffers', min(len(value)//4*3, limit)):
                        raw = aotx_base64(value, limit, 'messages.content')
                        result = await aotx_media_upload(state, principal, raw, mime)
                    del raw
                if result['phase'] != 6 or result['format'] != (1 if kind == 1 else 3):
                    raise aotx_error(409, 'The media source is not ready for this request.', 'media_state')
                media.append(result['id']); data = bytes.fromhex(result['sha256'])
            payload.extend(struct.pack('<II', kind, len(data))); payload.extend(data)
            if len(payload) > FRAME-HEAD: raise aotx_error(413, 'The message envelope is too large.', 'frame_limit')
    identity = uuid.uuid4().bytes
    try:
        await state.wire.call(principal, SUBMIT, epoch=info['epoch'], identity=identity,
            role=info['models'][body['model']]['role'], limit=tokens, temperature=temperature, top_p=top_p, payload=payload, control=control)
    except aotx_error as error:
        error.request_id = aotx_handle(info['epoch'], identity)
        error.media = tuple(dict.fromkeys(media))
        raise
    return aotx_submission(identity, info['epoch'], body['model'], body.get('stream', False),
        body.get('stream_options', {}).get('include_usage', False), tuple(dict.fromkeys(media)), info['limits']['output_bytes'])
