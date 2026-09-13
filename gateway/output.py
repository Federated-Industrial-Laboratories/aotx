# SPDX-License-Identifier: Apache-2.0
# Read owned device output and encode standard replies or resumable byte events.
import asyncio
import base64
import codecs
import time
from aiohttp import web
from .errors import aotx_error, aotx_bad
from .json_wire import aotx_encode
from .requests import aotx_handle
from .wire import READ, CANCEL, DONE, FAILED, CANCELLED

PHASES = ('free', 'queued', 'preparing', 'running', 'completed', 'failed', 'cancelled')


def aotx_usage(reply):
    return {'prompt_tokens': reply.prompt, 'completion_tokens': reply.sampled,
        'total_tokens': reply.prompt+reply.sampled}


def aotx_outcome(reply):
    if reply.phase == DONE and reply.finish in (1, 2) and not reply.code and not reply.cancelling:
        return 'stop' if reply.finish == 1 else 'length'
    code = reply.code if 400 <= reply.code <= 599 else 503
    raise aotx_error(code, 'The device request did not complete.',
        'request_cancelled' if reply.phase == CANCELLED else 'request_failed')


async def aotx_read(state, principal, epoch, identity, cursor=0, cancel=False):
    reply = await state.wire.call(principal, CANCEL if cancel else READ,
        epoch=epoch, identity=identity, cursor=cursor)
    if reply.epoch != epoch or reply.cursor != cursor or reply.phase not in range(1, 7) or cursor+len(reply.data) > reply.total:
        raise aotx_error(503, 'The output response is invalid.', 'invalid_device_response')
    return reply


def aotx_status(reply):
    return {'schema': 'aotx.request.v1', 'id': aotx_handle(reply.epoch, reply.identity),
        'runtime_epoch': str(reply.epoch), 'model_role': reply.role,
        'state': PHASES[reply.phase], 'cancel_requested': bool(reply.cancelling),
        'status': reply.code, 'persistence': 'ephemeral', 'usage': aotx_usage(reply),
        'output': {'encoding': 'base64', 'bytes': base64.b64encode(reply.data).decode('ascii'),
            'cursor': str(reply.cursor), 'next_cursor': str(reply.cursor+len(reply.data)), 'total_bytes': str(reply.total)},
        'finish_reason': ('stop' if reply.finish == 1 else 'length') if reply.phase == DONE and reply.finish else None}


async def aotx_outputs(state, principal, epoch, identity, cursor=0, first=None):
    deadline = time.monotonic()+state.config.limits['operation_seconds']
    reply = first
    while True:
        if reply is None: reply = await aotx_read(state, principal, epoch, identity, cursor)
        yield reply
        cursor += len(reply.data)
        if reply.phase >= DONE and cursor == reply.total: return
        if time.monotonic() >= deadline: raise aotx_error(504, 'The response deadline expired.', 'response_timeout')
        if cursor == reply.total: await asyncio.sleep(0.05)
        reply = None


async def aotx_completion(state, principal, submission):
    data = bytearray()
    async for reply in aotx_outputs(state, principal, submission.epoch, submission.identity):
        data.extend(reply.data)
        if len(data) > submission.output_bytes:
            raise aotx_error(413, 'The output exceeds the transport limit.', 'output_limit')
    finish = aotx_outcome(reply)
    try: text = data.decode('utf-8')
    except UnicodeError: raise aotx_error(503, 'The output is not valid UTF-8.', 'output_encoding') from None
    return {'id': 'chatcmpl-'+submission.identity.hex(), 'object': 'chat.completion',
        'created': int(time.time()), 'model': submission.model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text, 'refusal': None},
            'finish_reason': finish, 'logprobs': None}], 'usage': aotx_usage(reply)}


class aotx_events:
    def __init__(self, state, response):
        self.state, self.response = state, response

    async def write(self, data, event=None, identity=None):
        encoded = aotx_encode(data)
        prefix = (('event: '+event+'\n') if event else '') + (('id: '+identity+'\n') if identity else '')
        packet = prefix.encode('ascii')+b'data: '+encoded+b'\n\n'
        if len(packet) > self.state.config.limits['event_bytes']:
            raise aotx_error(503, 'The event exceeds its byte capacity.', 'event_limit')
        await asyncio.wait_for(self.response.write(packet), self.state.config.limits['write_seconds'])

    async def end(self, done=False):
        if done: await asyncio.wait_for(self.response.write(b'data: [DONE]\n\n'), self.state.config.limits['write_seconds'])
        await asyncio.wait_for(self.response.write_eof(), self.state.config.limits['write_seconds'])


async def aotx_stream(state, request, principal, epoch, identity, headers, submission=None, cursor=0):
    first = await aotx_read(state, principal, epoch, identity, cursor)
    response = web.StreamResponse(headers=dict(headers, **{'Content-Type': 'text/event-stream', 'X-Accel-Buffering': 'no'}))
    response.force_close()
    await asyncio.wait_for(response.prepare(request), state.config.limits['write_seconds'])
    events = aotx_events(state, response)
    size = min(4096, (state.config.limits['event_bytes']-768)//6)
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    handle = aotx_handle(epoch, identity)
    created, previous = int(time.time()), None

    def chunk(delta, finish=None, usage=None):
        data = {'id': 'chatcmpl-'+identity.hex(), 'object': 'chat.completion.chunk', 'created': created,
            'model': submission.model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish, 'logprobs': None}]}
        if usage is not None: data['choices'] = []; data['usage'] = usage
        elif submission.usage: data['usage'] = None
        return data

    try:
        if submission: await events.write(chunk({'role': 'assistant', 'content': ''}))
        async for reply in aotx_outputs(state, principal, epoch, identity, cursor, first):
            for at in range(0, len(reply.data), size):
                raw = reply.data[at:at+size]
                if submission:
                    text = decoder.decode(raw)
                    if text: await events.write(chunk({'content': text}))
                else:
                    offset = reply.cursor+at
                    await events.write({'schema': 'aotx.emission.v1', 'request_id': handle,
                        'runtime_epoch': str(epoch), 'source_kind': 'model', 'model_role': reply.role,
                        'offset': str(offset), 'next_cursor': str(offset+len(raw)), 'encoding': 'base64',
                        'bytes': base64.b64encode(raw).decode('ascii')}, 'emission', handle+':'+str(offset+len(raw)))
            terminal = reply.phase >= DONE and reply.cursor+len(reply.data) == reply.total
            marker = (reply.phase, reply.cancelling, reply.code)
            if not submission and (terminal or (reply.phase < DONE and marker != previous)):
                status = aotx_status(reply); status.pop('output')
                await events.write(status, 'status')
                previous = marker
            if terminal and submission:
                finish = aotx_outcome(reply)
                tail = decoder.decode(b'', final=True)
                if tail: await events.write(chunk({'content': tail}))
                await events.write(chunk({}, finish))
                if submission.usage: await events.write(chunk({}, usage=aotx_usage(reply)))
        await events.end(done=submission is not None)
    except (aotx_error, UnicodeError) as error:
        if isinstance(error, UnicodeError): error = aotx_error(503, 'The output is not valid UTF-8.', 'output_encoding')
        try: await events.write(error.body(), None if submission else 'error'); await events.end()
        except (ConnectionError, TimeoutError):
            if request.transport: request.transport.abort()
    except (ConnectionError, TimeoutError):
        if request.transport: request.transport.abort()
    return response


def aotx_cursor(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal() or len(value) > 20 or int(value) > 2**64-1:
        raise aotx_bad('cursor')
    return int(value)
