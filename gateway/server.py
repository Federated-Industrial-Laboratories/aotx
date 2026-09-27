# SPDX-License-Identifier: Apache-2.0
# Serve authenticated bounded HTTP requests through the scoped device client.
import asyncio
from contextlib import asynccontextmanager
import logging
import re
from aiohttp import web
from .capabilities import aotx_information, aotx_models, aotx_capabilities
from .errors import aotx_error, aotx_bad
from .fetch import aotx_fetcher
from .json_wire import aotx_encode, aotx_fields, aotx_json
from .limits import aotx_budget, aotx_read_body
from .media import aotx_media_id, aotx_media_list, aotx_media_status, aotx_media_frame, aotx_media_upload
from .output import aotx_completion, aotx_cursor, aotx_read, aotx_status, aotx_stream
from .requests import aotx_parse_handle, aotx_submit
from .wire import aotx_wire
from .shared import aotx_shared_route
from .affect import aotx_affect_route
from .policy import aotx_policy_route

LOG = logging.getLogger('aotx.gateway')


def aotx_media_headers(headers, identities):
    value = ','.join(identities)
    if value and len(value) <= 4096: headers['X-AOTX-Media-Ids'] = value


class aotx_state:
    def __init__(self, config):
        self.config = config
        self.budget = aotx_budget(config.limits)
        self.wire = aotx_wire(config.socket, config.limits['wire_connections'])
        self.fetcher = aotx_fetcher(config, self.budget)


class aotx_server(web.Server):
    def __init__(self, state):
        self.state = state
        self.transports = {}
        super().__init__(self.dispatch, handler_cancellation=True, keepalive_timeout=0,
            max_line_size=8190, max_headers=64, max_field_size=8190, lingering_time=0,
            read_bufsize=16384, auto_decompress=False, access_log=None)

    def connection_made(self, handler, transport):
        super().connection_made(handler, transport)
        if len(self.transports) >= self.state.config.limits['connections']:
            transport.close(); return
        timer = asyncio.get_running_loop().call_later(self.state.config.limits['header_seconds'], transport.close)
        self.transports[transport] = (handler, timer)

    def connection_lost(self, handler, exc=None):
        for transport, (owner, timer) in tuple(self.transports.items()):
            if owner is handler:
                if timer: timer.cancel()
                del self.transports[transport]
        super().connection_lost(handler, exc)

    async def dispatch(self, request):
        response = await self.handle_request(request)
        if isinstance(response, web.Response):
            try:
                seconds = self.state.config.limits['write_seconds']
                await asyncio.wait_for(response.prepare(request), seconds)
                await asyncio.wait_for(response.write_eof(), seconds)
            except (ConnectionError, TimeoutError):
                if request.transport: request.transport.abort()
        return response

    async def handle_request(self, request):
        headers = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Vary': 'Origin', 'Connection': 'close'}
        transport = request.transport
        if transport in self.transports:
            handler, timer = self.transports[transport]
            if timer: timer.cancel()
            self.transports[transport] = (handler, None)
        try:
            for name in ('Authorization', 'Origin', 'Content-Type', 'Expect', 'Last-Event-ID'):
                if len(request.headers.getall(name, [])) > 1: raise aotx_bad('headers')
            origin = request.headers.get('Origin')
            if origin:
                if origin not in self.state.config.origins: raise aotx_error(403, 'The origin is not permitted.', 'origin_forbidden')
                headers['Access-Control-Allow-Origin'] = origin
                headers['Access-Control-Expose-Headers'] = 'X-Request-ID, X-AOTX-Media-Ids, Retry-After'
            if request.method == 'OPTIONS':
                if not origin: raise aotx_bad('Origin')
                method = request.headers.get('Access-Control-Request-Method')
                requested = {v.strip().lower() for v in request.headers.get('Access-Control-Request-Headers', '').split(',') if v.strip()}
                if method not in ('GET', 'POST', 'DELETE') or requested - {'authorization', 'content-type', 'last-event-id'}:
                    raise aotx_bad('headers')
                headers.update({'Access-Control-Allow-Methods': 'GET, POST, DELETE',
                    'Access-Control-Allow-Headers': 'Authorization, Content-Type, Last-Event-ID'})
                return self.response(None, headers, 204)
            principal = self.state.config.authenticate(request.headers.get('Authorization'))
            if request.method in ('GET', 'DELETE'):
                if request.can_read_body: raise aotx_bad('body')
                if transport: transport.pause_reading()
            result = await self.route(request, principal, headers)
            return result
        except aotx_error as error:
            if error.request_id: headers['X-Request-ID'] = error.request_id
            aotx_media_headers(headers, error.media)
            if error.status == 401: headers['WWW-Authenticate'] = 'Bearer'
            if error.status in (429, 503): headers['Retry-After'] = '1'
            return self.response(error.body(), headers, error.status)
        except (ConnectionError, asyncio.CancelledError): raise
        except TimeoutError:
            return self.response(aotx_error(504, 'The operation deadline expired.', 'operation_timeout').body(), headers, 504)
        except Exception as error:
            LOG.error('The request failed (%s).', type(error).__name__)
            return self.response(aotx_error(500, 'The request failed.', 'internal_error').body(), headers, 500)

    @staticmethod
    def response(value, headers, status=200):
        response = web.Response(status=status, body=aotx_encode(value) if value is not None else None,
            content_type='application/json' if value is not None else None, headers=headers)
        response.force_close()
        return response

    @asynccontextmanager
    async def body(self, request, raw=False):
        limit = self.state.config.limits['upload_bytes' if raw else 'json_bytes']
        if not raw and request.content_type != 'application/json':
            raise aotx_error(415, 'This route requires a JSON body.', 'content_type')
        amount = limit if request.content_length is None else request.content_length
        if amount > limit: raise aotx_error(413, 'The request body is too large.', 'body_limit')
        async with self.state.budget.claim('body_buffers', amount) as credit:
            async with self.state.budget.claim('body_readers'):
                expect = request.headers.get('Expect')
                if expect:
                    if expect.lower() != '100-continue': raise aotx_error(417, 'The expectation is not supported.', 'expectation')
                    await asyncio.wait_for(request.writer.write(b'HTTP/1.1 100 Continue\r\n\r\n'),
                        self.state.config.limits['write_seconds'])
                    request.writer.output_size = 0
                data = await aotx_read_body(request, limit, self.state.config.limits['body_seconds'])
                credit.resize(len(data))
                value = data if raw else aotx_json(data)
            yield value

    async def route(self, request, principal, headers):
        state, path, method = self.state, request.path, request.method
        affect_result = await aotx_affect_route(self, request, principal, headers)
        if affect_result is not None: return affect_result
        policy_result = await aotx_policy_route(self, request, principal, headers)
        if policy_result is not None: return policy_result
        shared_result = await aotx_shared_route(self, request, principal, headers)
        if shared_result is not None: return shared_result
        match = re.fullmatch(r'/aotx/v1/requests/(req-[0-9a-f]{16}-[0-9a-f]{32})(/(cancel|events))?', path)
        allowed_query = {'cursor'} if method == 'GET' and (match or path == '/aotx/v1/media') else set()
        if set(request.query)-allowed_query or any(len(request.query.getall(k)) != 1 for k in request.query):
            raise aotx_bad('query')
        if method == 'GET' and path in ('/v1/models', '/aotx/v1/capabilities', '/aotx/v1/telemetry'):
            info = await aotx_information(state, principal, path.endswith('/telemetry'))
            if path == '/v1/models': value = aotx_models(info)
            elif path.endswith('/capabilities'): value = aotx_capabilities(state, info)
            else: value = {'schema': 'aotx.telemetry.v1', 'runtime_epoch': str(info['epoch']),
                'lineage': info['lineage'], 'sample': info['sample'], 'affect': None, 'expression': None}
            return self.response(value, headers)
        if method == 'POST' and path in ('/v1/chat/completions', '/aotx/v1/requests'):
            native = path.startswith('/aotx/')
            async with state.budget.claim('work'):
                async with self.body(request) as body: submission = await aotx_submit(state, principal, body, native)
                del body
                headers['X-Request-ID'] = submission.handle
                aotx_media_headers(headers, submission.media)
                if native:
                    return self.response({'schema': 'aotx.admission.v1', 'id': submission.handle,
                        'runtime_epoch': str(submission.epoch), 'state': 'accepted', 'persistence': 'ephemeral',
                        'media': list(submission.media)}, headers, 202)
                if submission.stream:
                    return await aotx_stream(state, request, principal, submission.epoch,
                        submission.identity, headers, submission)
                return self.response(await aotx_completion(state, principal, submission), headers)
        if match:
            handle, action = match[1], match[3]
            epoch, identity = aotx_parse_handle(handle)
            headers['X-Request-ID'] = handle
            cursor = aotx_cursor(request.query.get('cursor', '0'))
            if action == 'cancel' and method == 'POST':
                async with self.body(request) as body: aotx_fields(body, {})
                return self.response(aotx_status(await aotx_read(state, principal, epoch, identity, cancel=True)), headers)
            if action == 'events' and method == 'GET':
                last = request.headers.get('Last-Event-ID')
                if last:
                    if 'cursor' in request.query or not last.startswith(handle+':'): raise aotx_bad('Last-Event-ID')
                    cursor = aotx_cursor(last[len(handle)+1:])
                async with state.budget.claim('work'):
                    return await aotx_stream(state, request, principal, epoch, identity, headers, cursor=cursor)
            if action is None and method == 'GET':
                return self.response(aotx_status(await aotx_read(state, principal, epoch, identity, cursor)), headers)
        if path == '/aotx/v1/media' and method == 'POST':
            async with state.budget.claim('work'):
                async with self.body(request, raw=True) as data:
                    result = await aotx_media_upload(state, principal, data, request.content_type)
            return self.response(result, headers, 201)
        if path == '/aotx/v1/media' and method == 'GET':
            return self.response(await aotx_media_list(state, principal, aotx_cursor(request.query.get('cursor', '0'))), headers)
        if path == '/aotx/v1/media/import' and method == 'POST':
            async with state.budget.claim('work'):
                async with self.body(request) as body:
                    aotx_fields(body, {'url'}, {'url'})
                    async with state.fetcher.get(principal, body['url']) as (data, mime):
                        result = await aotx_media_upload(state, principal, data, mime)
            return self.response(result, headers, 201)
        if re.fullmatch('/aotx/v1/media/media-[0-9a-f]{32}', path):
            identity = aotx_media_id(path.rsplit('/', 1)[1])
            if method == 'GET': return self.response(await aotx_media_status(state, principal, identity), headers)
            if method == 'DELETE':
                await aotx_media_frame(state, principal, identity, 4)
                return self.response(None, headers, 204)
        raise aotx_error(404, 'The route is not available in this service profile.', 'route_not_found')
