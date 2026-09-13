# SPDX-License-Identifier: Apache-2.0
# Fetch bounded HTTPS bytes through a checked and pinned destination set.
import asyncio
import ipaddress
import re
import socket
import ssl
from contextlib import asynccontextmanager
from aiohttp import ClientSession, ClientTimeout, DummyCookieJar, TCPConnector, ClientError
from aiohttp.abc import AbstractResolver
from yarl import URL
from .errors import aotx_error, aotx_bad
from .json_wire import aotx_fields

V4_DENY = tuple(ipaddress.ip_network(n) for n in (
    '0.0.0.0/8', '10.0.0.0/8', '100.64.0.0/10', '127.0.0.0/8', '169.254.0.0/16',
    '172.16.0.0/12', '192.0.0.0/24', '192.0.2.0/24', '192.88.99.0/24',
    '192.168.0.0/16', '198.18.0.0/15', '198.51.100.0/24', '203.0.113.0/24',
    '224.0.0.0/4', '240.0.0.0/4'))
V6_GLOBAL = ipaddress.ip_network('2000::/3')
V6_DENY = tuple(ipaddress.ip_network(n) for n in ('2001::/23', '2001:db8::/32', '2002::/16', '3fff::/20'))


def aotx_public(address):
    if address.version == 6 and address.ipv4_mapped: address = address.ipv4_mapped
    if address.version == 4: return not any(address in n for n in V4_DENY)
    return address in V6_GLOBAL and not any(address in n for n in V6_DENY)


def aotx_url(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 8192 or any(ord(c) <= 32 or ord(c) == 127 for c in value) or '#' in value:
        raise aotx_bad('url')
    try:
        url = URL(value)
        host, port = url.raw_host, url.port
        if url.scheme != 'https' or url.user is not None or not host or '%' in host or host.endswith('.') or port is None:
            raise ValueError()
        try:
            address = ipaddress.ip_address(host)
            if address.compressed != host.lower(): raise ValueError()
        except ValueError:
            if ':' in host or re.fullmatch(r'(?:0[xX][0-9a-fA-F]+|[0-9]+)(?:\.(?:0[xX][0-9a-fA-F]+|[0-9]+))*', host):
                raise ValueError() from None
            if not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?', host): raise ValueError()
            if any(not label or len(label) > 63 or label.startswith('-') or label.endswith('-') for label in host.split('.')):
                raise ValueError()
        return url
    except (ValueError, TypeError, UnicodeError):
        raise aotx_bad('url', 'The URL is not supported.') from None


class aotx_resolver(AbstractResolver):
    def __init__(self, host, port, addresses):
        self.host, self.port, self.addresses = host, port, addresses

    async def resolve(self, host, port=0, family=socket.AF_INET):
        if host != self.host or port != self.port: raise OSError('Unbound destination.')
        return [{'hostname': host, 'host': str(a), 'port': port,
            'family': socket.AF_INET if a.version == 4 else socket.AF_INET6,
            'proto': socket.IPPROTO_TCP, 'flags': socket.AI_NUMERICHOST} for a in self.addresses]

    async def close(self): pass


class aotx_fetcher:
    def __init__(self, config, budget):
        self.config, self.budget = config, budget
        self.private = {}
        for item in config.urls.get('private', []):
            aotx_fields(item, {'origin', 'networks'}, {'origin', 'networks'}, 'urls.private')
            url = aotx_url(item['origin'])
            if str(url.origin()) != item['origin'] or not isinstance(item['networks'], list) or not item['networks']:
                raise aotx_bad('urls.private')
            try: networks = tuple(ipaddress.ip_network(n, strict=True) for n in item['networks'])
            except (ValueError, TypeError): raise aotx_bad('urls.private') from None
            self.private[str(url.origin())] = networks
        self.context = ssl.create_default_context()
        if config.urls.get('ca_file'): self.context.load_verify_locations(cafile=config.urls['ca_file'])
        self.active = set()

    @asynccontextmanager
    async def get(self, principal, value):
        if not principal.actions & 4: raise aotx_error(403, 'The grant does not permit URL imports.', 'url_forbidden')
        if principal.id in self.active: raise aotx_error(429, 'A URL import is already active.', 'fetch_limit')
        url = aotx_url(value)
        limit = min(self.config.limits['upload_bytes'], principal.media_bytes)
        async with self.budget.claim('fetches', limit):
            self.active.add(principal.id)
            try: yield await self._get(url, limit)
            finally: self.active.remove(principal.id)

    async def _get(self, url, limit):
        host, port = url.raw_host, url.port
        try:
            literal = ipaddress.ip_address(host)
            addresses = [literal]
        except ValueError:
            try:
                answer = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(
                    host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP), 5)
                addresses = sorted({ipaddress.ip_address(row[4][0]) for row in answer}, key=str)
            except (OSError, ValueError, TimeoutError):
                raise aotx_error(502, 'The source address is unavailable.', 'source_dns') from None
        networks = self.private.get(str(url.origin()), ())
        for address in addresses:
            permitted = any(address.version == n.version and address in n for n in networks)
            if not permitted and not (self.config.urls.get('public', True) and port == 443 and aotx_public(address)):
                raise aotx_error(403, 'The source destination is not permitted.', 'source_address')
        if not addresses: raise aotx_error(502, 'The source address is unavailable.', 'source_dns')
        resolver = aotx_resolver(host, port, addresses)
        connector = TCPConnector(resolver=resolver, use_dns_cache=False, force_close=True, ssl=self.context, limit=1)
        timeout = ClientTimeout(total=60, connect=10, sock_connect=10, sock_read=10)
        try:
            async with ClientSession(connector=connector, timeout=timeout, trust_env=False,
                cookie_jar=DummyCookieJar(), auto_decompress=False) as session:
                async with session.get(url, allow_redirects=False, headers={'Accept-Encoding': 'identity'}) as response:
                    if response.status != 200: raise aotx_error(502, 'The source did not return a complete object.', 'source_status')
                    if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                        raise aotx_error(415, 'Encoded source bodies are not supported.', 'source_encoding')
                    if response.content_length is not None and response.content_length > limit:
                        raise aotx_error(413, 'The source object is too large.', 'source_limit')
                    data = bytearray()
                    async for part in response.content.iter_chunked(65536):
                        if len(part) > limit-len(data): raise aotx_error(413, 'The source object is too large.', 'source_limit')
                        data.extend(part)
                    if response.content_length is not None and response.content_length != len(data):
                        raise aotx_error(502, 'The source length does not match its header.', 'source_length')
                    return data, response.content_type
        except (ClientError, OSError, TimeoutError):
            raise aotx_error(502, 'The source transfer failed.', 'source_transfer') from None
