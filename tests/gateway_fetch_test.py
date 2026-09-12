#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check actual HTTPS destinations, certificate checks and bounded source transfers.
# Inputs: The gateway environment and OpenSSL. Outputs: Test results. Exit: 0 pass, 1 failure.
import asyncio
from dataclasses import replace
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gateway.config import DEFAULTS, aotx_config, aotx_principal
from gateway.errors import aotx_error
from gateway.fetch import aotx_fetcher
from gateway.limits import aotx_budget


class aotx_fetch_tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='aotx-fetch-')
        root = Path(self.directory.name)
        self.cert, self.key = root/'cert.pem', root/'key.pem'
        generated = subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
            '-keyout', str(self.key), '-out', str(self.cert), '-days', '1', '-subj', '/CN=feed.test',
            '-addext', 'subjectAltName=DNS:feed.test,DNS:*.feed.test,DNS:localhost,IP:127.0.0.1,IP:::1'], capture_output=True)
        self.assertEqual(generated.returncode, 0)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.cert, self.key)
        self.hits = []
        async def source(request):
            self.hits.append((request.path, dict(request.headers), request.transport.get_extra_info('sockname')))
            if request.path == '/redirect': return web.Response(status=302, headers={'Location': '/bytes'})
            if request.path == '/encoded': return web.Response(body=b'abc', headers={'Content-Encoding': 'gzip'})
            if request.path == '/large': return web.Response(body=b'a'*2048)
            if request.path == '/chunked':
                response = web.StreamResponse(headers={'Content-Type': 'image/jpeg'})
                await response.prepare(request); await response.write(b'a'*2048); await response.write_eof(); return response
            if request.path == '/short':
                response = web.StreamResponse(headers={'Content-Length': '100'})
                await response.prepare(request); await response.write(b'abc'); request.transport.close(); return response
            data = ('source-'+request.query['case']).encode() if 'case' in request.query else b'abc'
            return web.Response(body=data, headers={'Content-Type': 'image/jpeg', 'Set-Cookie': 'source=1; Secure'})
        self.runner = web.ServerRunner(web.Server(source), shutdown_timeout=1)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, '127.0.0.1', 0, ssl_context=context)
        await self.site.start(); self.port = self.runner.addresses[0][1]
        self.ipv6 = web.TCPSite(self.runner, '::1', 0, ssl_context=context)
        await self.ipv6.start()
        self.ipv6_port = next(address[1] for address in self.runner.addresses if address[0] == '::1')
        self.origin = 'https://feed.test:'+str(self.port)
        self.config = aotx_config('/tmp/unused-service.sock', '127.0.0.1', 8081, (), {}, (), 1,
            dict(DEFAULTS, upload_bytes=1024, body_credit=2048), {'public': False,
            'private': [{'origin': self.origin, 'networks': ['127.0.0.1/32']}], 'ca_file': str(self.cert)}, None, None)
        self.principal = aotx_principal(b'a'*16, 1, (), 6, (), 0, 1, 1, 1, 1024)
        self.budget = aotx_budget(self.config.limits)
        self.fetcher = aotx_fetcher(self.config, self.budget)
        self.lookups = 0
        async def dns(host, port, **kw):
            self.lookups += 1
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '',
                ('127.0.0.1' if self.lookups == 1 else '127.0.0.2', port))]
        self.dns = patch.object(asyncio.get_running_loop(), 'getaddrinfo', side_effect=dns)
        self.dns.start()

    async def asyncTearDown(self):
        self.dns.stop()
        await self.runner.cleanup(); self.directory.cleanup()
        self.assertEqual(self.budget.bytes, 0)
        self.assertFalse(any(self.budget.counts.values()))

    async def test_distinct_source_batches(self):
        loop = asyncio.get_running_loop()
        for n in (1, 64):
            origins = ['https://p%d.feed.test:%d' % (i, self.port) for i in range(n)]
            principals = [replace(self.principal, id=(i+1).to_bytes(16, 'little')) for i in range(n)]
            limits = dict(self.config.limits, fetches=n, body_credit=n*1024)
            private = [{'origin': origin, 'networks': ['127.0.0.1/32']} for origin in origins]
            config = replace(self.config, limits=limits,
                urls={'public': False, 'private': private, 'ca_file': str(self.cert)})
            budget = aotx_budget(limits)
            lookups = {}
            mixed = False
            async def dns(host, port, **kw):
                lookups[host] = lookups.get(host, 0)+1
                addresses = ['127.0.0.1' if lookups[host] == 1 else '127.0.0.2']
                if mixed: addresses.append('10.0.0.2')
                return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (address, port)) for address in addresses]
            with patch.object(loop, 'getaddrinfo', side_effect=dns):
                fetcher = aotx_fetcher(config, budget)
                arrived, release = 0, asyncio.Event()
                async def positive(i):
                    nonlocal arrived
                    url = origins[i]+'/bytes?case='+str(i)
                    async with fetcher.get(principals[i], url) as (data, mime):
                        self.assertEqual((data, mime), (('source-'+str(i)).encode(), 'image/jpeg'))
                        with self.assertRaises(aotx_error) as error:
                            async with fetcher.get(principals[i], url): pass
                        self.assertEqual(error.exception.status, 429)
                        arrived += 1
                        if arrived == n:
                            self.assertEqual(budget.bytes, n*1024)
                            self.assertEqual(len(fetcher.active), n)
                            with self.assertRaises(aotx_error) as full:
                                async with fetcher.get(replace(principals[i], id=b'z'*16), url): pass
                            self.assertEqual(full.exception.status, 429)
                            release.set()
                        await asyncio.wait_for(release.wait(), 10)
                with patch.dict(os.environ, {'HTTPS_PROXY': 'http://127.0.0.2:1', 'HTTP_PROXY': 'http://127.0.0.2:1'}):
                    await asyncio.gather(*(positive(i) for i in range(n)))
                self.assertEqual(len(lookups), n); self.assertTrue(all(v == 1 for v in lookups.values()))
                self.assertTrue(all(h[2][0] == '127.0.0.1' for h in self.hits[-n:]))
                self.assertTrue(all(not {'Authorization', 'Cookie', 'Proxy-Authorization'} & set(h[1]) for h in self.hits[-n:]))
                self.assertEqual(budget.bytes, 0); self.assertFalse(fetcher.active)
                cases = [('/redirect', 502), ('/encoded', 415), ('/large', 413), ('/chunked', 413), ('/short', 502),
                    ('mixed', 403), ('grant', 403), ('certificate', 502)]
                for path, status in cases:
                    lookups.clear(); start = len(self.hits); mixed = path == 'mixed'
                    urls = config.urls if path not in ('grant', 'certificate') else {'public': False}
                    if path == 'certificate': urls = dict(urls, private=private)
                    fetcher = aotx_fetcher(replace(config, urls=urls), budget)
                    target = path if path.startswith('/') else '/bytes'
                    async def refused(i):
                        with self.assertRaises(aotx_error) as error:
                            async with fetcher.get(principals[i], origins[i]+target+'?case='+str(i)): pass
                        self.assertEqual(error.exception.status, status)
                    await asyncio.gather(*(refused(i) for i in range(n)))
                    self.assertEqual(len(lookups), n); self.assertTrue(all(v == 1 for v in lookups.values()))
                    self.assertEqual(len(self.hits)-start, n if path.startswith('/') else 0)
                    self.assertEqual(budget.bytes, 0); self.assertFalse(fetcher.active)
                    self.assertFalse(any(budget.counts.values()))
            origin = 'https://[::1]:'+str(self.ipv6_port)
            fetcher = aotx_fetcher(replace(config, urls={'public': False, 'ca_file': str(self.cert),
                'private': [{'origin': origin, 'networks': ['::1/128']}]}), budget)
            start, prior = len(self.hits), self.lookups
            async def ipv6(i):
                async with fetcher.get(principals[i], origin+'/bytes?case='+str(i)) as (data, mime):
                    self.assertEqual((data, mime), (('source-'+str(i)).encode(), 'image/jpeg'))
            await asyncio.gather(*(ipv6(i) for i in range(n)))
            self.assertEqual(len(self.hits)-start, n); self.assertEqual(self.lookups, prior)
            self.assertTrue(all(h[2][0] == '::1' for h in self.hits[start:]))
            self.assertEqual(budget.bytes, 0); self.assertFalse(fetcher.active)
            self.assertFalse(any(budget.counts.values()))

    async def test_actual_destination_is_pinned(self):
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://127.0.0.2:1', 'HTTP_PROXY': 'http://127.0.0.2:1'}):
            async with self.fetcher.get(self.principal, self.origin+'/bytes') as (data, mime):
                self.assertEqual((data, mime), (b'abc', 'image/jpeg'))
                self.assertEqual(self.budget.bytes, 1024)
        self.assertEqual(self.lookups, 1)
        self.assertEqual(self.hits[0][2][0], '127.0.0.1')
        self.assertFalse({'Authorization', 'Cookie', 'Proxy-Authorization'} & set(self.hits[0][1]))

    async def test_source_refusals(self):
        for path, status in (('/redirect', 502), ('/encoded', 415), ('/large', 413), ('/chunked', 413), ('/short', 502)):
            self.lookups = 0
            with self.assertRaises(aotx_error) as result:
                async with self.fetcher.get(self.principal, self.origin+path): pass
            self.assertEqual(result.exception.status, status, path)
        self.assertEqual([h[0] for h in self.hits], ['/redirect', '/encoded', '/large', '/chunked', '/short'])

    async def test_actual_ipv6_destination(self):
        origin = 'https://[::1]:'+str(self.ipv6_port)
        urls = {'public': False, 'private': [{'origin': origin, 'networks': ['::1/128']}],
            'ca_file': str(self.cert)}
        fetcher = aotx_fetcher(replace(self.config, urls=urls), self.budget)
        async with fetcher.get(self.principal, origin+'/bytes') as (data, mime):
            self.assertEqual((data, mime), (b'abc', 'image/jpeg'))
        self.assertEqual(self.hits[0][2][0], '::1')
        self.assertEqual(self.lookups, 0)

    async def test_mixed_dns_answers_prevent_connections(self):
        loop = asyncio.get_running_loop()
        async def mixed(host, port, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port))
                for ip in ('127.0.0.1', '10.0.0.2')]
        with patch.object(loop, 'getaddrinfo', side_effect=mixed):
            with self.assertRaises(aotx_error) as result:
                async with self.fetcher.get(self.principal, self.origin+'/bytes'): pass
        self.assertEqual(result.exception.status, 403); self.assertEqual(self.hits, [])

    async def test_certificates_and_private_grants(self):
        for urls, status in (({'public': False}, 403),
            ({'public': False, 'private': self.config.urls['private']}, 502)):
            self.lookups = 0
            fetcher = aotx_fetcher(replace(self.config, urls=urls), self.budget)
            with self.assertRaises(aotx_error) as result:
                async with fetcher.get(self.principal, self.origin+'/bytes'): pass
            self.assertEqual(result.exception.status, status)
        self.assertEqual(self.hits, [])

    async def test_byte_credit_and_principal_limit(self):
        async with self.fetcher.get(self.principal, self.origin+'/bytes'):
            with self.assertRaises(aotx_error) as result:
                async with self.fetcher.get(self.principal, self.origin+'/bytes'): pass
            self.assertEqual(result.exception.status, 429)
            async with self.budget.claim('body_buffers', 1024):
                with self.assertRaises(aotx_error) as result:
                    async with self.fetcher.get(replace(self.principal, id=b'b'*16), self.origin+'/bytes'): pass
                self.assertEqual(result.exception.status, 429)


if __name__ == '__main__': unittest.main(verbosity=2)
