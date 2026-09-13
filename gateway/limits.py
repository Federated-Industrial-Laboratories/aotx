# SPDX-License-Identifier: Apache-2.0
# Bound active transport work, source byte credit and body receive time.
import asyncio
from contextlib import asynccontextmanager
from .errors import aotx_error


class aotx_credit:
    def __init__(self, budget, amount): self.budget, self.amount = budget, amount

    def resize(self, amount):
        if amount < 0 or amount-self.amount > self.budget.limits['body_credit']-self.budget.bytes:
            raise aotx_error(429, 'The transport allocation is full.', 'transport_limit')
        self.budget.bytes += amount-self.amount
        self.amount = amount


class aotx_budget:
    def __init__(self, limits):
        self.limits = limits
        self.counts = {}
        self.bytes = 0

    @asynccontextmanager
    async def claim(self, kind, amount=0):
        cap = self.limits.get(kind, self.limits['connections'])
        if self.counts.get(kind, 0) >= cap or amount > self.limits['body_credit'] - self.bytes:
            raise aotx_error(429, 'The transport allocation is full.', 'transport_limit')
        self.counts[kind] = self.counts.get(kind, 0)+1
        self.bytes += amount
        credit = aotx_credit(self, amount)
        try: yield credit
        finally:
            self.counts[kind] -= 1
            self.bytes -= credit.amount


async def aotx_read_body(request, limit, seconds):
    if request.headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise aotx_error(415, 'Encoded request bodies are not supported.', 'content_encoding')
    if request.content_length is not None and request.content_length > limit:
        raise aotx_error(413, 'The request body is too large.', 'body_limit')
    data = bytearray()
    try:
        async with asyncio.timeout(seconds):
            while True:
                part = await asyncio.wait_for(request.content.read(min(65536, limit-len(data)+1)), 10)
                if not part: break
                if len(part) > limit-len(data): raise aotx_error(413, 'The request body is too large.', 'body_limit')
                data.extend(part)
    except TimeoutError:
        raise aotx_error(408, 'The request body deadline expired.', 'body_timeout') from None
    if request.content_length is not None and request.content_length != len(data):
        raise aotx_error(400, 'The body length does not match its header.', 'body_length')
    if request.transport: request.transport.pause_reading()
    return data
