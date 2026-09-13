# SPDX-License-Identifier: Apache-2.0
# Start the HTTP service or write its operator grant table.
# Inputs: A protected JSON configuration and command arguments.
# Outputs: HTTP responses or an atomic grant file. Exit codes: 0 success, 1 failure, 2 usage.
import argparse
import asyncio
import logging
import os
import signal
import ssl
import struct
import tempfile
from aiohttp import web
from .config import ROLES, aotx_load_config
from .errors import aotx_error
from .server import aotx_server, aotx_state
from .wire import HEAD, FRAME, GRANTS


def aotx_grants(config):
    frame = bytearray(HEAD+64*len(config.principals))
    if len(frame) > FRAME: raise ValueError('The grant table exceeds one control frame.')
    frame[:8] = b'AOTXAPI1'
    struct.pack_into('<I', frame, 8, GRANTS)
    struct.pack_into('<Q', frame, 32, config.revision)
    struct.pack_into('<I', frame, 76, len(config.principals))
    struct.pack_into('<I', frame, 88, len(frame)-HEAD)
    for i, p in enumerate(config.principals):
        models = sum(1 << role for role in {ROLES[config.models[m]['role']] for m in p.models})
        struct.pack_into('<16sQIIIIIIQ8x', frame, HEAD+64*i, p.id, p.revision, p.actions,
            models, p.pages, p.tokens, p.requests, p.media, p.media_bytes)
    return frame


def aotx_write_grants(config, path):
    data = aotx_grants(config)
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix='.aotx-grants-', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data); output.flush(); os.fsync(output.fileno())
        os.replace(temporary, path)
        parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try: os.fsync(parent)
        finally: os.close(parent)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


async def aotx_serve(config):
    state = aotx_state(config)
    server = aotx_server(state)
    runner = web.ServerRunner(server, shutdown_timeout=config.limits['write_seconds'])
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(sig, stopped.set)
    context = None
    if config.certificate:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(config.certificate, config.key)
    try:
        await runner.setup()
        site = web.TCPSite(runner, config.host, config.port, ssl_context=context, backlog=config.limits['connections'])
        await site.start()
        logging.getLogger('aotx.gateway').info('The gateway is ready.')
        await stopped.wait()
    finally:
        await runner.cleanup()
        await state.wire.close()


def aotx_main():
    parser = argparse.ArgumentParser(description='Serve scoped device inference over HTTP.')
    parser.add_argument('command', choices=('serve', 'grants'))
    parser.add_argument('--config', required=True, help='Read the protected JSON configuration.')
    parser.add_argument('--output', help='Write the grant table to this file.')
    args = parser.parse_args()
    if (args.command == 'grants') != bool(args.output): parser.error('The grants command requires --output.')
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    try:
        config = aotx_load_config(args.config)
        if args.command == 'grants': aotx_write_grants(config, args.output)
        else: asyncio.run(aotx_serve(config))
        return 0
    except (OSError, ValueError, aotx_error):
        logging.getLogger('aotx.gateway').error('The gateway configuration or local resource is unavailable.')
        return 1


if __name__ == '__main__': raise SystemExit(aotx_main())
