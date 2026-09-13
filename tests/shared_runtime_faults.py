#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check shared file recovery across verified writer stops and exact device boundaries.
# Inputs: Build, source, store, new short output path, batch count and prepared runtime file.
# Outputs: Owned process identities, exact file cuts and HTTP checks. Exit: 0 pass, 1 failure, 2 invalid arguments.
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import traceback
from aiohttp import ClientSession, ClientTimeout
from live_boot_test import wait
from runtime_boot_test import RuntimeTest, RuntimeRun, durable, sections
from gateway_runtime_test import aotx_http_run, aotx_ready
from shared_runtime_client import aotx_shared_client, PREFIX, TERMINAL
from shared_runtime_test import aotx_configuration


def aotx_process(pid):
    try:
        row = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=row[0], parent=int(row[1]), start=row[19], exit=int(row[49]))
    except FileNotFoundError: return None


class aotx_owned_drain:
    def __init__(self, run):
        self.run, self.fd, self.stopped = run, None, False
        self.boot = aotx_process(run.child.pid)
        run.test.check(self.boot is not None and self.boot['parent'] == os.getpid() and
            Path('/proc/' + str(run.child.pid) + '/exe').resolve() == (run.test.build / 'aotx_boot').resolve(), 'owned boot executable and parent')
        children = Path('/proc/' + str(run.child.pid) + '/task/' + str(run.child.pid) + '/children').read_text().split()
        matches = [int(p) for p in children if Path('/proc/' + p + '/exe').resolve() == (run.test.build / 'aotx_drain').resolve()]
        run.test.check(len(matches) == 1, 'one exact owned drain executable')
        self.identity = aotx_process(matches[0]); self.fd = os.pidfd_open(matches[0])
        self.verify(); run.test.record(owned_drain=self.identity, owned_boot=self.boot)

    def verify(self, parent=True):
        current = aotx_process(self.identity['pid'])
        if current is None or current['state'] == 'Z' or current['start'] != self.identity['start']:
            raise ProcessLookupError('The owned drain identity is no longer live.')
        if Path('/proc/' + str(current['pid']) + '/exe').resolve() != (self.run.test.build / 'aotx_drain').resolve():
            raise ProcessLookupError('The owned drain executable changed.')
        if parent:
            boot = aotx_process(self.boot['pid'])
            if boot is None or boot['start'] != self.boot['start'] or current['parent'] != boot['pid']:
                raise ProcessLookupError('The owned drain parent changed.')
        return current

    def pause(self):
        self.verify(); signal.pidfd_send_signal(self.fd, signal.SIGSTOP); self.stopped = True
        self.run.test.record(pid=self.identity['pid'], start=self.identity['start'], signal='SIGSTOP')
        current = wait(lambda: (r if (r := self.verify())['state'] in ('T', 't') else None), self.run.child, 30)
        self.run.test.check(True, 'owned drain is observably stopped', process=current)

    def kill(self):
        current = aotx_process(self.identity['pid'])
        if current is not None and current['start'] == self.identity['start'] and current['state'] != 'Z':
            self.verify(parent=False); signal.pidfd_send_signal(self.fd, signal.SIGKILL)
            self.run.test.record(pid=current['pid'], start=current['start'], signal='SIGKILL')
        def ended():
            row = aotx_process(self.identity['pid'])
            return {'absent': True} if row is None or row['start'] != self.identity['start'] else row if row['state'] == 'Z' else None
        result = wait(ended, seconds=30); self.stopped = False
        self.run.test.check(result.get('absent') or result['exit'] == signal.SIGKILL, 'owned drain kill status is separate from boot status', process=result)

    def close(self):
        if self.stopped: self.kill()
        if self.fd is not None: os.close(self.fd); self.fd = None


async def aotx_fault_clients(test, http, gateway, cfg, keys, count):
    url = 'http://127.0.0.1:' + str(cfg['port'])
    clients = [aotx_shared_client(test, http, url, keys[i], '%032x' % (i + 1)) for i in range(count)]
    await aotx_ready(test, gateway, http, url, clients[0].headers)
    return clients


async def aotx_fault_batch(test, clients, entries):
    async def one(client, entry):
        async with client.client.post(client.url + PREFIX + entry['path'], headers=client.headers, json=entry['body']) as response:
            value = await response.json(); status = response.status
            test.check(status in (200, 202, 429), 'one bounded canonical admission attempt', actor=entry['actor'], status=status, response=value)
            if status == 429:
                test.check(response.headers.get('Retry-After') == '1' and value['error']['type'] == 'rate_limit_error' and
                    value['error']['code'] in ('device_refused', 'transport_limit'), 'refused admission reports explicit pressure')
            else:
                test.check(value['accepted'] and value['sequence'] == entry['body']['sequence'], 'accepted input preserves its canonical sequence')
            entry.update(http_status=status, response=value)
    await asyncio.gather(*(one(c, e) for c, e in zip(clients, entries)))


async def aotx_fault_read(clients, entries):
    async def one(client, entry):
        if entry['http_status'] != 429:
            entry['cut'] = await client.request('GET', '/operations/' + entry['response']['id'], audit=False)
    await asyncio.gather(*(one(c, e) for c, e in zip(clients, entries)))


async def aotx_fault_wait(clients, entries, accept):
    end = time.monotonic() + 900
    while time.monotonic() < end:
        await aotx_fault_read(clients, entries)
        if accept(entries): return
        await asyncio.sleep(0.01)
    raise TimeoutError('The required device boundary was not observed.')


async def aotx_fault_capture(test, run, guard, cfg, path, keys, count, runtime, mode):
    gateway = aotx_http_run(test, path, mode)
    try:
        async with ClientSession(timeout=ClientTimeout(total=960)) as http:
            clients = await aotx_fault_clients(test, http, gateway, cfg, keys, count)
            for client in clients:
                person = await client.discover()
                if not person['registered']: await client.done('/participant')
            created = await asyncio.gather(*(c.create() for c in clients))
            durable(run); baseline = sections(test, runtime)
            people = await asyncio.gather(*(c.request('GET', '/participant') for c in clients))
            entries = []
            for i, (client, (space, conversation)) in enumerate(zip(clients, created)):
                body = client.command(text='List consecutive integers from 1 to 10000. Input ' + str(i) + '.', model='text', max_output_tokens=128)
                entries.append(dict(actor=i, space=space, conversation=conversation, person=people[i],
                    path='/conversations/' + conversation + '/inputs', body=body))
            (test.output / (mode + '-inputs.json')).write_text(json.dumps(entries, indent=2) + '\n')
            selected, before, after = [], [], []
            if mode == 'admit': guard.pause()
            await aotx_fault_batch(test, clients, entries)
            test.check(any(e['http_status'] != 429 for e in entries), 'at least one canonical input is accepted', mode=mode)
            if mode == 'admit':
                await aotx_fault_wait(clients, entries, lambda rows: any(e.get('cut', {}).get('device_committed') for e in rows))
                test.check(all(not e['cut']['saved_admission'] for e in entries if 'cut' in e), 'post-stop admissions have no saved acknowledgement')
            else:
                await aotx_fault_wait(clients, entries, lambda rows: any(e.get('cut', {}).get('state') == 'running' and e['cut']['saved_admission'] for e in rows))
                before = [e.get('cut') for e in entries]; guard.pause()
                await aotx_fault_read(clients, entries); after = [e.get('cut') for e in entries]
                selected = [e['actor'] for e in entries if e.get('cut', {}).get('state') == 'running' and e['cut']['saved_admission']]
                test.check(bool(selected), 'saved inputs remain observably running after the writer stops', selected=selected)
                await aotx_fault_wait(clients, entries, lambda rows: all(rows[i]['cut']['state'] in TERMINAL for i in selected))
                test.check(all(not entries[i]['cut']['saved_terminal'] for i in selected), 'selected device terminals occur after the fixed file cut')
                async def exact(client, entry):
                    if entry.get('cut', {}).get('state') in TERMINAL:
                        entry['cut'] = await client.terminal(entry['cut']['id'], saved=False)
                await asyncio.gather(*(exact(c, e) for c, e in zip(clients, entries)))
            result = dict(mode=mode, boot=run.boot, drain=guard.identity, entries=entries, selected=selected,
                before_pause=before, after_pause=after, baseline_memory=hashlib.sha256(baseline[2]).hexdigest())
            (test.output / (mode + '-cut.json')).write_text(json.dumps(result, indent=2) + '\n')
            guard.verify(); guard.kill(); run.stop(killed=True)
            return result
    finally:
        if guard.stopped:
            guard.kill()
            if not run.log.closed: run.stop(killed=True)
        gateway.close()


def aotx_fault_file(test, runtime, cut):
    saved = sections(test, runtime); replay = saved[7]
    get = lambda at: int.from_bytes(replay[at:at + 8], 'little')
    test.check(get(16) == int(cut['boot'], 16), 'complete file cut belongs to the stopped boot')
    cut['file'] = dict(boot=str(get(16)), source=str(get(72)), records=str(get(40)), state_hash=format(get(32), '016x'),
        replay_sha256=hashlib.sha256(replay).hexdigest(), memory_sha256=hashlib.sha256(saved[2]).hexdigest())
    for entry in cut['entries']:
        receipt = entry.get('cut')
        if receipt is None or not int(receipt['admission_source']) or int(receipt['admission_source']) > get(72): kind = 'absent'
        elif int(receipt['terminal_source']) and int(receipt['terminal_source']) <= get(72): kind = 'terminal'
        else: kind = 'interrupted'
        entry['recovery'] = kind
        if receipt:
            test.check((not receipt['saved_admission'] or kind != 'absent') and
                (not receipt['saved_terminal'] or kind == 'terminal'), 'acknowledged receipt is included in the exact file cut', actor=entry['actor'])
    if cut['mode'] == 'admit':
        test.check(all(e['recovery'] == 'absent' for e in cut['entries']) and cut['file']['memory_sha256'] == cut['baseline_memory'],
            'pre-admission file cut contains only the saved baseline')
    else:
        test.check(all(cut['entries'][i]['recovery'] == 'interrupted' for i in cut['selected']), 'selected saved admissions exclude every later terminal record')
    (test.output / (cut['mode'] + '-cut.json')).write_text(json.dumps(cut, indent=2) + '\n')
    return saved


async def aotx_fault_recover(test, run, cfg, path, keys, count, cut):
    gateway = aotx_http_run(test, path, cut['mode'] + '-recover')
    try:
        async with ClientSession(timeout=ClientTimeout(total=960)) as http:
            clients = await aotx_fault_clients(test, http, gateway, cfg, keys, count)
            people = await asyncio.gather(*(c.discover() for c in clients))
            async def one(client, entry, person):
                old = entry.get('cut'); kind = entry['recovery']; expected = int(entry['body']['sequence']) + (kind != 'absent')
                test.check(int(person['next_sequence']) == expected and person['retry_floor'] == entry['person']['retry_floor'],
                    'recovery restores the exact participant sequence and retry floor', actor=entry['actor'])
                events = await client.request('GET', '/conversations/' + entry['conversation'] + '/events')
                test.check(len(events['items']) == (kind != 'absent'), 'file-only recovery has exactly the recorded input count', actor=entry['actor'])
                if kind == 'absent':
                    if old: await client.request('GET', '/operations/' + old['id'], (404, 410))
                    retry = await client.request('POST', entry['path'], (200, 202), json=entry['body'])
                    actual = await client.terminal(retry['id'])
                    test.check(actual['state'] == 'completed' and actual['status'] == 200 and bool(base64.b64decode(actual['exact_output']).strip()),
                        'an absent admission can execute once after its exact retry', actor=entry['actor'])
                else:
                    actual = await client.terminal(old['id'])
                    if kind == 'interrupted':
                        test.check(actual['state'] == 'interrupted' and actual['status'] == 598 and actual['gap'],
                            'saved admission without terminal becomes an explicit interrupted result', actor=entry['actor'])
                    else:
                        test.check(all(actual[k] == old[k] for k in ('id', 'state', 'status', 'exact_output', 'usage', 'finish')),
                            'saved terminal preserves its exact bytes and usage', actor=entry['actor'])
                again = await client.request('POST', entry['path'], json=entry['body'])
                final = await client.terminal(again['id'])
                test.check(again['id'] == actual['id'] and all(final[k] == actual[k] for k in ('state', 'status', 'exact_output', 'usage', 'finish')),
                    'canonical retry preserves one opaque receipt and exact output', actor=entry['actor'])
                events = await client.request('GET', '/conversations/' + entry['conversation'] + '/events')
                test.check([r['id'] for r in events['items']] == [actual['id']], 'exact retry cannot append a second input')
                return dict(actor=entry['actor'], classification=kind, receipt=final)
            results = await asyncio.gather(*(one(c, e, p) for c, e, p in zip(clients, cut['entries'], people)))
            (test.output / (cut['mode'] + '-recovery.json')).write_text(json.dumps(results, indent=2) + '\n')
    finally: gateway.close()


def aotx_fault_main():
    parser = argparse.ArgumentParser()
    for name in ('build', 'source', 'store', 'output'): parser.add_argument(name, type=Path)
    parser.add_argument('count', type=int, choices=(1, 64))
    parser.add_argument('--runtime', type=Path, required=True, help='Use the prepared shared runtime file in place.')
    parser.add_argument('--ready-seconds', type=int, default=900, help='Set the file activation and replay time limit.')
    args = parser.parse_args()
    if args.ready_seconds < 1: parser.error('The readiness time limit must be positive.')
    if len(str(args.output.resolve() / 'terminal-recover-journal/service.sock').encode()) >= 108:
        parser.error('The output path exceeds the local socket path limit.')
    test = RuntimeTest(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve(), snapshot_every=64)
    start, status, guards = time.monotonic(), 0, []
    runtime = args.runtime.resolve()
    try:
        initial = sections(test, runtime)
        test.check(bool(int.from_bytes(initial[5][20:24], 'little') & 8), 'prepared file requires shared semantics')
        for index, mode in enumerate(('admit', 'terminal')):
            cfg, path, grants, keys = aotx_configuration(test, args.count, mode, 100 + index * 2)
            run = RuntimeRun(test, mode, runtime, extra=('--service-grants', grants)); run.ready(args.ready_seconds)
            guard = aotx_owned_drain(run); guards.append(guard)
            cut = asyncio.run(aotx_fault_capture(test, run, guard, cfg, path, keys, args.count, runtime, mode))
            saved = aotx_fault_file(test, runtime, cut); guard.close()
            test.check(run.log.closed and run.child.poll() is not None, 'old boot is closed before journal removal')
            shutil.rmtree(run.journal)
            label = mode + '-recover'
            cfg, path, grants, keys = aotx_configuration(test, args.count, label, 101 + index * 2)
            run = RuntimeRun(test, label, runtime, extra=('--service-grants', grants)); run.ready(args.ready_seconds)
            match = re.search(r'restore: applied (\d+) hash ([0-9a-f]+) decode_refused (\d+).* rejected (\d+)', run.path.read_text())
            test.check(match and int(match[1]) == int(cut['file']['records']) and int(match[2], 16) == int(cut['file']['state_hash'], 16) and
                not int(match[3]) and not int(match[4]), 'file-only recovery restores the exact record count and state hash', file=cut['file'])
            restored = sections(test, runtime)
            test.check(restored[2] == saved[2], 'file-only recovery cannot add unrecorded cognitive memory')
            asyncio.run(aotx_fault_recover(test, run, cfg, path, keys, args.count, cut))
            durable(run); run.stop(); shutil.rmtree(run.journal)
    except Exception as error:
        status = 1; (test.output / 'failure.txt').write_text(type(error).__name__ + ': ' + str(error) + '\n'); traceback.print_exc()
    finally:
        for guard in guards:
            try: guard.close()
            except Exception as error: status = 1; test.record(drain_cleanup_error=str(error))
        for run in reversed(test.active):
            if not run.log.closed:
                try: run.close()
                except Exception as error: status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    result = dict(status=status, checks=len(test.checks), seconds=time.monotonic() - start)
    (test.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result), flush=True)
    return status


if __name__ == '__main__': sys.exit(aotx_fault_main())
