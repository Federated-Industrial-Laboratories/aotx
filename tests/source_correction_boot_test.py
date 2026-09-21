#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Check a selected correction turn from recorded native memory transactions.
# Inputs: build, source, model store, captures, output, count, turn and slot.
# Outputs: GPU verdicts and exact choices. Exit: 0 pass, 1 failed, 2 bad arguments.
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from capacity_boot_test import batch
from checkpoint_boot_test import spawn
from live_boot_test import Test, Run
from retain_boot_test import transfers, compiled_limits
from source_boot_test import decisions, text
from text_boot_test import setup


def aotx_capture(test, f, path):
    data = path.read_bytes()
    count, stride = f.get(data, 8, 4), f.get(data, 40, 4)
    offset = 64 + count * stride
    test.check(data[:8] == b'AOTXICH2' and count == 64 and stride == 18672
        and f.get(data, 44, 4) == 0 and len(data) == offset + f.get(data, 48),
        'capture contains a complete accepted source batch')
    tail = data[offset:]
    test.check(tail[:8] == b'AOTXLOG1' and f.get(tail, 32) == f.get(data, 32) + 1,
        'capture contains its exact next memory transaction')
    test.record(capture=str(path), sha256=hashlib.sha256(data).hexdigest())
    return data, tail


def aotx_exercise(test, args):
    f = setup(test)
    paths = sorted(args.captures.glob('choice-*.bin'), key=lambda p: int(p.stem[7:]))
    test.check(len(paths) >= args.turn, 'recorded inputs cover the selected turn')
    captured = [aotx_capture(test, f, path) for path in paths[:args.turn]]
    f.LINEAGE = captured[0][0][16:32]
    inputs = test.output / 'inputs'
    checkpoint, memory = inputs / 'checkpoint', inputs / 'memory.aotxccir'
    checkpoint.write_bytes(f.image([], 0, 10))
    test.command([test.build / 'aotx_recall_cli_fixture', checkpoint, '-', memory, 1], 'pack')
    run = Run(test, 'selected', roles='language,embedding'); run.ready()
    spawn(run, args.count)
    run.operation('load', memory, 1, 0)
    for i, (_, tail) in enumerate(captured[:-1]):
        path = inputs / ('prior-' + str(i)); path.write_bytes(tail)
        run.operation('apply', path, 2, 0)
    original = captured[-1][0]
    cut = f.get(original, 32)
    slots = list(range(64)) if args.count == 64 else [args.slot]
    bind = batch(f, b'AOTXBND1', 64, cut, args.count)
    request = batch(f, b'AOTXLIV1', 8256, cut, args.count)
    for i, slot in enumerate(slots):
        saved = original[64 + slot * 18672:64 + (slot + 1) * 18672]
        at = 64 + i * 64
        f.put(bind, at, i, 4)
        bind[at + 8:at + 24] = saved[80:96]
        bind[at + 24:at + 40] = saved[96:112]
        bind[at + 40:at + 56] = saved[16:32]
        f.put(bind, at + 56, 512, 4); f.put(bind, at + 60, 2, 4)
        at = 64 + i * 8256
        request[at:at + 8256] = saved[:8256]
        f.put(request, at, i, 4); f.put(request, at + 32, 1)
        test.check(request[at + 64:at + 8256] == saved[64:8256],
            'prepared query bytes retain the exact saved input', original_slot=slot)
    (inputs / 'bind').write_bytes(bind); (inputs / 'query').write_bytes(request)
    run.operation('bind', inputs / 'bind', 3, args.count)
    run.operation('query', inputs / 'query', 4, args.count, seconds=args.operation_seconds)
    run.stop()
    limits = compiled_limits(test)
    choices = [p for op, _, p in transfers(test, run, f,
        {14: 64 + 64 * 18672 + limits['image_bytes']}) if op == 14]
    test.check(len(choices) == 1 and f.get(choices[0], 8, 4) == args.count,
        'only the selected turn generates a new semantic choice')
    choice = choices[0]
    expected = {}
    for slot in slots:
        source = text(slot, args.turn, 64)
        expected[source] = [part + '.' for part in source.split('. ')[:-1]] + [source.split('. ')[-1]]
    decisions(test, f, choices, args.count, expected)
    records = []
    for i, slot in enumerate(slots):
        at, saved_at = 64 + i * 18672, 64 + slot * 18672
        last, first = at + 9168, at + 14448
        result = json.loads(choice[last + 128:last + 128 + f.get(choice, last + 4, 4)])
        labels = json.loads(choice[first + 128:first + 128 + f.get(choice, first + 4, 4)])
        source = text(slot, args.turn, 64)
        test.check(labels == [[span, 'request' if span.startswith('Reply with') else 'statement']
            for span in expected[source]], 'selected turn keeps exact statement labels',
            kind='behavior', original_slot=slot)
        test.check(choice[at + 8256:at + 8784] == original[saved_at + 8256:saved_at + 8784],
            'preselection equals the original recorded selection', original_slot=slot)
        test.check(choice[at + 13920:at + 14448] == original[saved_at + 13920:saved_at + 14448],
            'correction targets equal the original recorded table', original_slot=slot)
        statements = [span for span, label in labels if label == 'statement']
        if args.turn == 3:
            old = saved_at + 9168
            prior = json.loads(original[old + 128:old + 128 + f.get(original, old + 4, 4)])
            test.check(result[:len(statements)] == prior[:len(statements)],
                'genuine correction keeps its exact assertion and target', kind='behavior',
                original_slot=slot, actual=result)
        else:
            test.check(result[:len(statements)] == [[3, span, 0] for span in statements],
                'new plans remain independent complete assertions', kind='behavior',
                original_slot=slot, actual=result)
        records.append(dict(original_slot=slot, first=labels, second=result))
    (test.output / 'decisions.json').write_text(json.dumps(records, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('build', 'source', 'store', 'captures', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('count', type=int, choices=(1, 64))
    parser.add_argument('--turn', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--slot', type=int, choices=range(64), default=20)
    parser.add_argument('--operation-seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.operation_seconds <= 7200: parser.error('Operation limit is outside 1 to 7200 seconds.')
    args.captures = args.captures.resolve()
    test = Test(args.build.resolve(), args.source.resolve(), args.store.resolve(), args.output.resolve(), snapshot_every=1024)
    status, start = 0, time.monotonic()
    try:
        aotx_exercise(test, args)
    except Exception as error:
        status = 1; (test.output / 'failure.txt').write_text(type(error).__name__ + ': ' + str(error) + '\n')
    finally:
        for run in test.active:
            if not run.log.closed:
                try: run.close()
                except Exception as error: status = 1; test.record(cleanup_error=str(error))
        test.flush_checks()
    failed = [row for row in test.checks if not row['passed']]
    result = dict(exit=int(bool(status or failed)), checks=len(test.checks), failed=len(failed),
        count=args.count, turn=args.turn, slot=args.slot, seconds=time.monotonic() - start)
    (test.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result['exit']


if __name__ == '__main__': sys.exit(main())
