#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a durable memory write and its replay before a new recall.
Inputs: an active conversation after its third input and its retained report.
Outputs: restore summaries, token dumps and comparison fields in the report.
Exit: exceptions identify a missing write, a replay difference or a failed recall.
"""
import re

import arch_process as process


def results(turn, tool):
    calls = [row for row in turn['records']
             if row.get('kind') == 'call' and row.get('tool') == tool]
    return [row for row in turn['records'] if row.get('kind') == 'result'
            and row.get('status') == 'ok'
            and any(call['request'] == row['request'] for call in calls)]


def restart(args, before, report):
    assert results(report['turns'][-1], 'memory_write'), 'the third input completed no memory write'
    history = {'records': [row for turn in report['turns'] for row in turn['records']]}
    writes = results(history, 'memory_write')
    assert all(row['text'] == 'the note is in memory' for row in writes), 'the memory result is not stable'
    last = report['turns'][-1]['last_turn']
    process.wait(lambda: len(before.turns()) == last, before.child)
    durable = before.summary('before-kill-summary')
    historical = before.tokens('before-kill-tokens', durable)
    assert len(process.sampled_turns(historical, 0)) == last, 'some turns are not durable'
    assert all(process.sampled_turns(historical, 0)), 'a durable turn has no sampled tokens'
    before.stop(killed=True)
    old = before.summary('summary')
    old_turns = before.turns()
    old_tokens = before.tokens('tokens', old)
    before.close()
    args.omit_restore = False
    after = type(before)(args, args.out / 'after', before.journal, restore=True)
    try:
        after.ready()
        replay = process.check_restore_log(after.log_path.read_text(), old)
        current = after.summary('after-replay-summary')
        assert current['boot'] != old['boot'] and current.get('restore_of') == old['boot']
        assert current.get('restore_hash') == old['state_hash']
        process.wait(lambda: len(after.turns()) == last, after.child)
        process.compare(old_turns, after.turns(), 'historical manifests')
        new_tokens = after.tokens('after-replay-tokens', current)
        replayed = [row for row in new_tokens if row['replayed'] == '1']
        keys = lambda data: [(row['slot'], row['position'], row['token'], row['flags']) for row in data]
        process.compare(keys(old_tokens), keys(replayed), 'replayed token prefix')
        events = process.rows(after.journal / after.boot / 'transcript/0.jsonl')
        restored_writes = results({'records': events}, 'memory_write')
        write_key = lambda rows: [(row['turn'], row['request'], row['status'], row['text']) for row in rows]
        process.compare(write_key(writes), write_key(restored_writes), 'restored write results')
        report['restore'] = {'before': old, 'after': current, 'turns': last,
                             'report': replay,
                             'tokens': len(replayed), 'writes': restored_writes,
                             'historical_manifests_equal': True, 'token_prefix_equal': True}
        print('memory restore: historical manifests, tokens and write results match', flush=True)
        return after
    except Exception:
        after.close()
        raise


def check_recall(report):
    recall = results(report['turns'][-1], 'memory_recall')
    code = r'(?<!\d)4827(?!\d)'
    assert recall and any(re.search(code, row['text']) for row in recall), 'restored recall lacks the saved code'
    replies = [row['text'] for row in report['turns'][-1]['records'] if row.get('kind') == 'reply']
    assert replies and re.search(code, replies[-1]), 'the final reply lacks the saved code'
    report['restore']['recall_results'] = recall
    report['restore']['recall_passed'] = True
    print('memory restore: new recall returns the saved code and the reply includes it', flush=True)


def restart_refusal(args, before, report):
    before.send('agent 0 pages 1')
    before.send('say Hello!')
    process.wait(lambda: 'say: the sequence did not open' in before.console.read_text(), before.child)
    process.wait(lambda: len(before.turns()) == 1, before.child)
    before.stop(killed=True)
    old = before.summary('refused-summary')
    turns = before.turns()
    assert len(turns) == 1 and int(turns[0]['tokens']) == 0, 'the refused turn has tokens'
    assert not before.tokens('refused-tokens', old), 'the refused run emitted tokens'
    before.close()
    args.omit_restore = False
    after = type(before)(args, args.out / 'after', before.journal, restore=True)
    try:
        after.ready()
        replay = process.check_restore_log(after.log_path.read_text(), old, expected_decode_refused=1)
        current = after.summary('after-refusal-summary')
        assert current['boot'] != old['boot'] and current.get('restore_of') == old['boot']
        assert current.get('restore_hash') == old['state_hash']
        process.wait(lambda: len(after.turns()) == len(turns), after.child)
        process.compare(turns, after.turns(), 'refused turn manifests')
        assert not after.tokens('after-refusal-tokens', current), 'restore added tokens to the refused run'
        report['refused_restore'] = {'before': old, 'after': current, 'report': replay,
                                    'turns': len(turns), 'tokens': 0, 'manifests_equal': True}
        after.send('agent 0 pages 64')
        print('refused restore: the failed turn matches; the page limit is now 64', flush=True)
        return after
    except Exception:
        after.close()
        raise
