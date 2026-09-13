# SPDX-License-Identifier: Apache-2.0
# Decode strict bounded UTF-8 JSON and check envelope fields.
import json
import math
from .errors import aotx_bad


def aotx_json(data: bytes, depth=8):
    try:
        text = data.decode('utf-8')
        inside = escaped = False
        nesting = 0
        for c in text:
            if inside:
                if escaped: escaped = False
                elif c == '\\': escaped = True
                elif c == '"': inside = False
            elif c == '"': inside = True
            elif c in '[{':
                nesting += 1
                if nesting > depth: raise ValueError()
            elif c in ']}': nesting -= 1

        def pairs(items):
            result = {}
            if len(items) > 128: raise ValueError()
            for key, value in items:
                if key in result: raise ValueError()
                result[key] = value
            return result

        def constant(_): raise ValueError()

        def check(value):
            if isinstance(value, str): value.encode('utf-8')
            elif isinstance(value, float) and not math.isfinite(value): raise ValueError()
            elif isinstance(value, dict):
                for key, item in value.items(): check(key); check(item)
            elif isinstance(value, list):
                for item in value: check(item)

        result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        check(result)
        if not isinstance(result, dict): raise ValueError()
        return result
    except (ValueError, UnicodeError, RecursionError):
        raise aotx_bad('body', 'The JSON body is invalid.') from None


def aotx_fields(value, allowed, required=(), param='body'):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise aotx_bad(param)
    return value


def aotx_integer(value, low, high, param):
    if type(value) is not int or not low <= value <= high: raise aotx_bad(param)
    return value


def aotx_number(value, low, high, param):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise aotx_bad(param)
    return float(value)


def aotx_encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
