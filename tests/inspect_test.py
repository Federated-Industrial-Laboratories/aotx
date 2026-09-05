#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# inspect_test.py: Check model file reports and refused input through the CLI.
# Input: The model program path and the fetch flag (on or off).
# Output: Case results and actual case counts.
# Exit codes: 0 passed, 1 failed, 2 invalid arguments.

import hashlib
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


U32, STRING, ARRAY = 4, 8, 9
MAX64 = (1 << 64) - 1


def number(value, width=8):
    return value.to_bytes(width, "little")


def text(value):
    value = value.encode("utf-8") if isinstance(value, str) else value
    return number(len(value)) + value


class Fixture:
    def __init__(self, architecture="llama", index=0, template=None, padding=0, layers=None):
        self.architecture = architecture
        self.layers = 1 + index % 3 if layers is None else layers
        self.hidden = 32
        self.vocabulary = 32
        self.template = template if template is not None else (
            "{{ bos_token }}{% for message in messages %}"
            "{{ message['content'] }}{% endfor %}" + f"\n{architecture}-{index}\n"
        ).encode("utf-8")
        self.metadata = [
            ("general.architecture", STRING, architecture),
            ("general.alignment", U32, 32),
            ("tokenizer.ggml.pre", STRING, "llama-bpe"),
            ("tokenizer.ggml.model", STRING, "gpt2"),
            ("tokenizer.ggml.tokens", ARRAY, [f"token_{n}" for n in range(32)]),
            ("tokenizer.chat_template", STRING, self.template),
            (f"{architecture}.block_count", U32, self.layers),
            (f"{architecture}.embedding_length", U32, self.hidden),
            (f"{architecture}.feed_forward_length", U32, 64),
            (f"{architecture}.attention.head_count", U32, 2),
            (f"{architecture}.attention.head_count_kv", U32, 1),
            (f"{architecture}.context_length", U32, 128),
        ]
        self.tensors = [("token_embd.weight", (32, 32), 0),
                        ("output_norm.weight", (32,), 0)]
        shapes = [("attn_norm", (32,)), ("attn_q", (32, 32)),
                  ("attn_k", (32, 16)), ("attn_v", (32, 16)),
                  ("attn_output", (32, 32)), ("ffn_norm", (32,)),
                  ("ffn_gate", (32, 64)), ("ffn_up", (32, 64)),
                  ("ffn_down", (64, 32))]
        if architecture == "qwen3":
            shapes += [("attn_q_norm", (16,)), ("attn_k_norm", (16,))]
        if architecture == "olmoe":
            shapes = [(name + "_exps", dims + (4,)) if name in ("ffn_gate", "ffn_up", "ffn_down")
                      else (name, dims) for name, dims in shapes]
            shapes += [("ffn_gate_inp", (32, 4)), ("attn_q_norm", (32,)), ("attn_k_norm", (16,))]
            self.metadata += [("olmoe.expert_count", U32, 4), ("olmoe.expert_used_count", U32, 2)]
        for layer in range(self.layers):
            self.tensors.extend((f"blk.{layer}.{name}.weight", dims, 0)
                                for name, dims in shapes)
        self.padding = padding

    def set_metadata(self, key, value):
        self.metadata = [(name, kind, value if name == key else old)
                         for name, kind, old in self.metadata]

    def encode(self):
        data = bytearray(b"GGUF" + number(3, 4) + number(len(self.tensors))
                         + number(len(self.metadata)))
        self.marks = {}
        for key, kind, value in self.metadata:
            self.marks[f"key:{key}"] = len(data)
            data += text(key) + number(kind, 4)
            self.marks[f"value:{key}"] = len(data)
            if kind == STRING:
                data += text(value)
            elif kind == ARRAY:
                data += number(STRING, 4) + number(len(value))
                for item in value:
                    data += text(item)
            else:
                data += number(value, 4)
        payload = bytearray(self.padding)
        for index, (name, dims, kind) in enumerate(self.tensors):
            self.marks[f"tensor:{index}"] = len(data)
            data += text(name)
            self.marks[f"dimensions:{index}"] = len(data)
            data += number(len(dims), 4)
            for dim in dims:
                data += number(dim)
            self.marks[f"type:{index}"] = len(data)
            data += number(kind, 4)
            self.marks[f"offset:{index}"] = len(data)
            data += number(len(payload))
            payload += struct.pack("<f", (index + 1) / 128) * math.prod(dims)
            payload += bytes(-len(payload) % 32)
        self.table_bytes = len(data)
        data += bytes(-len(data) % 32)
        self.header_bytes = len(data)
        data += payload
        return bytes(data)


class Checks:
    def __init__(self, executable):
        self.executable = str(Path(executable).resolve())
        self.total = 0
        self.failed = 0

    def case(self, name, action):
        self.total += 1
        try:
            action()
        except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
            self.failed += 1
            print(f"inspect: FAILED {name}: {error}", flush=True)
        else:
            print(f"inspect: ok {name}", flush=True)

    def run(self, source, code):
        env = dict(os.environ, NO_PROXY="127.0.0.1,localhost",
                   no_proxy="127.0.0.1,localhost")
        result = subprocess.run([self.executable, "inspect", str(source)],
                                capture_output=True, timeout=20, env=env)
        assert result.returncode == code, (
            f"{source}: expected exit {code}, got {result.returncode}; "
            f"stdout={result.stdout!r}; stderr={result.stderr!r}")
        assert all(32 <= byte < 127 or byte >= 160 or byte == 10 for byte in result.stdout), (
            f"unsafe output: {result.stdout!r}")
        return result.stdout.decode("utf-8"), result.stderr.decode("utf-8")

    def refused(self, source, code=2, reason=None):
        _, error = self.run(source, code)
        name = str(source).rsplit("/", 1)[-1]
        assert name in error, f"missing file name in error: {error!r}"
        remaining = error.replace(str(source), "").replace(name, "").replace("aotx_models", "").strip(" :\n")
        assert re.search(r"[A-Za-z]{3}", remaining), f"missing reason: {error!r}"
        if reason:
            assert re.search(reason, remaining, re.I), f"wrong reason: {error!r}"


def line(output, expected):
    assert expected in output.splitlines(), f"missing {expected!r}: {output!r}"


def value(output, key):
    matches = re.findall(rf"^{re.escape(key)}=(\d+)$", output, re.M)
    assert len(matches) == 1, f"missing or repeated {key}: {output!r}"
    return int(matches[0])


def report(checks, source, fixture, file_bytes, remote=False):
    output, _ = checks.run(source, 0)
    line(output, f"architecture={fixture.architecture}")
    line(output, "pre_tokenizer=llama-bpe supported=yes")
    line(output, "tokenizer_model=gpt2 supported=yes")
    line(output, f"tensors={len(fixture.tensors)}")
    line(output, f"block_type=F32 id=0 count={len(fixture.tensors)} supported=yes")
    line(output, f"layers={fixture.layers} hidden=32 vocabulary=32")
    layer_type = {"qwen3": "attention", "olmoe": "ffn_experts"}.get(
        fixture.architecture, "attention_no_qk_norm")
    line(output, f"layer_type={layer_type} count={fixture.layers}")
    line(output, "layer_sets_supported=yes unknown_tensors=0 layer_limit=64")
    digest = hashlib.sha256(fixture.template).hexdigest()
    line(output, f"chat_template_bytes={len(fixture.template)} chat_template_sha256={digest}")
    line(output, f"file_bytes={file_bytes}")
    line(output, f"header_bytes={fixture.table_bytes}")
    line(output, "build_support=yes")
    line(output, "run_verified=no")
    if remote:
        received = value(output, "received_bytes")
        assert fixture.table_bytes <= received < file_bytes // 2, output
    else:
        assert "received_bytes=" not in output, output
    return output


def local_cases(checks, root):
    for count in (1, 64):
        for architecture in ("llama", "qwen3", "olmoe"):
            for index in range(count):
                fixture = Fixture(architecture, index + count * 100)
                path = root / f"{architecture}-{count}-{index}.gguf"
                path.write_bytes(fixture.encode())
                checks.case(f"N={count} {architecture} file={index}",
                            lambda: report(checks, path, fixture, os.stat(path).st_size))
    checks.case("missing file", lambda: checks.refused(root / "absent.gguf", 1))
    base = Fixture()
    data = base.encode()
    marks = base.marks
    cuts = {
        "fixed-header": 17,
        "scalar": marks["value:general.alignment"] + 2,
        "string": marks["value:general.architecture"] + 10,
        "array-count": marks["value:tokenizer.ggml.tokens"] + 8,
        "array-string": marks["value:tokenizer.ggml.tokens"] + 22,
        "tensor-name": marks["tensor:0"] + 10,
        "tensor-dimensions": marks["dimensions:0"] + 7,
        "tensor-offset": marks["offset:0"] + 4,
        "tensor-payload": len(data) - 129,
    }
    for name, end in cuts.items():
        path = root / f"truncated-{name}.gguf"
        path.write_bytes(data[:end])
        checks.case(name, lambda: checks.refused(path, reason="end|short|truncat|header|file|tensor"))
    mutations = [
        ("metadata-count", 16, MAX64, 8),
        ("tensor-count", 8, MAX64, 8),
        ("key-length", marks["key:general.architecture"], MAX64, 8),
        ("string-length", marks["value:general.architecture"], MAX64, 8),
        ("array-count", marks["value:tokenizer.ggml.tokens"] + 4, MAX64, 8),
        ("array-string-length", marks["value:tokenizer.ggml.tokens"] + 12, MAX64, 8),
        ("tensor-name-length", marks["tensor:0"], MAX64, 8),
        ("dimension-count", marks["dimensions:0"], 0xFFFFFFFF, 4),
        ("dimension-product", marks["dimensions:0"] + 4, 1 << 63, 8),
        ("dimension-zero", marks["dimensions:0"] + 4, 0, 8),
        ("offset-overflow", marks["offset:0"], MAX64 - 31, 8),
        ("offset-past-file", marks["offset:0"], 1 << 32, 8),
        ("offset-alignment", marks["offset:0"], 1, 8),
        ("bad-magic", 0, 0, 4),
        ("bad-version", 4, 99, 4),
    ]
    for name, offset, replacement, width in mutations:
        path = root / f"invalid-{name}.gguf"
        path.write_bytes(data[:offset] + number(replacement, width) + data[offset + width:])
        checks.case(name, lambda: checks.refused(path))
    for kind in ("metadata", "tensor"):
        fixture = Fixture()
        if kind == "metadata":
            fixture.metadata.append(fixture.metadata[0])
        else:
            fixture.tensors.append(fixture.tensors[0])
        path = root / f"duplicate-{kind}.gguf"
        path.write_bytes(fixture.encode())
        checks.case(f"duplicate {kind}", lambda: checks.refused(path, reason="duplicat|repeat|same"))
    path = root / "header-limit.gguf"
    offset = marks["value:general.architecture"]
    path.write_bytes(data[:offset] + number(280000000))
    with path.open("r+b") as file:
        file.truncate(320 * 1024 * 1024)
    checks.case("header byte limit", lambda: checks.refused(path, reason="head|limit|large"))
    path = root / "allocation-limit.gguf"
    path.write_bytes(b"GGUF" + number(3, 4) + number(0) + number(5_000_000))
    with path.open("r+b") as file:
        file.truncate(320 * 1024 * 1024)
    checks.case("header allocation limit", lambda: checks.refused(path, reason="allocations.*512"))
    for architecture in ("llama", "qwen3"):
        fixture = Fixture(architecture, layers=64)
        path = root / f"{architecture}-64-layers.gguf"
        path.write_bytes(fixture.encode())
        checks.case(f"64 layers {architecture}",
                    lambda: report(checks, path, fixture, os.stat(path).st_size))
    for index, template in enumerate((b"", b"a" * 64, b"{{ content }}\x00\n\xc3\xa9")):
        fixture = Fixture(template=template)
        path = root / f"template-{index}.gguf"
        path.write_bytes(fixture.encode())
        checks.case(f"template bytes {index}",
                    lambda: report(checks, path, fixture, os.stat(path).st_size))


def unsupported_cases(checks, root):
    for name in ("pre", "model", "type", "bias", "expert", "layer-name", "escape", "architecture"):
        fixture = Fixture()
        if name in ("pre", "model"):
            fixture.set_metadata(f"tokenizer.ggml.{name}", "unknown")
        elif name == "type":
            tensor, dims, _ = fixture.tensors[-1]
            fixture.tensors[-1] = (tensor, dims, 0xFFFFFFFE)
        elif name in ("bias", "expert"):
            tensor = "blk.0.attn_q.bias" if name == "bias" else "blk.0.ffn_gate_exps.weight"
            fixture.tensors.append((tensor, (32,), 0))
        elif name == "layer-name":
            fixture.tensors = [(tensor.replace("blk.0.", "blk.00."), dims, kind)
                               for tensor, dims, kind in fixture.tensors]
        elif name == "escape":
            fixture.set_metadata("tokenizer.ggml.pre", "unknown\x1b[31m\n\t\r\x7f")
        else:
            fixture.set_metadata("general.architecture", "unknown")
        path = root / f"unsupported-{name}.gguf"
        path.write_bytes(fixture.encode())

        def check():
            output, _ = checks.run(path, 0)
            line(output, "build_support=no")
            line(output, "run_verified=no")
            if name == "type":
                line(output, "block_type=unknown id=4294967294 count=1 supported=no")
                line(output, f"block_type=F32 id=0 count={len(fixture.tensors) - 1} supported=yes")
            elif name in ("bias", "expert"):
                unknown = 1 if name == "bias" else 0
                line(output, f"layer_sets_supported=no unknown_tensors={unknown} layer_limit=64")
            elif name == "layer-name":
                line(output, "layer_sets_supported=no unknown_tensors=9 layer_limit=64")
            elif name in ("pre", "model"):
                key = "pre_tokenizer" if name == "pre" else "tokenizer_model"
                line(output, f"{key}=unknown supported=no")
            elif name == "escape":
                assert "\x7f" not in output, repr(output)
                assert re.search(r"^pre_tokenizer=unknown.+ supported=no$", output, re.M), output
        checks.case(f"unsupported {name}", check)
    for key in ("general.architecture", "tokenizer.ggml.pre", "tokenizer.ggml.model",
                "tokenizer.ggml.tokens", "llama.block_count", "llama.embedding_length"):
        for zero in (False, True):
            fixture = Fixture()
            if zero:
                old = next(item for item in fixture.metadata if item[0] == key)
                fixture.set_metadata(key, 0 if old[1] == U32 else ([] if old[1] == ARRAY else ""))
            else:
                fixture.metadata = [item for item in fixture.metadata if item[0] != key]
            path = root / f"{'zero' if zero else 'missing'}-{key}.gguf"
            path.write_bytes(fixture.encode())

            def check_missing():
                output, _ = checks.run(path, 0)
                line(output, "build_support=no")
                line(output, "run_verified=no")
            checks.case(path.stem, check_missing)
    for key in ("general.architecture", "tokenizer.ggml.pre", "tokenizer.ggml.model",
                "tokenizer.ggml.tokens", "llama.block_count", "llama.embedding_length"):
        fixture = Fixture()
        fixture.metadata = [(name, STRING if kind == U32 else U32,
                             "wrong" if kind == U32 else 7) if name == key else (name, kind, old)
                            for name, kind, old in fixture.metadata]
        path = root / f"wrong-type-{key}.gguf"
        path.write_bytes(fixture.encode())
        checks.case(path.stem, lambda: checks.refused(path, reason="type|string|number|array"))
    path = root / "empty.gguf"
    path.write_bytes(b"GGUF" + number(3, 4) + number(0) + number(0) + bytes(8))

    def check_empty():
        output, _ = checks.run(path, 0)
        line(output, "tensors=0")
        line(output, "layers=0 hidden=0 vocabulary=0")
        line(output, "build_support=no")
        line(output, "run_verified=no")
    checks.case("empty model", check_empty)


class RangeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, data):
        super().__init__(("127.0.0.1", 0), RangeHandler)
        self.data = data
        self.requests = []
        self.sent = 0
        self.stop_body = threading.Event()


class RangeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        server = self.server
        mode = self.path.split("/", 2)[1]
        raw_range = self.headers.get("Range", "")
        server.requests.append((self.path, raw_range))
        if mode == "redirect" or (mode == "resource-change" and len(server.requests) > 1):
            visits = sum(path.startswith("/redirect/") for path, _ in server.requests)
            target = "first" if mode == "redirect" and visits == 1 else "second"
            self.send_response(302)
            self.send_header("Location", f"/{target}/model.gguf")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if mode == "loop":
            self.send_response(302)
            self.send_header("Location", self.path)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", raw_range)
        if not match:
            self.send_error(400, "A bounded byte range is required")
            return
        start, end = map(int, match.groups())
        data = b"BAD!" + server.data[4:] if mode == "malformed" else server.data
        if mode == "second":
            data = data.replace(b"{{ messages }}", b"{{ messagEs }}")
        if start > end or end >= len(data):
            self.send_error(416, "The range is outside the file")
            return
        body = data[start:end + 1]
        self.send_response(200 if mode == "ignored" else 206)
        total = "0" if mode == "total" else str(len(data))
        range_start = start + 1 if mode == "range" else start
        self.send_header("Content-Range", f"bytes {range_start}-{end}/{total}")
        if mode != "missing":
            if mode == "modified":
                self.send_header("Last-Modified", "Wed, 01 Jan 2025 00:00:00 GMT")
            else:
                etag = '"file-two"' if mode == "changed" and len(server.requests) > 1 else '"file-one"'
                self.send_header("ETag", 'W/"file-one"' if mode == "weak" else etag)
        if mode == "ignored":
            body = data
        self.send_header("Content-Length", str(len(body) + (mode == "length")))
        self.send_header("Connection", "close")
        self.end_headers()
        if mode == "short":
            body = body[:-1]
        try:
            for offset in range(0, len(body), 16384):
                if server.stop_body.is_set():
                    break
                chunk = body[offset:offset + 16384]
                self.wfile.write(chunk)
                self.wfile.flush()
                server.sent += len(chunk)
                if mode == "ignored" and server.stop_body.wait(0.005):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass


def remote_cases(checks, fixture, fetch):
    data = fixture.encode()
    modes = ("valid", "modified", "ignored", "short", "length", "range", "total",
             "changed", "missing", "weak", "loop", "malformed", "redirect", "resource-change") if fetch else ("off",)
    for mode in modes:
        server = RangeServer(data)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                                  daemon=True)
        thread.start()
        source = f"http://127.0.0.1:{server.server_port}/{mode}/model.gguf"
        try:
            def check():
                if mode in ("valid", "modified", "redirect"):
                    output = report(checks, source, fixture, len(data), remote=True)
                    assert len(server.requests) >= 2, server.requests
                    requested_bytes = 0
                    for path, requested in server.requests:
                        if path.startswith("/redirect/"):
                            continue
                        match = re.fullmatch(r"bytes=(\d+)-(\d+)", requested)
                        assert match, requested
                        start, end = map(int, match.groups())
                        assert 0 <= start <= end < len(data), requested
                        requested_bytes += end - start + 1
                    assert value(output, "received_bytes") == requested_bytes, output
                    print(f"inspect: ranges={server.requests}", flush=True)
                    if mode == "redirect":
                        assert sum(path.startswith("/redirect/") for path, _ in server.requests) == 1
                        assert not any(path.startswith("/second/") for path, _ in server.requests)
                else:
                    checks.refused(source, 2 if mode == "malformed" else 1)
                    if mode == "off":
                        assert not server.requests, server.requests
                    if mode == "ignored":
                        assert server.sent < len(data) // 2, server.sent
                    if mode == "changed":
                        assert len(server.requests) >= 2, server.requests
                    if mode == "loop":
                        assert 1 <= len(server.requests) <= 11, server.requests
            checks.case(f"HTTP {mode}", check)
        finally:
            server.stop_body.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)




def main():
    if len(sys.argv) != 3 or sys.argv[2].lower() not in ("on", "off"):
        print("usage: inspect_test.py <aotx_models> <on|off>", file=sys.stderr)
        return 2
    checks = Checks(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="aotx-inspect-") as directory:
        root = Path(directory)
        local_cases(checks, root)
        unsupported_cases(checks, root)
        fixture = Fixture("qwen3", 64, template=b"{{ messages }}\n" * 8192,
                          padding=4 * 1024 * 1024)
        remote_cases(checks, fixture, sys.argv[2].lower() == "on")
    print(f"inspect: {checks.total} cases, {checks.total - checks.failed} passed, "
          f"{checks.failed} failed", flush=True)
    return int(checks.failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
