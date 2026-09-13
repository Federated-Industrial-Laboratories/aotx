// SPDX-License-Identifier: Apache-2.0
// Check standard JSON and SSE inference with an independently installed JavaScript SDK.
// Inputs: SDK module, client configuration and output path. Outputs: Results. Exit: 0 pass, 1 failure.
import { readFile, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

const [modulePath, configPath, outputPath] = process.argv.slice(2);
const { default: OpenAI } = await import(pathToFileURL(modulePath).href);
const config = JSON.parse(await readFile(configPath, 'utf8'));
const results = [];
const clients = config.keys.map(apiKey => new OpenAI({ apiKey, baseURL: config.baseURL,
  maxRetries: 0, timeout: 300000 }));
const body = { model: config.model, messages: [
  { role: 'system', content: 'Give a short direct answer.' },
  { role: 'user', content: 'What color is a clear daytime sky?' },
], temperature: 0, max_tokens: 16 };

function request(i) {
  const color = ['blue', 'green', 'red', 'yellow', 'orange', 'purple', 'white', 'black'][i % 8];
  return { ...body, messages: [{ role: 'user',
    content: `For client ${i}, repeat this label: case ${i} ${color}.` }] };
}
async function owned(i, identity) {
  assert(identity.startsWith('chatcmpl-'));
  const epoch = BigInt(config.runtime_epoch).toString(16).padStart(16, '0');
  const handle = `req-${epoch}-${identity.slice(9)}`;
  const url = config.baseURL.replace(/\/v1$/, '') + '/aotx/v1/requests/' + handle;
  const response = await fetch(url, { headers: { Authorization: 'Bearer '+config.keys[i] } });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.id, handle);
  assert.equal(result.state, 'completed');
  assert.equal(result.output.next_cursor, result.output.total_bytes);
  return result;
}
const models = await clients[0].models.list();
assert(models.data.some(m => m.id === config.model));
results.push({ check: 'model discovery', passed: true });
for (const n of config.batches) {
  assert(clients.length >= n);
  const replies = await Promise.all(clients.slice(0, n).map((c, i) => c.chat.completions.create(request(i))));
  assert.equal(new Set(replies.map(r => r.id)).size, n);
  for (const [i, reply] of replies.entries()) {
    const native = await owned(i, reply.id);
    assert.equal(reply.choices[0].message.content, Buffer.from(native.output.bytes, 'base64').toString('utf8'));
    for (const [key, value] of Object.entries(native.usage)) assert.equal(reply.usage[key], value);
    assert(reply.choices[0].message.content.trim());
    assert(reply.usage.prompt_tokens > 0 && reply.usage.completion_tokens > 0);
    assert(['stop', 'length'].includes(reply.choices[0].finish_reason));
  }
  results.push({ check: 'JSON batch', count: n, passed: true, replies });
  const outputs = await Promise.all(clients.slice(0, n).map(async (client, i) => {
    const stream = await client.chat.completions.create({ ...request(i), stream: true,
      stream_options: { include_usage: true } });
    const chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    assert.equal(chunks[0].choices[0].delta.role, 'assistant');
    assert.deepEqual(chunks.at(-1).choices, []);
    assert(chunks.at(-1).usage.completion_tokens > 0);
    assert(['stop', 'length'].includes(chunks.at(-2).choices[0].finish_reason));
    const native = await owned(i, chunks[0].id);
    assert(chunks.every(c => c.id === chunks[0].id));
    const text = chunks.filter(c => c.choices.length).map(c => c.choices[0].delta.content ?? '').join('');
    assert.equal(text, Buffer.from(native.output.bytes, 'base64').toString('utf8'));
    for (const [key, value] of Object.entries(native.usage)) assert.equal(chunks.at(-1).usage[key], value);
    assert.equal(chunks.at(-2).choices[0].finish_reason, native.finish_reason);
    return chunks;
  }));
  results.push({ check: 'SSE batch', count: n, passed: true, outputs });
}
await assert.rejects(clients[0].chat.completions.create({ ...body, tools: [] }),
  error => error.status === 400 && error.code === 'invalid_field');
results.push({ check: 'structured unsupported field error', passed: true });
await writeFile(outputPath, JSON.stringify(results, null, 2)+'\n');
console.log(`SDK: ${results.length} checks, 0 failures`);
