import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createAI, parseJson, PROVIDERS } from '../src/index.js'

// Un fetch falso que guarda lo que se le pidió y contesta lo que se le diga.
function fakeFetch (reply) {
  const calls = []
  const f = async (url, init) => {
    calls.push({ url, init })
    const r = typeof reply === 'function' ? reply(url, init) : reply
    if (r instanceof Error) throw r
    const { status = 200, body } = r
    return new Response(typeof body === 'string' ? body : JSON.stringify(body), { status })
  }
  f.calls = calls
  return f
}

test('a key is required, except for keyless providers; unknown providers are rejected', () => {
  assert.throws(() => createAI({ provider: 'openai' }), (e) => e.code === 'no-llm-key')
  assert.throws(() => createAI({ provider: 'nope', apiKey: 'k' }), (e) => e.code === 'unknown-provider')
  assert.throws(() => createAI({ provider: 'custom', apiKey: 'k' }), (e) => e.code === 'no-base-url')
  assert.equal(createAI({ provider: 'ollama' }).info.model, PROVIDERS.ollama.model)
})

test('openai family: chat completions with bearer key, system first, JSON mode', async () => {
  const f = fakeFetch({ body: { choices: [{ message: { content: '{"a":1}' } }] } })
  const ai = createAI({ provider: 'deepseek', apiKey: 'sk-1', fetch: f })
  assert.deepEqual(await ai.askJson({ system: 'sys', prompt: 'hi' }), { a: 1 })
  const { url, init } = f.calls[0]
  assert.equal(url, 'https://api.deepseek.com/chat/completions')
  assert.equal(init.headers.authorization, 'Bearer sk-1')
  const body = JSON.parse(init.body)
  assert.equal(body.model, 'deepseek-chat')
  assert.deepEqual(body.messages, [{ role: 'system', content: 'sys' }, { role: 'user', content: 'hi' }])
  assert.deepEqual(body.response_format, { type: 'json_object' })
})

test('ollama goes without an authorization header; custom takes its base URL', async () => {
  const f = fakeFetch({ body: { choices: [{ message: { content: 'ok' } }] } })
  await createAI({ provider: 'ollama', fetch: f }).ask({ prompt: 'x' })
  assert.equal(f.calls[0].init.headers.authorization, undefined)
  await createAI({ provider: 'custom', apiKey: 'k', baseUrl: 'https://llm.example/v1/', model: 'm', fetch: f }).ask({ prompt: 'x' })
  assert.equal(f.calls[1].url, 'https://llm.example/v1/chat/completions')
})

test('anthropic: messages API with the browser header, no temperature, fallbacks on', async () => {
  const f = fakeFetch({ body: { content: [{ type: 'text', text: 'hola' }], stop_reason: 'end_turn' } })
  const ai = createAI({ provider: 'anthropic', apiKey: 'sk-ant', fetch: f })
  assert.equal(ai.info.model, 'claude-opus-5-5')
  assert.equal(await ai.ask({ system: 's', prompt: 'p', maxTokens: 50, effort: 'low' }), 'hola')
  const { url, init } = f.calls[0]
  assert.equal(url, 'https://api.anthropic.com/v1/messages')
  assert.equal(init.headers['x-api-key'], 'sk-ant')
  assert.equal(init.headers['anthropic-dangerous-direct-browser-access'], 'true')
  assert.equal(init.headers['anthropic-version'], '2023-06-01')
  assert.equal(init.headers['anthropic-beta'], 'server-side-fallback-2026-07-01')
  const body = JSON.parse(init.body)
  assert.equal(body.system, 's')
  assert.equal(body.temperature, undefined)
  assert.equal(body.max_tokens, 4096, 'thinking counts toward max_tokens: never below 4096')
  assert.equal(body.fallbacks, 'default')
  assert.deepEqual(body.output_config, { effort: 'low' })
  assert.deepEqual(body.messages, [{ role: 'user', content: 'p' }])
  await assert.rejects(() => ai.transcribe(new Blob(['x'])), (e) => e.code === 'no-transcribe')
})

test('anthropic: a schema goes as output_config.format; a refusal has its own code', async () => {
  const schema = { type: 'object', properties: { a: { type: 'integer' } }, required: ['a'], additionalProperties: false }
  const f = fakeFetch({ body: { content: [{ type: 'text', text: '{"a":1}' }], stop_reason: 'end_turn' } })
  assert.deepEqual(await createAI({ provider: 'anthropic', apiKey: 'k', fetch: f }).askJson({ prompt: 'p', schema }), { a: 1 })
  const body = JSON.parse(f.calls[0].init.body)
  assert.deepEqual(body.output_config.format, { type: 'json_schema', schema })
  assert.equal(body.system, undefined, 'with a schema the JSON request does not go in the system prompt')
  const r = fakeFetch({ body: { content: [], stop_reason: 'refusal', stop_details: { type: 'refusal', category: 'cyber' } } })
  await assert.rejects(createAI({ provider: 'anthropic', apiKey: 'k', fetch: r }).ask({ prompt: 'p' }), (e) => e.code === 'llm-refused' && /cyber/.test(e.message))
})

test('openai: a schema is enforced only by OpenAI itself; compatibles get plain JSON mode', async () => {
  const schema = { type: 'object', properties: {}, additionalProperties: false }
  const f = fakeFetch({ body: { choices: [{ message: { content: '{}' } }] } })
  await createAI({ provider: 'openai', apiKey: 'k', fetch: f }).askJson({ prompt: 'p', schema })
  assert.equal(JSON.parse(f.calls[0].init.body).response_format.type, 'json_schema')
  await createAI({ provider: 'groq', apiKey: 'k', fetch: f }).askJson({ prompt: 'p', schema })
  assert.equal(JSON.parse(f.calls[1].init.body).response_format.type, 'json_object')
})

test('gemini: generateContent with the key in a header, roles mapped, JSON mime', async () => {
  const f = fakeFetch({ body: { candidates: [{ content: { parts: [{ text: '{"ok":true}' }] } }] } })
  const ai = createAI({ provider: 'gemini', apiKey: 'g-1', fetch: f })
  assert.deepEqual(await ai.askJson({ system: 's', messages: [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }] }), { ok: true })
  const { url, init } = f.calls[0]
  assert.match(url, /\/models\/gemini-2\.5-flash:generateContent$/)
  assert.equal(init.headers['x-goog-api-key'], 'g-1')
  const body = JSON.parse(init.body)
  assert.deepEqual(body.contents.map((c) => c.role), ['user', 'model'])
  assert.equal(body.systemInstruction.parts[0].text, 's')
  assert.equal(body.generationConfig.responseMimeType, 'application/json')
})

test('transcription: multipart for openai-compatible, inline audio for gemini', async () => {
  const f = fakeFetch({ body: { text: ' hola mundo ' } })
  const text = await createAI({ provider: 'groq', apiKey: 'k', fetch: f }).transcribe(new Blob([new Uint8Array([1, 2])], { type: 'audio/webm' }), { language: 'es' })
  assert.equal(text, 'hola mundo')
  assert.equal(f.calls[0].url, 'https://api.groq.com/openai/v1/audio/transcriptions')
  const form = f.calls[0].init.body
  assert.equal(form.get('model'), 'whisper-large-v3-turbo')
  assert.equal(form.get('language'), 'es')
  assert.equal(form.get('file').name, 'audio.webm')

  const g = fakeFetch({ body: { candidates: [{ content: { parts: [{ text: 'dicho' }] } }] } })
  assert.equal(await createAI({ provider: 'gemini', apiKey: 'k', fetch: g }).transcribe(new Blob([new Uint8Array([1, 2, 3])], { type: 'audio/ogg' })), 'dicho')
  const part = JSON.parse(g.calls[0].init.body).contents[0].parts[0]
  assert.deepEqual(part.inline_data, { mime_type: 'audio/ogg', data: 'AQID' })
  assert.equal(createAI({ provider: 'deepseek', apiKey: 'k' }).info.canTranscribe, false)
})

test('errors keep apart what is fixed differently', async () => {
  const ask = (reply) => createAI({ provider: 'openai', apiKey: 'k', fetch: fakeFetch(reply) }).ask({ prompt: 'x' })
  await assert.rejects(ask({ status: 401, body: 'bad key' }), (e) => e.code === 'llm-auth' && e.status === 401)
  await assert.rejects(ask({ status: 429, body: 'slow down' }), (e) => e.code === 'llm-rate-limited')
  await assert.rejects(ask({ status: 404, body: 'no model' }), (e) => e.code === 'llm-not-found')
  await assert.rejects(ask({ status: 500, body: 'boom' }), (e) => e.code === 'llm-http' && e.status === 500)
  await assert.rejects(ask({ body: { choices: [{ message: { content: '' }, finish_reason: 'length' }] } }), (e) => e.code === 'llm-empty' && /length/.test(e.message))
  await assert.rejects(ask(new TypeError('Failed to fetch')), (e) => e.code === 'llm-unreachable')
  const timeout = Object.assign(new Error('t'), { name: 'TimeoutError' })
  await assert.rejects(ask(timeout), (e) => e.code === 'llm-timeout')
  await assert.rejects(() => createAI({ provider: 'openai', apiKey: 'k' }).ask({}), (e) => e.code === 'no-prompt')
})

test('JSON is read as is, from a fenced block, or the first object; otherwise llm-json', () => {
  assert.deepEqual(parseJson('{"a":1}'), { a: 1 })
  assert.deepEqual(parseJson('Here:\n```json\n{"a":2}\n```'), { a: 2 })
  assert.deepEqual(parseJson('Sure! {"a":3} hope it helps'), { a: 3 })
  assert.throws(() => parseJson('no json here'), (e) => e.code === 'llm-json')
})
