// @dotrino/ai — hablar con un modelo de IA con la LLAVE DEL USUARIO.
//
// La llamada sale de donde está el usuario (su navegador, su PC), directo al proveedor que
// él eligió. No pasa por ningún servidor de Dotrino y aquí no hay ninguna llave de nadie más.
//
// Sin dependencias e isomórfico (fetch, FormData, Blob y AbortSignal existen en los
// navegadores y en Node ≥ 20). Tres familias de API cubren a los proveedores:
//
//   openai     OpenAI, DeepSeek, Groq, OpenRouter, Ollama y cualquier API compatible
//   anthropic  Claude
//   gemini     Google Gemini
//
// Lo que devuelve un modelo es una PROPUESTA: quien llama la comprueba en código antes de
// usarla. Este paquete solo garantiza que lo que entrega es texto, o JSON que se pudo leer.

import { codeError } from './errors.js'

export { codeError }

/** @typedef {'openai' | 'anthropic' | 'gemini'} Family */
/**
 * @typedef {object} Provider
 * @property {Family} family
 * @property {string} label
 * @property {string} baseUrl
 * @property {string} model
 * @property {string} [transcribeModel]
 * @property {boolean} [keyless]
 * @property {boolean} browser
 * @property {string} [keyUrl]
 */
/** @typedef {'openai' | 'deepseek' | 'groq' | 'openrouter' | 'ollama' | 'anthropic' | 'gemini' | 'custom'} ProviderName */
/** @typedef {{ role: 'user' | 'assistant', content: string }} Message */
/**
 * @typedef {object} AskRequest
 * @property {string} [system]
 * @property {string} [prompt]
 * @property {Message[]} [messages]
 * @property {number} [temperature]
 * @property {number} [maxTokens]
 * @property {'low' | 'medium' | 'high'} [effort]
 */
/** @typedef {AskRequest & { schema?: object }} AskJsonRequest */
/** @typedef {{ system: string, messages: Message[], temperature: number, maxTokens: number, effort?: 'low' | 'medium' | 'high' }} NormalizedRequest */
/** @typedef {{ language?: string, prompt?: string }} TranscribeOptions */
/**
 * @typedef {object} Config
 * @property {string} provider
 * @property {Family} family
 * @property {string} apiKey
 * @property {string} model
 * @property {string} baseUrl
 * @property {string | null} transcribeModel
 * @property {number} timeoutMs
 * @property {typeof fetch} fetch
 */
/**
 * @typedef {object} FamilyImpl
 * @property {(cfg: Config, req: NormalizedRequest, opts: { json: boolean, schema?: object }) => Promise<string>} chat
 * @property {(cfg: Config, audio: Blob, opts?: TranscribeOptions) => Promise<string>} [transcribe]
 */
/**
 * @typedef {object} AI
 * @property {Readonly<{ provider: string, label: string, model: string, baseUrl: string, canTranscribe: boolean }>} info
 * @property {(req: AskRequest) => Promise<string>} ask
 * @property {(req: AskJsonRequest) => Promise<any>} askJson  JSON leído del modelo: lo comprueba quien llama
 * @property {(audio: Blob, opts?: TranscribeOptions) => Promise<string>} transcribe
 */

/**
 * Proveedores conocidos. `model` es el de por defecto (barato y rápido: lo normal es
 * entender una frase, no escribir un ensayo); el usuario puede pasar otro.
 * `browser` dice si el proveedor deja llamar desde una página web (CORS): si es false, se
 * puede usar desde Node pero en el navegador la llamada no sale.
 */
/** @type {Readonly<Record<ProviderName, Provider>>} */
export const PROVIDERS = Object.freeze({
  openai: { family: 'openai', label: 'OpenAI', baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini', transcribeModel: 'whisper-1', browser: true, keyUrl: 'https://platform.openai.com/api-keys' },
  deepseek: { family: 'openai', label: 'DeepSeek', baseUrl: 'https://api.deepseek.com', model: 'deepseek-chat', browser: true, keyUrl: 'https://platform.deepseek.com/api_keys' },
  groq: { family: 'openai', label: 'Groq', baseUrl: 'https://api.groq.com/openai/v1', model: 'llama-3.3-70b-versatile', transcribeModel: 'whisper-large-v3-turbo', browser: true, keyUrl: 'https://console.groq.com/keys' },
  openrouter: { family: 'openai', label: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', model: 'openai/gpt-4o-mini', browser: true, keyUrl: 'https://openrouter.ai/keys' },
  ollama: { family: 'openai', label: 'Ollama', baseUrl: 'http://localhost:11434/v1', model: 'llama3.2', keyless: true, browser: true, keyUrl: 'https://ollama.com/download' },
  anthropic: { family: 'anthropic', label: 'Anthropic (Claude)', baseUrl: 'https://api.anthropic.com/v1', model: 'claude-opus-5-5', browser: true, keyUrl: 'https://console.anthropic.com/settings/keys' },
  gemini: { family: 'gemini', label: 'Google Gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-2.5-flash', transcribeModel: 'gemini-2.5-flash', browser: true, keyUrl: 'https://aistudio.google.com/apikey' },
  custom: { family: 'openai', label: 'OpenAI-compatible', baseUrl: '', model: '', browser: true },
})

const DEFAULT_TIMEOUT = 60_000

/**
 * @param {object} opts
 * @param {ProviderName} opts.provider
 * @param {string} [opts.apiKey]   obligatoria salvo en los proveedores `keyless` (Ollama)
 * @param {string} [opts.model]    el del proveedor si no se pasa
 * @param {string} [opts.baseUrl]  obligatoria en `custom`; en los demás cambia el servidor
 * @param {typeof fetch} [opts.fetch]
 * @param {number} [opts.timeoutMs]
 * @returns {AI}
 */
// Sin opciones, `provider` queda undefined y se lanza `unknown-provider` (el cast es solo para eso).
export function createAI ({ provider, apiKey = '', model, baseUrl, fetch: f, timeoutMs = DEFAULT_TIMEOUT } = /** @type {{ provider: ProviderName }} */ ({})) {
  const preset = PROVIDERS[provider]
  if (!preset) throw codeError('unknown-provider', `unknown AI provider: ${provider}`)
  /** @type {Config} */
  const cfg = {
    provider,
    family: preset.family,
    apiKey: String(apiKey || '').trim(),
    model: model || preset.model,
    baseUrl: String(baseUrl || preset.baseUrl).replace(/\/+$/, ''),
    transcribeModel: preset.transcribeModel || null,
    timeoutMs,
    // El `fetch` del navegador exige que `this` sea `window`: guardado en un objeto y llamado
    // como método lanza «Illegal invocation» (en Node no pasa, por eso no lo ve una prueba de
    // Node). Se envuelve para llamarlo siempre suelto.
    fetch: f || ((input, init) => globalThis.fetch(input, init)),
  }
  if (!cfg.apiKey && !preset.keyless) throw codeError('no-llm-key', `${provider} needs an API key`)
  if (!cfg.baseUrl) throw codeError('no-base-url', `${provider} needs a base URL`)
  if (!cfg.model) throw codeError('no-model', `${provider} needs a model name`)
  if (typeof globalThis.fetch !== 'function' && !f) throw codeError('no-fetch', 'no fetch implementation available')

  /** @type {FamilyImpl} */
  const family = FAMILIES[cfg.family]

  return {
    info: Object.freeze({ provider, label: preset.label, model: cfg.model, baseUrl: cfg.baseUrl, canTranscribe: !!family.transcribe && (!!cfg.transcribeModel || cfg.family === 'gemini') }),

    /**
     * Texto → texto.
     * @param {{ system?: string, prompt?: string, messages?: {role:'user'|'assistant', content:string}[], temperature?: number, maxTokens?: number, effort?: 'low'|'medium'|'high' }} req
     *   `temperature` no se manda a Claude (sus modelos actuales la rechazan); `effort` solo a Claude.
     * @returns {Promise<string>}
     */
    async ask (req) {
      return family.chat(cfg, normalize(req), { json: false })
    },

    /**
     * Texto → objeto JSON. Se le pide JSON al proveedor (donde hay modo JSON) y se lee lo
     * que devuelve; si no es JSON, lanza `llm-json` con el principio de la respuesta.
     * `schema` (JSON Schema, objetos con `additionalProperties: false`) lo hace cumplir el
     * proveedor donde se puede (Claude, OpenAI); en los demás solo pide JSON.
     * @param {AskJsonRequest} req
     * @returns {Promise<any>}
     */
    async askJson (req) {
      const text = await family.chat(cfg, normalize(req), { json: true, schema: req.schema })
      return parseJson(text)
    },

    /**
     * Audio → texto. Solo donde el proveedor lo da (OpenAI, Groq, Gemini); en los demás
     * lanza `no-transcribe`.
     * @param {Blob} audio
     * @param {TranscribeOptions} [opts]
     * @returns {Promise<string>}
     */
    async transcribe (audio, opts = {}) {
      if (!family.transcribe || (!cfg.transcribeModel && cfg.family !== 'gemini')) {
        throw codeError('no-transcribe', `${provider} cannot transcribe audio`)
      }
      return family.transcribe(cfg, audio, opts)
    },
  }
}

/**
 * @param {AskRequest} [req]
 * @returns {NormalizedRequest}
 */
function normalize (req = {}) {
  const messages = req.messages ? [...req.messages] : []
  if (req.prompt) messages.push({ role: 'user', content: String(req.prompt) })
  if (!messages.length) throw codeError('no-prompt', 'nothing to ask: pass `prompt` or `messages`')
  return { system: req.system || '', messages, temperature: req.temperature ?? 0.2, maxTokens: req.maxTokens ?? 1024, effort: req.effort }
}

/**
 * JSON tal cual, o el que viene dentro de un bloque ```json. Si no hay JSON, se dice.
 * @param {unknown} text
 * @returns {any} lo que dé JSON.parse: la forma la comprueba quien llama
 */
export function parseJson (text) {
  const src = String(text || '').trim()
  try {
    return JSON.parse(src)
  } catch { /* se intenta el bloque de abajo */ }
  const fenced = /```(?:json)?\s*([\s\S]*?)```/i.exec(src)
  const candidate = fenced ? fenced[1] : src.slice(src.indexOf('{'), src.lastIndexOf('}') + 1)
  try {
    if (candidate) return JSON.parse(candidate)
  } catch { /* abajo se lanza con el texto */ }
  throw codeError('llm-json', `the model did not answer JSON: ${src.slice(0, 200)}`)
}

// ---------- la llamada, con los errores separados por lo que hay que hacer ----------

/**
 * @param {Config} cfg
 * @param {string} url
 * @param {RequestInit} init
 * @returns {Promise<Response>}
 */
async function call (cfg, url, init) {
  let res
  try {
    res = await cfg.fetch(url, { ...init, signal: AbortSignal.timeout(cfg.timeoutMs) })
  } catch (err) {
    const e = /** @type {any} */ (err) // cualquier cosa lanzada por fetch
    if (e?.name === 'TimeoutError' || e?.name === 'AbortError') {
      throw codeError('llm-timeout', `${cfg.provider} did not answer within ${cfg.timeoutMs} ms`, { cause: e })
    }
    // En un navegador, sin red y «el proveedor no deja llamar desde una web» (CORS) dan el
    // mismo TypeError: no se puede saber cuál fue, y se dice así.
    throw codeError('llm-unreachable', `${cfg.provider} could not be reached (offline, or the provider blocks calls from web pages): ${e?.message || e}`, { cause: e })
  }
  if (res.ok) return res
  const body = (await res.text().catch(() => '')).slice(0, 300)
  if (res.status === 401 || res.status === 403) throw codeError('llm-auth', `${cfg.provider} rejected the key (HTTP ${res.status}): ${body}`, { status: res.status })
  if (res.status === 429) throw codeError('llm-rate-limited', `${cfg.provider} rate limit or quota reached (HTTP 429): ${body}`, { status: 429 })
  if (res.status === 404) throw codeError('llm-not-found', `${cfg.provider} does not know this model or URL (HTTP 404): ${body}`, { status: 404 })
  throw codeError('llm-http', `${cfg.provider} → HTTP ${res.status}: ${body}`, { status: res.status })
}

/**
 * @param {Config} cfg
 * @param {unknown} [why]
 * @returns {Error}
 */
function empty (cfg, why) {
  return codeError('llm-empty', `${cfg.provider} returned an empty answer${why ? ` (${why})` : ''}`)
}

// ---------- familias ----------

/** @type {FamilyImpl} */
const openai = {
  async chat (cfg, req, { json, schema }) {
    const messages = req.system ? [{ role: 'system', content: req.system }, ...req.messages] : req.messages
    /** @type {Record<string, unknown>} */
    const body = { model: cfg.model, messages, temperature: req.temperature, max_tokens: req.maxTokens }
    // El esquema estricto solo lo da OpenAI; los compatibles entienden el modo JSON simple.
    if (json && schema && cfg.provider === 'openai') body.response_format = { type: 'json_schema', json_schema: { name: 'answer', strict: true, schema } }
    else if (json) body.response_format = { type: 'json_object' }
    /** @type {Record<string, string>} */
    const headers = { 'content-type': 'application/json' }
    if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`
    const res = await call(cfg, `${cfg.baseUrl}/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) })
    const data = await res.json()
    const content = data?.choices?.[0]?.message?.content
    if (!content) throw empty(cfg, data?.choices?.[0]?.finish_reason)
    return content
  },

  async transcribe (cfg, audio, { language, prompt } = {}) {
    const form = new FormData()
    form.append('file', audio, /** @type {File} */ (audio).name || fileNameFor(audio.type))
    form.append('model', /** @type {string} */ (cfg.transcribeModel))
    if (language) form.append('language', language)
    if (prompt) form.append('prompt', prompt)
    /** @type {Record<string, string>} */
    const headers = {}
    if (cfg.apiKey) headers.authorization = `Bearer ${cfg.apiKey}`
    const res = await call(cfg, `${cfg.baseUrl}/audio/transcriptions`, { method: 'POST', headers, body: form })
    const data = await res.json()
    if (typeof data?.text !== 'string') throw empty(cfg, 'no text')
    return data.text.trim()
  },
}

/** @type {FamilyImpl} */
const anthropic = {
  async chat (cfg, req, { json, schema }) {
    // Los modelos actuales de Claude rechazan `temperature` (400): no se manda. La salida JSON
    // va por `output_config.format` cuando hay esquema; sin esquema se pide en el sistema.
    const system = json && !schema ? `${req.system}\n\nAnswer with a single JSON object and nothing else.`.trim() : req.system
    // Con el razonamiento siempre encendido, `max_tokens` incluye lo que piensa: un tope bajo
    // corta la respuesta antes de empezar.
    /** @type {Record<string, unknown>} */
    const body = { model: cfg.model, max_tokens: Math.max(req.maxTokens, 4096), messages: req.messages }
    if (system) body.system = system
    /** @type {Record<string, unknown>} */
    const output = {}
    if (schema) output.format = { type: 'json_schema', schema }
    if (req.effort) output.effort = req.effort
    if (Object.keys(output).length) body.output_config = output
    // Si un clasificador de seguridad declina, el servidor reintenta con otro modelo en vez
    // de devolver la negativa al usuario.
    body.fallbacks = 'default'
    const res = await call(cfg, `${cfg.baseUrl}/messages`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': cfg.apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-beta': 'server-side-fallback-2026-07-01',
        // La llave es del usuario y la llamada sale de su navegador: es justo el caso para
        // el que existe esta cabecera.
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    if (data?.stop_reason === 'refusal') {
      throw codeError('llm-refused', `${cfg.provider} declined to answer (${data?.stop_details?.category || 'no category'})`)
    }
    const text = (data?.content || []).filter((/** @type {any} */ b) => b.type === 'text').map((/** @type {any} */ b) => b.text).join('')
    if (!text) throw empty(cfg, data?.stop_reason)
    return text
  },
}

/**
 * @typedef {object} GeminiImpl
 * @property {FamilyImpl['chat']} chat
 * @property {NonNullable<FamilyImpl['transcribe']>} transcribe
 * @property {(cfg: Config, model: string, body: object) => Promise<string>} generate
 */
/** @type {GeminiImpl} */
const gemini = {
  async chat (cfg, req, { json }) {
    /** @type {{ contents: object[], generationConfig: Record<string, unknown>, systemInstruction?: object }} */
    const body = {
      contents: req.messages.map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
      generationConfig: { temperature: req.temperature, maxOutputTokens: req.maxTokens },
    }
    if (req.system) body.systemInstruction = { parts: [{ text: req.system }] }
    if (json) body.generationConfig.responseMimeType = 'application/json'
    return gemini.generate(cfg, cfg.model, body)
  },

  async transcribe (cfg, audio, { language, prompt } = {}) {
    const data = await toBase64(audio)
    const ask = ['Transcribe this audio verbatim. Answer with the transcription only.', language ? `Language: ${language}.` : '', prompt || ''].filter(Boolean).join(' ')
    const body = {
      contents: [{ role: 'user', parts: [{ inline_data: { mime_type: audio.type || 'audio/webm', data } }, { text: ask }] }],
      generationConfig: { temperature: 0 },
    }
    return (await gemini.generate(cfg, cfg.transcribeModel || cfg.model, body)).trim()
  },

  async generate (cfg, model, body) {
    const res = await call(cfg, `${cfg.baseUrl}/models/${encodeURIComponent(model)}:generateContent`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-goog-api-key': cfg.apiKey },
      body: JSON.stringify(body),
    })
    const data = await res.json()
    const cand = data?.candidates?.[0]
    const text = (cand?.content?.parts || []).map((/** @type {any} */ p) => p.text || '').join('')
    if (!text) throw empty(cfg, cand?.finishReason || data?.promptFeedback?.blockReason)
    return text
  },
}

/** @type {Record<Family, FamilyImpl>} */
const FAMILIES = { openai, anthropic, gemini }

/**
 * @param {string} [mime]
 * @returns {string}
 */
function fileNameFor (mime = '') {
  if (mime.includes('ogg')) return 'audio.ogg'
  if (mime.includes('mp4') || mime.includes('m4a') || mime.includes('aac')) return 'audio.m4a'
  if (mime.includes('mpeg') || mime.includes('mp3')) return 'audio.mp3'
  if (mime.includes('wav')) return 'audio.wav'
  return 'audio.webm'
}

/**
 * @param {Blob} blob
 * @returns {Promise<string>}
 */
async function toBase64 (blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer())
  let bin = ''
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(bin)
}
