# @dotrino/ai

> **Parte del ecosistema [Dotrino](https://dotrino.com).** Dotrino es un ecosistema de aplicaciones centradas en la privacidad de los datos: tu información es tuya, y las decisiones sobre ella también — qué compartes, con quién, cuándo y por qué. Sin anuncios, sin cookies, sin rastreo de datos, sin vender tu identidad a nadie.

Hablar con un modelo de IA **con la llave del usuario**. La llamada sale de donde está el
usuario (su navegador, su PC) directo al proveedor que él eligió: no pasa por ningún
servidor de Dotrino y aquí no hay ninguna llave de nadie más.

Sin dependencias e isomórfico (navegador y Node ≥ 20). En el wiki:
[wiki.dotrino.com/desarrollo/pilares](https://wiki.dotrino.com/desarrollo/pilares/).

| Proveedor | `provider` | Familia | Transcribe audio |
|---|---|---|---|
| OpenAI | `openai` | openai | sí (`whisper-1`) |
| DeepSeek | `deepseek` | openai | no |
| Groq | `groq` | openai | sí (`whisper-large-v3-turbo`) |
| OpenRouter | `openrouter` | openai | no |
| Ollama (local, sin llave) | `ollama` | openai | no |
| Cualquier API compatible con OpenAI | `custom` (+ `baseUrl`, `model`) | openai | no |
| Anthropic (Claude) | `anthropic` | anthropic | no |
| Google Gemini | `gemini` | gemini | sí |

## Uso

```js
import { createAI, PROVIDERS } from '@dotrino/ai'

const ai = createAI({ provider: 'deepseek', apiKey })          // model y baseUrl opcionales
const text = await ai.ask({ system: '…', prompt: 'hola' })
const obj  = await ai.askJson({ system: '…', prompt: 'almuerzo 4.50', schema })  // schema opcional
const said = await ai.transcribe(blob, { language: 'es' })     // si ai.info.canTranscribe
```

Lo que devuelve un modelo es una **propuesta**: quien llama la comprueba en código antes de
usarla.

## Errores (`e.code`)

| Código | Qué pasó | Qué se hace |
|---|---|---|
| `no-llm-key` · `no-base-url` · `no-model` · `unknown-provider` | falta configuración | pedirla en ajustes |
| `llm-auth` | el proveedor rechazó la llave (401/403) | cambiar la llave |
| `llm-rate-limited` | tope de uso o saldo (429) | esperar o recargar saldo |
| `llm-not-found` | el modelo o la URL no existen (404) | revisar el modelo |
| `llm-unreachable` | sin red, o el proveedor no deja llamar desde una web (CORS) | en el navegador no se puede saber cuál de las dos |
| `llm-timeout` | no contestó a tiempo | reintentar |
| `llm-refused` | Claude declinó la petición | reformular |
| `llm-empty` · `llm-json` · `llm-http` | respuesta vacía, no-JSON u otro HTTP | ver el mensaje |
| `no-transcribe` | el proveedor no transcribe audio | elegir otro |

## Notas por proveedor

- **Claude** se llama con la cabecera `anthropic-dangerous-direct-browser-access` (la llave
  es del usuario y la llamada sale de su navegador). No se manda `temperature`: los modelos
  actuales la rechazan. `max_tokens` nunca baja de 4096, porque el razonamiento cuenta
  dentro. Va con `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`): si un
  filtro de seguridad declina, el servidor reintenta con otro modelo. Con `schema`, la salida
  JSON la garantiza `output_config.format`.
- **OpenAI** garantiza el esquema (`response_format: json_schema`); los compatibles reciben
  el modo JSON simple.
- **Ollama** desde una página web necesita `OLLAMA_ORIGINS` con el origen de la app.

## Desarrollo

```sh
npm test
```

Se publica desde CI con un tag `vX.Y.Z` (`.github/workflows/release.yml`).

## Licencia

MIT — © Dotrino
