# Chapter 2 — Serving a Model over HTTP

This chapter takes you from "I have a `.gguf` file" to "I have a working OpenAI-compatible
API endpoint". We will start with the smallest command that works, then add one flag at a
time and explain what each buys you.

---

## 2.1 The smallest command that works

Open PowerShell in the llama.cpp folder and run:

```powershell
cd S:\OneDrive\Tools\llamacpp

.\llama-server.exe -m "S:\HuggingFace\lmstudio\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
```

That is genuinely all you need. llama.cpp will:

1. read the GGUF header and print the architecture,
2. **auto-fit** the model to your available VRAM (deciding `-ngl` and `--tensor-split` for you),
3. pick a context size,
4. start listening on `http://127.0.0.1:8080`.

Open that URL in a browser and you get a built-in chat web UI. Congratulations — you are
serving a 35-billion-parameter model.

> **Teaching point.** Start here *every time you meet a new model.* Before you tune
> anything, confirm the model loads and answers coherently. Tuning a broken setup is the
> most common way to waste an afternoon.

### Reading the startup log

Three lines matter. Find them before you do anything else:

```
print_info: n_ctx_train  = 262144        ← the largest context the model was TRAINED for
print_info: n_layer      = 40            ← how many layers you are splitting
main: model loaded
main: listening on http://127.0.0.1:8080  ← it worked
```

`n_ctx_train` is your hard ceiling for quality. You *can* ask for more with RoPE scaling,
but you should not, unless you enjoy debugging incoherence.

---

## 2.2 Making it reachable and predictable

The defaults are fine for a first look and wrong for daily use. Three additions:

```powershell
.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf" `
  --host 0.0.0.0 `
  --port 9010 `
  --alias qwen3.6-35b
```

| Flag | Why |
| --- | --- |
| `--host 0.0.0.0` | Listen on all interfaces, so other machines (or a container, or your phone) can reach it. The default `127.0.0.1` is local-only. |
| `--port 9010` | Pick a port you'll remember. 8080 collides with half the software you own. |
| `--alias qwen3.6-35b` | The name reported to API clients as the "model". Some clients display it; some require it to match a configured string. |

> **Security note.** `--host 0.0.0.0` exposes an unauthenticated endpoint that will happily
> run inference for anybody who can route to your machine. On a home LAN behind a router
> that is usually fine. If you are anywhere less trusted, add `--api-key <secret>` and send
> `Authorization: Bearer <secret>`. Note the startup warning llama.cpp prints:
> *"CORS is set to allow all origins ('\*') and no API key is set."* It is not joking.

---

## 2.3 Talking to it

### Check it is alive

```powershell
curl.exe http://localhost:9010/health
```

Expect `{"status":"ok"}`. While the model is still loading you get a 503 instead — useful
for scripts that need to wait.

### The OpenAI-compatible endpoint

This is the one you will actually use, because every LLM tool on earth speaks it:

```powershell
curl.exe http://localhost:9010/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"qwen3.6-35b\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain a KV cache in two sentences.\"}],\"max_tokens\":200}'
```

Escaping JSON inside PowerShell quoting is miserable. Use a here-string instead — this is
the idiom to remember:

```powershell
$body = @'
{
  "model": "qwen3.6-35b",
  "messages": [
    {"role": "user", "content": "Explain a KV cache in two sentences."}
  ],
  "max_tokens": 200,
  "temperature": 0.6
}
'@

Invoke-RestMethod -Uri http://localhost:9010/v1/chat/completions `
  -Method Post -ContentType 'application/json' -Body $body |
  ForEach-Object { $_.choices[0].message.content }
```

### Streaming

Add `"stream": true` and you get server-sent events — tokens as they are produced, which is
what makes a chat UI feel responsive:

```powershell
curl.exe -N http://localhost:9010/v1/chat/completions `
  -H "Content-Type: application/json" `
  --data-binary "@request.json"
```

(`-N` disables curl's buffering; without it you see nothing until the end.)

### The useful endpoints

| Endpoint | Method | What it gives you |
| --- | --- | --- |
| `/health` | GET | Loading / ready |
| `/v1/chat/completions` | POST | OpenAI chat API — **use this one** |
| `/v1/completions` | POST | OpenAI legacy raw completion |
| `/v1/models` | GET | Model list, for clients that probe it |
| `/completion` | POST | llama.cpp's own richer API (exposes every sampler) |
| `/props` | GET | Live server settings, including the chat template |
| `/slots` | GET | **Per-slot state — your best debugging tool** |
| `/metrics` | GET | Prometheus metrics (needs `--metrics`) |
| `/tokenize`, `/detokenize` | POST | Token counting — how you find out a prompt is 57,000 tokens |
| `/` | GET | The built-in web UI |

`GET /slots` deserves special mention. It shows you what each slot is holding right now,
which is how you diagnose "why is it re-processing my whole prompt every time?".

---

## 2.4 Slots, continuous batching, and prompt caching

This is the part that turns a working server into a *fast* one, and it is the feature that
LM Studio hides from you.

### Slots

A **slot** is an independent conversation with its own KV cache. llama-server splits your
`-c` context budget across `-np` slots:

```
-c 130048 -np 4   →   4 slots × 32512 tokens each
-c 130048 -np 1   →   1 slot  × 130048 tokens
```

Read that twice, because it is the most common configuration error there is. **If you ask
for a 130k context and four slots, no single request can use more than 32k.** A 50,000-token
prompt will be rejected or truncated, and you will blame the model.

So the rule is:

> Choose `-np` by asking: *how many long conversations do I need at the same time?*
> For one coding assistant, that number is 1.

With `-kvu` / `--kv-unified` the accounting becomes shared rather than statically divided,
which is more flexible; it is the default in recent builds. Check the startup log for
`n_slots` and `kv_unified` and trust what it prints over what you assumed.

### Continuous batching

`-cb` (on by default) lets the server interleave several requests in one GPU batch. Two
users typing at once get nearly the throughput of one, instead of taking turns. Leave it on.

### Prompt caching — the biggest win available to you

llama-server remembers the KV state of previous prompts and reuses the **longest common
prefix** with a new request. For an agentic coding workload this is transformative, because
every request repeats a large, identical system prompt and file context.

Real numbers measured on this machine:

```
Cold request:   57,559 prompt tokens   →   32.5 s
Next request:   same prefix + a small edit
                f_sim_best = 0.987, f_keep = 0.995
                only 790 tokens actually evaluated   →   0.96 s
```

That is a **34× reduction in wait time**, from a feature you get for free. In the server log
you will see the decision being made:

```
selected slot by LCP similarity, f_sim_best = 0.987, f_keep = 0.995
```

- `f_sim_best` — how much of the new prompt matched the best slot. Above 0.95 is excellent.
- `f_keep` — how much of that slot's cache survived.

Two flags control the caching:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-cram N` / `--cache-ram N` | 8192 MiB | Host-RAM budget for **cached prompt states beyond what fits in the slots**. With 64 GiB of RAM you can afford far more than 8 GiB. `-1` means unlimited. |
| `--cache-reuse N` | | Minimum chunk size to salvage from a partially-matching cache via KV shifting. |

> **Teaching point.** `--cache-prompt` and `-kvo` appear in older guides. In build 10509
> they are still *accepted* (so old commands don't break) but they no longer do anything —
> the behaviour they used to enable is now the default. Don't cargo-cult them.

Raising the prompt cache is close to free performance if you have spare system RAM:

```powershell
-cram 24576        # let llama.cpp keep ~24 GiB of prompt states in host RAM
```

---

## 2.5 Controlling the chat template and thinking mode

Qwen3.6 is a reasoning model: it emits a thinking block before its answer. You usually want
to control that.

```powershell
--jinja                                             # use the model's own template (default on)
--chat-template-kwargs "{\"enable_thinking\":false}"   # turn thinking off entirely
--reasoning-format none                             # leave think tags in the raw text
--reasoning-budget 0                                # allow thinking, but end it immediately
```

For a coding assistant that must answer fast, `enable_thinking:false` is often the right
call. For hard reasoning tasks, leave it on and raise `--reasoning-budget`.

Sampling settings recommended by the model authors:

| Mode | Flags |
| --- | --- |
| Thinking, general | `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0` |
| Thinking, precise coding | `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` |
| Non-thinking | `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5` |

These are server-side defaults; an API client that sends its own `temperature` overrides them.

---

## 2.6 A realistic production command

Putting the chapter together — this is a complete, sensible server for a single-user coding
assistant. Chapter 7 has the tuned per-model variants; this one is the *shape* to remember.

```powershell
cd S:\OneDrive\Tools\llamacpp

.\llama-server.exe `
  -m "S:\HuggingFace\lmstudio\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf" `
  --alias qwen3.6-35b `
  --host 0.0.0.0 --port 9010 `
  -c 130048 -np 1 `
  -ngl 999 -sm layer -ts 12,29 -fa on `
  -b 2048 -ub 512 `
  -ctk q8_0 -ctv q8_0 `
  -cram 24576 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

Grouped by purpose, so you can see there are only four ideas here:

```
WHAT to serve      -m, --alias
WHERE to serve     --host, --port
HOW MUCH context   -c, -np, -ctk, -ctv
HOW FAST           -ngl, -sm, -ts, -fa, -b, -ub, -cram
```

---

## 2.7 Pointing a client at it

Because the API is OpenAI-shaped, most tools need only a base URL.

**Generic OpenAI-compatible client:**
```
Base URL:  http://localhost:9010/v1
API key:   (anything non-empty, e.g. "none")
Model:     qwen3.6-35b
```

**VS Code / GitHub Copilot-style extensions** that support a custom OpenAI endpoint: use the
same base URL. Then watch your server log — you should see `f_sim_best` climbing towards
0.99 on successive requests, which tells you prompt caching is doing its job. If you instead
see the full prompt reprocessed every time, check `-np` (see §2.4) before anything else.

**Python:**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9010/v1", api_key="none")
print(client.chat.completions.create(
    model="qwen3.6-35b",
    messages=[{"role": "user", "content": "hello"}],
).choices[0].message.content)
```

---

## 2.8 Running it as a background service

For daily use you want it running without a terminal window pinned open:

```powershell
$args = @(
  '-m','S:\HuggingFace\lmstudio\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf',
  '--host','0.0.0.0','--port','9010',
  '-c','130048','-np','1','-ngl','999','-sm','layer','-ts','12,29',
  '-fa','on','-b','2048','-ub','512','-ctk','q8_0','-ctv','q8_0'
)
Start-Process -FilePath 'S:\OneDrive\Tools\llamacpp\llama-server.exe' `
  -ArgumentList $args -WindowStyle Hidden `
  -RedirectStandardError 'S:\OneDrive\Tools\llamacpp\wiki\benchmarks\server.log'
```

**Keep that log.** It is where `f_sim_best`, prefill timings, and OOM warnings appear, and
you will want it the first time something is mysteriously slow.

To stop it:

```powershell
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process
```

Ready-made scripts are in [`scripts/`](scripts/).

---

Next: [Chapter 3 — The parameters that matter](03-parameters.md), where every flag above is
explained properly, along with the ones that will quietly ruin your performance.
