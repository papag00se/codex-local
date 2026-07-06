# Recommended settings per model

Living reference for the 4 models under test on `127.0.0.1:18084`. Companion to
[local-model-services.md](local-model-services.md) (how to launch/switch them). Three layers, in
**precedence order** (later overrides earlier for a given request):

1. **Server-side** — llama-server launcher flags (`~/bin/llama-<model>-*-server`).
   The model's baseline: quant, context, KV, GPU, and *default* sampling.
2. **Client config** — `.codex-multi/config.toml` `[models.<role>]`. Sampling here
   is sent **per request and overrides the launcher's `--temp/--top-p/--top-k`**.
   Keyed by ROLE (classifier / light_coder / light_reasoner / compactor), not by
   model — the same loaded model serves every role with different sampling.
3. **Per-request** — what the harness puts on the wire. The role's sampling **and**
   its `reasoning` flag (sent as `chat_template_kwargs.enable_thinking`), so one loaded
   instance can serve reasoning-on and reasoning-off roles at once — no restart, no
   per-launch flag. Requires a gating template server-side. See "Reasoning control".

> **Quant caveat:** the three 9Bs are **Q6_K**; Gemma is **Q4_K_M** (4-bit). Some of
> Gemma's weaker eval showing may be the lower quant, not the model.

---

## 1. Server-side (launcher flags)

All four launchers (`~/bin/llama-<model>-*-server`) are **uniform** — same context, GPU,
prefill, sampling, and port. They differ ONLY by the model and its chat template.

**Uniform flags (identical in every launcher):**
`-c 49152` (48K) · `-b 2048 -ub 512` · `-np 1` · `--device CUDA0 -ngl auto -sm none -mg 0`
(single 3080, pinned via `CUDA_VISIBLE_DEVICES`) · `-ctk q8_0 -ctv q8_0` · `--no-host
--no-mmproj -fa on --no-warmup --jinja` · `--temp 1.0 --top-p 0.95 --top-k 20` ·
`--reasoning auto` · `--host 127.0.0.1 --port 18084`.

**Per-model — model + chat template only:**

| Model | Quant | Model source | Chat template |
|-------|-------|--------------|---------------|
| **Qwopus 3.5 9B** | Q6_K | `-hf Jackrong/Qwopus3.5-9B-v3-GGUF:Q6_K` | `qwopus-toggle.jinja` (stock hardwires `<think>`) |
| **Ornith 1.0 9B** | Q6_K | `-hf deepreinforce-ai/Ornith-1.0-9B-GGUF:Q6_K` | `ornith-toggle.jinja` |
| **Qwythos 9B** | Q6_K | `-m …/Qwythos-…-Q6_K.gguf` | *(none — embedded template gates)* |
| **Gemma 4 12B** | Q4_K_M | `-m …/gemma4-v2-Q4_K_M.gguf` | `gemma-toggle.jinja` |

Templates live in `~/shepherd-eval/templates/` and are applied with `--chat-template-file`
(all but Qwythos). See "Reasoning control".

- **`-ub 512` is required** — a larger prefill micro-batch overflows the 10 GB 3080.
- **All four fit the single 3080** — the 9Bs are ~7.3 GB (Q6_K), Gemma ~7.4 GB (Q4_K_M);
  no multi-GPU split. `--no-mmproj` drops vision weights (text-only coding) to save VRAM.

Notes:
- **Launcher sampling is a FALLBACK** for the coding path — the config's `light_coder`
  temp (0.1) overrides it per request (see below). It still applies to any request that
  doesn't set sampling, and defines the baseline for ad-hoc `curl` tests. Per-model
  sampling tuning (e.g. a cooler reasoner) belongs in the client config, not the launcher.
- **KV cache** is q8_0 for all (halves KV VRAM vs f16, negligible quality cost).
- **Reasoning is per-request** (config `reasoning` → `chat_template_kwargs.enable_thinking`),
  which is why every launcher carries `--reasoning auto` (leaves the flag to the request)
  plus the gating template. Leave `--reasoning-budget` at default — a launch-time
  `--reasoning-budget 0` pins the whole instance no-think. See "Reasoning control".

---

## 2. Client config — `.codex-multi/config.toml` (per ROLE, overrides launcher sampling)

These are what the coder/reasoner **actually run at** (sent on every request). The same
loaded model serves all roles; only the sampling + role differ. All roles point at
`endpoint = "http://127.0.0.1:18084/v1"`, `provider = "openai-compat"`, `model =
"ornith_1_9b_q6"` (a label — llama-server serves whatever GGUF is loaded, ignores it).

| Role | temp | top_p | top_k | repeat_penalty | max_tokens | reasoning | timeout |
|------|------|-------|-------|----------------|-----------|-----------|---------|
| `classifier` | 0.0 | 1.0 | 1 | — | — | off | 30s |
| **`light_coder`** (main coding) | **0.1** | 0.95 | 20 | 1.1 | 4096 | auto | 7200s |
| `light_reasoner` | 0.4 | 0.95 | 40 | — | 4096 | auto | 7200s |
| `compactor` | 0.0 | 1.0 | 1 | — | — | auto | 7200s |

- **Coding runs near-greedy (temp 0.1)** for determinism — this OVERRIDES the launcher's
  higher temp. If you want to change how the coder samples, edit `light_coder` here, not
  the launcher.
- **Classifier/compactor are fully greedy (temp 0.0, top_k 1)** — structured snap outputs.
- **`max_tokens` vs `output_reserve` — two separate knobs (don't conflate them).**
  `max_tokens` is the conventional **hard output cap**, and it's **unset = uncapped by
  default** so a big `write_file` generates to completion instead of truncating mid-content
  (that truncation, reported as "wrote N bytes", drove a rewrite loop). `output_reserve` is
  the input-side **window reserve** — it guarantees the model ≥ N tokens of room by trimming
  input to `n_ctx − N − margin`; give file-writing roles a generous value. `timeout_seconds`
  (not `max_tokens`) is the wall-clock bound — set it high on slow CPUs; the mission is
  agentic coding on low-end hardware where long waits are acceptable.
- `model_context_window` is **not set** — the harness auto-detects real `n_ctx` from
  `/props` at startup and drives native compaction off it. Don't hardcode it.
- `[routing] local_only = true` blocks the cloud tiers so everything routes to the local
  roles.

---

## 3. Per-request settings — reasoning now rides here too

The harness sends the role's `temperature / top_p / top_k / repeat_penalty / max_tokens`
per request (overriding the launcher). **Reasoning is now one of them** — the role's
`reasoning` field goes on every request as `chat_template_kwargs.enable_thinking`, so one
loaded instance serves reasoning-on and reasoning-off roles simultaneously (no restart).

`reasoning` is **tri-state**, with the conventional meaning of a thinking flag:

| `reasoning` | On the wire | Effect |
|-------------|-------------|--------|
| `off` | `chat_template_kwargs.enable_thinking = false` | force no-think |
| `on`  | `chat_template_kwargs.enable_thinking = true`  | force think |
| `auto` (or unset/unrecognized) | *(omitted)* | the model's template default decides |

- **Ollama-flavor** endpoints get the same value on the top-level `think` field instead;
  both flavors **omit** the flag entirely on `auto`.
- **Requires a gating template** (see "Reasoning control"). Against a template that
  hardwires `<think>` (stock Qwopus), `enable_thinking` is ignored and `off` won't
  suppress — that was the old "request toggles don't work" behavior, a *template*
  limitation, not the wire. It's fixed for any template that gates on `enable_thinking`.
- Code: `apply_reasoning_kwargs` / `apply_ollama_think` in
  `codex-rs/routing/src/ollama.rs`; string→tri-state in `thinking_from_reasoning`
  (`config.rs`). Cloud tiers still read `reasoning` as an effort level.

---

## Reasoning control (OFF / ON / AUTO) — per-request, needs a gating template

Reasoning is driven **per request** by each role's `reasoning` config (Section 3). The one
server-side requirement: the loaded model must run a chat template that **gates on
`enable_thinking`**, so the per-request flag actually changes the generation prompt.

| Model | Gating template | Launch |
|-------|-----------------|--------|
| **Qwythos** | embedded template already gates | nothing — no `--chat-template-file` needed |
| **Ornith / Gemma** | their `reason-on` templates already gate | `--chat-template-file ~/shepherd-eval/templates/{ornith,gemma}-toggle.jinja` |
| **Qwopus** | stock hardwires `<think>`; use the merged toggle | `--chat-template-file ~/shepherd-eval/templates/qwopus-toggle.jinja` |

The `*-toggle.jinja` templates unify each model's `nothink` + `reason-on` into ONE template
gated on `enable_thinking` (verified by rendering: `enable_thinking=false` is byte-exact to
the old `nothink` file, `=true` to `reason-on`). Launch with the toggle template and **leave
`--reasoning-budget` at default** — a launch-time `--reasoning-budget 0` clamps thinking off
for the whole instance and defeats per-request control. Each model's think token differs
(Qwen `<think>`, Ornith deepseek, Gemma channel `<|channel>thought`).

**Superseded (blunt whole-instance override, still valid):** launching with
`--reasoning-budget 0`, or swapping in a `nothink`/`reason-on` template per launch, forces
ONE mode for every request. Only reach for it to deliberately pin a whole instance, or when
a model's template can't gate.

**Eval verdict on reasoning (F17):** it's base-model-dependent. Strong base (Qwythos)
tolerates OFF (faster, decisive); weaker bases (Ornith, Gemma) collapse OFF (loop, drop
constraints). Qwopus does better OFF but that's a *fabricated* off-mode (no native
off-branch). **Default to ON** unless you've verified a given model is fine OFF.

---

## Quick reference — verify what's actually live
```bash
# which model + context
curl -s 127.0.0.1:18084/props | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["model_path"].split("/")[-1], "n_ctx=", d.get("default_generation_settings",{}).get("n_ctx"))'
# reasoning per-request toggle (needs a gating template loaded; true -> >0, false -> 0)
for t in true false; do printf 'enable_thinking=%s -> ' "$t"
  curl -s 127.0.0.1:18084/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"2+2? think first\"}],\"stream\":false,\"max_tokens\":200,\"chat_template_kwargs\":{\"enable_thinking\":$t}}" \
    | python3 -c 'import sys,json;print("reasoning_len:", len(json.load(sys.stdin)["choices"][0]["message"].get("reasoning_content") or ""))'
done
```
