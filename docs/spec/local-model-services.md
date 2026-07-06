# Local model service launches (port 18084)

> Recommended per-model settings (server / config.toml / per-request): [model-settings.md](model-settings.md).

Seven models, each a systemd service. **Only one runs at a time** (they share port
18084 and the RTX 3080). `Conflicts=` is set (bidirectional), so **starting one
auto-stops the others** — no manual stop needed to switch.

**Established (verified) — 9Bs + Gemma:**

| Service | Model | Launcher | Per-request reasoning toggle |
|---------|-------|----------|------------------------------|
| `llama-qwopus-q6`   | Qwopus 3.5 9B Q6_K | `~/bin/llama-qwopus-q6-server` | needs `qwopus-toggle.jinja` (stock hardwires `<think>`) |
| `llama-ornith-q6`   | Ornith 1.0 9B Q6_K | `~/bin/llama-ornith-q6-server` | use `ornith-toggle.jinja` (already gates) |
| `llama-gemma4-q4km` | Gemma 4 12B agentic Q4_K_M | `~/bin/llama-gemma4-agentic-q4km-server` | use `gemma-toggle.jinja` (already gates) |
| `llama-qwythos-q6`  | Qwythos-9B-Claude-Mythos-5-1M Q6_K | `~/bin/llama-qwythos-q6-server` | ready — embedded template gates |

**Prepared (MoE trials, not yet loaded/tested):** installed + `daemon-reload`ed but
never started. Model downloads (~7 GB each) happen on first `systemctl start` via the
pinned `--hf-file`. Alias is the `--alias`; the `_moe_` label marks a mixture-of-experts
model. Reasoning toggle uses the embedded template + `--reasoning auto` — **verify
`enable_thinking` gating after first load** (`curl /props`) and add a `*-toggle.jinja`
if the embedded template doesn't gate (like Qwopus). Arch support in the current
llama.cpp build (b8881) is unverified for these newer/niche archs — first start is the test.

> **Arch-test results (2026-07-05) — all three MoE trials tested:**
> - **`fabliq_8b_moe_q6` ✅ loads** (arch supported on b8881), n_ctx 48K. Its embedded
>   template did NOT gate reasoning-off (the model emits `<think>` regardless of
>   `enable_thinking`), so **fixed with a toggle template** — `fabliq-toggle.jinja`
>   (wired into `~/bin/llama-fabliq-q6-server` via `--chat-template-file`). It primes an
>   empty `<think></think>` on `enable_thinking=false` so the model **genuinely skips
>   thinking** (verified: off → empty think block, no reasoning generated); the harness
>   strips the empty residue. The on-path is byte-identical to the embedded template.
>   **Now the active local model** (all four roles, `.codex-multi/config.toml`).
> - **`lfm2.5_8b_moe_q6` ✅ loads**, n_ctx 48K — same non-gating template, **fixed the same
>   way** with `lfm25-toggle.jinja` (verified: off skips thinking).
> - **`mellum2_12b_moe_opus_q4` ❌ FAILS on b8881** — `unknown model architecture: 'mellum'`.
>   GGUF is complete (7.6 GB); the build just lacks the arch, and it crash-restart-loops on
>   start (and via `Conflicts=` stops whatever was running). **The arch IS supported upstream:**
>   llama.cpp [PR #23966 "model: add Mellum architecture"](https://github.com/ggml-org/llama.cpp/pull/23966)
>   was **merged to master 2026-06-02**. Our local checkout (`/home/jesse/src/llama.cpp-cuda`,
>   commit `0dedb9ef7`, **b8881, dated 2026-04-21**) predates it by ~6 weeks — confirmed via
>   `git merge-base --is-ancestor` (merge NOT in local HEAD). **To enable Mellum: `git pull` the
>   checkout past 2026-06-02 and rebuild the CUDA `llama-server`, then re-verify every other
>   model still loads** (6 weeks of upstream churn). Until then, don't switch to it.

| Service | Model (alias) | Launcher | GGUF (`--hf-file`) |
|---------|---------------|----------|--------------------|
| `llama-mellum2-q4` | Mellum2-12B-A2.5B MoE Opus-Thinking Q4_K_M (`mellum2_12b_moe_opus_q4`) — ❌ needs llama.cpp ≥ 2026-06-02 (PR #23966); b8881 too old | `~/bin/llama-mellum2-q4-server` | `yuxinlu1/…-GGUF` → `mellum2-claude-Q4_K_M.gguf` |
| `llama-fabliq-q6`  | Fabliq-8B-Agent MoE i1-Q6_K (`fabliq_8b_moe_q6`) — ✅ loads; `fabliq-toggle.jinja` gates reasoning | `~/bin/llama-fabliq-q6-server` | `mradermacher/Fabliq-8B-Agent-i1-GGUF` → `Fabliq-8B-Agent.i1-Q6_K.gguf` |
| `llama-lfm25-q6`   | LFM2.5-8B-A1B MoE UD-Q6_K_XL (`lfm2.5_8b_moe_q6`) — ✅ loads; `lfm25-toggle.jinja` gates reasoning | `~/bin/llama-lfm25-q6-server` | `unsloth/LFM2.5-8B-A1B-GGUF` → `LFM2.5-8B-A1B-UD-Q6_K_XL.gguf` |

All seven services bind `127.0.0.1:18084` and share the uniform launcher shape
(48K ctx, q8 KV, single 3080, `-ub 512`) — see [model-settings.md](model-settings.md).
`-hf` models cache under `~/.cache/huggingface/hub/` (or `~/.cache/llama.cpp/`).

## Switch models
```bash
sudo systemctl start llama-qwythos-q6      # auto-stops whichever was running
curl -s 127.0.0.1:18084/props | python3 -c 'import sys,json;print(json.load(sys.stdin)["model_path"])'
```

## Boot default
Only `llama-qwopus-q6` is **enabled** (auto-starts at boot). The other three are
manual-start. To change the boot default:
```bash
sudo systemctl disable llama-qwopus-q6
sudo systemctl enable  llama-qwythos-q6
```
(Keep exactly one enabled, or they'll contend for the port at boot.)

## Reasoning ON / OFF / AUTO
Reasoning is driven **per request** by the role's `reasoning` config (`off`/`on`/`auto`
→ `chat_template_kwargs.enable_thinking`; see [model-settings.md](model-settings.md)).
One running instance can serve reasoning-on and reasoning-off roles at once — **provided
it loads a chat template that gates on `enable_thinking`**:

- **Qwythos** — embedded template already gates. Launch as-is; per-request toggle works.
- **Ornith / Gemma** — their `reason-on` templates already gate. Launch with
  `--chat-template-file ~/shepherd-eval/templates/{ornith,gemma}-toggle.jinja`.
- **Qwopus** — stock hardwires `<think>`. Launch with
  `--chat-template-file ~/shepherd-eval/templates/qwopus-toggle.jinja` (the merged
  gated template — `enable_thinking=false` renders byte-exact to the old `nothink`
  file, `=true` to `reason-on`).
- **Fabliq / LFM2.5** — embedded template ignores `enable_thinking` (the model emits
  `<think>` regardless). Launch with `--chat-template-file
  ~/shepherd-eval/templates/{fabliq,lfm25}-toggle.jinja` (already wired into their
  launchers). The gate is minimal: it only adds an empty `<think></think>` prime on
  `enable_thinking=false` so the model skips thinking; the on-path is byte-identical to
  the embedded template. Verified: on → clean `reasoning_content`; off → empty think.

Leave `--reasoning-budget` at default — `--reasoning-budget 0` clamps thinking off for
the whole instance and defeats per-request control. `--reasoning-format deepseek/auto`
is orthogonal (parses `<think>` into `reasoning_content`; keep it for clean separation).
Verify with the per-request toggle probe in [model-settings.md](model-settings.md).

**Whole-instance override (fallback):** to pin ONE mode for every request regardless of
role, launch with `--reasoning-budget 0` (models that gate) or a `nothink` template swap
(Qwopus, `~/shepherd-eval/templates/qwopus-nothink.jinja`). Superseded by the per-request
path above for normal multi-role use.

## Notes
- Gemma is the only Q4_K_M (4-bit) model; the three 9Bs are Q6_K (6-bit). All four are
  ~7.3–7.4 GB and run on the **single RTX 3080** with a `-ub 512` prefill — no multi-GPU
  split.
- **All four launchers are uniform** (48K ctx, q8 KV, 3080-only, `-ub 512`, port 18084);
  they differ only by model + chat template (see [model-settings.md](model-settings.md)).
  Gemma now hardcodes port 18084 like the rest — the old `GEMMA4_AGENTIC_PORT` env
  indirection is gone, so the systemd unit's port override (if any) is now redundant.
- Timestamped known-good launcher snapshots live in `/home/jesse/llama-service-configs/`.
