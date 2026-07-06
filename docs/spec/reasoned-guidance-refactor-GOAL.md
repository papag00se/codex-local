GOAL: Converge Shephard's interventions on grounded reasoned guidance, and consolidate the disjointed intervention code.

You are working in the `codex-local` Rust research harness (`codex-rs/`). Its job is to make a small (~8–12B) local model do autonomous agentic coding by intervening around it. Across many iterations the intervention code has become disjointed, and live sessions prove the core problem: the guards DETECT loops correctly, but the bare-text "nudge" interventions are IGNORED, so the model flails. One recent session burned 510K tokens on a trivial loop and ended in a false completion.

This effort has two inseparable thrusts:

1. Convert every STEERING nudge into grounded reasoned guidance. A bare directive a weak model ignores becomes a reasoner-authored next step — but ONLY when the reasoner is fed FRESH GROUND TRUTH: the actual files re-read from disk, the actual test/probe output, the actual repeated call and its error. NEVER let the reasoner reason over the model's own claims or a clean probe dressed up as a diagnosis — that hallucinates (it once told the coder to "add an X-API-Key header" for a runtime TypeError). If there is no fresh signal, do not call the reasoner. Keep MECHANICAL interventions mechanical — the hard block on identical infinite writes, output repair (`\n`/JSON), truncation-resume, and the rumination CUTOFF are NOT reasoned (their job is to stop or repair, not to steer).

2. Consolidate the disjointed intervention code into one coherent grounded-reasoned-guidance layer backed by one ground-truth provider, deleting the cruft accumulated across iterations.

The detailed, ordered task list — with per-task scope, current-vs-target behavior, acceptance criteria, and the files involved — is in `docs/spec/reasoned-guidance-refactor.md`. READ IT FIRST and work it top to bottom. The highest-value conversion (the context excise → reasoned + ground-truth rebuild) is the worked example that establishes the pattern; generalize from it.

WORKING METHOD (every task):
- One task at a time, in order. Do not batch or skip ahead.
- First READ the code paths the task names and understand what is really there — verify against the code and, where useful, the live-session logs at ~/.codex/log/codex-tui.log. No conjecture; if you are unsure of the truth, go find it before changing anything.
- Fix the ROOT cause upstream. No band-aids, fallbacks, or mitigations. If the real fix is out of scope, stop and say so.
- Any reasoned intervention you add must be wired to the shared ground-truth provider — if you add a reasoner call, you also add what grounds it.
- Keep reasoner rebuilds SMALL and bounded (token cap + tight, single-answer prompt); a weak reasoner told to "summarize everything" will itself ramble.
- Add or adjust tests for the behavior you changed. Then build: `cargo build -p codex-cli` (this deploys the codex-debug binary; lib-only builds do NOT). Confirm tests pass.
- Update the matching spec doc (docs/spec/*) as part of the task — the specs are the deliverable, not an afterthought.
- Describe scope by surface area, risk, and dependencies — never in units of time.
- After finishing a task, append a one-line result to the Progress log in the task-list file, so work is resumable if interrupted.

DONE (whole effort):
- Every steering nudge either routes through the grounded reasoned-guidance layer (fed fresh truth) or is a deliberately-kept mechanical intervention, documented as such.
- The context excise rebuilds a clean, ground-truth working context via the reasoner, not a canned reframe.
- One guidance layer, one ground-truth provider; duplicate/dead intervention code removed.
- The full `cargo build -p codex-cli` passes and the specs match the code.
