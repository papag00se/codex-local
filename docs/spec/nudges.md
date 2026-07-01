# Nudge ideas — read-mode / research-loop detectors

[< Spec Index](index.md) · catalog: [shephard.md](shephard.md) · why + code pointers: [local-coder-massaging.md](local-coder-massaging.md)

Forward proposals (not yet built). These target a failure mode the current detectors
miss: the model spins in **read / research mode** — searching, fetching, reading —
gathering information forever without ever acting or making progress.

*Live example that motivated these:* a session firing `web_search "api.handle.me
swagger.yml …"` and `web_fetch raw.githubusercontent.com` dozens of times over many
minutes — every call a variation of the same question, all fetches 404, never a
single file written.

## `NEW` — how these relate to `detect_tunnel_vision` (complementary, not substitutes)

The tunnel-vision detector already exists and already does most of what a
"circling" detector should. The clarifying insight from working through it:

- **`detect_tunnel_vision`** counts *revisits to the SAME well-defined target* and
  resets on any *new* target. It catches circling a **fixed set** — including
  **write** loops (re-writing one file 8×). Its blind spot: a stream of *different*
  targets (varied search queries, guessed URLs) keeps resetting it, so it sleeps
  through a research loop that never repeats itself.
- **Read-without-write (#1 below)** counts *reads since the last write* and resets
  ONLY on a **write**. It doesn't care whether the targets repeat — so it fires on
  exactly the varied loop tunnel-vision misses.

So they cover **disjoint** failure modes and neither subsumes the other:

| detector | fires on | resets on | catches the loop that motivated this? |
|---|---|---|---|
| `detect_tunnel_vision` (exists) | revisiting a **fixed** target set (incl. writes) | a new target | **no** — varied queries reset it |
| read-without-write (**`NEW`**, #1) | many reads, **zero writes** (targets may all differ) | a **write** | **yes** |

**Takeaway:** #1 (read-without-write) is the genuinely *additive* detector — it is
not a refinement of tunnel-vision, it's its **read-space sibling**. #2 and #3 below
are narrower special-cases that sit between the two. The one change tunnel-vision
itself needs is the reset rule below (approach-not-target), which is why it missed
the `api.handle.me` loop.

## The governing principle: reset on change of APPROACH, not change of target

This is the fix for the tunnel-vision detector's blind spot. `detect_tunnel_vision`
resets its counter on any new well-defined target — so a stream of *different* search
queries (each a "new target") kept resetting it, and it never fired on a research
loop even at threshold 8. The queries changed; the approach didn't.

So for every detector below, **the counter resets ONLY when the model takes a
genuinely different approach — never on a mere new target.** A new query, a new URL,
a new file to read is not progress; only a *category* change is:

| detector | resets on |
|---|---|
| reads-without-writes | a **write** |
| same-prefix searches  | a **different search prefix** |
| failing fetches       | a **2xx result** |

All three are deterministic — counting ops, comparing a fixed-length query prefix,
reading an HTTP status. No fuzzy inference, so they clear the "no guessing" bar.

## 1. Read-without-write loop  `NEW` — the meta detector

*This is the one tunnel-vision does **not** already cover: it counts ANY read-type
op (of any target) with no write, so it catches varied research loops. The reset is
a write — the only real signal of progress.*


- **Signal:** N read-type operations (`web_search`, `web_fetch`, `read_file`,
  `list_dir`, grep) with similar parameters and **zero write operations**
  (`write_file` / `edit_file` / `apply_patch`) among them.
- **Nudge:** *"You seem to be looking for answers you aren't finding. Try searching
  something else or take a different approach."*
- **Reset:** on any write. The counter is "reads since the last write."

## 2. Same-prefix search loop  (nudge → then hard block)

- **Signal:** N `web_search` calls whose first `X` words are identical (e.g. every
  query beginning `api.handle.me swagger.yml`). `X` is tunable (~3–5 words).
- **Nudge:** *"Your search doesn't seem to be yielding the results you're looking
  for. Change your search string. Your next search that starts with '&lt;X&gt;' will
  be denied."*
- **Enforcement (a massage, not just a nudge):** if the model repeats the prefix
  after the warning, **block the call** — do not execute it; return the denial as the
  tool result. A hard stop, like the repeated-call override.
- **Reset:** on a search whose prefix differs from the warned one. The warned prefix
  stays "hot" until the model actually moves off it.

## 3. Failing-fetch loop

- **Signal:** N `web_fetch` calls with **no 2xx result** — all 4xx/5xx/errors, i.e.
  guessing at URLs that don't exist (repeated 404s on `raw.githubusercontent.com`).
- **Nudge:** *"You seem to be guessing at what to fetch. Try searching instead or
  take a different approach."*
- **Reset:** on a 2xx fetch. (Parts may already exist via the repetition guards; this
  is the targeted, HTTP-status-aware version.)

---

These sit alongside `detect_tunnel_vision` (see [shephard.md](shephard.md), *Nudges*):
tunnel-vision catches circling a fixed set of **edit** targets; these catch circling
in **read** space. The shared lesson is the reset rule — **progress is a change of
approach, not a change of target** — and the tunnel-vision detector should adopt it
too (its target-reset is why it missed the loop above).
