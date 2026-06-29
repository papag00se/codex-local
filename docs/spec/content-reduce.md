# Content Reduction (`content_reduce`)

[< Spec Index](index.md) | [Local Coder Massaging](local-coder-massaging.md)

> **Status: design proposal, not built.** A MIME-aware, lossless-first content
> reducer that bounds large tool outputs **at the source** so they never bloat
> the transcript — replacing the destructive blind truncation that trim does
> downstream.

## Why

A small local model has a tiny context window, so something must keep tool
outputs from filling it. Today that "something" is destructive:

- `trim::enforce_token_budget` finds the largest truncatable message and **halves
  it — keeps the head, throws away the tail** — leaving a `[… N chars truncated …]`
  marker. For a 74 KB `web_fetch` of an API spec, the model sees the front and
  loses the back, blindly, with no idea *what* was cut.
- `drop_oldest_until_fit` deletes whole messages to force-fit (and infamously
  dropped the user's request → Ornith's template raised `No user query found`).

Both are "fit it cheaply by deleting content." The principled alternative is to
**reduce content where it originates, in a type-aware and mostly-lossless way**,
so junk the model can't use never enters the transcript and trim never has to
guillotine. The dominant source of that junk is `web_fetch` (raw HTML/JSON pages),
so that's where this lands first.

## Principles

1. **Bound at the source.** Reduce in the tool that produces the output (e.g.
   `web_fetch`), before it's recorded into the conversation.
2. **Lossless-first, lossy-last.** Escalate: apply lossless transforms always;
   reach for lossy ones only if the result is *still* over a per-output cap.
3. **Structure is sacred.** Never break JSON/YAML — operate on the **parse tree**
   and re-serialize, never regex-on-raw-bytes.
4. **Never destroy meaning.** Never strip a negation; never touch a token that
   could be an identifier, number, symbol, URL, path, or code — even inside prose.

## The boundary

A standalone, pure utility:

```
content_reduce(content: &str, content_type: Option<&str>, cap_tokens: usize) -> String
```

No I/O, deterministic, unit-testable. Returns `content` unchanged when it's
already under `cap_tokens`. The caller (a tool handler) supplies the response's
`Content-Type` and a per-output token cap.

## MIME dispatch & escalation ladder

| Content-Type | Transform | Loss |
|---|---|---|
| `text/html`, `application/xhtml+xml` | Extract readable text (scripts/styles/markup stripped → lightweight markdown) | ~none (markup is noise) |
| `application/json`, `+json` | Parse → **minify** → (if still over cap) compress confirmed-prose string nodes → re-serialize | lossless, then guarded |
| `application/yaml`, `text/yaml`, `+yaml` | Parse → strip comments + minify → (if still over cap) compress confirmed-prose string nodes → re-serialize | lossless, then guarded |
| `text/*` (plain, markdown, csv…) | (if over cap) guarded function-word strip on the whole text | guarded |
| anything else / binary | Already handled upstream: `[non-text response: N bytes]` | n/a |

Lossless tiers (dehtml, minify, strip-comments) run **unconditionally** because
they don't lose information. The **guarded stripper** only runs when a tier left
the output still over `cap_tokens`.

## Components

### 1. HTML extractor

Strip `<script>`/`<style>`/`<template>` and the tag soup; extract the main
content as lightweight markdown so headings, lists, and links survive. This is
the classic "readability" problem — a battle-tested crate is fine (don't hand-roll
boilerplate detection). A 74 KB page is typically ~3–5 KB of real text, so this
is the **highest-value, lowest-risk** tier and is almost never destructive.

### 2. JSON / YAML tree reducer

1. Parse into a tree (serde_json / a YAML parser).
2. **Minify** — drop indentation/whitespace (lossless). YAML: also strip comments.
3. If still over cap: walk the tree; for each **string value**, if it classifies
   as prose (below), run the guarded stripper on its contents in place.
4. Re-serialize (minified) through a real encoder.

Because only the *content* of identified string nodes changes and the serializer
re-emits, **structure cannot break** — escaping, nesting, quoting stay valid by
construction. This is what makes "compress descriptions inside JSON" safe.

### 3. The guarded stripper (the one shared piece)

The same routine used for plain text and for prose nodes inside structured data —
the only difference per MIME type is *where* you point it. It works **token by
token** and removes a token **only if it is certainly throwaway**:

> **Strip** a token only if: all-lowercase, **purely alphabetic**, in the
> function-word set, and **not** a protected word.
>
> **Preserve verbatim** any token containing a digit, an uppercase letter, an
> underscore, internal punctuation, mixed case, `ALL_CAPS`, or `/ . : \` — i.e.
> anything that could be an identifier, number, symbol, URL, path, or code.

Safe by construction: the stripper can physically only delete members of a small
known-junk set, so an embedded identifier, status code, or negation is
untouchable even when wedged between two articles.

**Function-word set** (strip candidates): articles (`a/an/the`), common
prepositions (`of/to/in/on/at/for/with/from/by/as/into/onto/over/under`),
auxiliary verbs (`is/are/was/were/be/been/being/am/do/does/did/has/have/had`),
interjections, and *mild* fillers. Tuned for "most of the savings at a fraction of
the risk" — articles + prepositions + auxiliaries are the bulk.

**Protected words** (never strip, even though alphabetic & lowercase):
- **Negations / meaning-flippers:** `not no never none neither nor cannot` and any
  `…n't` (stripping one flips "do **not** delete prod" → "delete prod").
- **Logic words:** `or and if when unless else then` — they carry conditional/
  set logic that matters in API descriptions ("returns X **or** Y").

### 4. Prose classification (for structured string nodes)

A string value inside JSON/YAML is compressed only when **both** signals hold:

**Signal 1 — the key is a known-prose field.** Whitelist:
`description summary title comment doc documentation details note notes overview
abstract help message text body longDescription`.
Hard-exclude (never compress, even if prose-looking):
`pattern format enum example default const $ref ref url uri href path cmd command
code id name key type value version operationId`.

**Signal 2 — the content sniffs as natural language: *dominantly* letters +
spaces, several words, sentence-shaped.** Crucially, embedded identifiers/numbers
(`user_id`, `404`, `UUID`, `HANDLE_NOT_FOUND`) **do not disqualify** a value —
only *dominance* by non-prose characters does (a bare regex, a URL, a single
`snake_case` token, JSON-in-a-string all fail the sniff and are skipped).

Requiring **both** keeps false positives near zero: Signal 1 alone is unsafe (a
`description` could hold `"^[a-z]+$"` — caught by Signal 2's sniff), and Signal 2
alone is unsafe (a prose-looking `default` value might be real data — caught by
Signal 1's exclude list).

For **plain text** (`text/*`, HTML-extracted text) there are no field names, so
only Signal 2 gates whether the guarded stripper runs on the whole text.

> Example. Field `description` = *"Resolves an Ada Handle to its Cardano address;
> returns 404 when the handle is not found, with payment_address in the body."*
> → strip `an to its the with in` → keep `Ada Handle Cardano 404 payment_address`
> (caps/digit/snake_case), keep `not` (negation), keep `returns/when` →
> *"Resolves Ada Handle Cardano address; returns 404 when handle not found,
> payment_address body."* ~30 % lighter, every meaningful token intact.

## Wiring

- **`web_fetch` first.** After the fetch, before returning the body to the model,
  run `content_reduce(body, content_type, cap)`. That's where ~90 % of context
  junk originates.
- **Shared later.** Expose `content_reduce` so any tool with an oversized output
  (a giant `exec` dump, a large file read) can opt in with its own cap.

## Safety properties

- **Structure can't break** — JSON/YAML go through parse → edit-tree → serialize.
- **The stripper can't eat meaning** — it removes only from a fixed function-word
  set and preserves every digit/identifier/symbol/negation token.
- **Lossless before lossy** — dehtml/minify always; word-stripping only on
  genuine prose that's still over cap.
- **Residual risk** is bounded to "shortened a verbose human description," which is
  recoverable (the model can re-fetch a specific part) — a different risk class
  than corrupting data.

## What this lets us simplify in trim

With large outputs bounded at the source, `enforce_token_budget`'s blind
head-keep/tail-drop becomes a rare last resort (or is removed), and `drop_oldest`
goes away in favor of routing real overflow through compaction. See the
trim/compaction consolidation discussion — `content_reduce` is the "bound at
source" half; compaction is the "summarize the rest" half. Trim returns to being
purely mechanical and non-destructive.

## Empirical results (prototype, on the live api.handle.me)

Tested the JSON/YAML and HTML tiers against the actual payloads `web_fetch` saw:

| Input | Before | After | Saved | Notes |
|---|---|---|---|---|
| `swagger.yml` (raw OpenAPI spec) | 18.5k tok | 17.5k tok | **6%** | prose-strip on a ~94%-structure file; stripper correct (negations + `api-key` survived), just little prose to take |
| `/` homepage (HTML) | 14.6k tok | 9.1k tok | **38%** | stripping Swagger-UI chrome; on a prose-heavy docs *article* this is typically 80–90% |

**Lesson:** the HTML tier is the workhorse (markup is the bulk); the structured-prose
tier is safe-but-small (structure dominates and we correctly refuse to touch it).
**Crucially, neither compression tier makes a 74 KB spec "small"** — the model needs
most of that structure. For genuinely large structured data the lever is not
compression but **selection** (hand the model the relevant slice), which motivates
`find` below.

## HTTP status is part of the response

A reduced body is not the whole story: the **HTTP status** must reach the model, or
a 404/403/empty page reads as a blank success and the model keeps guessing similar
URLs. (Observed: a 9B fetching a dozen non-existent `raw.githubusercontent` doc
paths in a row, each returning `404: Not Found`, because the harness surfaced only
the tiny body under a `Fetched:` header that looked successful.) Two rules:

1. **Every result leads with `HTTP <code> <reason>`** — `HTTP 200 OK`, `HTTP 404 Not
   Found`. The status is never hidden, but the **body is never suppressed either**:
   a non-2xx body is still content and still paginated/`find`-able (e.g.
   `api.handle.me` serves real API documentation *on* its 404 page — throwing that
   away would be the bug). An empty body says so explicitly ("body was EMPTY …
   retrying returns the same").
2. **Stop-guessing is earned, not reflexive.** A single 404 is just reported — the
   model may have a good reason to try one URL. Only after **3 consecutive non-2xx
   fetches** (a process-wide streak, reset by any 2xx) does the result append a
   nudge: *"N web_fetch calls in a row have failed — if you're guessing URLs, STOP;
   find the correct one via local_web_search."* This distinguishes a one-off bad URL
   from genuine thrashing, instead of scolding on the first miss.

The point is the same as `find`'s no-match guidance: **the model must never mistake
"nothing useful came back" for "the page had nothing in it"** — while still getting
whatever the server actually returned.

## Pagination & find-within

`content_reduce` shrinks; it doesn't *select*. When the reduced body is still over
the cap, two affordances let the model pull the rest **on demand instead of losing
the tail** — both backed by a small **session-scoped cache** keyed by URL (fetch +
reduce once, serve slices), same pattern as the system-summary cache.

### `cursor=` — linear pagination (the backstop)

A truncated result returns page 1 plus a **loud, literal continuation**:

```
[page 1/9 — 8000 of 71000 tokens shown]
<content…>
⚠️ More remains. To read the next page, call:
web_fetch(url="…/swagger.yml", cursor="p2_a91f")
```

**The cursor is a character offset, presented as a token the model copies
verbatim.** The model never computes it — the harness emits `cursor="c8000"`, the
model copies it. Two deliberate choices:
- **Offset, not page-number-plus-hash.** A hash makes the cursor brittle: any
  change to the cached body fails the match and resets to page 1. A raw offset
  *degrades gracefully* — if the body shifted, the offset still lands somewhere
  sensible (a small gap/overlap, snapped to a line) and returns content rather
  than erroring. Worst case is a minor misalignment, not a dead end.
- **Opaque-but-copyable.** It looks like `c8000`, so the model can't be tempted to
  "do the math" (which a 9B fumbles); it just copies the literal token from the
  result. The result shows the char range (`chars 8000–16000 of 71000`) so it
  knows where it is and that more remains, and puts the exact call in the text.

### `find="…"` — targeted retrieval (the headline for a 9B)

`web_fetch(url, find="handles/{handle}")` returns a **small, complete, in-context
chunk** — MIME-aware, not a blind character window:

- **JSON / YAML →** the matched node's **subtree** plus its **ancestor spine**
  (root-to-match path, siblings collapsed, so the model knows *where* it is), and
  **one hop of `$ref` resolution** (inline the schemas the subtree points to, with
  a depth/cycle guard) so the chunk is self-contained. Example — `find="handles"`:
  ```yaml
  paths:
    /handles/{handle}:          # ancestor spine (siblings elided)
      get:
        summary: Resolve a handle to its holder + address
        parameters: [ {name: handle, in: path, required: true} ]
        responses:
          '200': { schema: { type: object, properties: {...} } }  # $ref resolved one hop
  ```
  A ±N-char window would be *wrong* here — it splits structure, returns invalid
  YAML, and misses the referenced schema.
- **Text / Markdown →** snap to a coherent unit: the **Markdown section under the
  nearest heading**, or the matching **paragraph / ±N lines** — falling back to a
  ±256-char window only for unstructured blobs, expanded to whitespace so no token
  is severed.
- **HTML →** extracted to text/markdown first (the HTML tier), then text behavior.

**Matching & ranking** (this is where a 9B lives or dies):
- Match case-insensitively against **keys *and* values** (structured) or the text
  (plain). Substring is the most 9B-friendly contract.
- **Rank** — "handle" matches everywhere in a handles API: exact key > key
  substring > value substring, and prefer **structural** nodes (path / operation /
  schema) over deep leaf strings. Return the **top K** (≈3) subtrees, capped by a
  total size budget.
- **No match →** never return empty; return the **available top-level keys /
  section headings** ("did you mean `paths`, `components/schemas`, `info`?") so it
  can refine.
- **Too many →** return the K best + "N more; narrow your `find`".

### 9B comprehension (the honest part)

- **`find` >> linear paging for a 9B.** `find` is a *single intent* ("get the part
  I need"); paging is a *stateful loop* ("remember you're mid-document and keep
  going"). The loop is where a 9B fails — silently: it reads page 1, answers, and
  never asks for page 2. That **silent incompleteness** is worse than an error
  because it looks like success. So make `find` the headline, `cursor` the backstop.
- The existing loop guards catch the *other* failure (re-fetching page 1 forever);
  they do **not** catch silent incompleteness — only a good `find` affordance does.

## Open questions (decide before building)

1. **HTML extractor dependency** — which readability/`html2text` crate, vs. a
   minimal `<script>/<style>`-strip + tag-strip for v1.
2. **YAML dependency** — add a YAML parser (e.g. `serde_yaml`) for the tree path,
   or treat YAML as plain text for v1 and only do JSON structurally.
3. **`cap_tokens` value** — per-output cap (e.g. a few k tokens) and whether it's
   configurable per tool.
4. **Scope of the shared reducer** — `web_fetch` only at first, or also
   `local_web_search` results and large `exec` outputs from the start.
5. **Markdown fidelity** — how much HTML structure (tables, code blocks) to
   preserve vs. flatten to text.
6. **Cache model** — lifetime/eviction of the fetched-body cache (per-session LRU,
   size cap) and whether `cursor`/`find` re-fetch on a miss or fail closed.
7. ~~**`cursor` opacity** — encode position + a content hash?~~ **Decided:** plain
   char-offset token (`c<offset>`), **no hash** — a changed body degrades to a
   small gap/overlap instead of a hard miss. The model copies it verbatim.
8. **`find` query contract** — substring vs. path-expression (`paths./handles`)
   vs. both; how `top K` and the size budget are tuned; whether `find` and `cursor`
   compose (`find` within a page).
9. **`$ref` resolution depth** — one hop (proposed) vs. configurable; how to render
   a cycle or an unresolvable external `$ref`.
10. **Phasing** — ship `content_reduce` (compression) first, then `find`
    (selection), then `cursor` (pagination)? `find` is the highest-value/9B-safest,
    so it may deserve to come second, right after the HTML tier.
