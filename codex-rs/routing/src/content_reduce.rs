//! MIME-aware, lossless-first reduction of large tool outputs.
//!
//! **Research-vehicle implementation.** The authoritative, language-agnostic
//! design lives in `docs/spec/content-reduce.md` — that spec is what migrates to
//! the Python service. This Rust version exists to validate the logic against
//! real payloads and to bound `web_fetch` output in the current harness. Keep the
//! logic here a faithful, crate-light mirror of the spec (hand-rolled HTML, the
//! universal JSON primitive via serde_json, YAML treated as text for now — the
//! Python port does YAML structurally with pyyaml).
//!
//! Escalation per the spec: lossless transforms first (HTML→text, JSON minify);
//! the lossy guarded stripper only runs when a tier left the output over the cap.

use serde_json::Value as JsonValue;

/// chars/4 token estimate (same crude estimate the trimmer uses).
fn est_tokens(s: &str) -> usize {
    s.chars().count() / 4
}

/// Reduce `content` to fit roughly `cap_tokens`, dispatching on `content_type`.
/// Returns the input unchanged when it already fits.
pub fn content_reduce(content: &str, content_type: Option<&str>, cap_tokens: usize) -> String {
    if est_tokens(content) <= cap_tokens {
        return content.to_string();
    }
    let ct = content_type.unwrap_or("").to_ascii_lowercase();
    if ct.contains("html") || ct.contains("xml") {
        let text = html_to_text(content);
        if est_tokens(&text) > cap_tokens {
            strip_prose_text(&text)
        } else {
            text
        }
    } else if ct.contains("json") {
        reduce_json(content, cap_tokens).unwrap_or_else(|| content.to_string())
    } else {
        // text/*, yaml, unknown → guarded prose strip (only ran because over cap)
        strip_prose_text(content)
    }
}

/// Lossless-only reduction (HTML→text, JSON minify) — no prose stripping, no
/// size gate. Used by the `web_fetch` nav path: we keep the *full* content and
/// let the model page/`find` through it, which is non-destructive, rather than
/// lossily stripping to cram it into one response.
pub fn reduce_lossless(content: &str, content_type: Option<&str>) -> String {
    let ct = content_type.unwrap_or("").to_ascii_lowercase();
    if ct.contains("html") || ct.contains("xml") {
        html_to_text(content)
    } else if ct.contains("json") {
        serde_json::from_str::<JsonValue>(content)
            .ok()
            .and_then(|v| serde_json::to_string(&v).ok())
            .unwrap_or_else(|| content.to_string())
    } else {
        content.to_string()
    }
}

// ---------------------------------------------------------------------------
// JSON tier: parse → minify (lossless) → strip prose nodes (lossy) → re-serialize
// ---------------------------------------------------------------------------

fn reduce_json(content: &str, cap_tokens: usize) -> Option<String> {
    let mut v: JsonValue = serde_json::from_str(content).ok()?;
    let minified = serde_json::to_string(&v).ok()?;
    if est_tokens(&minified) <= cap_tokens {
        return Some(minified); // lossless was enough
    }
    strip_prose_nodes(&mut v, None);
    serde_json::to_string(&v).ok()
}

/// Walk the tree; compress only string values that are (Signal 1) under a
/// prose-named key AND (Signal 2) sniff as natural language. Editing the tree +
/// re-serializing means structure cannot break.
fn strip_prose_nodes(v: &mut JsonValue, key: Option<&str>) {
    match v {
        JsonValue::Object(map) => {
            for (k, val) in map.iter_mut() {
                strip_prose_nodes(val, Some(k));
            }
        }
        JsonValue::Array(arr) => {
            for val in arr.iter_mut() {
                strip_prose_nodes(val, key);
            }
        }
        JsonValue::String(s) => {
            if let Some(k) = key
                && is_prose_field(k)
                && looks_like_prose(s)
            {
                *s = strip_prose_text(s);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// HTML tier: strip script/style/template/comments + tags, decode entities, ws
// ---------------------------------------------------------------------------

fn html_to_text(html: &str) -> String {
    let chars: Vec<char> = html.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(n / 2);
    let mut i = 0;
    while i < n {
        if chars[i] == '<' {
            if starts_with_ci(&chars, i, "<!--") {
                i = find_ci(&chars, i + 4, "-->").map(|p| p + 3).unwrap_or(n);
                continue;
            }
            if let Some(tag) = match_block_tag(&chars, i) {
                // skip the whole <tag …> … </tag> block
                let close = format!("</{tag}");
                let after_open = find_ci(&chars, i, ">").map(|p| p + 1).unwrap_or(n);
                i = find_ci(&chars, after_open, &close)
                    .and_then(|p| find_ci(&chars, p, ">").map(|q| q + 1))
                    .unwrap_or(n);
                out.push(' ');
                continue;
            }
            // generic tag → drop, emit a space so words don't fuse
            i = find_ci(&chars, i, ">").map(|p| p + 1).unwrap_or(n);
            out.push(' ');
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    collapse_ws(&decode_entities(&out))
}

/// `<script>`/`<style>`/`<template>`/`<noscript>` opener at `i`? → its tag name.
fn match_block_tag(chars: &[char], i: usize) -> Option<&'static str> {
    for tag in ["script", "style", "template", "noscript"] {
        let open = format!("<{tag}");
        if starts_with_ci(chars, i, &open) {
            // next char must be whitespace or '>' (so we don't match <styles…>)
            let after = chars.get(i + open.chars().count());
            if matches!(after, Some(c) if c.is_whitespace() || *c == '>') {
                return Some(tag);
            }
        }
    }
    None
}

fn starts_with_ci(chars: &[char], at: usize, needle: &str) -> bool {
    let nd: Vec<char> = needle.chars().collect();
    if at + nd.len() > chars.len() {
        return false;
    }
    nd.iter()
        .enumerate()
        .all(|(k, c)| chars[at + k].eq_ignore_ascii_case(c))
}

fn find_ci(chars: &[char], from: usize, needle: &str) -> Option<usize> {
    let nd: Vec<char> = needle.chars().collect();
    if nd.is_empty() || from >= chars.len() {
        return None;
    }
    (from..=chars.len().saturating_sub(nd.len())).find(|&p| {
        nd.iter()
            .enumerate()
            .all(|(k, c)| chars[p + k].eq_ignore_ascii_case(c))
    })
}

fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut blanks = 0u8; // 0=none, 1=space, 2=newline-run
    for c in s.chars() {
        if c == '\n' || c == '\r' {
            blanks = blanks.max(2);
        } else if c.is_whitespace() {
            blanks = blanks.max(1);
        } else {
            match blanks {
                2 => out.push('\n'),
                1 => out.push(' '),
                _ => {}
            }
            blanks = 0;
            out.push(c);
        }
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// The one guarded stripper (shared by plain text and prose JSON/YAML nodes)
// ---------------------------------------------------------------------------

/// Strip only certain-junk function words; preserve any token that could carry
/// meaning (digit/uppercase/underscore/symbol) and every negation/logic word.
fn strip_prose_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for piece in split_keep_ws(text) {
        if piece.chars().all(char::is_whitespace) {
            out.push_str(piece);
            continue;
        }
        let (lead, core, trail) = peel(piece);
        if !core.is_empty() && is_strippable(core) {
            out.push_str(lead);
            out.push_str(trail); // drop the word, keep attached punctuation
        } else {
            out.push_str(piece);
        }
    }
    collapse_inline_spaces(&out)
}

/// A token is strippable only if all-lowercase-alphabetic, in the function-word
/// set, and not protected. Anything with a digit, uppercase, `_`, or punctuation
/// fails `is_alphabetic`/`is_lowercase` and is kept verbatim.
fn is_strippable(tok: &str) -> bool {
    if !tok.chars().all(|c| c.is_alphabetic() && c.is_lowercase()) {
        return false;
    }
    !is_protected(tok) && is_function_word(tok)
}

fn split_keep_ws(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_ws = None;
    for (idx, c) in s.char_indices() {
        let ws = c.is_whitespace();
        match in_ws {
            Some(prev) if prev != ws => {
                out.push(&s[start..idx]);
                start = idx;
                in_ws = Some(ws);
            }
            None => in_ws = Some(ws),
            _ => {}
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Split a token into (leading punctuation, core, trailing punctuation).
fn peel(tok: &str) -> (&str, &str, &str) {
    let lead_end = tok
        .char_indices()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, _)| i)
        .unwrap_or(tok.len());
    let trail_start = tok
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(lead_end);
    (
        &tok[..lead_end],
        &tok[lead_end..trail_start],
        &tok[trail_start..],
    )
}

fn collapse_inline_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            pending_space = true;
        } else {
            if pending_space && !out.is_empty() && !out.ends_with('\n') {
                // drop a space that sits right before punctuation a stripped word left
                if !matches!(c, ',' | '.' | ';' | ':' | ')' | ']') {
                    out.push(' ');
                }
            }
            pending_space = false;
            out.push(c);
        }
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// Word lists (mirror docs/spec/content-reduce.md)
// ---------------------------------------------------------------------------

fn is_function_word(w: &str) -> bool {
    matches!(
        w,
        // articles
        "a" | "an" | "the"
        // possessive determiners
        | "its" | "his" | "her" | "their" | "our" | "your" | "my"
        // prepositions
        | "of" | "to" | "in" | "on" | "at" | "for" | "with" | "from" | "by" | "as"
        | "into" | "onto" | "over" | "under" | "via"
        // auxiliaries
        | "is" | "are" | "was" | "were" | "be" | "been" | "being" | "am"
        | "do" | "does" | "did" | "has" | "have" | "had"
        // interjections / fillers
        | "oh" | "ah" | "um" | "uh" | "well"
    )
}

/// Never stripped even though alphabetic+lowercase: negations and logic words.
fn is_protected(w: &str) -> bool {
    matches!(
        w,
        "not"
            | "no"
            | "never"
            | "none"
            | "neither"
            | "nor"
            | "cannot"
            | "or"
            | "and"
            | "if"
            | "when"
            | "unless"
            | "else"
            | "then"
    )
}

fn is_prose_field(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    const PROSE: &[&str] = &[
        "description",
        "summary",
        "title",
        "comment",
        "doc",
        "documentation",
        "details",
        "note",
        "notes",
        "overview",
        "abstract",
        "help",
        "message",
        "text",
        "body",
        "longdescription",
    ];
    const EXCLUDE: &[&str] = &[
        "pattern",
        "format",
        "enum",
        "example",
        "default",
        "const",
        "$ref",
        "ref",
        "url",
        "uri",
        "href",
        "path",
        "cmd",
        "command",
        "code",
        "id",
        "name",
        "key",
        "type",
        "value",
        "version",
        "operationid",
    ];
    PROSE.contains(&k.as_str()) && !EXCLUDE.contains(&k.as_str())
}

/// Signal 2: dominantly letters+spaces, several words, sentence-shaped.
/// Embedded identifiers/numbers don't disqualify — only dominance by non-prose.
fn looks_like_prose(s: &str) -> bool {
    if s.split_whitespace().count() < 4 {
        return false;
    }
    let total = s.chars().count().max(1);
    let proseish = s
        .chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .count();
    proseish * 100 / total >= 75
}

// ---------------------------------------------------------------------------
// Pagination (cursor backstop): char-offset slices, line-boundary snapped
// ---------------------------------------------------------------------------

/// Slice `content` from char `offset` for ~`cap_tokens`, snapped back to a line
/// boundary so a line is never split. Returns `(slice, next_offset, total_chars)`;
/// `next_offset == total_chars` means there's no more. Offset-based (not page-
/// numbered) so a slightly-changed body still yields a sensible slice rather than
/// a hard miss. char-based so multibyte text is safe.
pub fn page_from(content: &str, offset: usize, cap_tokens: usize) -> (String, usize, usize) {
    let chars: Vec<char> = content.chars().collect();
    let total = chars.len();
    let start = offset.min(total);
    let page_chars = cap_tokens.saturating_mul(4).max(1);
    let mut end = (start + page_chars).min(total);
    if end < total
        && let Some(off) = chars[start..end].iter().rposition(|&c| c == '\n')
        && off > 0
    {
        end = start + off + 1;
    }
    if end <= start {
        end = (start + page_chars).min(total);
    }
    (chars[start..end].iter().collect(), end, total)
}

// ---------------------------------------------------------------------------
// find (selection): MIME-aware targeted retrieval
// ---------------------------------------------------------------------------

const FIND_TOP_K: usize = 3;

/// Return the slice(s) of `content` relevant to `query`, MIME-aware: a JSON
/// subtree + ancestor spine + one-hop `$ref` resolution, or a text section.
pub fn find_in(
    content: &str,
    content_type: Option<&str>,
    query: &str,
    cap_tokens: usize,
) -> String {
    // Models routinely wrap the find term in quotes (`"Handle"`) as natural
    // emphasis, but matching is a literal substring — so the quote characters make
    // it miss every time on content that has no literal quotes (parsed JSON keys,
    // reduced text). Strip surrounding quotes/whitespace so the bare keyword is
    // what's matched. (Observed: a model looped a dozen fetches re-quoting terms
    // against a doc it had already fetched, each returning "no match".)
    let query = query
        .trim()
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .trim();
    let ct = content_type.unwrap_or("").to_ascii_lowercase();
    if ct.contains("json")
        && let Ok(root) = serde_json::from_str::<JsonValue>(content)
    {
        return find_json(&root, query, cap_tokens);
    }
    find_text(content, query, cap_tokens)
}

#[derive(Clone)]
enum Seg {
    Key(String),
    Idx(usize),
}

struct FoundNode {
    path: Vec<Seg>,
    key_match: bool,
    container: bool,
}

fn find_json(root: &JsonValue, query: &str, cap_tokens: usize) -> String {
    let q = query.to_ascii_lowercase();
    let mut matches = Vec::new();
    let mut path = Vec::new();
    collect_json_matches(root, &q, &mut path, &mut matches);

    if matches.is_empty() {
        let keys = top_level_keys(root);
        return format!(
            "find \"{query}\": no match. Available top-level keys: {}.",
            keys.join(", ")
        );
    }
    // rank: key match first, container over leaf, shallower path first.
    matches.sort_by(|a, b| {
        b.key_match
            .cmp(&a.key_match)
            .then(b.container.cmp(&a.container))
            .then(a.path.len().cmp(&b.path.len()))
    });
    let total = matches.len();
    let mut out = String::new();
    let mut shown = 0;
    let mut used = 0;
    for m in &matches {
        if shown >= FIND_TOP_K {
            break;
        }
        let rendered = render_json_match(root, m);
        let t = est_tokens(&rendered);
        if shown > 0 && used + t > cap_tokens {
            break;
        }
        if shown > 0 {
            out.push_str("\n\n");
        }
        out.push_str(&rendered);
        used += t;
        shown += 1;
    }
    if total > shown {
        out.push_str(&format!(
            "\n\n[{} more match(es); narrow your find]",
            total - shown
        ));
    }
    out
}

fn collect_json_matches(node: &JsonValue, q: &str, path: &mut Vec<Seg>, out: &mut Vec<FoundNode>) {
    match node {
        JsonValue::Object(map) => {
            for (k, v) in map {
                let container = v.is_object() || v.is_array();
                if k.to_ascii_lowercase().contains(q) {
                    let mut p = path.clone();
                    p.push(Seg::Key(k.clone()));
                    out.push(FoundNode {
                        path: p,
                        key_match: true,
                        container,
                    });
                } else if let JsonValue::String(s) = v
                    && s.to_ascii_lowercase().contains(q)
                {
                    let mut p = path.clone();
                    p.push(Seg::Key(k.clone()));
                    out.push(FoundNode {
                        path: p,
                        key_match: false,
                        container: false,
                    });
                }
                path.push(Seg::Key(k.clone()));
                collect_json_matches(v, q, path, out);
                path.pop();
            }
        }
        JsonValue::Array(arr) => {
            for (i, v) in arr.iter().enumerate() {
                path.push(Seg::Idx(i));
                collect_json_matches(v, q, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}

fn render_json_match(root: &JsonValue, m: &FoundNode) -> String {
    let header = m
        .path
        .iter()
        .map(|s| match s {
            Seg::Key(k) => k.clone(),
            Seg::Idx(i) => format!("[{i}]"),
        })
        .collect::<Vec<_>>()
        .join(" > ");
    let mut sub = get_at(root, &m.path).cloned().unwrap_or(JsonValue::Null);
    resolve_refs(&mut sub, root, 1);
    format!(
        "# {header}\n{}",
        serde_json::to_string_pretty(&sub).unwrap_or_default()
    )
}

fn get_at<'a>(root: &'a JsonValue, path: &[Seg]) -> Option<&'a JsonValue> {
    let mut cur = root;
    for seg in path {
        cur = match seg {
            Seg::Key(k) => cur.get(k)?,
            Seg::Idx(i) => cur.get(i)?,
        };
    }
    Some(cur)
}

/// Inline `$ref` targets within `node` up to `depth` hops (cycle-safe by depth).
fn resolve_refs(node: &mut JsonValue, root: &JsonValue, depth: u8) {
    if depth == 0 {
        return;
    }
    match node {
        JsonValue::Object(map) => {
            if let Some(JsonValue::String(r)) = map.get("$ref")
                && let Some(target) = resolve_ref_path(root, r)
            {
                *node = target.clone();
                resolve_refs(node, root, depth - 1);
                return;
            }
            for v in map.values_mut() {
                resolve_refs(v, root, depth);
            }
        }
        JsonValue::Array(arr) => {
            for v in arr.iter_mut() {
                resolve_refs(v, root, depth);
            }
        }
        _ => {}
    }
}

fn resolve_ref_path<'a>(root: &'a JsonValue, r: &str) -> Option<&'a JsonValue> {
    let r = r.strip_prefix("#/")?;
    let mut cur = root;
    for part in r.split('/') {
        let part = part.replace("~1", "/").replace("~0", "~"); // JSON-pointer unescape
        cur = cur.get(&part)?;
    }
    Some(cur)
}

fn top_level_keys(root: &JsonValue) -> Vec<String> {
    match root {
        JsonValue::Object(map) => map.keys().take(20).cloned().collect(),
        JsonValue::Array(_) => vec!["[array]".to_string()],
        _ => vec![],
    }
}

fn find_text(content: &str, query: &str, cap_tokens: usize) -> String {
    let q = query.to_ascii_lowercase();
    let lc = content.to_ascii_lowercase(); // ASCII-only fold → byte offsets align
    let per = (cap_tokens.saturating_mul(4) / FIND_TOP_K.max(1)).max(256);
    let mut results: Vec<String> = Vec::new();
    let mut from = 0;
    while let Some(rel) = lc[from..].find(&q) {
        let at = from + rel;
        let slice = extract_around(content, at, per);
        if !results
            .iter()
            .any(|r| r.contains(&slice) || slice.contains(r.as_str()))
        {
            results.push(slice);
        }
        from = at + q.len();
        if results.len() >= FIND_TOP_K {
            break;
        }
    }
    if results.is_empty() {
        let heads: Vec<String> = content
            .lines()
            .filter(|l| l.trim_start().starts_with('#'))
            .take(15)
            .map(|s| s.trim().to_string())
            .collect();
        if heads.is_empty() {
            return format!("find \"{query}\": no match.");
        }
        return format!(
            "find \"{query}\": no match. Sections:\n{}",
            heads.join("\n")
        );
    }
    results.join("\n\n---\n\n")
}

/// The paragraph (blank-line bounded) around byte offset `at`, capped to `~budget`
/// bytes and snapped to char boundaries.
fn extract_around(content: &str, at: usize, budget: usize) -> String {
    let mut lo = content[..at].rfind("\n\n").map(|p| p + 2).unwrap_or(0);
    let mut hi = content[at..]
        .find("\n\n")
        .map(|p| at + p)
        .unwrap_or(content.len());
    if hi.saturating_sub(lo) > budget {
        lo = at.saturating_sub(budget / 2).max(lo);
        hi = (at + budget / 2).min(hi);
        while lo < content.len() && !content.is_char_boundary(lo) {
            lo += 1;
        }
        while hi > 0 && !content.is_char_boundary(hi) {
            hi -= 1;
        }
    }
    content[lo..hi.max(lo)].trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_function_words_keeps_meaning() {
        let s = "Resolves an Ada Handle to its Cardano address; returns 404 when the handle is not found, with payment_address in the body.";
        let out = strip_prose_text(s);
        // negation, identifiers, digits, logic words survive
        for keep in [
            "not",
            "Ada",
            "Handle",
            "Cardano",
            "404",
            "payment_address",
            "when",
        ] {
            assert!(out.contains(keep), "must keep `{keep}`: {out}");
        }
        // function words gone
        for gone in [" an ", " to ", " its ", " with "] {
            assert!(!out.contains(gone), "must drop `{gone}`: {out}");
        }
        assert!(out.len() < s.len());
    }

    #[test]
    fn json_structure_never_breaks_and_only_prose_fields_change() {
        let src = r#"{"description":"This is a long human readable explanation of the thing that does work","pattern":"^[a-z]+ or [0-9]+$","example":"do not change this value at all please"}"#;
        let out = reduce_json(src, 1).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).expect("still valid JSON");
        // pattern (excluded field) and example (excluded field) are untouched
        assert_eq!(v["pattern"], "^[a-z]+ or [0-9]+$");
        assert_eq!(v["example"], "do not change this value at all please");
        // description (prose field) got shorter but kept its content words
        let d = v["description"].as_str().unwrap();
        assert!(d.contains("human") && d.contains("readable") && d.contains("explanation"));
        assert!(
            d.len() < "This is a long human readable explanation of the thing that does work".len()
        );
    }

    #[test]
    fn html_strips_script_style_and_tags() {
        let html = "<html><head><style>.x{color:red}</style><script>alert(1)</script></head><body><h1>Title</h1><p>Hello <b>world</b> &amp; friends</p></body></html>";
        let out = html_to_text(html);
        assert!(out.contains("Title"));
        assert!(out.contains("Hello") && out.contains("world") && out.contains("& friends"));
        assert!(!out.contains("alert") && !out.contains("color:red") && !out.contains('<'));
    }

    #[test]
    fn under_cap_is_unchanged() {
        let s = "small output";
        assert_eq!(content_reduce(s, Some("text/plain"), 1000), s);
    }
    #[test]
    fn find_json_returns_subtree_spine_and_resolves_ref() {
        let spec = r##"{"paths":{"/handles/{handle}":{"get":{"summary":"Resolve a handle","responses":{"200":{"schema":{"$ref":"#/defs/Handle"}}}}}},"defs":{"Handle":{"type":"object","properties":{"name":{"type":"string"}}}}}"##;
        let out = super::find_in(spec, Some("application/json"), "handles", 4000);
        assert!(
            out.contains("paths > /handles/{handle}"),
            "spine header: {out}"
        );
        assert!(out.contains("Resolve a handle"), "subtree content: {out}");
        // $ref was inlined one hop -> the Handle schema is present, not a bare $ref
        assert!(
            out.contains("properties") && out.contains("\"name\""),
            "ref resolved: {out}"
        );
        assert!(!out.contains("$ref"), "no dangling $ref: {out}");
    }

    #[test]
    fn find_no_match_lists_top_keys() {
        let spec = r#"{"paths":{},"components":{},"info":{}}"#;
        let out = super::find_in(spec, Some("application/json"), "zzzznope", 4000);
        assert!(out.contains("no match"));
        assert!(out.contains("paths") && out.contains("components") && out.contains("info"));
    }

    #[test]
    fn find_strips_surrounding_quotes_so_a_quoted_term_matches() {
        let spec = r#"{"components":{"schemas":{"Holder":{"type":"object"}}}}"#;
        // The bare term matches...
        let bare = super::find_in(spec, Some("application/json"), "Holder", 4000);
        assert!(!bare.contains("no match"), "bare should match: {bare}");
        // ...and the quoted term (the model's emphasis) must match identically,
        // not be defeated by the literal quote characters.
        for q in [r#""Holder""#, r#"'Holder'"#, r#"  "Holder"  "#] {
            let out = super::find_in(spec, Some("application/json"), q, 4000);
            assert!(
                !out.contains("no match"),
                "quoted {q:?} should match: {out}"
            );
        }
    }

    #[test]
    fn page_from_offset_walks_and_reports_remaining() {
        let body = (0..50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (p1, next, total) = super::page_from(&body, 0, 20); // ~80 chars/page
        assert!(p1.starts_with("line 0"));
        assert!(
            next > 0 && next < total,
            "more remains: next={next} total={total}"
        );
        let (p2, _, _) = super::page_from(&body, next, 20);
        assert_ne!(p1, p2);
        assert!(p2.starts_with("line "));
        // offset past the end → empty slice, next==total (graceful, no panic)
        let (end, n2, t2) = super::page_from(&body, total + 999, 20);
        assert!(end.is_empty() && n2 == t2 && t2 == total);
    }
}
