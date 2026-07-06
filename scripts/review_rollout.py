#!/usr/bin/env python3
"""review_rollout.py — render a Codex rollout (.jsonl) as a human-scannable
timeline that makes LOOPS visible to the naked eye.

Each event becomes one condensed line. The target column is aligned so repeated
targets stack into a vertical streak; a climbing `xN` counter per (tool,target)
shows a loop building; and a loop tally up top surfaces the worst offenders at a
glance. `write_file`->shell base64 blobs are decoded back to `write_file <path>`.

Usage:
  review_rollout.py [rollout.jsonl]     # a specific rollout
  review_rollout.py                     # auto-pick the newest under ~/.codex/sessions
  review_rollout.py --no-color ...      # plain ASCII (for piping to a file)
Stdlib only.
"""
import sys, os, re, json, glob, base64
from collections import Counter, defaultdict
from datetime import datetime

# ---- tiny helpers -----------------------------------------------------------

def find_latest():
    files = glob.glob(os.path.expanduser("~/.codex/sessions/**/rollout-*.jsonl"), recursive=True)
    return max(files, key=os.path.getmtime) if files else None

def short(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"

def ts_seconds(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None

def decode_shephard_path(cmd):
    m = re.search(r"# shephard-write:(\S+)", cmd or "")
    if not m:
        return None
    try:
        return base64.b64decode(m.group(1)).decode("utf-8", "replace")
    except Exception:
        return None

def firstline(s):
    for ln in str(s).splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""

def exec_summary(out):
    """Pull the meaningful result line from a command's output — the pytest
    summary, an error, or the write confirmation — not a streaming placeholder."""
    lines = [l.strip() for l in str(out).splitlines() if l.strip()]
    for l in reversed(lines):
        if re.search(r"\d+ (passed|failed)|[Ee]rror|Traceback|No such|not found"
                     r"|wrote \d+ bytes|externally-managed|assert|FAILED|PASSED", l):
            return l
    return lines[-1] if lines else ""

# ---- classify a tool call into (tool, target, detail) -----------------------

def classify(name, args_raw):
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
    except Exception:
        args = {}
    if name in ("shell", "local_shell"):
        cmd = args.get("command")
        cmdstr = cmd[-1] if isinstance(cmd, list) and cmd else str(cmd)
        p = decode_shephard_path(cmdstr)
        if p:  # a write_file we lowered to base64 shell
            return ("write_file", os.path.basename(p), p)
        return ("shell", short(cmdstr, 46), cmdstr)
    if name in ("local_web_search", "web_search"):
        return ("web_search", '"' + short(args.get("query", args.get("q", "")), 40) + '"', "")
    if name == "web_fetch":
        url = args.get("url", "")
        return ("web_fetch", re.sub(r"^https?://", "", url).split("/")[0] or short(url, 40), url)
    if name in ("exec_command",):
        cmd = args.get("cmd") or args.get("command")
        cmdstr = cmd[-1] if isinstance(cmd, list) and cmd else str(cmd)
        return ("exec", short(cmdstr, 46), cmdstr)
    if name in ("write_file", "create_file", "edit_file", "read_file", "str_replace"):
        return (name, os.path.basename(args.get("path", args.get("file_path", "?"))), "")
    if name == "apply_patch":
        return ("apply_patch", short(args.get("input", "")[:40], 40), "")
    if name == "update_plan":
        return ("update_plan", "", "")
    return (name, short(json.dumps(args), 40), "")

# ---- color ------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and "--no-color" not in sys.argv
# Show reasoning + plan blocks. The harness persists the local model's thinking and
# the reasoned-guidance plan to the rollout as `reasoning` items (payload.type ==
# "reasoning", summary[].text = label, content[].text = body). Off by default so
# the tail stays terse; --reasoning turns them on.
SHOW_REASONING = "--reasoning" in sys.argv
def c(code, s):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s
TOOL_COLORS = {
    "write_file": "36", "edit_file": "36", "shell": "90", "exec": "32",
    "web_search": "33", "web_fetch": "34", "read_file": "35", "apply_patch": "31",
}

# ---- live follow ------------------------------------------------------------

def emit_live(o):
    """One parsed line per meaningful event, for a live tail. Skips noise
    (token counts, paired outputs) and file-write shell-ends (shown as the call)."""
    p = o.get("payload") or {}
    t, pt = o.get("type"), p.get("type")
    rel = ts_seconds(o.get("timestamp"))
    clock = c("90", datetime.fromtimestamp(rel).strftime("%H:%M:%S") if rel else "--:--:--")
    if pt == "function_call":
        tool, target, detail = classify(p.get("name", "?"), p.get("arguments"))
        # web_fetch: show the FULL url (classify's detail) so you can see exactly
        # which file/path is being fetched, not just the host.
        shown = detail if (tool == "web_fetch" and detail) else short(target, 60)
        print(f"{clock}  {c(TOOL_COLORS.get(tool,'0'), ('▶ '+tool).ljust(14))} {shown}", flush=True)
    elif pt == "exec_command_end":
        cmd = p.get("command"); s = cmd[-1] if isinstance(cmd, list) and cmd else str(cmd)
        if "base64 -d" in s or "shephard-write" in s:
            return  # a file write — already surfaced as its write_file call
        xc = p.get("exit_code")
        summ = exec_summary(p.get("aggregated_output") or p.get("stdout") or "")
        col = "32" if xc in (0, None) else "31"
        print(f"{clock}  {c(col, ('◀ exit '+str(xc)).ljust(14))} {c('90', short(summ,76))}", flush=True)
    elif pt == "agent_message" or (t == "response_item" and pt == "message" and p.get("role") == "assistant"):
        msg = p.get("message") or "".join(x.get("text","") for x in (p.get("content") or []) if isinstance(x, dict))
        if msg.strip():
            print(f"{clock}  {c('37','≡ say'.ljust(14))} {short(msg,76)}", flush=True)
    elif pt == "user_message" or (t == "response_item" and pt == "message" and p.get("role") == "user"):
        msg = p.get("message") or "".join(x.get("text","") for x in (p.get("content") or []) if isinstance(x, dict))
        print(f"{clock}  {c('1;37','? USER'.ljust(14))} {short(msg,76)}", flush=True)
    elif pt == "reasoning" and SHOW_REASONING:
        # Persisted reasoning items: the plan-first block and the coder's per-turn
        # thinking. summary[].text is the label; content[].text is the body.
        label = " · ".join(s.get("text", "") for s in (p.get("summary") or []) if isinstance(s, dict)) or "reasoning"
        body = "\n".join(x.get("text", "") for x in (p.get("content") or []) if isinstance(x, dict)).strip()
        lines = body.splitlines()
        head = f"{clock}  {c('95', ('◇ ' + label).ljust(14))}"
        if lines:
            print(f"{head} {c('95', lines[0])}", flush=True)
            for ln in lines[1:]:
                print(f"{' ' * 27}{c('90', ln)}", flush=True)
        else:
            print(head, flush=True)
    # token_count / function_call_output are skipped as noise


# Loop-guard firings are NOT in the rollout — they're prelude injections that get
# logged to the TUI log. To show them in the tail we read that log too and merge by
# timestamp. Path matches the harness's tracing sink.
LOG_PATH = os.path.expanduser("~/.codex/log/codex-tui.log")

def session_id_from_path(path):
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
                  os.path.basename(path or ""))
    return m.group(1) if m else None

def parse_guard_log_line(line, session_id=None):
    """A loop-guard firing in codex-tui.log → (epoch_seconds, label), else None.
    Matches the generic state_extract alert (always present) and the specific
    `Loop guard fired:` notice (which carries a guard= label). Filtered to the
    given session via its thread_id so other sessions' guards don't bleed in."""
    if "Loop guard fired" not in line and "Repetition alert fired" not in line:
        return None
    if session_id and session_id not in line:
        return None
    tsm = re.match(r"\s*(\S+)", line)
    tsec = ts_seconds(tsm.group(1)) if tsm else None
    if "Loop guard fired" in line:
        g = re.search(r'guard\s*=\s*"?([a-z_]+)"?', line)
        if g:
            label = g.group(1).replace("_", "-")
        else:
            d = re.search(r"Loop guard fired:\s*(.+?)(?:\s+\w+=|$)", line)
            label = d.group(1).strip() if d else "loop-guard"
    else:
        cnt = re.search(r"count=(\d+)", line)
        tn = re.search(r"tool_name=(\S*)", line)
        who = tn.group(1) if (tn and tn.group(1)) else "reads/circling"
        label = f"repetition alert ×{cnt.group(1) if cnt else '?'} ({who})"
    return (tsec, label)

def emit_guard(tsec, label):
    clock = c("90", datetime.fromtimestamp(tsec).strftime("%H:%M:%S") if tsec else "--:--:--")
    print(f"{clock}  {c('1;33', ('⚠ GUARD').ljust(14))} {c('1;33', short(label, 76))}", flush=True)


def follow():
    import time
    path = None
    f = None
    logf = None
    session_id = None

    def open_log_tail():
        # Open the shared TUI log and seek to END: only NEW guard lines stream
        # while tailing (historical ones are already shown by replay_full).
        nonlocal logf
        if logf:
            logf.close()
            logf = None
        if os.path.exists(LOG_PATH):
            logf = open(LOG_PATH)
            logf.seek(0, os.SEEK_END)

    def replay_full(newest):
        # Merge the WHOLE rollout with this session's guard log lines by timestamp,
        # so guards land inline where they fired instead of after the fact.
        nonlocal session_id
        session_id = session_id_from_path(newest)
        merged = []  # (epoch, tiebreak, kind, payload)
        with open(newest) as rf:
            for line in rf:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                merged.append((ts_seconds(o.get("timestamp")) or 0.0, 0, "ev", o))
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH) as lf:
                for line in lf:
                    g = parse_guard_log_line(line, session_id)
                    if g and g[0] is not None:
                        merged.append((g[0], 1, "guard", g[1]))
        merged.sort(key=lambda x: (x[0], x[1]))
        for tsec, _, kind, payload in merged:
            if kind == "ev":
                emit_live(payload)
            else:
                emit_guard(tsec, payload)

    def open_newest():
        nonlocal path, f
        newest = find_latest()
        if newest and newest != path:
            if f:
                f.close()
            path = newest
            replay_full(newest)
            f = open(newest)
            f.seek(0, os.SEEK_END)  # tail from EOF; replay showed the history
            open_log_tail()
            print(c("1", f"── live: {os.path.basename(newest)}  (Ctrl-C to stop) ──"), flush=True)

    open_newest()
    if not f:
        sys.exit("no rollout found under ~/.codex/sessions")
    try:
        while True:
            progressed = False
            line = f.readline()
            if line:
                progressed = True
                try:
                    emit_live(json.loads(line))
                except Exception:
                    pass
            if logf:
                lline = logf.readline()
                if lline:
                    progressed = True
                    g = parse_guard_log_line(lline, session_id)
                    if g and g[0] is not None:
                        emit_guard(*g)
            if not progressed:
                time.sleep(0.5)
                open_newest()  # auto-switch if you restart into a new session
    except KeyboardInterrupt:
        print()

# ---- main -------------------------------------------------------------------

def main():
    if "--follow" in sys.argv or "-f" in sys.argv:
        follow()
        return
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = argv[0] if argv else find_latest()
    if not path or not os.path.exists(path):
        sys.exit("no rollout found; pass a path")

    events = []
    for line in open(path):
        try:
            events.append(json.loads(line))
        except Exception:
            pass

    # Results pair by ORDER, not call_id: the local coder reuses the call_id
    # "local_call_0" for EVERY call, so call_id pairing collapses (every line
    # showed the last write's output). function_call and function_call_output
    # alternate 1:1, so the i-th output pairs with the i-th call.
    outputs = []
    for o in events:
        p = o.get("payload") or {}
        if p.get("type") != "function_call_output":
            continue
        out = p.get("output"); ex = None
        if isinstance(out, str):
            try:
                j = json.loads(out); ex = (j.get("metadata") or {}).get("exit_code"); out = j.get("output", out)
            except Exception:
                pass
        outputs.append((f"exit {ex}" if ex not in (None, 0) else "ok", firstline(out)))

    t0 = ts_seconds(events[0].get("timestamp")) if events else None
    counts = Counter()           # cumulative per (tool,target) -> the loop signal
    tally = Counter()            # for the summary
    tool_use = Counter()
    lines = []
    oi = 0  # index into `outputs`, advanced per function_call (order-paired)

    for o in events:
        p = o.get("payload") or {}
        t = o.get("type"); pt = p.get("type")
        rel = ts_seconds(o.get("timestamp"))
        clock = f"t+{int((rel - t0)//60):02d}:{int((rel - t0)%60):02d}" if (rel and t0) else "  --  "

        if pt == "user_message" or (t == "response_item" and pt == "message" and p.get("role") == "user"):
            msg = p.get("message") or "".join(x.get("text", "") for x in (p.get("content") or []) if isinstance(x, dict))
            lines.append((clock, "USER", c("1;37", short(msg, 74)), "", ""))
            counts.clear()  # a new user turn = a fresh footprint
        elif pt == "agent_message" or (t == "response_item" and pt == "message" and p.get("role") == "assistant"):
            msg = p.get("message") or "".join(x.get("text", "") for x in (p.get("content") or []) if isinstance(x, dict))
            if msg.strip():
                lines.append((clock, "say", c("37", short(msg, 70)), "", ""))
        elif t == "response_item" and pt == "function_call":
            tool, target, _ = classify(p.get("name", "?"), p.get("arguments"))
            key = (tool, target)
            counts[key] += 1; tally[key] += 1; tool_use[tool] += 1
            n = counts[key]
            status, out = outputs[oi] if oi < len(outputs) else ("", "")
            oi += 1
            lines.append((clock, tool, target, n, (status + "  " + short(out, 34)).strip()))

    # ---- render -------------------------------------------------------------
    total_s = (ts_seconds(events[-1].get("timestamp")) - t0) if (t0 and events) else 0
    print(c("1", f"\n{os.path.basename(path)}"))
    print(f"{len(events)} events  |  {int(total_s//60)}m {int(total_s%60)}s")
    print("tools: " + "  ".join(f"{c(TOOL_COLORS.get(k,'0'), k)}×{v}" for k, v in tool_use.most_common()))
    loops = [(k, n) for k, n in tally.most_common() if n >= 3]
    if loops:
        print(c("1;31", "\nLOOPS (same target touched ≥3×):"))
        for (tool, target), n in loops:
            bar = "█" * min(n, 40)
            print(f"  {c('1;31' if n>=8 else '33', f'{n:>3}×')}  {tool:<12} {short(target,34):<34} {c('31', bar)}")
    print(c("1", "\n── timeline " + "─" * 60))
    for clock, tool, target, n, extra in lines:
        if tool in ("USER", "say"):
            tag = c("1;37", "USER") if tool == "USER" else c("90", " say")
            print(f"{c('90', clock)}  {tag}  {target}")
            continue
        col = TOOL_COLORS.get(tool, "0")
        nstr = f"×{n}"
        loopmark = c("1;31", " ◄ LOOP") if n >= 8 else (c("31", " ◄") if n >= 5 else "")
        print(f"{c('90', clock)}  {c(col, tool.ljust(12))} {short(target,34):<34} "
              f"{c('1;31' if n>=8 else ('33' if n>=5 else '90'), nstr.rjust(4))}  {c('90', extra)}{loopmark}")
    print()

if __name__ == "__main__":
    main()
