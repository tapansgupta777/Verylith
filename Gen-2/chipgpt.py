import os
import json
import argparse
import subprocess
import hashlib
import re
import time
import random
import shutil
import concurrent.futures
import traceback
from builtins import print as _builtin_print
from collections import OrderedDict
from openai import OpenAI
from groq import Groq


_CONSOLE_SYMBOLS = {
    "⚠️": "[WARN]", "❌": "[FAIL]", "✅": "[PASS]", "📡": "[API]",
    "⚡": "[CACHE]", "🔄": "[ROUND]", "🧠": "[MODEL]", "🔥": "[STRONG]",
    "📝": "[SPEC]", "📐": "[ARCH]", "📋": "[TB]", "🔬": "[VERIFY]",
    "🧬": "[MUTATE]", "🔎": "[REVIEW]", "🔒": "[LOCK]", "🔓": "[UNLOCK]",
    "🧩": "[TYPE]", "🧷": "[LINK]", "🧾": "[REGISTRY]", "📁": "[OUTPUT]",
    "🎉": "[DONE]", "🚨": "[STOP]", "⏳": "[WAIT]", "🔧": "[INFO]",
    "🔁": "[RETRY]", "📌": "[ASSUMPTION]", "📊": "[SUMMARY]", "🚀": "",
    "—": "-", "–": "-", "→": "->", "←": "<-", "─": "-", "═": "=",
}


def _ascii_console_text(value):
    text = str(value)
    for symbol, replacement in _CONSOLE_SYMBOLS.items():
        text = text.replace(symbol, replacement)
    return text.encode("ascii", "ignore").decode("ascii")


def print(*values, **kwargs):
    """Keep interactive CLI logs readable and portable across terminals."""
    return _builtin_print(*(_ascii_console_text(value) for value in values), **kwargs)

# External skeleton library — versioned, testable independently
from skeletons import (
    detect_design_type,
    looks_like_hierarchical_integration,
    get_skeleton,
    check_skeleton_invariants,
    has_ready_valid_ports,
    build_protocol_assertions,
    SKELETON_LIBRARY,
)
from linker import (
    resolve_link_plan,
    register_verified_module,
    yosys_read_commands,
    link_error,
    format_link_summary,
    format_registered_module_signatures,
)

# ==============================================================================
# MODEL TIER CONFIGURATION
# ==============================================================================
FAST_MODEL   = "gpt-4o-mini"
STRONG_MODEL = "gpt-5.4-mini"

REASONING_MODELS = {
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "gpt-5.4-mini",
}

def _is_reasoning_model(model_name: str) -> bool:
    return model_name.strip() in REASONING_MODELS

MAX_TOKENS_FAST   = 4096
MAX_TOKENS_STRONG = 16384   # kept at 16384 — RTL + TB generation needs the space

# ==============================================================================
# LRU CACHE
# ==============================================================================
import threading

class BoundedCache:
    """True LRU cache — thread-safe, prevents memory explosion on long runs."""
    def __init__(self, max_size=100):
        self.cache    = OrderedDict()
        self.max_size = max_size
        self._lock    = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key, val):
        with self._lock:
            self.cache[key] = val
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def __contains__(self, key):
        with self._lock:
            return key in self.cache

ARCH_CACHE = BoundedCache(50)
RTL_CACHE  = BoundedCache(200)
TB_CACHE   = BoundedCache(200)

STRUCTURAL_TB_FAILURES = {
    "TB_TIMEOUT", "TB_SYNTAX_ERROR", "TB_STRUCTURAL"
}

PORT_NAME_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "of", "on", "or", "the", "to", "with",
    "input", "output", "inout", "wire", "reg", "logic", "signal",
    "signals", "port", "ports", "bus", "bit", "bits", "byte", "bytes",
    "serial", "parallel", "stream", "streaming", "active", "logical",
}

PROTOCOL_COMPOSITION_TYPES = {
    "axi_stream_uart_tx",
}
HIERARCHICAL_TYPES = {
    "hierarchical_top",
}

# ==============================================================================
# OPENAI CLIENT
# ==============================================================================

def get_openai_client(timeout_s: int = 180) -> OpenAI:
    """Create an OpenAI client with the specified timeout.
    Stateless — each call creates a fresh client so the timeout is always respected.
    OpenAI clients are lightweight (just httpx session config), so this is fine.
    """
    return OpenAI(timeout=timeout_s)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def code_hash(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)
    normalized = re.sub(r'\s+', ' ', text.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def get_cache_key(*args) -> str:
    combined = "".join(code_hash(str(a)) for a in args)
    return code_hash(combined)

def truncate_log(log: str, max_lines: int = 15) -> str:
    lines = log.strip().split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + "\n... [ERRORS TRUNCATED]"
    return log


def truncate_middle(text: str, max_chars: int = 6000) -> str:
    """Keep prompt payloads bounded while preserving both module header and fix tail."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    half = max(1, max_chars // 2)
    return (
        text[:half]
        + "\n\n... [TRUNCATED MIDDLE] ...\n\n"
        + text[-half:]
    )


def get_tooling_insights(design_type: str) -> str:
    """Return universal and design-type-specific lessons from prior resolved runs."""
    path = os.path.join(".", "workspace", "tooling_insights.md")
    if not os.path.exists(path):
        return ""

    tag = f"[{(design_type or 'generic').lower()}]"
    selected = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip()
                if not line.strip():
                    continue
                line_l = line.lower()
                if tag in line_l:
                    selected.append(line)
                    continue
                # Any bracketed design tag makes the line scoped. Lines without
                # bracketed tags are universal lessons and apply to all designs.
                if not re.search(r"\[[^\]]+\]", line):
                    selected.append(line)
    except OSError:
        return ""

    return "\n".join(selected).strip()


def get_registry_module_names() -> list:
    """Return verified module names from workspace/registry.json, if present."""
    path = os.path.join(".", "workspace", "registry.json")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        return []
    return sorted(str(name) for name in modules.keys())


def extract_primary_error(log: str) -> str:
    lines = log.strip().split('\n')
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in
               ("error", "mismatch", "fail", "syntax", "%warning", "assertion failed")):
            start = max(0, i - 1)
            end   = min(len(lines), i + 3)
            return "\n".join(lines[start:end])
    return truncate_log(log, 5)

def classify_failure(log: str) -> str:
    # Strip COVER_HIT lines — they are coverage markers, not failure descriptions.
    # Without this, "COVER_HIT:reset_during_shift" would trigger SHIFT_LOGIC
    # even when the actual failure is a reset bug.
    stripped = "\n".join(
        line for line in log.splitlines()
        if not line.strip().upper().startswith("COVER_HIT")
    )
    l = stripped.lower()
    path_l = l.replace("\\", "/")
    for marker, category in (
        ("validator_false_rejection:", "VALIDATOR_FALSE_REJECTION"),
        ("oracle_error:", "ORACLE_ERROR"),
        ("contract_violation:", "CONTRACT_VIOLATION"),
        ("implementation_syntax_error:", "IMPLEMENTATION_SYNTAX_ERROR"),
        ("structural_error:", "STRUCTURAL_ERROR"),
        ("functional_failure:", "FUNCTIONAL_FAILURE"),
    ):
        if marker in l:
            return category
    if "yosys_timeout" in l:
        return "YOSYS_TIMEOUT"
    if "memory_not_inferred" in l or "memory array mapped to registers" in l or "replacing memory" in l:
        return "MEMORY_NOT_INFERRED"
    if "timeout" in l or "simulation timeout" in l:
        return "TB_TIMEOUT"
    if "sequential_optimized_away" in l or "0 dffs" in l or "0 registers synthesized" in l:
        return "SEQUENTIAL_OPTIMIZED_AWAY"
    if "unoptflat" in l or ("loop" in l and "comb" in l):
        return "COMBINATIONAL_LOOP"
    if "compile" in l or "syntax" in l:
        if "/src/" in path_l or re.search(r'\bcandidate_\d+/src/', path_l):
            return "RTL_SYNTAX_ERROR"
        if "/tb/" in path_l or re.search(r'\bcandidate_\d+/tb/', path_l) or "_tb.v" in path_l:
            return "TB_SYNTAX_ERROR"
        return "RTL_SYNTAX_ERROR"
    if "latch" in l:
        return "LATCH_INFERRED"
    if "bit extraction" in l and "requires" in l and "bit index" in l:
        return "BIT_EXTRACTION"
    if "width" in l or "size" in l or "expects" in l:
        return "WIDTH_MISMATCH"
    if any(kw in l for kw in ("s_ready", "m_ready", "m_valid", "s_valid",
                               "tready", "tvalid", "tdata",
                               "handshake", "backpressure", "buffer", "skid")):
        return "HANDSHAKE_LOGIC"
    # FSM/UART output failures — check before X/Z regex to avoid false matches
    if any(kw in l for kw in ("tx_out", "start bit", "stop bit", "data bit",
                               "baud", "uart", "serial bit")):
        return "FSM_OUTPUT_WRONG"
    # Handshake/ready-valid failures — check before OFF_BY_ONE numeric diff
    if any(kw in l for kw in ("s_ready", "m_ready", "m_valid", "s_valid",
                               "handshake", "backpressure", "buffer", "skid")):
        return "HANDSHAKE_LOGIC"
    # X/Z propagation: tightened regex — only match clear X/Z indicators
    if re.search(r'\bgot\s+z\b|\bvalue\s*=\s*z\b', l):
        return "UNDRIVEN_OUTPUT"
    if re.search(r'\bxx+\b|\bgot\s+x\b|=\s*x\b', l):
        return "X_PROPAGATION_BUG"
    if "stuck" in l or "never changes" in l:
        return "STATE_MACHINE_STUCK"

    m = re.search(
        r'expected\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+).*?got\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+)', l
    )
    if m:
        try:
            g1, g2 = m.group(1), m.group(2)
            if re.search(r'[xzXZ]', g1 + g2):
                pass
            else:
                has_hex = ('0x' in l or any(c in 'abcdef'
                           for c in g1.lower() + g2.lower()))
                base = 16 if has_hex else 10
                exp  = int(g1, base)
                got  = int(g2, base)
                # Guard: OFF_BY_ONE only for multi-bit values (>1).
                # Single-bit 0 vs 1 differences are handshake/flag bugs.
                if abs(exp - got) == 1 and max(exp, got) > 1:
                    return "OFF_BY_ONE"
        except (ValueError, OverflowError):
            pass

    if "priority" in l or "instead of" in l or "order" in l:
        return "PRIORITY_ORDER_BUG"
    if "shift" in l or "serial" in l or "sipo" in l or "siso" in l or "piso" in l:
        return "SHIFT_LOGIC"
    if "increment" in l or "count up" in l or "count_up" in l:
        return "INCREMENT_LOGIC"
    if "decrement" in l or "count down" in l or "count_down" in l:
        return "DECREMENT_LOGIC"
    if "overflow" in l:
        return "OVERFLOW_FLAG"
    if "underflow" in l:
        return "UNDERFLOW_FLAG"
    # UART/FSM-specific reset failure: FSM didn't return to IDLE after async reset
    if re.search(r'tx_ready.*got\s+0|after.*reset.*tx_ready|recovery.*reset', l):
        return "RESET_AFTER_TX"
    if "reset" in l:
        return "RESET_LOGIC"
    if "load" in l:
        return "LOAD_LOGIC"
    if "assert" in l or "assertion" in l or "invariant" in l:
        return "PROTOCOL_ASSERTION_FAILED"
    if re.search(r'(expected|mismatch|fail)', l):
        return "LOGIC_MISMATCH"
    return "TB_STRUCTURAL"

def get_error_fingerprint(failure_type: str, full_log: str) -> str:
    primary   = extract_primary_error(full_log)
    line_nums = re.findall(r'\.v:(\d+):', primary)
    line_sig  = "_".join(line_nums[:3]) if line_nums else code_hash(primary)[:8]
    return f"{failure_type}_{line_sig}"

def localize_bug(v_code: str, failure_type: str, error_log: str = "") -> str:
    """Extract RTL lines most likely to contain the bug.

    Strategy 1: Parse the failing signal name from the error log and find its
    driver in the RTL — this gives the exact assignment that needs to change.
    Strategy 2: Fall back to keyword search by failure type.
    Returns a compact snippet for the mutation prompt.
    """
    lines    = v_code.split('\n')
    snippets = []

    # Strategy 1: extract signal name from error log
    # Try multiple patterns from most-specific to least-specific:
    #   "expected s_ready=0"  →  group 1 = s_ready
    #   "expected parallel_out"  →  group 1 = parallel_out
    #   "FAIL: count mismatch"  →  group 1 = count
    if error_log:
        signal = None
        for pattern in [
            r"Signal unoptimizable:\s+[^:]+:\s*([A-Za-z_]\w*)",
            r'(?:expected|got)\s+(\w+)\s*[=!<>]',          # expected signal=val
            r'(?:expected|got)\s+(\w+)\s+(?:0x|8\'|[0-9])',# expected signal 0x..
            r'\bFAIL[^:]*:\s*\w+\s+(\w+)\s+',              # FAIL: msg signal ...
            r'(?:expected|got)\s+(\w+)',                     # expected signal (bare)
        ]:
            m = re.search(pattern, error_log, re.IGNORECASE)
            if m:
                candidate = m.group(1).lower()
                # Skip generic non-signal words
                if candidate not in ('0', '1', 'true', 'false', 'high', 'low',
                                     'test', 'after', 'at', 'when', 'got', 'expected',
                                     'fail', 'pass', 'error', 'mismatch', 'assertion'):
                    signal = candidate
                    break

        if signal:
            for i, line in enumerate(lines):
                if re.search(
                    rf'\bassign\s+{re.escape(signal)}\b'
                    rf'|\b{re.escape(signal)}\s*<='
                    rf'|\b{re.escape(signal)}\s*=(?!=)',
                    line, re.IGNORECASE
                ):
                    start = max(0, i - 1)
                    end   = min(len(lines), i + 3)
                    snippets.append("\n".join(lines[start:end]).strip())

    # Strategy 2: keyword fallback by failure type
    if not snippets:
        keywords = []
        if failure_type == "INCREMENT_LOGIC":       keywords = ["+ 1", "+1", "increment", "add"]
        elif failure_type == "DECREMENT_LOGIC":     keywords = ["- 1", "-1", "decrement", "sub"]
        elif failure_type == "OVERFLOW_FLAG":       keywords = ["overflow", "255", "hff", "== 255"]
        elif failure_type == "UNDERFLOW_FLAG":      keywords = ["underflow", "== 0"]
        elif failure_type == "RESET_LOGIC":         keywords = ["rst", "reset"]
        elif failure_type == "LOAD_LOGIC":          keywords = ["load"]
        elif failure_type == "PRIORITY_ORDER_BUG":  keywords = ["if", "else if"]
        elif failure_type == "SHIFT_LOGIC":         keywords = ["shift", "serial", "{"]
        elif failure_type == "OFF_BY_ONE":          keywords = ["<", "<=", ">=", "=="]
        elif failure_type in ("LOGIC_MISMATCH", "PROTOCOL_ASSERTION_FAILED"):
            keywords = ["assign", "ready", "valid", "buffer"]
        elif failure_type in ("STATE_MACHINE_STUCK",):
            keywords = ["always", "state", "next_state"]
        elif failure_type == "COMBINATIONAL_LOOP":
            keywords = ["assign", "always @(*)", "ready", "valid", "grant", "next"]

        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in keywords):
                start = max(0, i - 1)
                end   = min(len(lines), i + 2)
                snippets.append("\n".join(lines[start:end]).strip())

    if not snippets:
        return ""

    unique_snippets = list(dict.fromkeys(snippets))
    return "...\n" + "\n...\n".join(unique_snippets[:4]) + "\n..."


def build_delta_mutation(v_code: str, failure_type: str, error_log: str) -> str:
    """Build a focused delta snippet for the mutation prompt.

    Instead of sending the full RTL, extract just the failing driver lines
    and the exact assertion that failed. This tightens the LLM's attention
    to the specific change needed.
    """
    if not error_log:
        return ""

    # Extract the first FAIL line as the specific assertion
    fail_line = ""
    for line in error_log.splitlines():
        if re.search(r'\bFAIL\b|\bfail\b|expected.*got|mismatch', line, re.I):
            fail_line = line.strip()
            break

    localized = localize_bug(v_code, failure_type, error_log)
    if not fail_line and not localized:
        return ""

    parts = []
    if fail_line:
        parts.append(f"FAILING ASSERTION:\n  {fail_line}")
    if localized:
        parts.append(f"DRIVER CODE TO FIX:\n{localized}")

    return "\n\n".join(parts)

def parse_llm_json(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        print("   ⚠️  LLM returned empty response.")
        return {}
    try:
        triple_tick = "`" * 3
        clean = re.sub(
            rf'{triple_tick}(?:json)?\s*|{triple_tick}\s*$', '',
            raw_text.strip(), flags=re.MULTILINE
        ).strip()
        clean = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', clean)
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group(), strict=False)
        return json.loads(clean, strict=False)
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON Parse Error: {e}")
        return {}

def auto_patch_param_width(v_code: str) -> str:
    """DISABLED: auto-widening was breaking internally-consistent LLM-generated code.
    The patcher widened register declarations but not the sized literals in assignments,
    creating internal inconsistency that Verilator rejects. The LLM now handles this
    via explicit prompt rules (unsized literals, correct counter widths).
    """
    return v_code


def clean_code_string(raw_code: str) -> str:
    if not raw_code:
        return ""
    clean = raw_code.strip()
    triple_tick = "`" * 3
    for fence in (f"{triple_tick}verilog", f"{triple_tick}systemverilog",
                  f"{triple_tick}v", triple_tick):
        clean = clean.replace(fence, "")
    return clean.strip()


def normalize_testbench_text(tb_code: str) -> str:
    """Repair safe, syntax-only testbench formatting issues before validation."""
    if not tb_code:
        return ""
    tb_code = re.sub(r'```[a-zA-Z0-9_+-]*\n?', '', tb_code).strip()
    tb_code = re.sub(r'(?m)^(\s*)timescale\b', r'\1`timescale', tb_code, count=1)
    if tb_code and not re.search(r'(?m)^\s*`timescale\b', tb_code):
        tb_code = "`timescale 1ns/1ps\n" + tb_code.lstrip()
    return tb_code.strip()


def _safe_path_token(token: str) -> str:
    token = str(token or "artifact")
    token = re.sub(r'[^A-Za-z0-9_.-]+', '_', token)
    return token.strip("._") or "artifact"


def save_debug_artifact(
    design_name: str,
    category: str,
    round_id: int,
    cand_idx: int,
    filename: str,
    content: str,
) -> str:
    """Persist pre-verification artifacts so generator/TB failures are inspectable."""
    if not content:
        return ""
    safe_design = _safe_path_token(design_name)
    safe_category = _safe_path_token(category)
    safe_filename = _safe_path_token(filename)
    base = os.path.join(".", "workspace", safe_design, "debug", f"round_{round_id}", safe_category)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"cand_{cand_idx}_{safe_filename}")
    with open(path, "w") as f:
        f.write(content)
    return path


def extract_requested_port_names(prompt: str) -> list:
    """Best-effort parser for explicit user interface lists.

    The prompt often uses descriptive phrases such as "the serial output tx".
    A first-identifier parser turns that into the bogus port name "the".
    Treat each comma/and item as a phrase and select the last identifier that
    is not an English/descriptive word.
    """
    def _is_empty_port_clause(raw: str) -> bool:
        return bool(re.match(
            r'^\s*(?:none\b|no\s+(?:ports?|outputs?|inputs?|external\s+ports?|external\s+outputs?|signals?)\b|n/?a\b|not\s+applicable\b)',
            raw or "",
            re.IGNORECASE,
        ))

    def _parse_port_items(raw: str) -> list:
        if _is_empty_port_clause(raw):
            return []
        raw = re.sub(r'\band\b', ',', raw, flags=re.IGNORECASE)
        parsed = []
        for item in raw.split(","):
            if _is_empty_port_clause(item):
                continue
            tokens = re.findall(r'[A-Za-z_]\w*', item)
            if not tokens:
                continue
            chosen = None
            for tok in reversed(tokens):
                if tok.lower() not in PORT_NAME_STOPWORDS:
                    chosen = tok
                    break
            if chosen is None:
                chosen = tokens[-1]
            if chosen and chosen not in parsed:
                parsed.append(chosen)
        return parsed

    m = re.search(
        r'\bInterface\s*:\s*(.*?)(?:\.\s|Parameter\s*:|When\s+|$)',
        prompt,
        re.IGNORECASE | re.DOTALL,
    )
    names = []
    if m:
        names.extend(_parse_port_items(m.group(1)))

    # Many combinational prompts use "Inputs: ... Output: ..." rather than a
    # single Interface list. Parse those labeled clauses without guessing from
    # arbitrary prose.
    for label in ("Inputs?", "Outputs?"):
        for m in re.finditer(
            rf'\b{label}\s*:\s*(.*?)(?:\.\s|;|\b(?:Inputs?|Outputs?|Parameter|Supported|When|Use)\s*:|$)',
            prompt,
            re.IGNORECASE | re.DOTALL,
        ):
            for name in _parse_port_items(m.group(1)):
                if name not in names:
                    names.append(name)
    return names


def has_protocol_oracle(design_type: str) -> bool:
    return design_type in PROTOCOL_COMPOSITION_TYPES


def _iter_always_blocks(v_code: str):
    """Yield (sensitivity/header, block_text) for Verilog always blocks."""
    text = re.sub(r'/\*.*?\*/', '', v_code, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)
    for m in re.finditer(r'always\s*@\s*\((.*?)\)', text, re.IGNORECASE | re.DOTALL):
        header = m.group(1)
        begin_m = re.search(r'\bbegin\b', text[m.end():], re.IGNORECASE)
        if not begin_m:
            semi = text.find(';', m.end())
            end_pos = len(text) if semi == -1 else semi + 1
            yield header, text[m.start():end_pos]
            continue

        pos = m.end() + begin_m.start()
        depth = 0
        end_pos = len(text)
        for tok in re.finditer(r'\bbegin\b|\bend\b', text[pos:], re.IGNORECASE):
            word = tok.group(0).lower()
            depth += 1 if word == "begin" else -1
            if depth == 0:
                end_pos = pos + tok.end()
                break
        yield header, text[m.start():end_pos]


def _check_blocking_mix(v_code: str) -> str:
    """Return the variable assigned with both blocking and non-blocking style."""
    nonblock_vars = set()
    block_vars = set()
    ignored = {
        '', 'begin', 'end', 'if', 'else', 'case', 'default', 'for',
        'while', 'repeat', 'integer', 'reg'
    }

    for _header, block in _iter_always_blocks(v_code):
        for m in re.finditer(r'\b([A-Za-z_]\w*)(?:\s*\[[^\]]+\])?\s*<=', block):
            name = m.group(1).strip()
            if name not in ignored:
                nonblock_vars.add(name)
        for m in re.finditer(r'(?<![<>=!])\b([A-Za-z_]\w*)(?:\s*\[[^\]]+\])?\s*=(?!=)', block):
            name = m.group(1).strip()
            if name not in ignored:
                block_vars.add(name)

    conflict = nonblock_vars & block_vars
    if conflict:
        return sorted(conflict)[0]
    return ""


def _parse_verilog_int_literal(value: str):
    value = value.strip().replace("_", "")
    sized = re.match(r"(?:(\d+)\s*)?'([dDhHbBoO])([0-9a-fA-FxzXZ]+)$", value)
    if sized:
        base_ch = sized.group(2).lower()
        digits = sized.group(3)
        if re.search(r'[xz]', digits, re.IGNORECASE):
            return None
        base = {"d": 10, "h": 16, "b": 2, "o": 8}[base_ch]
        return int(digits, base)
    if re.match(r'^\d+$', value):
        return int(value, 10)
    return None


def _check_duplicate_state_encodings(v_code: str) -> str:
    """Return a message if common FSM state names share the same localparam value."""
    state_name_re = re.compile(
        r'\b(?:IDLE|RESET|START|DATA|STOP|DONE|WAIT|BUSY|EMPTY|FULL|READ|WRITE|'
        r'ADDR|RESP|STATE_\w+|S\d+)\b',
        re.IGNORECASE,
    )
    seen = {}
    for m in re.finditer(
        r'\blocalparam\b(?:\s*\[[^\]]+\])?\s+([A-Za-z_]\w*)\s*=\s*([^,;\n]+)',
        v_code,
        re.IGNORECASE,
    ):
        name = m.group(1)
        if not state_name_re.search(name):
            continue
        value = _parse_verilog_int_literal(m.group(2))
        if value is None:
            continue
        if value in seen:
            return f"duplicate FSM state encoding: {seen[value]} and {name} both equal {m.group(2).strip()}"
        seen[value] = name
    return ""


def _check_ungated_uart_phase_transition(v_code: str) -> str:
    """Catch UART START/DATA/STOP state exits that ignore the baud counter."""
    if not re.search(r'\b(?:START|DATA|STOP)\b', v_code):
        return ""
    checks = [
        ("START", "DATA"),
        ("DATA", "STOP"),
        ("STOP", "IDLE"),
    ]
    for src, dst in checks:
        pattern = (
            rf'(?:else\s+)?if\s*\([^)]*\b{src}\b[^)]*\)\s*begin'
            rf'(?P<body>.*?)\bend\b'
        )
        for m in re.finditer(pattern, v_code, re.IGNORECASE | re.DOTALL):
            body = m.group("body")
            if re.search(rf'<=\s*\b{dst}\b', body) and not re.search(
                r'\b(?:baud|tick|TICKS_PER_BAUD|CLKS_PER_BIT|CLKS_PER_BAUD)\b',
                body,
                re.IGNORECASE,
            ):
                return f"UART {src}->{dst} transition is not gated by baud counter"
    return ""


# ==============================================================================
# CORE LLM ROUTER
# ==============================================================================
def run_llm(
    prompt:           str,
    system_prompt:    str,
    provider:         str,
    max_api_retries:  int  = 3,
    cand_idx:         int  = 0,
    force_fast_model: bool = False,
    silent:           bool = False,
) -> dict:
    temps   = [0.2, 0.5, 0.8, 1.0]
    efforts = ["low", "medium", "high"]

    model_name   = FAST_MODEL if force_fast_model else STRONG_MODEL
    is_reasoning = _is_reasoning_model(model_name)
    max_tokens   = MAX_TOKENS_FAST if force_fast_model else MAX_TOKENS_STRONG

    for attempt in range(max_api_retries):
        try:
            if provider == "openai":
                client    = get_openai_client(timeout_s=180 if not force_fast_model else 90)
                temp_idx  = min(cand_idx + attempt, len(temps) - 1)
                kwargs    = {
                    "model":    model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                }
                if is_reasoning:
                    kwargs["max_completion_tokens"] = max_tokens
                    kwargs["reasoning_effort"]      = efforts[temp_idx % 3]
                    # Reasoning models don't support response_format=json_object.
                    # Inject a mandatory JSON instruction into the system prompt instead.
                    for msg in kwargs["messages"]:
                        if msg["role"] == "system":
                            msg["content"] = (
                                msg["content"].rstrip()
                                + "\n\nCRITICAL: Your response MUST be valid JSON only. "
                                "Output nothing before or after the JSON object. "
                                "No markdown fences, no explanation, no preamble."
                            )
                            break
                else:
                    kwargs["max_tokens"]      = max_tokens
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["temperature"]     = temps[temp_idx]

                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling {model_name} (Attempt {attempt+1})...")

                response = client.chat.completions.create(**kwargs)
                content  = response.choices[0].message.content

                if not content or not content.strip():
                    print(f"      ⚠️  Thread {cand_idx+1}: Empty content. Retrying...")
                    continue

                return parse_llm_json(content)

            elif provider == "groq":
                client   = Groq()
                temp_idx = min(cand_idx + attempt, len(temps) - 1)
                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling Groq llama-3.3-70b...")
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=temps[temp_idx],
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=90,
                )
                return parse_llm_json(response.choices[0].message.content)

        except Exception as e:
            err = str(e).lower()
            if "400" in err and ("unsupported parameter" in err or "invalid_request" in err):
                print(f"   ❌ Thread {cand_idx+1}: Bad request: {e}")
                return {}
            elif "429" in err or "quota" in err:
                wait = 35 * (attempt + 1)
                print(f"   ⏳ Thread {cand_idx+1}: Rate limit. Sleeping {wait}s...")
                time.sleep(wait)
            elif "timeout" in err:
                print(f"   ⚠️  Thread {cand_idx+1}: API timeout. Retrying...")
            else:
                print(f"   ⚠️  Thread {cand_idx+1}: Fatal API error: {e}")
                time.sleep(5)

    return {}

# ==============================================================================
# AGENT: SOC SYSTEM ARCHITECT
# ==============================================================================
def decompose_soc_architecture(prompt: str, provider: str) -> dict:
    """Plan missing verified modules and the final hierarchical top wrapper."""
    available_modules = get_registry_module_names()
    signatures = format_registered_module_signatures(" ".join(available_modules))
    if not signatures:
        signatures = "(no registered module signatures found)"

    system_prompt = f"""
You are an elite SoC Systems Architect. Break down the user's SoC request into a dependency graph.

AVAILABLE VERIFIED MODULES: {available_modules}

VERIFIED PORT SIGNATURES:
{signatures}

Identify any missing sub-modules that need to be generated from scratch.
CRITICAL: For every missing module, your standalone prompt MUST dictate the EXACT port names to be used, so your top-level wrapper knows exactly how to wire them.
Finally, write a strict, highly detailed prompt to generate the top-level 'hierarchical_top' wrapper that instantiates and wires everything together.
Output ONLY JSON.

JSON schema:
{{
  "missing_modules": [
    {{
      "name": "module_name",
      "prompt": "Standalone prompt. Must include exact port names (e.g., Inputs: clk_in, rst_n. Outputs: clk_out) and behavior."
    }}
  ],
  "top_level_prompt": "Design a hierarchical_top wrapper... Include strict wiring instructions matching the exact port signatures."
}}
"""
    user_prompt = f"""
USER SOC REQUEST:
{prompt}

Return only the JSON object described by the schema.
"""
    result = run_llm(
        user_prompt,
        system_prompt,
        provider,
        max_api_retries=2,
        force_fast_model=True,
        silent=False,
    )

    if not isinstance(result, dict):
        result = {}

    missing = result.get("missing_modules", [])
    if not isinstance(missing, list):
        missing = []
    normalized_missing = []
    seen = set()
    for item in missing:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        mod_prompt = str(item.get("prompt", "")).strip()
        if not name or not mod_prompt:
            continue
        if name in seen:
            continue
        seen.add(name)
        normalized_missing.append({"name": name, "prompt": mod_prompt})

    top_level_prompt = str(result.get("top_level_prompt", "")).strip()
    if top_level_prompt and "hierarchical_top" not in top_level_prompt.lower():
        top_level_prompt = "Design a hierarchical_top wrapper. " + top_level_prompt

    return {
        "missing_modules": normalized_missing,
        "top_level_prompt": top_level_prompt,
    }

# ==============================================================================
# EXPERIENCE MEMORY AGENT
# ==============================================================================
def commit_tooling_insight(
    design_name: str,
    design_type: str,
    worst_error: str,
    failing_code: str,
    passing_code: str,
    provider: str,
) -> None:
    system_prompt = (
        "You are a hardware RTL expert. Given a failing Verilog module, the "
        "simulation/synthesis error it produced, and the fixed working version, "
        "extract ONE specific structural rule that fixed the bug. Format your "
        "rule as 'For [design_type]: [specific rule]'. Output ONLY JSON matching "
        "this schema: {\"insight_rule\": \"...\"}"
    )
    dtype = design_type or "generic"
    user_prompt = f"""
Design name: {design_name}
Design type: {dtype}

ERROR PRODUCED BY FAILING VERSION:
{truncate_middle(worst_error or "", 4000)}

FAILING VERILOG:
```verilog
{truncate_middle(failing_code or "", 7000)}
```

FIXED PASSING VERILOG:
```verilog
{truncate_middle(passing_code or "", 7000)}
```

Extract one reusable structural rule. The rule must begin exactly with:
For [{dtype}]:
"""
    try:
        result = run_llm(
            user_prompt,
            system_prompt,
            provider,
            cand_idx=0,
            force_fast_model=True,
            silent=True,
        )
        insight_rule = str((result or {}).get("insight_rule", "")).strip()
        if not insight_rule:
            return
        insight_rule = " ".join(insight_rule.split())
        os.makedirs(os.path.join(".", "workspace"), exist_ok=True)
        with open(
            os.path.join(".", "workspace", "tooling_insights.md"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(insight_rule + "\n")
    except Exception as e:
        print(f"   ⚠️  Memory Agent failed: {e}")


# ==============================================================================
# AGENT: CLARIFIER
# ==============================================================================
def generate_specification(prompt: str, provider: str) -> dict:
    print(f"   📝 Routing to CLARIFIER AGENT ({provider.upper()}) [{FAST_MODEL}]...")
    system_prompt = """
You are a Lead Hardware Specifications Engineer and an Encyclopedic Hardware Knowledge Base.
Translate the user's ambiguous request into a strict, disambiguated technical specification.

SPEC-SYNC PROTOCOL:
You are the ONLY agent allowed to resolve user ambiguity. Downstream RTL and
testbench agents will receive your master specification, not the raw user prompt.
Therefore you MUST NOT leave vague phrases unresolved.

Aggressively hunt for ambiguity:
- Undefined opcode maps / truth tables / ALU operations
- Missing reset polarity or reset value
- Vague FSM transition priority
- Missing flag semantics such as carry, borrow, overflow, zero, ready, valid
- Undefined behavior for divide-by-zero, empty/full FIFO access, simultaneous read/write
- Modeling style constraints such as dataflow, structural, behavioral, gate-level

If ambiguity exists, autonomously choose a sane textbook/default behavior and
record it under `assumptions_made`. Never write "TBD", "as needed", "etc.",
"possible operations", or "implementation dependent". The master spec must be
strict enough that RTL and testbench agents implement the SAME behavior.

For an ALU with N operations but no opcode table, define the full opcode map.
For a 16-operation ALU, assign every opcode 0..15 exactly once, define result
width, flag behavior, divide-by-zero/mod-by-zero behavior if used, and whether
arithmetic wraps, saturates, or reports overflow. If the user requests dataflow
modeling, state that the DUT must use continuous assign statements only: no
always blocks, no procedural assignments, and no reg-driven outputs.

CRITICAL GENERALIZATION RULE:
If the user asks for a standard component (e.g., 'Skid Buffer', 'Async FIFO', 'Arbiter',
'UART', 'SPI', 'AXI', 'shift register', 'counter'), DO NOT just restate their prompt.
You MUST aggressively expand `formalized_request` using your internal knowledge:
  1. List the textbook logic equations and architectural structures required.
  2. Name every required internal register (e.g., buffer_data, buffer_valid, read_ptr).
  3. State rigid protocol invariants explicitly:
     - For ready/valid: "valid may not deassert until ready is high (once valid asserts,
       data must hold until accepted). s_ready must deassert when internal buffer is full
       and downstream is not consuming."
     - For FIFO: "full flag prevents writes, empty flag prevents reads."
  4. Describe the exact combinational conditions for every output flag.
This expansion is critical — downstream agents must not guess implicit dependencies.

AMS SPEC-SYNC REQUIREMENTS:
For SAR ADC controllers, explicitly specify or record a named default for: resolution,
comparator polarity, the exact comparator sampling event, registered DAC update timing,
sample/settle timing and settle-cycle count, start acceptance, start-while-busy behavior,
one-cycle EOC/done timing, and reset behavior. State the deterministic transaction:
START -> initialize -> set current MSB/bit trial -> registered DAC update -> settle ->
sample comparator -> retain/clear the current bit -> select next bit -> final result ->
exactly one DONE cycle -> IDLE. Treat comparator resolve time and CDAC settling accuracy
as PHYSICAL_ASSUMPTIONs, not RTL-proven facts.

For SPI register banks, explicitly specify or record defaults for: SPI mode (CPOL/CPHA),
which SCLK edge samples MOSI and shifts MISO, frame format/length, bit order, R/W encoding,
address and data widths, register map, CS_n incomplete-frame abort, readback behavior,
shadow versus active register behavior, atomic commit semantics, SCLK/CLK relationship,
the CDC update protocol, and reset. A one-cycle IO_UPDATE pulse is not acceptable when it
can be missed. Describe a toggle/event protocol and the bundled-data stability contract;
do not claim a two-flop synchronizer alone makes a multi-bit data bus universally safe.

For non-overlapping clock controllers, explicitly specify or record defaults for: source
reference clock/control-domain relationship, phase sequencing, dead time in supported clock
cycles, reset state, enable/disable behavior, and whether CDC exists. State the RTL invariant
!(phi1 && phi2), but list nanosecond dead time, skew, PVT, and charge injection as
PHYSICAL_ASSUMPTIONs rather than RTL guarantees. Never describe synchronizing a clock as
ordinary data unless the user explicitly requests a separate clock-control CDC protocol.

Use the following invariant categories for AMS work whenever relevant: RTL_INVARIANT,
CDC_INVARIANT, PROTOCOL_INVARIANT, NUMERICAL_INVARIANT, and PHYSICAL_ASSUMPTION.
When a material parameter is missing, choose a conservative documented default and append
the choice verbatim to assumptions_made; never leave it implicit.

Return pure JSON:
{
  "formalized_request": "...",
  "master_specification": "...",
  "data_widths": "...",
  "sequential_or_combinational": "...",
  "clocking_scheme": "...",
  "reset_behavior": "...",
  "overflow_behavior": "...",
  "protocol_invariants": "...",
  "required_internal_registers": "...",
  "operation_table": "...",
  "truth_table": "...",
  "modeling_style_constraints": "...",
  "testbench_obligations": "...",
  "rtl_invariants": "...",
  "cdc_invariants": "...",
  "numerical_invariants": "...",
  "physical_assumptions": "...",
  "ams_contract": {"kind": "...", "parameters": {"...": "..."}},
  "assumptions_made": ["..."]
}
"""
    return run_llm(prompt, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)


def _requires_dataflow_modeling(text: str) -> bool:
    text = (text or "").lower()
    return any(kw in text for kw in (
        "dataflow modelling", "dataflow modeling", "data flow modelling",
        "data flow modeling", "strictly dataflow", "strict dataflow",
        "continuous assign", "continuous assignment",
    ))


def _name_is_clock_like(name: str) -> bool:
    n = str(name or "").lower()
    return "clk" in n or "clock" in n


def _architecture_has_clock_port(architecture: dict) -> bool:
    return any(
        _name_is_clock_like(p.get("name", ""))
        for p in (architecture or {}).get("port_interface", [])
        if isinstance(p, dict)
    )


def _spec_has_clock_port(clarified_spec: dict = None, requested_port_names: list = None) -> bool:
    requested = list(requested_port_names or [])
    spec = clarified_spec or {}
    requested.extend(str(p) for p in spec.get("_user_requested_ports", []) if isinstance(p, str))
    return any(_name_is_clock_like(name) for name in requested)


def _rtl_text_has_edge_always(v_code: str) -> bool:
    code = re.sub(r'/\*.*?\*/', '', v_code or "", flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return bool(re.search(r'\balways\s*@\s*\([^)]*\b(?:posedge|negedge)\b', code, re.IGNORECASE))


def _clocked_or_sequential_intent(
    text: str = "",
    architecture: dict = None,
    clarified_spec: dict = None,
    requested_port_names: list = None,
) -> bool:
    spec = clarified_spec or {}
    fields = " ".join(
        str(spec.get(k, ""))
        for k in (
            "formalized_request",
            "sequential_or_combinational",
            "clocking_scheme",
            "reset_behavior",
            "required_internal_registers",
            "protocol_invariants",
        )
    )
    combined = f"{text or ''} {fields}".lower()
    text_mentions_clock = bool(re.search(r'\b\w*(?:clk|clock)\w*\b', combined))
    text_mentions_clock = text_mentions_clock and not _text_says_combinational(combined)
    return (
        _architecture_has_clock_port(architecture)
        or _spec_has_clock_port(spec, requested_port_names)
        or text_mentions_clock
        or _text_says_sequential(combined)
    )


def _should_enforce_strict_dataflow(
    text: str,
    architecture: dict = None,
    clarified_spec: dict = None,
    requested_port_names: list = None,
    v_code: str = None,
) -> bool:
    """Strict dataflow only applies to genuinely clockless combinational designs."""
    if not _requires_dataflow_modeling(text):
        return False
    if _clocked_or_sequential_intent(text, architecture, clarified_spec, requested_port_names):
        return False
    if v_code and _rtl_text_has_edge_always(v_code):
        return False
    return True


def _architecture_style_blob(architecture: dict) -> str:
    arch = architecture or {}
    return " ".join(
        str(arch.get(k, ""))
        for k in ("style", "implementation_style", "template_type")
    )


def _override_clocked_dataflow_architecture(architecture: dict) -> bool:
    """Convert impossible clocked-dataflow blueprints into sequential RTL style."""
    if not architecture:
        return False
    if "continuous_assign_dataflow" not in _architecture_style_blob(architecture):
        return False
    if not _architecture_has_clock_port(architecture):
        return False

    architecture["implementation_style"] = "behavioral_single_always_block"
    if "style" in architecture:
        architecture["style"] = "behavioral_single_always_block"
    if architecture.get("template_type") == "continuous_assign_dataflow":
        architecture["template_type"] = "behavioral_single_always_block"
    return True


def _has_explicit_opcode_map(text: str) -> bool:
    text = text or ""
    opcode_hits = re.findall(
        r'\b(?:op|opcode|alu_ctrl|select|sel)\s*(?:=|==|:|->)?\s*'
        r"(?:4\s*'?\s*[hb]\s*)?[0-9a-fA-F]+\s*(?:=|:|->|-)",
        text,
        re.IGNORECASE,
    )
    numbered_hits = re.findall(r'\b(?:[0-9]|1[0-5])\s*(?:=|:|->|-)\s*[A-Za-z]', text)
    return len(opcode_hits) >= 4 or len(numbered_hits) >= 8


def _default_4bit_alu16_operation_table() -> str:
    return """Authoritative 4-bit ALU opcode map. Inputs A[3:0], B[3:0], op[3:0].
All result values are 4-bit. Arithmetic wraps modulo 16; overflow/carry flag ov
reports unsigned carry/borrow/range/divide error as specified below.
op=0 ADD: result=(A+B)[3:0], ov=carry out of bit 3.
op=1 SUB: result=(A-B)[3:0], ov=1 when A<B (borrow), else 0.
op=2 INC_A: result=(A+1)[3:0], ov=1 when A==15, else 0.
op=3 DEC_A: result=(A-1)[3:0], ov=1 when A==0, else 0.
op=4 MUL_LOW: result=(A*B)[3:0], ov=1 when (A*B)>15, else 0.
op=5 DIV: if B==0 result=0 and ov=1; else result=(A/B)[3:0] and ov=0.
op=6 MOD: if B==0 result=0 and ov=1; else result=(A%B)[3:0] and ov=0.
op=7 AND: result=A&B, ov=0.
op=8 OR: result=A|B, ov=0.
op=9 XOR: result=A^B, ov=0.
op=10 NOT_A: result=~A, ov=0.
op=11 SHL1_A: result=(A<<1)[3:0], ov=A[3].
op=12 SHR1_A: result=A>>1 logical, ov=0.
op=13 MIN_UNSIGNED: result=(A<B)?A:B, ov=0.
op=14 MAX_UNSIGNED: result=(A>B)?A:B, ov=0.
op=15 PASS_B: result=B, ov=0."""


def _append_spec_text(spec: dict, key: str, additions: list) -> None:
    """Append unique, explicit contract clauses without erasing clarifier output."""
    current = str(spec.get(key, "")).strip()
    lines = [line.strip() for line in current.splitlines() if line.strip()]
    for addition in additions:
        addition = str(addition or "").strip()
        if addition and addition not in lines:
            lines.append(addition)
    if lines:
        spec[key] = "\n".join(lines)


def _find_ams_integer(text: str, patterns: tuple, default: int) -> tuple:
    """Return (value, explicitly_specified) for bounded AMS integer parameters."""
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value, True
    return default, False


_AMS_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def _ams_number(token: str):
    token = str(token or "").strip().lower()
    if token.isdigit():
        return int(token)
    return _AMS_NUMBER_WORDS.get(token)


def _find_ams_number(text: str, patterns: tuple):
    for pattern in patterns:
        match = re.search(pattern, text or "", re.IGNORECASE)
        if match:
            value = _ams_number(match.group(1))
            if value is not None and value > 0:
                return value, True
    return None, False


def _sar_existing_value(spec: dict, name: str):
    contract = spec.get("ams_contract")
    if not isinstance(contract, dict):
        return None
    parameters = contract.get("parameters")
    if not isinstance(parameters, dict):
        return None
    value = parameters.get(name)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        return int(m.group(0)) if m else None
    if isinstance(value, (int, bool)):
        return int(value)
    return None


def _sar_contract_hash(contract: dict) -> str:
    payload = dict(contract or {})
    payload.pop("contract_hash", None)
    payload.pop("provenance", None)
    payload.pop("unresolved_fields", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _set_sar_contract_field(contract, key, value, provenance, source, raw_value=None):
    contract[key] = value
    provenance[key] = {
        "status": source,
        "raw_value": raw_value,
    }


def _normalise_sar_contract(spec: dict, raw_prompt: str, assumptions: list) -> None:
    """Build a canonical SAR contract; explicit raw-user values always win."""
    raw = raw_prompt or ""
    clarify_text = " ".join(
        str(spec.get(key, ""))
        for key in ("formalized_request", "master_specification", "ams_contract")
    )
    old_contract = spec.get("ams_contract")
    if not isinstance(old_contract, dict):
        old_contract = {}
    clarifier_assumptions = spec.get("assumptions_made", [])
    if isinstance(clarifier_assumptions, str):
        clarifier_assumptions = [clarifier_assumptions]
    clarifier_assumptions = " ".join(str(item) for item in clarifier_assumptions)

    def clarifier_defaulted(field_pattern):
        return bool(re.search(
            rf"(?:sar[^.\n]{{0,80}})?(?:{field_pattern})[^.\n]{{0,100}}(?:default|unspecified|not specified)",
            clarifier_assumptions,
            re.IGNORECASE,
        ))
    provenance = {}
    contract = {
        "kind": "sar_adc_controller",
        "version": "sar-controller-contract-v2",
    }

    number = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    resolution, explicit = _find_ams_number(raw, (
        rf"\b{number}\s*[- ]?bit\s+(?:wide\s+)?(?:sar\s+)?(?:adc|controller)\b",
        rf"\b(?:sar\s+)?adc\s*[,;:]?\s*{number}\s*[- ]?bits?\b",
        rf"\bresolution\s*(?:=|:|is)?\s*{number}\s*bits?\b",
        rf"\b{number}\s*[- ]?bit\s+sar\b",
    ))
    if explicit:
        resolution_source, resolution_raw = "explicit_user", resolution
    else:
        resolution = _sar_existing_value(spec, "resolution")
        if resolution is None:
            resolution, inferred = _find_ams_number(clarify_text, (
                rf"\bresolution\s*(?:=|:|is)?\s*{number}\s*bits?\b",
                rf"\b{number}\s*[- ]?bit\s+(?:sar\s+)?(?:adc|controller)\b",
            ))
            resolution_source = "inferred" if inferred else "defaulted"
        else:
            resolution_source = "defaulted" if clarifier_defaulted("resolution") else "inferred"
        resolution_raw = None
    if resolution is None:
        resolution, resolution_source = 8, "defaulted"
    if resolution < 1 or resolution > 32:
        resolution_source = "unresolved"
        resolution = 8
    _set_sar_contract_field(contract, "resolution", resolution, provenance, resolution_source, resolution_raw)

    settle_cycles, explicit = _find_ams_number(raw, (
        rf"\b(?:allow|wait|use|provide)\s+{number}\s+(?:full\s+)?(?:clock\s+)?cycles?\s+for\s+(?:the\s+)?(?:cdac\s+)?settling\b",
        rf"\b{number}\s+(?:full\s+)?(?:clock\s+)?cycles?\s+(?:of|for)\s+(?:cdac\s+)?settling\b",
        rf"\b{number}\s+(?:clock\s+)?cycles?\s+(?:of\s+)?(?:the\s+)?(?:cdac\s+)?settling\b",
        rf"\b{number}\s+settle\s+cycles?\b",
        rf"\bsettle(?:\s+for|\s+after|\s+time)?\s*(?:=|:|is)?\s*{number}\s*(?:clock\s+)?cycles?\b",
        rf"\b{number}\s+settling\s+cycles?\b",
        rf"\bsettle_cycles\s*=\s*{number}\b",
        rf"\b{number}\s+clocks?\s+after\s+(?:the\s+)?dac\s+update\b",
    ))
    if explicit:
        settle_source, settle_raw = "explicit_user", settle_cycles
    else:
        settle_cycles = _sar_existing_value(spec, "settle_cycles")
        if settle_cycles is None:
            settle_cycles, inferred = _find_ams_number(clarify_text, (
                rf"\bsettle_cycles\s*=\s*{number}\b",
                rf"\b{number}\s+(?:clock\s+)?cycles?\s+(?:of|for)\s+settling\b",
            ))
            settle_source = "inferred" if inferred else "defaulted"
        else:
            settle_source = "defaulted" if clarifier_defaulted(r"settle(?:_cycles|ment| interval)?") else "inferred"
        settle_raw = None
    if settle_cycles is None:
        settle_cycles, settle_source = 1, "defaulted"
    _set_sar_contract_field(contract, "settle_cycles", settle_cycles, provenance, settle_source, settle_raw)

    keep_high_match = re.search(
        r"(?:comparator|cmp)(?:\s+output)?[^.\n]{0,120}?(?:high|logic\s*1|=\s*1)[^.\n]{0,100}?(?:keep\w*|kept|retain\w*|accept\w*)|"
        r"(?:high|logic\s*1|comparator\s*=\s*1)[^.\n]{0,100}?(?:means?|indicates?|when)[^.\n]{0,60}?(?:keep\w*|kept|retain\w*|accept\w*)|"
        r"(?:comparator|cmp)(?:\s+output)?[^.\n]{0,100}(?:is|=)\s*1[^.\n]{0,100}(?:keep\w*|kept|retain\w*|accept\w*)",
        raw, re.IGNORECASE,
    )
    keep_low_match = re.search(
        r"(?:comparator|cmp)(?:\s+output)?[^.\n]{0,120}?(?:low|logic\s*0|=\s*0)[^.\n]{0,100}?(?:keep\w*|kept|retain\w*|accept\w*)|"
        r"(?:low|logic\s*0|comparator\s*=\s*0)[^.\n]{0,100}?(?:means?|indicates?|when)[^.\n]{0,60}?(?:keep\w*|kept|retain\w*|accept\w*)",
        raw, re.IGNORECASE,
    )
    high_clear_match = re.search(
        r"(?:comparator|cmp)(?:\s+output)?[^.\n]{0,120}?(?:high|logic\s*1|=\s*1)[^.\n]{0,100}?(?:clear\w*|reject\w*|discard\w*)|"
        r"(?:high|logic\s*1|comparator\s*=\s*1)[^.\n]{0,100}?(?:means?|indicates?)[^.\n]{0,60}?(?:clear\w*|reject\w*|discard\w*)",
        raw, re.IGNORECASE,
    )
    if keep_low_match or high_clear_match:
        high_keeps, cmp_source = False, "explicit_user"
    elif keep_high_match:
        high_keeps, cmp_source = True, "explicit_user"
    elif re.search(
        r"comparator(?:\s+output)?[^.\n]{0,100}(?:high|logic\s*1)[^.\n]{0,80}vdac\s*>\s*vin",
        raw, re.IGNORECASE,
    ):
        high_keeps, cmp_source = False, "inferred"
    elif re.search(
        r"comparator(?:\s+output)?[^.\n]{0,100}(?:high|logic\s*1)[^.\n]{0,80}vin\s*>\s*vdac",
        raw, re.IGNORECASE,
    ):
        high_keeps, cmp_source = True, "inferred"
    else:
        old_value = old_contract.get("parameters", {}).get("comparator_high_keeps_bit") if isinstance(old_contract.get("parameters"), dict) else None
        if isinstance(old_value, str):
            old_value = old_value.lower() in {"1", "true", "yes", "high keeps", "high means keep"}
        if isinstance(old_value, bool):
            high_keeps, cmp_source = old_value, "defaulted" if clarifier_defaulted(r"comparator(?: polarity)?") else "inferred"
        elif re.search(r"comparator[^.\n]{0,100}(?:low|0)[^.\n]{0,50}(?:keep\w*|kept|retain\w*)|(?:low|0)[^.\n]{0,60}(?:keep\w*|kept|retain\w*)|comparator[^.\n]{0,100}(?:high|1)[^.\n]{0,50}(?:clear\w*|reject\w*)", clarify_text, re.IGNORECASE):
            high_keeps, cmp_source = False, "inferred"
        else:
            high_keeps, cmp_source = True, "defaulted"
    _set_sar_contract_field(contract, "comparator_high_keeps_bit", bool(high_keeps), provenance, cmp_source)

    physical_relation = None
    if re.search(r"\bvin\s*>\s*vdac\b", raw, re.IGNORECASE):
        physical_relation = "VIN > VDAC"
    elif re.search(r"\bvdac\s*>\s*vin\b", raw, re.IGNORECASE):
        physical_relation = "VDAC > VIN"
    else:
        physical_relation = "unspecified analog transfer relation"
    if physical_relation == "unspecified analog transfer relation":
        provenance["comparator.physical_relation"] = {"status": "unresolved", "raw_value": None}
    else:
        provenance["comparator.physical_relation"] = {"status": "explicit_user", "raw_value": physical_relation}

    if re.search(r"(?:sample|sampling)[^.\n]{0,80}(?:rising|positive)\s+edge|(?:rising|positive)\s+edge[^.\n]{0,80}(?:sample|sampling)", raw, re.IGNORECASE):
        sample_event, sample_source = "posedge clk", "explicit_user"
    elif re.search(r"(?:sample|sampling)[^.\n]{0,80}(?:falling|negative)\s+edge|(?:falling|negative)\s+edge[^.\n]{0,80}(?:sample|sampling)", raw, re.IGNORECASE):
        sample_event, sample_source = "negedge clk", "explicit_user"
    elif re.search(r"sample[^.\n]{0,80}after[^.\n]{0,50}sett", raw, re.IGNORECASE):
        sample_event, sample_source = "posedge clk", "inferred"
    else:
        sample_event, sample_source = "posedge clk", "defaulted"
    _set_sar_contract_field(contract, "comparator_sample_event", sample_event, provenance, sample_source)

    contract["parameters"] = {
        "resolution": resolution,
        "settle_cycles": settle_cycles,
        "comparator_high_keeps_bit": bool(high_keeps),
    }
    contract["comparator"] = {
        "physical_behavior": physical_relation,
        "logical_output_polarity": {
            "active_level": 1 if high_keeps else 0,
            "active_level_meaning": "trial_bit_should_be_kept",
            "logic_1_means": "trial_bit_should_be_kept" if high_keeps else "trial_bit_should_be_cleared",
            "logic_0_means": "trial_bit_should_be_cleared" if high_keeps else "trial_bit_should_be_kept",
        },
        "sar_decision": {
            "on_logic_1": "keep_trial_bit" if high_keeps else "clear_trial_bit",
            "on_logic_0": "clear_trial_bit" if high_keeps else "keep_trial_bit",
            "comparator_value_to_keep_trial": 1 if high_keeps else 0,
            "comparator_value_to_clear_trial": 0 if high_keeps else 1,
        },
    }
    contract["timing"] = {
        "sample_event": sample_event,
        "dac_trial_update": "posedge clk in S_SET_TRIAL; dac_code changes to trial_code | selected_bit",
        "settling_interval": f"{settle_cycles} complete clk periods after the DAC update edge",
        "sample_offset": f"{sample_event} at trial_update_cycle + {settle_cycles + 1} clock periods",
    }
    contract["transaction"] = {
        "start_acceptance": "start is sampled high in S_IDLE at a rising clk edge; busy rises on that edge",
        "first_trial": "MSB trial is registered at the second rising edge after start acceptance",
        "decision": "sampled comparator applies to the currently selected trial bit",
        "bit_order": "MSB to LSB, exactly once each",
        "adc_out_commit": "adc_out receives the final decided code on the LSB sampling edge",
        "done": "done rises on the next rising edge after the final comparator sample and stays high for one clk period",
        "busy": "busy rises on start acceptance and falls on the same edge done rises",
        "start_while_busy": "ignored; conversion does not restart",
        "reset": "asynchronous active-low reset clears all controller outputs and state",
    }
    contract["provenance"] = provenance
    contract["unresolved_fields"] = [
        key for key, item in provenance.items() if item.get("status") == "unresolved"
    ]

    # Remove contradictory Clarifier assumptions for values explicitly fixed by raw text.
    assumptions[:] = [a for a in assumptions if not (
        isinstance(a, str)
        and re.search(r"SAR.*(?:resolution|settle|comparator|sampling)|(?:resolution|settle).*SAR", a, re.IGNORECASE)
    )]
    values = {
        "resolution": (resolution, resolution_source),
        "settle_cycles": (settle_cycles, settle_source),
        "comparator_high_keeps_bit": (high_keeps, cmp_source),
        "comparator_sample_event": (sample_event, sample_source),
    }
    for key, (value, source) in values.items():
        assumptions.append(f"[{source.upper()}] SAR {key}={value}.")
    if provenance["comparator.physical_relation"]["status"] == "unresolved":
        assumptions.append("[UNRESOLVED] Analog comparator transfer relation is unspecified; the digital contract consumes comparator decisions only.")

    spec["ams_contract"] = contract
    cycle_contract = (
        f"SAR ADC controller, resolution={resolution}; comparator high "
        f"{'keeps' if high_keeps else 'clears'} the current trial bit; "
        f"SETTLE_CYCLES={settle_cycles}; comparator sampled at {sample_event}. "
        "At rising edge C, start is accepted only in IDLE and busy rises. At C+2, "
        "the registered MSB trial appears on dac_code. For every trial update at "
        f"cycle N, hold that DAC code for {settle_cycles} complete clock periods "
        f"and sample the comparator at N+{settle_cycles + 1}; that sample decides "
        "the currently selected bit. Then select the next lower bit. Commit adc_out "
        "on the LSB sample edge. On the next rising edge, assert done and clear busy; "
        "hold done for exactly one period. Ignore start while busy. Active-low reset "
        "asynchronously clears all state and outputs."
    )
    spec["rtl_invariants"] = "\n".join((
        f"RTL_INVARIANT: resolution={resolution}; settle_cycles={settle_cycles}; comparator_high_keeps_bit={str(high_keeps).lower()}.",
        "RTL_INVARIANT: DAC trial update at S_SET_TRIAL; sample only at the specified event after the complete settle interval.",
        "RTL_INVARIANT: decision applies to the current bit; visit MSB through LSB once; adc_out commits on final sample; done is one cycle.",
        "RTL_INVARIANT: busy rises on accepted start, falls with done; start while busy is ignored.",
    ))
    spec["numerical_invariants"] = "The comparator decision trace must produce the exact binary-search result; bits are tested MSB-to-LSB and each decision applies to its current trial bit."
    spec["physical_assumptions"] = "Comparator resolution and CDAC settling accuracy are physical assumptions; RTL tests only the digital sample event and decision sequencing."
    spec["testbench_obligations"] = "The independent oracle must use this exact contract version/hash, resolution, settle_cycles, comparator decision mapping, and sample event. It must test busy/start, progression, settle timing, result, reset, repeat conversion, and one-cycle done."
    spec["formalized_request"] = cycle_contract
    spec["master_specification"] = cycle_contract
    spec["data_widths"] = f"dac_code and adc_out are exactly {resolution} bits wide."
    spec["clocking_scheme"] = "Single synchronous state machine clocked by clk; comparator is sampled at the contract event."
    spec["sequential_or_combinational"] = "Synchronous registered FSM; all SAR state/output updates occur on specified clock edges."
    contract["contract_hash"] = _sar_contract_hash(contract)


def _sar_contract_parameters(clarified_spec: dict) -> dict:
    contract = (clarified_spec or {}).get("ams_contract", {})
    parameters = contract.get("parameters", {}) if isinstance(contract, dict) else {}
    return {
        "resolution": int(parameters.get("resolution", 8)),
        "settle_cycles": int(parameters.get("settle_cycles", 1)),
        "comparator_high_keeps_bit": bool(parameters.get("comparator_high_keeps_bit", True)),
        "sample_event": str(contract.get("timing", {}).get("sample_event", "posedge clk")),
        "contract_version": str(contract.get("version", "sar-controller-contract-v2")),
        "contract_hash": str(contract.get("contract_hash", "")),
    }


def _specialize_sar_skeleton(skeleton: str, clarified_spec: dict) -> str:
    config = _sar_contract_parameters(clarified_spec)
    skeleton = re.sub(
        r"(parameter\s+RESOLUTION\s*=\s*)[^,\n]+",
        rf"\g<1>{config['resolution']}", skeleton, count=1,
    )
    skeleton = re.sub(
        r"(parameter\s+SETTLE_CYCLES\s*=\s*)[^,\n]+",
        rf"\g<1>{config['settle_cycles']}", skeleton, count=1,
    )
    skeleton = re.sub(
        r"(parameter\s+CMP_HIGH_KEEPS_BIT\s*=\s*)[^\n]+",
        rf"\g<1>{int(config['comparator_high_keeps_bit'])}", skeleton, count=1,
    )
    return (
        f"// VERYLITH_SAR_CONTRACT version={config['contract_version']} hash={config['contract_hash']}\n"
        + skeleton
    )


def _attach_sar_contract_identity(v_code: str, clarified_spec: dict) -> str:
    config = _sar_contract_parameters(clarified_spec)
    if re.search(r"(?m)^\s*//\s*VERYLITH_SAR_CONTRACT\b", v_code or ""):
        return v_code
    return (
        f"// VERYLITH_SAR_CONTRACT version={config['contract_version']} hash={config['contract_hash']}\n"
        + (v_code or "")
    )


def validate_sar_contract_rtl(v_code: str, clarified_spec: dict):
    """Check SAR contract metadata, public interface, and clocked implementation shape.

    This intentionally does not inspect FSM labels, internal registers, parameter
    spelling, or source ordering. The deterministic black-box oracle is the
    behavioral acceptance check.
    """
    config = _sar_contract_parameters(clarified_spec)
    if config["sample_event"] != "posedge clk":
        return False, (
            f"CONTRACT_VIOLATION: SAR path currently supports posedge clk sampling; "
            f"Spec-Sync sample_event={config['sample_event']}"
        )
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", "", v_code or "", flags=re.DOTALL)
    marker = re.search(
        r"(?m)^\s*//\s*VERYLITH_SAR_CONTRACT\s+version=([^\s]+)\s+hash=([0-9a-f]+)\s*$",
        v_code or "",
        re.IGNORECASE,
    )
    if not marker or marker.group(1) != config["contract_version"] or marker.group(2) != config["contract_hash"]:
        return False, "CONTRACT_VIOLATION: SAR RTL contract identity/version/hash does not match Spec-Sync"
    if not config["contract_hash"]:
        return False, "CONTRACT_VIOLATION: canonical SAR contract hash is missing"

    # Keep the module API and clocked nature as the only source-level contract.
    module_match = re.search(r"\bmodule\s+[A-Za-z_]\w*\b", clean)
    if not module_match:
        return False, "STRUCTURAL_ERROR: SAR module declaration is missing"
    pos = module_match.end()
    while pos < len(clean) and clean[pos].isspace():
        pos += 1
    if pos < len(clean) and clean[pos] == "#":
        pos += 1
        while pos < len(clean) and clean[pos].isspace():
            pos += 1
        if pos >= len(clean) or clean[pos] != "(":
            return False, "STRUCTURAL_ERROR: malformed SAR parameter header"
        depth = 0
        while pos < len(clean):
            depth += (clean[pos] == "(") - (clean[pos] == ")")
            pos += 1
            if depth == 0:
                break
    while pos < len(clean) and clean[pos].isspace():
        pos += 1
    if pos >= len(clean) or clean[pos] != "(":
        return False, "STRUCTURAL_ERROR: SAR public port list is missing"
    start = pos + 1
    depth = 1
    pos += 1
    while pos < len(clean) and depth:
        depth += (clean[pos] == "(") - (clean[pos] == ")")
        pos += 1
    if depth:
        return False, "STRUCTURAL_ERROR: unterminated SAR public port list"
    ports = _parse_ansi_module_ports(clean[start:pos - 1])
    expected_ports = {
        "clk": "input", "rst_n": "input", "start": "input", "comparator": "input",
        "dac_code": "output", "adc_out": "output", "busy": "output", "done": "output",
    }
    actual_ports = {item["name"].lower(): item["direction"].lower() for item in ports}
    if actual_ports != expected_ports:
        return False, "STRUCTURAL_ERROR: SAR public ports/directions differ from the canonical interface"
    if not re.search(r"\balways\s*@\s*\([^)]*\bposedge\s+clk\b", clean, re.IGNORECASE):
        return False, "STRUCTURAL_ERROR: SAR implementation has no clk-edge sequential process"
    return True, "ok"


def validate_sar_oracle_contract(tb_code: str, clarified_spec: dict):
    """Reject a SAR oracle whose declared configuration differs from Spec-Sync."""
    config = _sar_contract_parameters(clarified_spec)
    clean = re.sub(r"//[^\n]*", "", tb_code or "")
    expected = {
        "RESOLUTION": config["resolution"],
        "SETTLE_CYCLES": config["settle_cycles"],
        "CMP_HIGH_KEEPS_BIT": int(config["comparator_high_keeps_bit"]),
    }
    for name, value in expected.items():
        match = re.search(rf"\blocalparam\s+{name}\s*=\s*(\d+)\b", clean, re.IGNORECASE)
        if not match:
            return False, f"CONTRACT_ERROR: SAR oracle is missing canonical {name}={value}"
        if int(match.group(1)) != value:
            return False, f"CONTRACT_ERROR: Spec-Sync {name}={value}; oracle {name}={match.group(1)}"
    if config["contract_hash"] not in (tb_code or ""):
        return False, "CONTRACT_ERROR: SAR oracle contract hash/version does not match Spec-Sync"
    if f"contract_version={config['contract_version']}" not in (tb_code or ""):
        return False, "CONTRACT_ERROR: SAR oracle contract version does not match Spec-Sync"
    return True, "ok"


def _set_ams_parameter(
    contract: dict,
    name: str,
    value: str,
    explicit: bool,
    assumptions: list,
    default_reason: str,
) -> None:
    parameters = contract.setdefault("parameters", {})
    existing = parameters.get(name)
    if existing not in (None, "", [], {}):
        return
    parameters[name] = value
    if not explicit:
        assumptions.append(default_reason)


def _apply_ams_contract(spec: dict, raw_prompt: str, assumptions: list) -> None:
    """Add canonical, testable AMS semantics and label non-RTL assumptions."""
    spec_text = " ".join(str(value) for value in spec.values())
    combined = f"{raw_prompt or ''} {spec_text}"
    design_type = detect_design_type(spec, raw_prompt)
    if design_type not in {
        "sar_adc_controller", "spi_register_bank", "non_overlapping_clock",
    }:
        return

    existing_contract = spec.get("ams_contract")
    contract = dict(existing_contract) if isinstance(existing_contract, dict) else {}
    contract["kind"] = design_type
    spec["ams_contract"] = contract

    if design_type == "sar_adc_controller":
        _normalise_sar_contract(spec, raw_prompt, assumptions)

    elif design_type == "spi_register_bank":
        mode_match = re.search(r"(?:spi\s*)?mode\s*([0-3])", combined, re.IGNORECASE)
        mode = int(mode_match.group(1)) if mode_match else 0
        addr_w, addr_explicit = _find_ams_integer(
            combined,
            (r"address(?:\s+field)?\s*(?:width)?\s*(?:=|:|is)?\s*(\d+)\s*bits?",),
            7,
        )
        data_w, data_explicit = _find_ams_integer(
            combined,
            (r"data(?:\s+field)?\s*(?:width)?\s*(?:=|:|is)?\s*(\d+)\s*bits?",),
            16,
        )
        frame_w, frame_explicit = _find_ams_integer(
            combined,
            (r"(?:frame(?:\s+length)?|spi frame)\s*(?:=|:|is)?\s*(\d+)\s*bits?",),
            1 + addr_w + data_w,
        )
        _set_ams_parameter(contract, "spi_mode", f"mode {mode}", bool(mode_match), assumptions,
                           "SPI mode was not specified; defaulted to mode 0 (CPOL=0, CPHA=0).")
        _set_ams_parameter(contract, "address_width", f"{addr_w} bits", addr_explicit, assumptions,
                           "SPI address width was not specified; defaulted to 7 bits.")
        _set_ams_parameter(contract, "data_width", f"{data_w} bits", data_explicit, assumptions,
                           "SPI data width was not specified; defaulted to 16 bits.")
        _set_ams_parameter(contract, "frame_length", f"{frame_w} bits", frame_explicit, assumptions,
                           "SPI frame length was not specified; defaulted to [R/W][ADDR(7)][DATA(16)] = 24 bits.")
        _set_ams_parameter(contract, "bit_order", "MSB first", bool(re.search(r"(?:msb|lsb)[ -]?first", combined, re.IGNORECASE)), assumptions,
                           "SPI bit order was not specified; defaulted to MSB first.")
        _set_ams_parameter(contract, "write_encoding", "R/W=0 writes; R/W=1 reads", bool(re.search(r"r\s*/\s*w|read.*write|write.*read", combined, re.IGNORECASE)), assumptions,
                           "SPI R/W encoding was not specified; defaulted to R/W=0 write and R/W=1 read.")
        _set_ams_parameter(contract, "cdc_commit", "SCLK-domain bundled-data toggle plus destination acknowledgement toggle", False, assumptions,
                           "SPI CDC update mechanism was not specified; defaulted to a toggle plus acknowledgement protocol that holds commit data stable.")
        _append_spec_text(spec, "formalized_request", [
            "Use SPI mode 0 by default: sample MOSI on SCLK rising edges and launch MISO on SCLK falling edges. A complete accepted write stages shadow data, toggles one commit event, and updates active registers exactly once in clk only after the event crosses the CDC boundary. Incomplete CS_n-aborted frames change no shadow or active register.",
            "Canonical interface: inputs clk, rst_n, sclk, cs_n, mosi; outputs miso, active_reg0[DATA_W-1:0], active_reg1[DATA_W-1:0].",
        ])
        _append_spec_text(spec, "protocol_invariants", ["The default frame is [R/W][ADDR][DATA], MSB first. CS_n aborts an incomplete frame without resetting the design. Readback is explicitly pipelined from shadow registers in the following frame."])
        _append_spec_text(spec, "cdc_invariants", ["Every accepted commit toggle produces exactly one clk-domain active-register update.", "Commit address/data remain stable from source toggle until destination acknowledgement returns; a new write while outstanding is ignored rather than overwriting the pending payload."])
        _append_spec_text(spec, "rtl_invariants", ["Active registers change only in the clk domain and only for accepted commit events."])
        _append_spec_text(spec, "physical_assumptions", ["Bundled-data timing and CDC implementation constraints must guarantee the held payload is stable around destination capture; two-flop event synchronization alone is not a universal multi-bit bus proof."])
        _append_spec_text(spec, "testbench_obligations", ["Independently verify mode/edge timing, exact frame length, write decoding, CS_n abort, shadow-to-active atomic update, exactly-once commit delivery, readback behavior, reset, and no active change before commit."])

    else:
        dead_cycles, dead_explicit = _find_ams_integer(
            combined,
            (r"dead[ -]?time\s*(?:=|:|is)?\s*(\d+)\s*cycles?", r"dead[ -]?cycles?\s*(?:=|:|is)?\s*(\d+)"),
            1,
        )
        _set_ams_parameter(contract, "reference_clock", "clk is the already synchronous control/reference clock", bool(re.search(r"(?:reference|source)\s+clock", combined, re.IGNORECASE)), assumptions,
                           "Non-overlapping clock reference-domain relationship was not specified; defaulted to a synchronous control clock named clk.")
        _set_ams_parameter(contract, "dead_cycles", str(dead_cycles), dead_explicit, assumptions,
                           "Non-overlapping clock dead time was not specified; defaulted to one full control-clock cycle with both phases low.")
        _set_ams_parameter(contract, "enable_behavior", "enable is sampled in clk; disable forces both phases low at the next clk edge", bool(re.search(r"enable|disable", combined, re.IGNORECASE)), assumptions,
                           "Non-overlapping clock enable/disable behavior was not specified; defaulted to synchronous enable and next-edge disable.")
        _append_spec_text(spec, "formalized_request", ["Implement registered break-before-make sequencing: phi1 high for one control cycle, programmed dead cycles with both low, phi2 high for one control cycle, programmed dead cycles with both low, then repeat while enabled.", "Canonical interface: inputs clk, rst_n, enable; outputs phi1, phi2."])
        _append_spec_text(spec, "sequential_or_combinational", ["Synchronous registered FSM driven by the reference/control clk; no direct combinational complementary phase generation."])
        _append_spec_text(spec, "reset_behavior", ["Active-low reset forces phi1=0, phi2=0, dead counter=0, and the controller OFF."])
        _append_spec_text(spec, "rtl_invariants", ["!(phi1 && phi2) on every RTL cycle.", "phi1 and phi2 are registered outputs."])
        _append_spec_text(spec, "protocol_invariants", ["Each phase transition inserts exactly DEAD_CYCLES full clk cycles where both phases are low. Disabling leaves both phases low."])
        _append_spec_text(spec, "physical_assumptions", ["The logical cycle-level dead interval is sufficient after clock-tree skew, propagation delay, PVT variation, and analog-switch charge injection; RTL does not establish nanosecond physical non-overlap."])
        _append_spec_text(spec, "testbench_obligations", ["Assert !(phi1 && phi2) every simulated control cycle; verify the reset state, phase order, exact programmed dead cycles, and enable/disable behavior."])


def normalize_clarified_spec(clarified_spec: dict, raw_prompt: str) -> dict:
    """Patch common ambiguous specs before any downstream agent sees them."""
    spec = dict(clarified_spec or {})
    raw_l = (raw_prompt or "").lower()
    spec_text = " ".join(str(v) for v in spec.values() if not isinstance(v, (list, dict)))

    assumptions = spec.get("assumptions_made") or []
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    elif not isinstance(assumptions, list):
        assumptions = []

    alu_16_ambiguous = (
        "alu" in raw_l
        and re.search(r'\b(?:16|sixteen)\b', raw_l)
        and any(kw in raw_l for kw in ("operation", "operations", "opcodes", "opcode"))
        and not _has_explicit_opcode_map(raw_prompt + " " + spec_text)
    )
    if alu_16_ambiguous:
        op_table = _default_4bit_alu16_operation_table()
        assumptions = [
            a for a in assumptions
            if not re.search(
                r'opcode|operation|divide|division|modulo|undefined|error signal|e\.g\.',
                str(a),
                re.IGNORECASE,
            )
        ]
        spec["formalized_request"] = (
            "Design a 4-bit purely combinational ALU. Conventional interface: "
            "A[3:0], B[3:0], op[3:0], result[3:0], ov. The opcode map and ov "
            "semantics are fully defined by operation_table; no other opcode "
            "behavior is allowed."
        )
        spec["operation_table"] = op_table
        assumptions.append(
            "User requested a 16-operation ALU without defining the opcode map; "
            "Clarifier selected the canonical 4-bit ALU opcode table in operation_table."
        )
        assumptions.append(
            "Division and modulo by zero are defined exactly: result=0 and ov=1."
        )
        spec["overflow_behavior"] = (
            "Use the operation_table ov definition exactly for every opcode; logical "
            "operations set ov=0, division/modulo by zero set result=0 and ov=1."
        )
        spec["testbench_obligations"] = (
            "The testbench golden model must use operation_table exactly. It must not "
            "invent any alternate opcode meaning."
        )

    dataflow_requested = _requires_dataflow_modeling(raw_prompt + " " + spec_text)
    if dataflow_requested and not _clocked_or_sequential_intent(
        raw_prompt + " " + spec_text,
        clarified_spec=spec,
        requested_port_names=spec.get("_user_requested_ports", []),
    ):
        spec["modeling_style_constraints"] = (
            "Strict dataflow modeling: synthesizable DUT must use continuous assign "
            "statements/conditional expressions only. No always blocks, no procedural "
            "assignments, no output reg declarations, and no internal reg variables."
        )
        assumptions.append(
            "Strict dataflow modeling is treated as a hard structural constraint."
        )
    elif dataflow_requested:
        if _requires_dataflow_modeling(str(spec.get("modeling_style_constraints", ""))):
            spec["modeling_style_constraints"] = (
                "Clocked/sequential design: use synthesizable edge-triggered always "
                "blocks for state and continuous assigns only for simple combinational "
                "glue. Dataflow-only modeling is impossible for clock dividers, "
                "counters, FIFOs, and other stateful modules."
            )
        assumptions = [
            a for a in assumptions
            if not _requires_dataflow_modeling(str(a))
        ]
        assumptions.append(
            "Dataflow-only modeling was not enforced because the design has clocked/sequential intent."
        )

    _apply_ams_contract(spec, raw_prompt, assumptions)

    if assumptions:
        spec["assumptions_made"] = list(dict.fromkeys(str(a) for a in assumptions if str(a).strip()))

    return spec


def build_master_spec_prompt(clarified_spec: dict, design_name: str) -> str:
    """Canonical downstream prompt. Raw user text is intentionally omitted."""
    ordered_keys = [
        "formalized_request",
        "master_specification",
        "data_widths",
        "sequential_or_combinational",
        "clocking_scheme",
        "reset_behavior",
        "overflow_behavior",
        "protocol_invariants",
        "required_internal_registers",
        "operation_table",
        "truth_table",
        "modeling_style_constraints",
        "testbench_obligations",
        "rtl_invariants",
        "cdc_invariants",
        "numerical_invariants",
        "physical_assumptions",
        "ams_contract",
        "assumptions_made",
        "_user_requested_ports",
        "_detected_design_type",
    ]
    lines = [
        "MASTER SPECIFICATION DOCUMENT - AUTHORITATIVE",
        f"Target module name: {design_name}",
        "Downstream agents must implement and test this document only.",
        "The raw user prompt is intentionally not authoritative beyond this point.",
    ]
    for key in ordered_keys:
        if key not in (clarified_spec or {}):
            continue
        value = clarified_spec.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, dict)):
            value = json.dumps(value, indent=2, sort_keys=True)
        lines.append(f"\n[{key}]\n{value}")
    return "\n".join(lines)


def _text_says_combinational(text: str) -> bool:
    text = (text or "").lower()
    return any(kw in text for kw in (
        "purely combinational", "pure combinational", "combinational only",
        "clockless", "no clock", "no clocking", "no reset or clock",
        "does not use a clock", "without a clock",
    ))


def _text_says_sequential(text: str) -> bool:
    text = (text or "").lower()
    return any(kw in text for kw in (
        "posedge", "negedge", "edge-triggered", "clocked", "synchronous",
        "sequential", "state register", "state machine", "fsm", "pipeline",
        "register file", "fifo", "counter", "ram", "sram", "memory array",
    ))


def _rtl_has_edge_triggered_storage(v_file: str) -> bool:
    try:
        with open(v_file, "r") as f:
            code = f.read()
    except OSError:
        return False
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return bool(re.search(r'\balways\s*@\s*\([^)]*\b(?:posedge|negedge)\b', code, re.IGNORECASE))


def expects_sequential_storage(
    base_prompt: str,
    v_file: str = None,
    design_type: str = None,
    clarified_spec: dict = None,
) -> bool:
    """True only when 0 DFFs should be considered a synthesis failure."""
    if _should_enforce_strict_dataflow(base_prompt, clarified_spec=clarified_spec):
        return False

    spec = clarified_spec or {}
    seq_field = str(spec.get("sequential_or_combinational", ""))
    clock_field = str(spec.get("clocking_scheme", ""))
    style_field = str(spec.get("modeling_style_constraints", ""))
    formal_field = str(spec.get("formalized_request", ""))
    combined_fields = " ".join([seq_field, clock_field, style_field, formal_field])

    if _text_says_combinational(seq_field) or _text_says_combinational(style_field):
        return False
    if _text_says_combinational(combined_fields) and not _text_says_sequential(seq_field):
        return False

    if v_file and _rtl_has_edge_triggered_storage(v_file):
        return True

    requested_ports = [
        str(p).lower()
        for p in spec.get("_user_requested_ports", [])
        if isinstance(p, str)
    ]
    if any(_name_is_clock_like(p) for p in requested_ports):
        return True

    if _text_says_sequential(seq_field) or _text_says_sequential(clock_field):
        return True

    # For legacy callers without structured spec, use a conservative fallback:
    # explicit edge-sensitive language means sequential; bare "clock" in a
    # section header or "no clock" text does not.
    fallback = re.sub(r'\[[^\]]+\]', ' ', base_prompt or '')
    if _text_says_combinational(fallback):
        return False
    return bool(re.search(r'\b(posedge|negedge|always\s*@\s*\([^)]*edge)\b', fallback, re.IGNORECASE))

# ==============================================================================
# AGENT: ARCHITECT
# ==============================================================================
ARCH_STYLES = [
    "behavioral_single_always_block",
    "explicit_datapath_and_controller",
    "pipelined_registered_stages",
    "decoupled_control_and_status",
    "register_chain_optimized",
    "onehot_fsm_encoded",
    "registered_output_buffered",
]

def generate_architecture(
    spec:              dict,
    provider:          str,
    previous_failures: str = None,
    attempt_idx:       int = 0,
    forced_style:      str = None,
) -> dict:
    cache_key = get_cache_key(json.dumps(spec), previous_failures, attempt_idx, forced_style)
    if cache_key in ARCH_CACHE:
        print(f"   ⚡ Cache Hit: Reusing Micro-Architecture blueprint.")
        return ARCH_CACHE.get(cache_key)

    print(f"   📐 Routing to MICRO-ARCHITECT AGENT ({provider.upper()}) [{FAST_MODEL}]...")
    error_context = ""
    if previous_failures:
        error_context = f"""
CRITICAL FEEDBACK FROM PREVIOUS ARCHITECTURES:
{previous_failures}
Do NOT repeat the exact same datapath, pipelining, or mux structures.
"""

    style_directive = (
        f"\nYou MUST strictly adhere to the `{forced_style}` implementation paradigm.\n"
        if forced_style else ""
    )

    salt = random.randint(10000, 99999)
    system_prompt = f"""
You are an elite SoC Micro-Architect. Design a strictly synthesizable hardware architecture.
{error_context}{style_directive}

CRITICAL CLASSIFICATION:
- DATAPATH: counters, FIFOs, ALUs, datapaths, memories, shift registers, arithmetic units.
- FSM: protocol controllers, sequence detectors, handshake controllers, state machines.
A counter with overflow/underflow flags is always DATAPATH, never FSM.
- Pure combinational blocks (ALU, decoder, encoder, mux, comparator, immediate
  generator, combinational arithmetic) MUST NOT invent clock/reset ports. For
  those, set clock_and_reset to null, registers to [], memory_blocks to [], and
  describe behavior with datapath_nodes, muxes, and flag_logic only.
- Sequential DATAPATH blocks (register files, FIFOs, counters, pipelines, RAMs)
  must include their real clock/reset ports only when required by the spec or
  chosen structure. Do not add a reset port unless the interface/spec includes it
  or the skeleton requires it.
- If the spec requests strict dataflow modeling, use template_type
  "continuous_assign_dataflow", registers=[], memory_blocks=[], no clock/reset
  unless explicitly present in the interface, and describe all behavior as
  combinational equations. Do not choose a behavioral always-block architecture.
- If input JSON contains `_detected_design_type: "axi_stream_uart_tx"`, this is a
  protocol composition, not a generic datapath. Use module_class "FSM" and
  template_type "axi_stream_uart_tx_bridge". The architecture MUST include:
  state register IDLE/START/DATA/STOP, data latch, 3-bit data bit index,
  32-bit baud counter, combinational s_axis_tready=(state==IDLE), UART tx idle
  high, start bit low, 8 data bits LSB first, stop bit high. The handshake
  capture occurs only in IDLE when s_axis_tvalid and s_axis_tready are true.
- If input JSON contains `_detected_design_type: "hierarchical_top"`, this is a
  top-level integration/wrapper. Use module_class "DATAPATH" and template_type
  "hierarchical_integration". Do NOT classify it as a pure ALU/decoder just
  because those submodules are named. Include every top-level port exactly,
  including clk/reset if present. Treat named submodules as external verified
  dependencies to instantiate and wire; do not inline or redesign their internals.
  For hierarchical_top, memory_blocks means only RAM arrays physically declared
   inside the top module. Verified child memories such as SRAM/data_memory
   instances are external submodules, not local memory_blocks.
- If input JSON contains `_detected_design_type: "sar_adc_controller"`, use a
  synchronous registered controller architecture. Preserve the canonical sequence
  S_IDLE -> S_INIT -> S_SET_TRIAL -> S_SETTLE -> S_DECIDE -> S_DONE. List state,
  bit_index, settle_count, trial_code, dac_code, adc_out, busy, and done. The
  comparator is sampled only in S_DECIDE after the configured settle interval;
  do not describe analog settling as an RTL-proven condition.
- If input JSON contains `_detected_design_type: "spi_register_bank"`, use a
  two-domain architecture with mode-specific SCLK sample/shift behavior, shadow
  registers in the SCLK domain, active registers in the clk domain, and a bundled-
  data commit toggle plus acknowledgement. Never model an update as a one-cycle
  pulse that may be missed, and never call a two-flop synchronizer by itself a
  complete multi-bit bus CDC solution.
- If input JSON contains `_detected_design_type: "non_overlapping_clock"`, use a
  registered FSM with OFF, PHI1, DEAD_12, PHI2, and DEAD_21 phases. `clk` is the
  synchronous reference/control clock. The architecture must make !(phi1 && phi2)
  a logical RTL invariant and keep physical dead-time/skew claims in assumptions.

You MUST list ALL ports from the specification in port_interface.
If the input JSON contains `_user_requested_ports`, the port_interface names
MUST match that list exactly. Do not rename resetn to rst_n, data_in to tx_data,
or busy to ready unless the user used those exact names.

ALGORITHM DESCRIPTION REQUIREMENT:
Describe the behavioral algorithm at pseudocode level. For every register, give its exact
next-state logic. For every output flag, give its exact combinational condition.
Use `control_logic`, `flag_logic`, and `priority_order` fields.

Return pure JSON exactly matching this schema:
{{
    "module_class": "DATAPATH",
    "template_type": "counter",
    "implementation_style": "{forced_style or 'behavioral_single_always_block'}",
    "clock_and_reset": {{"clock": "clk", "reset": "rst_n", "active_low": true}},
    "port_interface": [{{"name": "clk", "direction": "input", "width": 1}}],
    "registers": [{{"name": "count", "width": 8, "reset_value": "0"}}],
    "memory_blocks": [],
    "datapath_nodes": [
        {{"op": "+", "dst": "next_count", "a": "count", "b": "1"}}
    ],
    "muxes": [
        {{
            "dst": "count_next_final",
            "control_conditions": [
                {{"when": "load_en", "route": "load_data"}},
                {{"default": "count"}}
            ]
        }}
    ],
    "control_logic": [
        {{
            "register": "count",
            "algorithm": "if load_en → count = load_data; else if enable and up_down → count = count + 1; else if enable and !up_down → count = count - 1; else count unchanged"
        }}
    ],
    "flag_logic": [
        {{"signal": "overflow", "type": "combinational", "condition": "count == 255 AND enable AND up_down"}}
    ],
    "priority_order": ["rst_n (async, active-low)", "load_en", "enable"],
    "fsm_specific": {{"state_encoding": {{}}, "next_state_logic": []}}
}}

For pure combinational modules, use:
  "clock_and_reset": null,
  "registers": [],
  "memory_blocks": [],
  "priority_order": ["combinational decode"]

[VARIATION SALT: {salt}]
"""
    result = run_llm(json.dumps(spec), system_prompt, provider,
                     cand_idx=attempt_idx, force_fast_model=True, silent=True)
    if result:
        ARCH_CACHE.put(cache_key, result)
    return result

def validate_architecture(
    arch: dict,
    expected_min_ports: int = 3,
    expected_port_names: list = None,
    clarified_spec: dict = None,
    user_prompt: str = "",
) -> bool:
    ok, _reason = validate_architecture_with_reason(
        arch,
        expected_min_ports=expected_min_ports,
        expected_port_names=expected_port_names,
        clarified_spec=clarified_spec,
        user_prompt=user_prompt,
    )
    return ok


def validate_architecture_with_reason(
    arch: dict,
    expected_min_ports: int = 3,
    expected_port_names: list = None,
    clarified_spec: dict = None,
    user_prompt: str = "",
) -> tuple:
    if not arch or "module_class" not in arch:
        return False, "missing module_class"
    ports = arch.get("port_interface", [])
    exact_ports_required = expected_port_names is not None and len(expected_port_names) > 0
    if exact_ports_required:
        if not isinstance(ports, list):
            return False, "port_interface is not a list"
    elif not ports or len(ports) < expected_min_ports:
        return False, f"expected at least {expected_min_ports} ports, got {len(ports) if isinstance(ports, list) else 'non-list'}"
    port_names = [p.get("name", "").lower() for p in ports]
    if any((not n) or (n in PORT_NAME_STOPWORDS and n not in {"a", "b", "x", "y"}) for n in port_names):
        return False, f"invalid/descriptive port name in {port_names}"
    if exact_ports_required:
        actual = {p.get("name", "") for p in ports}
        expected = set(expected_port_names)
        if actual != expected:
            return False, f"port mismatch expected={sorted(expected)} actual={sorted(actual)}"

    has_clk = any("clk" in n or "clock" in n for n in port_names)
    has_rst = any("rst" in n or "reset" in n for n in port_names)
    cr = arch.get("clock_and_reset") or {}
    cr_clock = str(cr.get("clock", "")).strip() if isinstance(cr, dict) else ""
    cr_reset = str(cr.get("reset", "")).strip() if isinstance(cr, dict) else ""

    spec_text = " ".join(
        str((clarified_spec or {}).get(k, ""))
        for k in (
            "formalized_request", "sequential_or_combinational",
            "clocking_scheme", "protocol_invariants",
        )
    )
    combined = (spec_text + " " + user_prompt).lower()
    expected_names_l = {n.lower() for n in (expected_port_names or [])}
    expected_has_clk = any("clk" in n or "clock" in n for n in expected_names_l)
    expected_has_rst = any("rst" in n or "reset" in n for n in expected_names_l)
    hierarchical_hint = (
        (clarified_spec or {}).get("_detected_design_type") in HIERARCHICAL_TYPES
        or looks_like_hierarchical_integration(clarified_spec or {}, user_prompt)
    )

    combinational_hint = any(kw in combined for kw in (
        "purely combinational", "combinational", "clockless", "no clock",
        "alu", "decoder", "encoder", "mux", "multiplexer", "comparator",
        "boolean", "priority encoder",
    ))
    if hierarchical_hint:
        combinational_hint = False
    sequential_hint = any(kw in combined for kw in (
        "clock", "posedge", "negedge", "reset", "rst", "sequential",
        "synchronous", "state machine", "fsm", "fifo", "register file",
        "pipeline", "valid", "ready",
    )) or hierarchical_hint
    arch_has_state = (
        arch.get("module_class") == "FSM"
        or bool(arch.get("memory_blocks"))
        or bool(arch.get("fsm_specific", {}).get("states"))
        or bool(arch.get("fsm_specific", {}).get("state_encoding"))
    )
    arch_registers = [
        str(r.get("name", "")).lower()
        for r in arch.get("registers", [])
        if isinstance(r, dict)
    ]
    arch_has_registers = bool(arch_registers)

    if cr_clock and not has_clk:
        return False, f"clock_and_reset names clock {cr_clock!r}, but no clock-like port exists"
    if cr_reset and not has_rst:
        return False, f"clock_and_reset names reset {cr_reset!r}, but no reset-like port exists"

    sequential_required = (
        expected_has_clk
        or arch_has_state
        or (arch_has_registers and not combinational_hint)
        or (sequential_hint and not combinational_hint)
    )
    if sequential_required and not has_clk:
        return False, "sequential architecture/spec requires a clock-like port"
    if expected_has_rst and not has_rst:
        return False, "requested ports include reset, but blueprint omitted a reset-like port"
    if arch["module_class"] == "FSM":
        fsm = arch.get("fsm_specific", {})
        # Only hard-reject if BOTH fields are completely absent (empty dict and empty list)
        # A minimal FSM with just state names but no full transition table is still usable.
        se = fsm.get("state_encoding", {})
        nl = fsm.get("next_state_logic", [])
        if not se and not nl and not fsm.get("states"):
            return False, "FSM blueprint missing states/state_encoding/next_state_logic"
    return True, "ok"

# ==============================================================================
# AGENT: REVIEWER
# ==============================================================================
def review_hardware(v_code: str, provider: str) -> dict:
    system_prompt = """
You are an elite Silicon Verification Engineer.
Analyze this Verilog RTL strictly for:
  1. Inferred latches (missing defaults in combinational always blocks)
  2. Missing reset conditions on sequential registers
  3. Combinational feedback loops

CRITICAL RULES — violating any of these will cause the patch to be rejected:
  - DO NOT rewrite the module. Output only a minimal patch.
  - NEVER convert assign statements to reg assignments.
  - NEVER add a new always block. Fix latches by adding defaults INSIDE existing always blocks.
  - NEVER drive a register in an always @(*) block if it is already driven in an always @(posedge clk) block.
    This creates BLKANDNBLK conflicts that make the design unsynthesizable.
  - If the code is clean, return {"status": "PASSED", "fixed_code": ""}.
  - If patching, return {"status": "REJECTED", "fixed_code": "<full corrected module>"}.

CORRECT latch fix pattern (add default at top of existing block):
  always @(*) begin
    out = 1'b0;        // ← add this default line
    if (cond) out = 1'b1;
  end

WRONG pattern (NEVER do this):
  always @(*) begin    // ← NEVER add a new always block to fix a latch
    reg_driven_by_posedge_block = 0;
  end

Return pure JSON: {"status": "...", "fixed_code": "..."}
"""
    return run_llm(v_code, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)



# ==============================================================================
# AGENT: RTL GENERATOR
# ==============================================================================
def generate_rtl(
    prompt:          str,
    provider:        str,
    design_name:     str,
    architecture:    dict,
    clarified_spec:  dict  = None,
    previous_ports:  str   = None,
    previous_error:  str   = None,
    best_v_code:     str   = None,
    cand_idx:        int   = 0,
    design_type_hint: str  = None,   # passed from main loop to avoid re-detection
) -> dict:
    # Only cache fresh (non-mutation) calls. Mutations are intentionally unique —
    # caching them defeats the entire purpose of the evolution engine and causes
    # the generator to loop on identical candidates after the first failure.
    is_mutation = bool(previous_error and best_v_code)
    cache_key   = get_cache_key(prompt, architecture, cand_idx) if not is_mutation else None
    if cache_key and cache_key in RTL_CACHE:
        print(f"      ⚡ Cache Hit: Reusing RTL for Thread {cand_idx+1}.")
        return RTL_CACHE.get(cache_key)

    design_type = design_type_hint or detect_design_type(clarified_spec or {}, prompt)

    port_rule = (
        f"2. STRICT INTERFACE LOCK: You MUST reuse this exact port list verbatim:\n"
        f"   {previous_ports}"
        if previous_ports else
        "2. STRICT INTERFACE: Port declarations MUST perfectly match the Architect's `port_interface` array."
    )

    error_context = ""
    insights = get_tooling_insights(design_type)
    insights_section = (
        f"\n[LESSONS LEARNED FROM PAST RUNS - FOLLOW STRICTLY]:\n{insights}\n"
        if insights else ""
    )
    if previous_error and best_v_code:
        failure_type = classify_failure(previous_error)
        delta        = build_delta_mutation(best_v_code, failure_type, previous_error)
        delta_str    = f"\n{delta}\n" if delta else ""

        # Design-type-specific mutation guidance — only injected when relevant.
        # For bridge/adapter designs (design_type=="generic"), also check if the
        # prompt mentions a known peripheral — inject protocol hints without forcing
        # the skeleton structure.
        _dtype = design_type  # captured from detect above
        _prompt_lower = prompt.lower()
        _dsrules = ""
        if _dtype in {"uart_tx", "axi_stream_uart_tx"} or ("uart" in _prompt_lower and _dtype == "generic"):
            _dsrules = """UART TX PROTOCOL RULES (applies to any UART transmitter, including AXI bridges):
- FRAME COUNTER: A UART frame has 10 phases (start + 8 data + stop).
  Use a 4-bit counter reg [3:0] phase to count 0-9.
  Use a separate 3-bit index or slice when indexing the 8-bit data register.
- BAUD COUNTER: Use reg [31:0] baud_counter for any unknown TICKS_PER_BAUD parameter.
  Compare as: if (baud_counter == TICKS_PER_BAUD - 1).
- STATE TRANSITIONS: In STOP state, count FULL baud period before returning to IDLE.
  STOP→IDLE only when baud_counter == TICKS_PER_BAUD - 1 (or equivalent).
- READY SIGNAL — THIS IS THE #1 AXI-UART BUG. Read carefully:
  The correct architecture drives s_axis_tready COMBINATIONALLY from state:
    assign s_axis_tready = (state == IDLE);
  This guarantees ready drops the SAME cycle state leaves IDLE — no extra cycle.

  WRONG pattern (LLM most common mistake — registered busy flag):
    reg busy;
    assign s_axis_tready = !busy;
    always @(posedge clk) begin
      if (tvalid && tready) begin busy <= 1; end  // busy goes high NEXT cycle
    end
    // Result: tready stays 1 for one extra cycle → testbench FAILS

  CORRECT pattern (combinational from state):
    localparam IDLE = 2'd0, START = 2'd1, ...;
    reg [1:0] state;
    assign s_axis_tready = (state == IDLE);  // drops same cycle state changes
    always @(posedge aclk or negedge aresetn) begin
      if (!aresetn) state <= IDLE;
      else case (state)
        IDLE: if (s_axis_tvalid) begin data_reg <= s_axis_tdata; state <= START; end
        ...
      endcase
    end

  Reset block still needs: state <= IDLE (so assign gives tready=1 after reset).
  Do NOT have both an assign and a reg assignment for the same ready signal.
"""
        elif _dtype == "skid_buffer":
            _dsrules = """SKID BUFFER SPECIFIC RULES:
- s_ready MUST be: assign s_ready = !buf_valid || m_ready;
- Drain path priority: buf_valid check BEFORE s_valid check inside if(m_ready||!m_valid).
- Push path: write to buf_data/buf_valid ONLY — never overwrite m_data in push path.
"""

        error_context = f"""
MUTATION REQUIRED — previous RTL failed.
FAILURE CLASSIFICATION: {failure_type}
SIMULATION OUTPUT:
{previous_error}
{delta_str}
TARGETED FIX RULES based on failure type:
- SHIFT_LOGIC            → check concatenation direction: left shift LSB-in is
                           out <= {{out[6:0], serial_in}}; right shift MSB-in is
                           out <= {{serial_in, out[7:1]}}; verify matches spec.
- INCREMENT_LOGIC        → modify only the increment counting path
- DECREMENT_LOGIC        → modify only the decrement counting path
- OFF_BY_ONE             → check boundary conditions (< vs <=) and initialization
- OVERFLOW_FLAG          → fix only the overflow flag assign statement
- UNDERFLOW_FLAG         → fix only the underflow flag assign statement
- RESET_AFTER_TX         → FSM did not return to IDLE after async reset.
                           Every output register and state register MUST be explicitly
                           assigned in the if (!rst_n) block. Never rely on state=0
                           or any implicit default — every reg needs an explicit reset value.
                           Check: is the output register (e.g. tx_ready, s_axis_tready,
                           or equivalent "ready" signal) assigned in the reset block?
- RESET_LOGIC            → fix only the reset branch in always @(posedge clk or negedge rst_n)
- LOAD_LOGIC             → fix only the load_en priority branch
- PRIORITY_ORDER_BUG     → reorder if/else statements to match spec priority
- PROTOCOL_ASSERTION_FAILED → a handshake invariant was violated.
                           Read the FAIL message to identify which signal is wrong.


                           "ready not deasserted immediately after data capture" / tready=1 when it should be 0:
                           ready must go LOW in the SAME clock cycle as data capture.
                           The handshake fires at posedge when tvalid=1 AND tready=1.
                           In that SAME always block, you MUST set tready<=0.
                           WRONG (ready stays high one extra cycle — second word captured):
                             if (s_axis_tvalid) begin data<=tdata; state<=START; end
                           CORRECT (ready deasserts same cycle as capture):
                             if (s_axis_tvalid && s_axis_tready) begin
                               data<=s_axis_tdata; s_axis_tready<=0; state<=START;
                             end

                           "ready not reasserted after frame" / "tready=0" after transmission:
                           The ready signal (s_axis_tready, tx_ready, etc.) was never set back
                           to 1 after the UART frame completed. Check THREE places:
                           (a) Reset block: ready must be initialized to 1 (device starts ready)
                           (b) IDLE state assignment: always set ready=1 in IDLE state or
                               when entering IDLE — never rely on it staying 1 from before.
                           (c) State transition to IDLE: the cycle that transitions STOP→IDLE
                               must also execute ready<=1 in the same clock edge.
                           Correct pattern:
                             STOP: if (baud_done) begin state<=IDLE; s_axis_tready<=1'b1; end
                           Wrong pattern (forgets ready):
                             STOP: if (baud_done) state<=IDLE;

                           "valid never drops" / "data mutated while valid":
                           (1) valid must stay asserted until ready is seen;
                           (2) data must not change while valid=1 and ready=0.
- LOGIC_MISMATCH         → read the expected vs got values carefully.
                           If the failing signal is identified in the log, trace it back
                           to its driver in the RTL and fix only that assignment.
                           For ready/valid pipeline designs where the RTL contains signals
                           named s_ready, m_ready, s_valid, m_valid: verify s_ready is
                           driven as: assign s_ready = !buffer_valid || m_ready;
                           (NOT assign s_ready = m_ready — that omits the buffered case)
                           Only apply this rule if those signal names are present.
- UNDRIVEN_OUTPUT        → an output port reads as 'z' (high-impedance) meaning it has
                           no driver. Fix: declare it as 'output reg' and add an explicit
                           assignment in the reset block and in the sequential logic.
                           Example: 'output reg m_valid' with 'm_valid <= 1\'b0' in reset.
- HANDSHAKE_LOGIC        → a ready/valid handshake signal has the wrong value.
                           THREE distinct sub-bugs — identify which matches the log:

                           SUB-BUG A: s_ready wrong
                           MANDATORY: assign s_ready = !buf_valid || m_ready;
                           Not !buf_valid alone (drops m_ready).
                           Not m_ready alone (loses buffer term).

                           SUB-BUG B: buffered data not appearing (m_valid=0 after backpressure)
                           Log: "expected buffered ... m_valid=0"
                           Drain path MUST be:
                             if (m_ready || !m_valid) begin
                               if (buf_valid) begin
                                 m_valid <= 1'b1; m_data <= buf_data; buf_valid <= 1'b0;
                               end else if (s_valid) begin
                                 m_valid <= 1'b1; m_data <= s_data;
                               end else begin m_valid <= 1'b0; end
                             end
                           Common mistake: setting m_data without m_valid<=1'b1.

                           SUB-BUG C: m_data mutated under backpressure
                           Log: "ASSERTION FAILED: m_data mutated while m_valid=1 m_ready=0"
                           Push path must write to buf_data ONLY, never m_data:
                             if (s_valid && s_ready && !(m_ready || !m_valid)) begin
                               buf_valid <= 1'b1; buf_data <= s_data;
                             end
- FSM_OUTPUT_WRONG       → a state machine output has the wrong value.
                           General approach:
                           (1) Find which state the FSM should be in when the output is wrong.
                           (2) Check that state's output assignment — is the value correct?
                           (3) Check that the FSM actually transitions into that state.
                               Missing or wrong transition condition → FSM stays in wrong state.
                           (4) Check the baud/timing counter: must count 0 to N-1 before
                               transitioning. Off-by-one means transitioning one cycle early/late.
                           (5) Check state register update: ensure state<=next_state fires
                               unconditionally in the sequential always block.
                           (6) Verify every output register has an explicit value in every
                               state — no undriven paths that rely on previous state value.
- BIT_EXTRACTION         → array index is wider than the array requires.
                           Verilator: "Bit extraction of var[N:0] requires M bit index, not K bits"
                           means your index variable is K bits but the array only needs M bits.

                           UART/SERIAL FRAME COUNTER NOTE — this is the most common cause:
                           A UART frame has 10 phases (start + 8 data + stop), so the frame
                           counter needs 4 bits (reg [3:0]) to count 0-9.
                           But the DATA array is only 8 bits, so indexing it needs 3 bits.
                           SOLUTION: Use a 4-bit frame counter but slice it when indexing data:
                             reg [3:0] phase;           // counts 0-9: start(0) data(1-8) stop(9)
                             tx_out <= tx_data[phase[2:0] - 1]; // slice for 8-bit array index
                           OR use a separate 3-bit data index alongside the 4-bit frame counter.
                           NEVER use a 3-bit counter for a 10-phase UART frame (max value 7 < 9).

                           General fixes (non-UART):
                           Option A — Narrow the counter to match array size:
                             reg [2:0] bit_counter;   // for 8-element array (indices 0-7)
                           Option B — Slice at point of use (keeps counter wide for other uses):
                             assign tx = shift_reg[bit_counter[2:0]];  // slice to 3 bits at use
                           Look at the Verilator error line — it shows exactly where the mismatch is.
- WIDTH_MISMATCH         → signal has wrong bit width for its context.
                           RULE 1 — INDEX WIDTH: To index into a reg[N-1:0] array, the index
                               variable must be exactly $clog2(N) bits wide.
                               reg [7:0] data needs a [2:0] index (3 bits, values 0-7).
                               reg [15:0] data needs a [3:0] index (4 bits, values 0-15).
                               Check the Verilator error: "requires X bit index, not Y bits"
                               means your index is Y bits but needs to be X bits.
                           RULE 2 — COMPARISON WIDTH: Both sides of == must have matching widths.
                               reg [2:0] counter == 3'd7   ✓ (both 3 bits)
                               reg [2:0] counter == 4'd7   ✗ (3 vs 4 bits — WIDTH error)
                               NEVER value-overflow: 3-bit register can hold 0-7 maximum.
                               Comparing reg[2:0] against 8 or higher is always wrong.
                               Fix: widen the register OR reduce the comparison value.
                           RULE 3 — LITERAL FORMAT: Never combine number prefixes.
                               3'd2 or 2'b10 or 2'h2 — pick ONE format per literal.
                               3'd2'b10 is illegal syntax (two prefixes concatenated).
                           RULE 4 — PARAMETER COMPARISONS: comparing against a parameter is safe
                               because Verilog auto-widens for parameter comparisons.
                               if (counter == PARAM - 1) is legal regardless of counter width.
- X_PROPAGATION_BUG      → explicitly initialize the signal in the reset block
- COMBINATIONAL_LOOP     → break the cyclic combinational dependency
- RTL_SYNTAX_ERROR       → fix illegal Verilog in the DUT source only. Common causes:
                           declarations inside begin/end blocks, malformed literals,
                           or SystemVerilog-only syntax in synthesizable RTL.
- TB_SYNTAX_ERROR        → do not mutate RTL; regenerate/repair the testbench only.
- SEQUENTIAL_OPTIMIZED_AWAY → Yosys removed all registers. Ensure registered state
                           or data registers directly affect output ports, reset and
                           transition logic are reachable, and the sequential always
                           block is not dead code.
- YOSYS_TIMEOUT          → synthesis shape exploded. If this design contains memory,
                           use a synchronous SRAM/BRAM inference idiom: no async
                           memory reads, no memory reset loop, no combinational
                           read-before-write merge. Use per-byte writes inside the
                           posedge block.
- MEMORY_NOT_INFERRED    → memory mapped to registers/muxes instead of a memory cell.
                           Restructure to a synchronous memory template.
- TB_TIMEOUT             → check for combinational loops or missing $finish paths
- TB_STRUCTURAL          → fix RTL defensively (clean interface, all outputs driven)

You MUST ONLY modify logic related to the failure type. Do not touch unrelated blocks.
{_dsrules}PREVIOUS BEST VERILOG (mutate this, do not rewrite from scratch):
{best_v_code}
"""
    elif previous_error:
        error_context = (
            f"SYNTAX ERROR IN PREVIOUS ATTEMPT:\n{previous_error}\n"
            "Fix only the line causing this error. Do not restructure the module."
        )

    # Skeleton injection from external library (skeletons.py)
    design_type = design_type_hint or detect_design_type(clarified_spec or {}, prompt)
    # Infer width from architecture if available
    width = 8
    for reg in architecture.get("registers", []):
        if reg.get("width"):
            try:
                width = int(reg["width"])
                break
            except (ValueError, TypeError):
                pass

    skeleton = get_skeleton(design_type, design_name, width)
    if skeleton and design_type == "sar_adc_controller":
        skeleton = _specialize_sar_skeleton(skeleton, clarified_spec or {})
        sar_config = _sar_contract_parameters(clarified_spec or {})
        skeleton = (
            f"// CANONICAL SAR CONTRACT: version={sar_config['contract_version']} "
            f"hash={sar_config['contract_hash']} resolution={sar_config['resolution']} "
            f"settle_cycles={sar_config['settle_cycles']} "
            f"comparator_high_keeps_bit={int(sar_config['comparator_high_keeps_bit'])}\n"
            + skeleton
        )
    skeleton_section = ""
    if skeleton and design_type == "sar_adc_controller":
        skeleton_section = f"""
PREFERRED SAR REFERENCE IMPLEMENTATION — advisory, not a required source layout.
Use its algorithm as guidance, but you may choose different state names, internal
signals, state encodings, reset coding, or FSM organization. Do not reproduce the
reference merely to satisfy source-pattern checks. The canonical SAR contract below
is authoritative; the independent black-box oracle determines cycle-level behavior.
Preserve the exact public interface, MSB-to-LSB trial behavior, comparator convention,
settling interval, reset/start/busy/done semantics, and contract identity comment.

{skeleton}

"""
    elif skeleton and not is_mutation:
        skeleton_section = f"""
STRUCTURAL SKELETON — you MUST use this as your starting point.
This skeleton encodes invariants that are FROZEN and must not be changed:
the sensitivity list, reset structure, non-blocking discipline, and (for
skid buffers) the s_ready equation and drain/push priority.
Only adapt signal widths, port names, and behavioral branches to match the spec.
DO NOT remove reset initializations. DO NOT introduce blocking (=) assignments
inside always @(posedge clk) blocks.

{skeleton}

"""
    elif skeleton and is_mutation and design_type == "sar_adc_controller":
        skeleton_section = f"""
REFERENCE SAR IMPLEMENTATION — advisory only; source layout is not part of the
contract. Preserve the canonical behavior, interface, and contract identity while
making only a targeted fix where possible.
{skeleton}
"""
    elif skeleton and is_mutation:
        skeleton_section = f"""
REFERENCE SKELETON for {design_type} — compare your mutation against this.
Ensure the frozen structural patterns are preserved:
{skeleton}

"""
    protocol_section = ""
    if design_type == "axi_stream_uart_tx":
        protocol_section = """
PROTOCOL COMPOSITION TEMPLATE - AXI4-Stream sink to UART TX:
- This is a controller/serializer FSM, not a pure datapath.
- Required states: IDLE, START, DATA, STOP.
- Required registers: state, data_reg[7:0], bit_index[2:0], baud_counter[31:0].
- Reset: state=IDLE, tx=1, baud_counter=0, bit_index=0, data_reg=0.
- s_axis_tready must be combinational from the IDLE state:
    assign s_axis_tready = (state == IDLE);
  Do not also assign s_axis_tready in an always block.
- IDLE: tx must be 1. If s_axis_tvalid is 1, latch s_axis_tdata, reset counters,
  and enter START. Since ready is state==IDLE, this is the AXI handshake.
- START: tx must be 0 for exactly TICKS_PER_BAUD clocks.
- DATA: tx must be data_reg[bit_index], LSB first, each bit for exactly
  TICKS_PER_BAUD clocks. Only increment bit_index after a full baud cell.
- STOP: tx must be 1 for exactly TICKS_PER_BAUD clocks, then return to IDLE.
- Ignore/hold off any new AXI byte while not IDLE by keeping s_axis_tready low.
"""
    hierarchy_section = ""
    if design_type in HIERARCHICAL_TYPES or looks_like_hierarchical_integration(clarified_spec or {}, prompt):
        signature_block = format_registered_module_signatures(prompt)
        if signature_block:
            signature_block = "\n" + signature_block + "\n"
        hierarchy_section = f"""
HIERARCHICAL INTEGRATION RULES:
- Generate ONLY the requested top module `{design_name}`.
- Instantiate requested verified submodules by their exact module names.
- Do NOT paste, rewrite, or redefine submodule internals in this file.
- Use named port connections: .port_name(signal_name). Never rely on positional ports.
- Connect every port shown in the registered submodule interface. If an output is
  unused, connect it to a named dummy wire. Do not omit submodule ports.
- Prefer connecting unused submodule outputs to named dummy wires rather than .port().
  Example: wire alu_zero_unused; rv32i_alu u_alu (..., .zero(alu_zero_unused));
- The compiler/linker will provide registered dependency sources from workspace/registry.json.
- If memory is provided by a verified submodule, instantiate that submodule and
  wire its ports. Do not declare a local reg-array copy of that memory in the wrapper.
- If an instruction asks for a small local decode/mux block in the top, implement only
  that top-level glue logic with Verilog-2001 combinational always @(*) or assign statements.
- Do not add local edge-triggered always blocks, reset flops, or address pipeline
  registers in a wrapper unless the spec explicitly requests new top-level state.
  Use wires/assign statements for address translation, chip selects, and muxing.
{signature_block}"""
    style_section = ""
    if _should_enforce_strict_dataflow(prompt, architecture, clarified_spec):
        style_section = """
STRICT DATAFLOW MODELING RULES:
- Implement the DUT using continuous assign statements and conditional operators only.
- Do NOT use always blocks, initial blocks, tasks, functions, procedural assignments,
  output reg declarations, or internal reg variables.
- Declare outputs as wire-style outputs, compute intermediate wires with assign,
  and use assign result = ... / assign flag = ... for final outputs.
"""

    alg_sections = []

    if architecture.get("control_logic"):
        lines = [
            f"  Register '{cl.get('register')}': {cl.get('algorithm')}"
            for cl in architecture["control_logic"]
        ]
        alg_sections.append(
            "REGISTER NEXT-STATE ALGORITHMS (transcribe these exactly):\n" + "\n".join(lines)
        )

    if architecture.get("flag_logic"):
        lines = [
            f"  Signal '{fl.get('signal')}' [{fl.get('type','combinational')}]: "
            f"assert when {fl.get('condition')}"
            for fl in architecture["flag_logic"]
        ]
        alg_sections.append(
            "FLAG LOGIC (combinational → assign; sequential → register):\n" + "\n".join(lines)
        )

    if architecture.get("priority_order"):
        alg_sections.append(
            "CONTROL PRIORITY (implement as nested if-else, highest first):\n"
            + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(architecture["priority_order"]))
        )

    if architecture.get("datapath_nodes"):
        lines = [
            f"  Node '{n.get('dst')}': {n.get('a')} {n.get('op')} {n.get('b')}"
            for n in architecture["datapath_nodes"]
        ]
        alg_sections.append(
            "DATAPATH NODES (implement as continuous assigns or combinational blocks):\n"
            + "\n".join(lines)
        )

    if architecture.get("muxes"):
        lines = []
        for m in architecture["muxes"]:
            lines.append(f"  MUX Target '{m.get('dst', '?')}':")
            for cond in m.get("control_conditions", []):
                if "default" in cond:
                    lines.append(f"    else → {cond['default']}")
                else:
                    lines.append(f"    if ({cond.get('when')}) → {cond.get('route')}")
        alg_sections.append(
            "MULTIPLEXERS & ROUTING (implement as prioritized if/else):\n" + "\n".join(lines)
        )

    if architecture.get("memory_blocks"):
        if design_type in HIERARCHICAL_TYPES:
            alg_sections.append(
                "HIERARCHICAL MEMORY INSTANCE PLAN:\n  "
                "Treat memory_blocks as local RAM declarations only when the top module itself "
                "must own storage. For wrapper/integration designs, named memories are usually "
                "verified submodule instances to instantiate and wire, not reg arrays to redeclare.\n  "
                + str(architecture["memory_blocks"])
            )
        else:
            alg_sections.append(
                "MEMORY ARCHITECTURE (implement RAM/FIFO using provided pointers):\n  "
                + str(architecture["memory_blocks"])
            )

    if architecture.get("module_class") == "FSM":
        alg_sections.append(
            "FSM STRUCTURE: Use case(state) for state transitions. "
            "Cleanly separate state_reg and next_state."
        )

    algorithm_section = ""
    if alg_sections:
        algorithm_section = (
            "\nALGORITHM DESCRIPTION — implement by transcription, no invention:\n"
            + "\n\n".join(alg_sections) + "\n"
        )

    rtl_salt = random.randint(10000, 99999)

    system_prompt = f"""
You are an elite RTL design engineer. Generate ONLY the synthesizable Verilog RTL module.
Do NOT generate a testbench — a separate agent handles that.
The user message is a MASTER SPECIFICATION emitted by the Clarifier Agent. Treat
it as authoritative and do not infer alternate opcode maps, truth tables, flag
semantics, or protocol behavior.
{insights_section}{error_context}{skeleton_section}{protocol_section}{hierarchy_section}{style_section}{algorithm_section}
Implement exactly this architecture:
{json.dumps(architecture, indent=2)}

SILICON RULES — all mandatory:
1. Top module named exactly: `{design_name}`
{port_rule}
3. Verilog-2001 only. No SystemVerilog. No initial blocks in synthesizable modules.
4. Sequential always @(posedge clk): non-blocking (<=) ONLY.
   Combinational always @(*): blocking (=) ONLY.
   NEVER mix blocking and non-blocking on the same variable — BLKANDNBLK is fatal.

   WRONG (causes BLKANDNBLK — never do this):
     always @(posedge clk) begin state = next_state; end  // blocking in sequential
     always @(*) begin state <= IDLE; end                  // nonblocking in combinational
     always @(posedge clk) begin count <= count + 1; end
     always @(*) begin count = 0; end                      // second driver on count

   CORRECT (one driver per register, always in sequential block):
     always @(posedge clk or negedge rst_n) begin
         if (!rst_n) state <= IDLE;
         else        state <= next_state;
     end

4b. COUNTER AND REGISTER WIDTH RULES — mandatory for Verilator compatibility:
   When a counter is compared against a module parameter (e.g. TICKS_PER_BAUD),
   declare the counter wide enough to hold the parameter's maximum value.
   Use 32-bit counters for parameters whose size is unknown at design time:
     reg [31:0] baud_counter;   // safe for any TICKS_PER_BAUD value

   NEVER use explicitly-sized small literals for wide registers:
   WRONG:  baud_counter <= 16'd0;   // 16-bit literal into 32-bit reg → WIDTH error
   WRONG:  baud_counter + 16'd1     // 16-bit addend with 32-bit reg → WIDTH error
   CORRECT: baud_counter <= 0;       // unsized — Verilog auto-sizes to match LHS
   CORRECT: baud_counter <= baud_counter + 1;  // unsized integer constant
   CORRECT: baud_counter <= 'd0;    // explicitly unsized decimal

   The same applies to wires computed from wide registers:
   WRONG:  wire [15:0] baud_next = baud_counter + 16'd1;  // width mismatch
   CORRECT: wire [31:0] baud_next = baud_counter + 1;      // matches reg width

   PARAMETER AND LOCALPARAM WIDTH: NEVER give parameters explicit bit widths.
   WRONG:  parameter [15:0] TICKS_PER_BAUD = 434;  // 16-bit param compared to 32-bit counter → WIDTH error
   WRONG:  localparam [15:0] MAX_COUNT = 100;        // same problem
   CORRECT: parameter TICKS_PER_BAUD = 434;          // implicit 32-bit — safe to compare with any counter
   CORRECT: localparam MAX_COUNT = 100;               // implicit 32-bit
   Rule: leave all numeric parameters/localparams without bit-width prefixes.
5. No latches: assign defaults to all signals at top of every always @(*) block.
6. Parameters: module {design_name} #(parameter WIDTH=8) (input clk, ...);
7. No expression part-selects: NEVER apply [N:0] slicing to an expression result.
   WRONG: wire [2:0] idx = (bit_counter - 1)[2:0];   // expression part-select — illegal
   WRONG: assign tx = data[(counter - 1)[2:0]];        // same error inside assignment
   CORRECT: wire [3:0] tmp = bit_counter - 1;          // compute in intermediate wire
             wire [2:0] idx = tmp[2:0];                 // THEN slice the wire
   CORRECT: wire [2:0] idx = bit_counter[2:0] - 1;     // slice BEFORE the operation
   The rule: you can slice a named signal, never an expression.
8. ASYNC RESET: active-low means reset when signal is LOW.
   Standard form:  always @(posedge clk or negedge rst_n)
                   if (!rst_n) begin ... end else begin ... end
   AXI variant:    always @(posedge aclk or negedge aresetn)
                   if (!aresetn) begin ... end else begin ... end
   Use whichever signal names match the port list. The STRUCTURE is identical.
   Reset every control/state/valid/ready/output register explicitly. Datapath
   payload memories/register arrays may remain unreset ONLY when the spec says
   they are reset-less or when the structure is a register file/RAM whose valid
   control determines observability. Never leave control state uninitialized.
   Uninitialized control regs produce X/Z behavior in simulation — a fatal error.
9. SINGLE DRIVER: Every reg driven by exactly ONE always block.
10. COMBINATIONAL FLAGS: If the spec says a flag "goes high WHEN condition",
    implement as: assign flag = (condition); — wire, not reg.
11. ARRAY INDEX WIDTH: When indexing into an N-bit array with a counter,
    the counter MUST be exactly $clog2(N) bits wide — no wider.
    Indexing reg[7:0] requires a 3-bit index (values 0-7).
    Indexing reg[15:0] requires a 4-bit index (values 0-15).
    WRONG: reg [3:0] bit_ctr; assign tx = data[bit_ctr];  // 4-bit index into 8-bit array
    CORRECT option A: reg [2:0] bit_ctr;                  // declare 3-bit counter
    CORRECT option B: assign tx = data[bit_ctr[2:0]];     // slice at point of use
    Verilator ERROR: "Bit extraction of var[7:0] requires 3 bit index, not 4 bits"
    means your counter is too wide for the array it indexes.

12. TRI-STATE / INOUT RULE:
    If the design has an inout port such as I2C sda/scl, NEVER drive that port
    procedurally inside an always block. Use open-drain continuous assignment:
      assign sda = sda_oe ? 1'b0 : 1'bz;
    Internal sequential logic may update sda_oe, but only assign drives the pin.
13. SIGNED ARITHMETIC RULE:
    For arithmetic shift right (SRA), signed comparisons, or signed multiply/add,
    cast operands explicitly with $signed(). Use $signed(a) >>> shamt for SRA.
    Plain >> is logical shift and is wrong for signed arithmetic right shift.
14. COMBINATIONAL CASE DEFAULT RULE:
    Every case/casez/casex inside always @(*) MUST include a default: branch
    and assign safe defaults to all combinational outputs before the case.
    Missing default branches infer latches and will be rejected.

[VARIATION SALT: {rtl_salt}]

Return pure JSON:
{{
  "structural_reasoning": "...",
  "verilog_code": "..."
}}
"""
    result = run_llm(prompt, system_prompt, provider, max_api_retries=2, cand_idx=cand_idx)
    # NOTE: RTL_CACHE.put() is intentionally NOT called here.
    # Caching happens in the main loop AFTER all validation checks pass
    # (port match, BLKANDNBLK check, skeleton invariant check).
    # Caching invalid RTL here would cause cache hits to replay failed candidates forever.
    # Attach cache_key so the main loop can write to cache after validation.
    if result and cache_key:
        result["_cache_key"] = cache_key
    return result

# ==============================================================================
# AGENT: TESTBENCH GENERATOR (black-box, receives port interface not RTL)
# ==============================================================================
def _port_width_decl(width) -> str:
    try:
        w = int(width)
    except (TypeError, ValueError):
        w = 1
    return "" if w <= 1 else f"[{w-1}:0] "


def _find_port(port_interface: list, predicate):
    for p in port_interface or []:
        name = str(p.get("name", ""))
        if predicate(name.lower(), p):
            return p
    return None


def _port_direction(p: dict) -> str:
    return str(p.get("direction", "")).lower()


def _extract_uart_baud_param(prompt: str) -> str:
    for name in ("TICKS_PER_BAUD", "CLKS_PER_BIT", "CLKS_PER_BAUD"):
        if re.search(rf'\b{name}\b', prompt, re.IGNORECASE):
            return name
    m = re.search(r'\bParameter\s*:\s*([A-Za-z_]\w*)', prompt, re.IGNORECASE)
    return m.group(1) if m else "TICKS_PER_BAUD"


def _clock_ports(port_interface: list) -> list:
    return [
        p for p in port_interface or []
        if _port_direction(p) == "input"
        and _name_is_clock_like(str(p.get("name", "")).lower())
    ]


def _inout_ports(port_interface: list) -> list:
    return [p for p in port_interface or [] if _port_direction(p) == "inout"]


def _derive_timeout_cycles(prompt: str) -> int:
    nums = []
    for m in re.finditer(r'\b([0-9][0-9_,]*)\s*(?:clock\s*)?cycles?\b', prompt, re.IGNORECASE):
        try:
            nums.append(int(m.group(1).replace(",", "").replace("_", "")))
        except ValueError:
            pass
    for m in re.finditer(r'\b(?:TICKS_PER_BAUD|CLKS_PER_BIT|CLKS_PER_BAUD)\b', prompt, re.IGNORECASE):
        nums.append(1000)
    base = max(nums) if nums else 1000
    return max(10000, min(base * 20, 5000000))


def build_tb_environment_guidance(prompt: str, port_interface: list) -> str:
    clocks = [p.get("name") for p in _clock_ports(port_interface)]
    inouts = [p.get("name") for p in _inout_ports(port_interface)]
    timeout_cycles = _derive_timeout_cycles(prompt)

    if clocks:
        clk_lines = []
        for idx, clk in enumerate(clocks):
            half_period = 5 + (idx * 2)
            clk_lines.append(
                f"  - Declare reg {clk}; initialize {clk}=0; generate always #{half_period} {clk}=~{clk};"
            )
        primary_clk = clocks[0]
    else:
        clk_lines = ["  - No clock-like port was found. Do not invent clk unless the port_interface contains it."]
        primary_clk = None

    timeout_rule = (
        f"Use a cycle-based timeout: integer timeout_count; "
        f"repeat ({timeout_cycles}) @(posedge {primary_clk}); then FAIL/finish."
        if primary_clk else
        "Use a bounded absolute-time timeout because no clock-like port exists."
    )

    inout_rule = ""
    if inouts:
        examples = []
        for name in inouts:
            examples.append(
                f"  tri1 {name}; reg tb_drive_{name}_low; "
                f"assign {name} = tb_drive_{name}_low ? 1'b0 : 1'bz;"
            )
        inout_rule = (
            "\nINOUT/TRI-STATE TESTBENCH RULE:\n"
            "For inout pins, model external pullups/pulldowns with tri1 and a separate driver reg.\n"
            "Never assign to an inout procedurally inside initial/always blocks.\n"
            + "\n".join(examples)
        )

    return (
        "\nDYNAMIC TESTBENCH ENVIRONMENT:\n"
        "Clock ports detected:\n" + "\n".join(clk_lines) + "\n"
        f"Timeout guidance: {timeout_rule}\n"
        f"Derived timeout cycle budget: {timeout_cycles}.\n"
        + inout_rule + "\n"
    )


def build_uart_tx_testbench(prompt: str, design_name: str, port_interface: list) -> str:
    """Deterministic UART TX checker for skeleton-matched UART transmitters."""
    clk_p = _find_port(port_interface, lambda n, _p: "clk" in n or "clock" in n)
    rst_p = _find_port(port_interface, lambda n, _p: "rst" in n or "reset" in n)
    start_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input" and (n in {"start", "tx_valid", "valid"} or "start" in n),
    )
    data_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input"
        and str(p.get("width", 1)) in {"8", "8.0"}
        and ("data" in n or "byte" in n),
    )
    tx_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "output" and (n == "tx" or n in {"tx_out", "txd"} or "tx" in n),
    )
    busy_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "output" and "busy" in n)
    ready_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "output" and "ready" in n)

    if not all([clk_p, rst_p, start_p, data_p, tx_p]):
        return ""

    clk = clk_p["name"]
    rst = rst_p["name"]
    start = start_p["name"]
    data = data_p["name"]
    tx = tx_p["name"]
    busy = busy_p["name"] if busy_p else None
    ready = ready_p["name"] if ready_p else None
    baud_param = _extract_uart_baud_param(prompt)

    decls = []
    resets = []
    conns = []
    for p in port_interface:
        name = p.get("name")
        direction = p.get("direction")
        width = _port_width_decl(p.get("width", 1))
        kind = "reg" if _port_direction(p) == "input" else "wire"
        decls.append(f"  {kind} {width}{name};")
        conns.append(f"    .{name}({name})")
        if _port_direction(p) == "input":
            resets.append(f"    {name} = 0;")

    busy_checks = []
    idle_checks = []
    if busy:
        busy_checks.append(f"""        if ({busy} !== 1'b1) begin
          $display("FAIL: busy low during frame bit %0d tick %0d", bit_idx, tick_idx);
          $finish;
        end""")
        idle_checks.append(f"""    if ({busy} !== 1'b0) begin
      $display("FAIL: busy did not drop after stop bit");
      $finish;
    end""")
    if ready:
        busy_checks.append(f"""        if ({ready} !== 1'b0) begin
          $display("FAIL: ready high during frame bit %0d tick %0d", bit_idx, tick_idx);
          $finish;
        end""")
        idle_checks.append(f"""    if ({ready} !== 1'b1) begin
      $display("FAIL: ready did not reassert after stop bit");
      $finish;
    end""")

    conn_text = ",\n".join(conns)
    busy_text = "\n".join(busy_checks)
    idle_text = "\n".join(idle_checks)
    reset_text = "\n".join(resets)
    decl_text = "\n".join(decls)

    return f"""`timescale 1ns/1ps

module tb_{design_name};
  localparam {baud_param} = 4;

{decl_text}
  integer bit_idx;
  integer tick_idx;
  reg expected_bit;

  {design_name} #(.{baud_param}({baud_param})) dut (
{conn_text}
  );

  initial begin
    {clk} = 0;
  end
  always #5 {clk} = ~{clk};

  initial begin
    #100000;
    $display("FAIL: Timeout");
    $finish;
  end

  function frame_bit;
    input [7:0] value;
    input integer pos;
    begin
      case (pos)
        0: frame_bit = 1'b0;
        1: frame_bit = value[0];
        2: frame_bit = value[1];
        3: frame_bit = value[2];
        4: frame_bit = value[3];
        5: frame_bit = value[4];
        6: frame_bit = value[5];
        7: frame_bit = value[6];
        8: frame_bit = value[7];
        default: frame_bit = 1'b1;
      endcase
    end
  endfunction

  task send_and_check;
    input [7:0] tx_data;
    begin
      @(negedge {clk});
      {data} = tx_data;
      {start} = 1'b1;
      @(negedge {clk});
      {start} = 1'b0;

      for (bit_idx = 0; bit_idx < 10; bit_idx = bit_idx + 1) begin
        for (tick_idx = 0; tick_idx < {baud_param}; tick_idx = tick_idx + 1) begin
          expected_bit = frame_bit(tx_data, bit_idx);
          if ({tx} !== expected_bit) begin
            $display("FAIL: tx mismatch bit %0d tick %0d expected %b got %b", bit_idx, tick_idx, expected_bit, {tx});
            $finish;
          end
{busy_text}
          if (bit_idx == 3 && tick_idx == 0) begin
            {data} = ~tx_data;
            {start} = 1'b1;
          end else if (bit_idx == 3 && tick_idx == 1) begin
            {start} = 1'b0;
          end
          @(negedge {clk});
        end
      end

      {start} = 1'b0;
{idle_text}
      if ({tx} !== 1'b1) begin
        $display("FAIL: tx not idle high after frame, got %b", {tx});
        $finish;
      end
    end
  endtask

  initial begin
{reset_text}
    {rst} = 1'b0;
    repeat (2) @(negedge {clk});
    {rst} = 1'b1;
    repeat (2) @(negedge {clk});

    if ({tx} !== 1'b1) begin
      $display("FAIL: tx not idle high after reset, got %b", {tx});
      $finish;
    end
{idle_text}

    send_and_check(8'hA5);
    repeat (2) @(negedge {clk});
    send_and_check(8'h3C);
    $display("COVER_HIT:uart_two_frames");
    $display("COVER_HIT:start_ignored_while_busy");
    $display("SIMULATION_SUCCESS");
    $finish;
  end
endmodule"""


def build_axi_stream_uart_tx_testbench(prompt: str, design_name: str, port_interface: list) -> str:
    """Deterministic checker for AXI4-Stream sink feeding a UART TX engine.

    This composes two protocol oracles:
      - AXI-stream target accepts data only on tvalid && tready.
      - UART TX emits start, eight LSB-first data bits, and stop for fixed baud cells.
    """
    clk_p = _find_port(port_interface, lambda n, _p: n in {"aclk", "clk"} or "clock" in n)
    rst_p = _find_port(port_interface, lambda n, _p: n in {"aresetn", "resetn", "rst_n"} or "reset" in n or "rst" in n)
    valid_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input" and ("tvalid" in n or n in {"valid", "s_valid"}),
    )
    data_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input"
        and ("tdata" in n or "data" in n)
        and str(p.get("width", 1)) in {"8", "8.0"},
    )
    ready_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "output" and ("tready" in n or n in {"ready", "s_ready"}),
    )
    tx_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "output" and n in {"tx", "tx_out", "txd", "uart_tx"},
    )

    if not all([clk_p, rst_p, valid_p, data_p, ready_p, tx_p]):
        return ""

    clk = clk_p["name"]
    rst = rst_p["name"]
    valid = valid_p["name"]
    data = data_p["name"]
    ready = ready_p["name"]
    tx = tx_p["name"]
    baud_param = _extract_uart_baud_param(prompt)

    decls = []
    resets = []
    conns = []
    for p in port_interface:
        name = p.get("name")
        width = _port_width_decl(p.get("width", 1))
        kind = "reg" if _port_direction(p) == "input" else "wire"
        decls.append(f"  {kind} {width}{name};")
        conns.append(f"    .{name}({name})")
        if _port_direction(p) == "input":
            resets.append(f"    {name} = 0;")

    conn_text = ",\n".join(conns)
    reset_text = "\n".join(resets)
    decl_text = "\n".join(decls)

    return f"""`timescale 1ns/1ps

module tb_{design_name};
  localparam {baud_param} = 4;

{decl_text}
  integer bit_idx;
  integer tick_idx;
  reg expected_bit;

  {design_name} #(.{baud_param}({baud_param})) dut (
{conn_text}
  );

  initial begin
    {clk} = 0;
  end
  always #5 {clk} = ~{clk};

  initial begin
    #200000;
    $display("FAIL: Timeout");
    $finish;
  end

  function frame_bit;
    input [7:0] value;
    input integer pos;
    begin
      case (pos)
        0: frame_bit = 1'b0;
        1: frame_bit = value[0];
        2: frame_bit = value[1];
        3: frame_bit = value[2];
        4: frame_bit = value[3];
        5: frame_bit = value[4];
        6: frame_bit = value[5];
        7: frame_bit = value[6];
        8: frame_bit = value[7];
        default: frame_bit = 1'b1;
      endcase
    end
  endfunction

  task check_uart_frame;
    input [7:0] tx_data;
    begin
      for (bit_idx = 0; bit_idx < 10; bit_idx = bit_idx + 1) begin
        for (tick_idx = 0; tick_idx < {baud_param}; tick_idx = tick_idx + 1) begin
          expected_bit = frame_bit(tx_data, bit_idx);
          if ({tx} !== expected_bit) begin
            $display("FAIL: UART frame bit %0d tick %0d expected tx=%b got %b", bit_idx, tick_idx, expected_bit, {tx});
            $finish;
          end
          if ({ready} !== 1'b0) begin
            $display("FAIL: s_axis_tready high during UART frame bit %0d tick %0d", bit_idx, tick_idx);
            $finish;
          end
          @(negedge {clk});
        end
      end
      if ({tx} !== 1'b1) begin
        $display("FAIL: UART idle not high after frame, got %b", {tx});
        $finish;
      end
      if ({ready} !== 1'b1) begin
        $display("FAIL: s_axis_tready not reasserted after stop bit, got %b", {ready});
        $finish;
      end
    end
  endtask

  task send_byte;
    input [7:0] tx_data;
    begin
      @(negedge {clk});
      if ({ready} !== 1'b1) begin
        $display("FAIL: s_axis_tready not high before handshake, got %b", {ready});
        $finish;
      end
      {data} = tx_data;
      {valid} = 1'b1;
      @(negedge {clk});
      if ({ready} !== 1'b0) begin
        $display("FAIL: s_axis_tready not deasserted immediately after capture, got %b", {ready});
        $finish;
      end
      {valid} = 1'b0;
      check_uart_frame(tx_data);
    end
  endtask

  task send_with_pending_byte;
    input [7:0] first_byte;
    input [7:0] pending_byte;
    begin
      @(negedge {clk});
      if ({ready} !== 1'b1) begin
        $display("FAIL: s_axis_tready not high before first handshake, got %b", {ready});
        $finish;
      end
      {data} = first_byte;
      {valid} = 1'b1;
      @(negedge {clk});
      if ({ready} !== 1'b0) begin
        $display("FAIL: s_axis_tready not deasserted immediately after first capture, got %b", {ready});
        $finish;
      end
      {data} = pending_byte;
      {valid} = 1'b1;
      check_uart_frame(first_byte);

      @(negedge {clk});
      if ({ready} !== 1'b0) begin
        $display("FAIL: pending byte was not captured after ready returned, got ready=%b", {ready});
        $finish;
      end
      {valid} = 1'b0;
      check_uart_frame(pending_byte);
      $display("COVER_HIT:axis_pending_byte_after_backpressure");
    end
  endtask

  initial begin
{reset_text}
    {rst} = 1'b0;
    repeat (2) @(negedge {clk});
    {rst} = 1'b1;
    repeat (2) @(negedge {clk});

    if ({tx} !== 1'b1) begin
      $display("FAIL: UART idle not high after reset, got %b", {tx});
      $finish;
    end
    if ({ready} !== 1'b1) begin
      $display("FAIL: s_axis_tready not high after reset, got %b", {ready});
      $finish;
    end

    send_byte(8'hA5);
    repeat (2) @(negedge {clk});
    send_with_pending_byte(8'h3C, 8'h00);
    repeat (2) @(negedge {clk});
    send_byte(8'hFF);
    $display("COVER_HIT:axis_uart_three_frames");
    $display("SIMULATION_SUCCESS");
    $finish;
  end
endmodule"""


def build_rv32i_decoder_testbench(prompt: str, design_name: str, port_interface: list) -> str:
    """Deterministic oracle for a combinational RV32I field decoder/immediate unit."""
    instr_p = _find_port(
        port_interface,
        lambda n, p: "input" in _port_direction(p)
        and ("instr" in n or "instruction" in n)
        and str(p.get("width", 1)) in {"32", "32.0"},
    )
    outputs = {}
    for name in (
        "opcode", "rd", "funct3", "rs1", "rs2", "funct7", "imm_out",
        "reg_write", "alu_src", "branch", "mem_read", "mem_write",
    ):
        outputs[name] = _find_port(
            port_interface,
            lambda n, p, expected=name: "output" in _port_direction(p) and n == expected,
        )

    if not instr_p or not all(outputs.values()):
        return ""

    instr = instr_p["name"]
    names = {k: v["name"] for k, v in outputs.items()}

    decls = []
    resets = []
    conns = []
    for p in port_interface:
        name = p.get("name")
        width = _port_width_decl(p.get("width", 1))
        kind = "reg" if "input" in _port_direction(p) else "wire"
        decls.append(f"  {kind} {width}{name};")
        conns.append(f"    .{name}({name})")
        if "input" in _port_direction(p):
            resets.append(f"    {name} = 0;")

    decl_text = "\n".join(decls)
    conn_text = ",\n".join(conns)
    reset_text = "\n".join(resets)

    return f"""`timescale 1ns/1ps

module tb_{design_name};
{decl_text}

  reg [31:0] exp_instr;
  reg [6:0] exp_opcode;
  reg [4:0] exp_rd;
  reg [2:0] exp_funct3;
  reg [4:0] exp_rs1;
  reg [4:0] exp_rs2;
  reg [6:0] exp_funct7;
  reg [31:0] exp_imm;
  reg exp_reg_write;
  reg exp_alu_src;
  reg exp_branch;
  reg exp_mem_read;
  reg exp_mem_write;

  {design_name} dut (
{conn_text}
  );

  task check_fields;
    input [31:0] inst;
    input [6:0] want_opcode;
    input [4:0] want_rd;
    input [2:0] want_funct3;
    input [4:0] want_rs1;
    input [4:0] want_rs2;
    input [6:0] want_funct7;
    input [31:0] want_imm;
    begin
      {instr} = inst;
      #5;
      if ({names["opcode"]} !== want_opcode) begin
        $display("FAIL: opcode expected %b got %b", want_opcode, {names["opcode"]});
        $finish;
      end
      if ({names["rd"]} !== want_rd) begin
        $display("FAIL: rd expected %0d got %0d", want_rd, {names["rd"]});
        $finish;
      end
      if ({names["funct3"]} !== want_funct3) begin
        $display("FAIL: funct3 expected %b got %b", want_funct3, {names["funct3"]});
        $finish;
      end
      if ({names["rs1"]} !== want_rs1) begin
        $display("FAIL: rs1 expected %0d got %0d", want_rs1, {names["rs1"]});
        $finish;
      end
      if ({names["rs2"]} !== want_rs2) begin
        $display("FAIL: rs2 expected %0d got %0d", want_rs2, {names["rs2"]});
        $finish;
      end
      if ({names["funct7"]} !== want_funct7) begin
        $display("FAIL: funct7 expected %b got %b", want_funct7, {names["funct7"]});
        $finish;
      end
      if ({names["imm_out"]} !== want_imm) begin
        $display("FAIL: imm_out expected %h got %h", want_imm, {names["imm_out"]});
        $finish;
      end
    end
  endtask

  task check_control;
    input want_reg_write;
    input want_alu_src;
    input want_branch;
    input want_mem_read;
    input want_mem_write;
    begin
      #1;
      if ({names["reg_write"]} !== want_reg_write) begin
        $display("FAIL: reg_write expected %b got %b", want_reg_write, {names["reg_write"]});
        $finish;
      end
      if ({names["alu_src"]} !== want_alu_src) begin
        $display("FAIL: alu_src expected %b got %b", want_alu_src, {names["alu_src"]});
        $finish;
      end
      if ({names["branch"]} !== want_branch) begin
        $display("FAIL: branch expected %b got %b", want_branch, {names["branch"]});
        $finish;
      end
      if ({names["mem_read"]} !== want_mem_read) begin
        $display("FAIL: mem_read expected %b got %b", want_mem_read, {names["mem_read"]});
        $finish;
      end
      if ({names["mem_write"]} !== want_mem_write) begin
        $display("FAIL: mem_write expected %b got %b", want_mem_write, {names["mem_write"]});
        $finish;
      end
    end
  endtask

  initial begin
{reset_text}
    #5;

    check_fields(32'h00b50533, 7'b0110011, 5'd10, 3'b000, 5'd10, 5'd11, 7'b0000000, 32'd0);
    check_control(1'b1, 1'b0, 1'b0, 1'b0, 1'b0);
    $display("COVER_HIT:rv32i_r_type");

    check_fields(32'hfff10093, 7'b0010011, 5'd1, 3'b000, 5'd2, 5'd31, 7'b1111111, 32'hffffffff);
    check_control(1'b1, 1'b1, 1'b0, 1'b0, 1'b0);
    $display("COVER_HIT:rv32i_i_type");

    check_fields(32'hff822183, 7'b0000011, 5'd3, 3'b010, 5'd4, 5'd24, 7'b1111111, 32'hfffffff8);
    check_control(1'b1, 1'b1, 1'b0, 1'b1, 1'b0);
    $display("COVER_HIT:rv32i_load");

    check_fields(32'h00532623, 7'b0100011, 5'd12, 3'b010, 5'd6, 5'd5, 7'b0000000, 32'd12);
    check_control(1'b0, 1'b1, 1'b0, 1'b0, 1'b1);
    $display("COVER_HIT:rv32i_store");

    check_fields(32'hfe838ee3, 7'b1100011, 5'd29, 3'b000, 5'd7, 5'd8, 7'b1111111, 32'hfffffffc);
    check_control(1'b0, 1'b0, 1'b1, 1'b0, 1'b0);
    $display("COVER_HIT:rv32i_branch");

    check_fields(32'h123454b7, 7'b0110111, 5'd9, 3'b101, 5'd8, 5'd3, 7'b0001001, 32'h12345000);
    $display("COVER_HIT:rv32i_u_type");

    check_fields(32'h001000ef, 7'b1101111, 5'd1, 3'b000, 5'd0, 5'd1, 7'b0000000, 32'd2048);
    $display("COVER_HIT:rv32i_j_type");

    $display("SIMULATION_SUCCESS");
    $finish;
  end
endmodule"""


def build_sram_byte_en_testbench(prompt: str, design_name: str, port_interface: list) -> str:
    """Deterministic checker for synchronous 32-bit byte-enable data memories."""
    clk_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "input" and (n == "clk" or "clock" in n))
    rst_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "input" and ("rst" in n or "reset" in n))
    mem_read_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "input" and n in {"mem_read", "read_en", "ren", "rd_en"})
    mem_write_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "input" and n in {"mem_write", "write_en", "wen", "wr_en"})
    addr_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "input" and n == "addr")
    wdata_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "input" and n in {"wdata", "write_data", "wr_data"})
    byte_en_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "input" and ("byte_en" in n or "byte_enable" in n or "be" == n))
    rdata_p = _find_port(port_interface, lambda n, p: _port_direction(p) == "output" and n in {"rdata", "read_data", "rd_data"})

    if not all([clk_p, rst_p, mem_read_p, mem_write_p, addr_p, wdata_p, byte_en_p, rdata_p]):
        return ""

    clk = clk_p["name"]
    rst = rst_p["name"]
    mem_read = mem_read_p["name"]
    mem_write = mem_write_p["name"]
    addr = addr_p["name"]
    wdata = wdata_p["name"]
    byte_en = byte_en_p["name"]
    rdata = rdata_p["name"]

    decls = []
    resets = []
    conns = []
    for p in port_interface:
        name = p.get("name")
        width = _port_width_decl(p.get("width", 1))
        kind = "reg" if _port_direction(p) == "input" else "wire"
        decls.append(f"  {kind} {width}{name};")
        conns.append(f"    .{name}({name})")
        if _port_direction(p) == "input":
            resets.append(f"    {name} = 0;")

    decl_text = "\n".join(decls)
    conn_text = ",\n".join(conns)
    reset_text = "\n".join(resets)

    return f"""`timescale 1ns/1ps

module tb_{design_name};
  localparam MEM_DEPTH = 1024;

{decl_text}

  {design_name} #(.MEM_DEPTH(MEM_DEPTH)) dut (
{conn_text}
  );

  initial begin
    {clk} = 0;
  end
  always #5 {clk} = ~{clk};

  initial begin
    #200000;
    $display("FAIL: Timeout");
    $finish;
  end

  task write_word;
    input [31:0] wr_addr;
    input [31:0] wr_data;
    input [3:0] wr_be;
    begin
      @(negedge {clk});
      {addr} = wr_addr;
      {wdata} = wr_data;
      {byte_en} = wr_be;
      {mem_write} = 1'b1;
      {mem_read} = 1'b0;
      @(negedge {clk});
      {mem_write} = 1'b0;
      {byte_en} = 4'b0000;
      {wdata} = 32'd0;
    end
  endtask

  task read_expect;
    input [31:0] rd_addr;
    input [31:0] expected;
    begin
      @(negedge {clk});
      {addr} = rd_addr;
      {mem_read} = 1'b1;
      {mem_write} = 1'b0;
      {byte_en} = 4'b0000;
      @(negedge {clk});
      if ({rdata} !== expected) begin
        $display("FAIL: read addr=%08h expected=%08h got=%08h", rd_addr, expected, {rdata});
        $finish;
      end
      {mem_read} = 1'b0;
      @(negedge {clk});
      if ({rdata} !== 32'd0) begin
        $display("FAIL: rdata should clear to 0 when mem_read=0, got=%08h", {rdata});
        $finish;
      end
    end
  endtask

  initial begin
{reset_text}
    {rst} = 1'b0;
    repeat (2) @(negedge {clk});
    {rst} = 1'b1;
    repeat (2) @(negedge {clk});

    if ({rdata} !== 32'd0) begin
      $display("FAIL: rdata not zero after reset/no-read, got=%08h", {rdata});
      $finish;
    end
    $display("COVER_HIT:sram_reset_output_zero");

    write_word(32'h00000004, 32'hDEADBEEF, 4'b1111);
    read_expect(32'h00000004, 32'hDEADBEEF);
    $display("COVER_HIT:sram_full_word_write");

    write_word(32'h00000008, 32'h11223344, 4'b1111);
    read_expect(32'h00000008, 32'h11223344);
    write_word(32'h00000008, 32'hAABBCCDD, 4'b0101);
    read_expect(32'h00000008, 32'h11BB33DD);
    $display("COVER_HIT:sram_byte_mask_0101");

    write_word(32'h00000008, 32'hAABBCCDD, 4'b1010);
    read_expect(32'h00000008, 32'hAABBCCDD);
    $display("COVER_HIT:sram_byte_mask_1010");

    write_word(32'h00000008, 32'hFFFFFFFF, 4'b0000);
    read_expect(32'h00000008, 32'hAABBCCDD);
    $display("COVER_HIT:sram_zero_byte_enable_nochange");

    write_word(32'h0000000C, 32'hCAFEBABE, 4'b1111);
    read_expect(32'h0000000C, 32'hCAFEBABE);
    read_expect(32'h00000004, 32'hDEADBEEF);
    $display("COVER_HIT:sram_word_addressing");

    $display("SIMULATION_SUCCESS");
    $finish;
  end
endmodule"""


def build_async_fifo_testbench(prompt: str, design_name: str, port_interface: list) -> str:
    """Deterministic checker for dual-clock async FIFOs."""
    wr_clk_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input" and "wr" in n and ("clk" in n or "clock" in n),
    )
    rd_clk_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input" and "rd" in n and ("clk" in n or "clock" in n),
    )
    reset_ps = [
        p for p in port_interface or []
        if _port_direction(p) == "input"
        and re.search(r'(?:rst|reset)', str(p.get("name", "")).lower())
    ]
    wr_en_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input" and n in {"wr_en", "write_en", "wen"},
    )
    rd_en_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input" and n in {"rd_en", "read_en", "ren"},
    )
    wr_data_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "input"
        and (n in {"wr_data", "write_data", "wdata", "din", "data_in"} or ("wr" in n and "data" in n)),
    )
    rd_data_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "output"
        and (n in {"rd_data", "read_data", "rdata", "dout", "data_out"} or ("rd" in n and "data" in n)),
    )
    full_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "output" and n == "full",
    )
    empty_p = _find_port(
        port_interface,
        lambda n, p: _port_direction(p) == "output" and n == "empty",
    )

    if not all([wr_clk_p, rd_clk_p, reset_ps, wr_en_p, rd_en_p, wr_data_p, rd_data_p, full_p, empty_p]):
        return ""

    wr_clk = wr_clk_p["name"]
    rd_clk = rd_clk_p["name"]
    wr_en = wr_en_p["name"]
    rd_en = rd_en_p["name"]
    wr_data = wr_data_p["name"]
    rd_data = rd_data_p["name"]
    full = full_p["name"]
    empty = empty_p["name"]

    try:
        data_w = int(wr_data_p.get("width", rd_data_p.get("width", 8)))
    except (TypeError, ValueError):
        data_w = 8
    data_w = max(data_w, 1)

    decls = []
    resets = []
    conns = []
    reset_assert_lines = []
    reset_deassert_lines = []
    for p in port_interface:
        name = p.get("name")
        width = _port_width_decl(p.get("width", 1))
        kind = "reg" if _port_direction(p) == "input" else "wire"
        decls.append(f"  {kind} {width}{name};")
        conns.append(f"    .{name}({name})")
        if _port_direction(p) == "input":
            resets.append(f"    {name} = 0;")

    for p in reset_ps:
        name = p.get("name")
        n_l = str(name).lower()
        active_low = bool(re.search(r'(?:_n$|rstn$|resetn$|aresetn$)', n_l))
        reset_assert_lines.append(f"    {name} = 1'b{'0' if active_low else '1'};")
        reset_deassert_lines.append(f"    {name} = 1'b{'1' if active_low else '0'};")

    decl_text = "\n".join(decls)
    conn_text = ",\n".join(conns)
    reset_text = "\n".join(resets)
    reset_assert_text = "\n".join(reset_assert_lines)
    reset_deassert_text = "\n".join(reset_deassert_lines)

    return f"""`timescale 1ns/1ps

module tb_{design_name};
  localparam DATA_W = {data_w};

{decl_text}
  reg [DATA_W-1:0] expected [0:15];
  integer idx;
  integer wait_cycles;
  reg wait_done;

  {design_name} dut (
{conn_text}
  );

  initial begin
    {wr_clk} = 0;
  end
  always #5 {wr_clk} = ~{wr_clk};

  initial begin
    {rd_clk} = 0;
  end
  always #7 {rd_clk} = ~{rd_clk};

  initial begin
    #300000;
    $display("FAIL: Timeout");
    $finish;
  end

  task write_value;
    input [DATA_W-1:0] value;
    begin
      wait_done = 1'b0;
      for (wait_cycles = 0; wait_cycles < 120; wait_cycles = wait_cycles + 1) begin
        @(negedge {wr_clk});
        if ({full} === 1'b0) begin
          wait_done = 1'b1;
          wait_cycles = 120;
        end
      end
      if (!wait_done) begin
        $display("FAIL: full stayed high before write value=%02h", value);
        $finish;
      end
      {wr_data} = value;
      {wr_en} = 1'b1;
      @(negedge {wr_clk});
      {wr_en} = 1'b0;
      {wr_data} = 0;
    end
  endtask

  task read_expect;
    input [DATA_W-1:0] value;
    input [31:0] index;
    begin
      wait_done = 1'b0;
      for (wait_cycles = 0; wait_cycles < 160; wait_cycles = wait_cycles + 1) begin
        @(negedge {rd_clk});
        if ({empty} === 1'b0) begin
          wait_done = 1'b1;
          wait_cycles = 160;
        end
      end
      if (!wait_done) begin
        $display("FAIL: empty stayed high before read index %0d", index);
        $finish;
      end
      {rd_en} = 1'b1;
      @(negedge {rd_clk});
      {rd_en} = 1'b0;
      if ({rd_data} !== value) begin
        $display("FAIL: read mismatch at index %0d expected=%02h got=%02h", index, value, {rd_data});
        $finish;
      end
    end
  endtask

  initial begin
{reset_text}
    wait_done = 1'b0;
    for (idx = 0; idx < 16; idx = idx + 1) begin
      expected[idx] = idx + 5;
    end

{reset_assert_text}
    repeat (4) @(negedge {wr_clk});
{reset_deassert_text}
    repeat (8) @(negedge {rd_clk});

    if ({full} !== 1'b0 || {empty} !== 1'b1) begin
      $display("FAIL: reset flags expected full=0 empty=1, got full=%b empty=%b", {full}, {empty});
      $finish;
    end
    $display("COVER_HIT:async_fifo_reset_empty");

    for (idx = 0; idx < 16; idx = idx + 1) begin
      write_value(expected[idx]);
    end
    $display("COVER_HIT:write_to_full_wraparound");

    @(negedge {wr_clk});
    if ({full} !== 1'b1) begin
      $display("FAIL: FIFO should be full after 16 writes, got full=%b", {full});
      $finish;
    end

    {wr_data} = 8'hEE;
    {wr_en} = 1'b1;
    @(negedge {wr_clk});
    {wr_en} = 1'b0;
    {wr_data} = 0;
    @(negedge {wr_clk});
    if ({full} !== 1'b1) begin
      $display("FAIL: full deasserted after overflow write attempt, got full=%b", {full});
      $finish;
    end
    $display("COVER_HIT:full_blocks_overflow_write");

    repeat (20) @(negedge {rd_clk});

    for (idx = 0; idx < 16; idx = idx + 1) begin
      read_expect(expected[idx], idx);
    end

    if ({empty} !== 1'b1) begin
      repeat (8) @(negedge {rd_clk});
      if ({empty} !== 1'b1) begin
        $display("FAIL: FIFO should be empty after 16 reads, got empty=%b", {empty});
        $finish;
      end
    end

    $display("COVER_HIT:async_fifo_passed");
    $display("SIMULATION_SUCCESS");
    $finish;
  end
endmodule"""


def _has_contract_ports(port_interface: list, expected: dict) -> bool:
    """Require deterministic AMS oracles to bind only their documented API."""
    actual = {
        str(port.get("name", "")).lower(): _port_direction(port)
        for port in (port_interface or [])
        if isinstance(port, dict)
    }
    return all(actual.get(name) == direction for name, direction in expected.items())


def build_sar_adc_controller_testbench(clarified_spec: dict, design_name: str, port_interface: list) -> str:
    """Independent black-box oracle configured only from canonical Spec-Sync."""
    expected_ports = {
        "clk": "input", "rst_n": "input", "start": "input", "comparator": "input",
        "dac_code": "output", "adc_out": "output", "busy": "output", "done": "output",
    }
    if not _has_contract_ports(port_interface, expected_ports):
        return ""

    config = _sar_contract_parameters(clarified_spec)
    resolution = config["resolution"]
    settle_cycles = config["settle_cycles"]
    high_keeps = config["comparator_high_keeps_bit"]
    if config["sample_event"] != "posedge clk" or not config["contract_hash"]:
        return ""
    target = ((1 << resolution) * 5) // 8
    target2 = ((1 << resolution) * 3) // 8
    wrong_comparator = 0 if high_keeps else 1

    return f"""`timescale 1ns/1ps

module tb_{design_name};
  // SPECIFICATION: contract_version={config['contract_version']} contract_hash={config['contract_hash']}
  // ORACLE CONFIG: resolution={resolution} settle_cycles={settle_cycles} comparator_high_keeps_bit={str(high_keeps).lower()} sample_event={config['sample_event']}
  localparam RESOLUTION = {resolution};
  localparam SETTLE_CYCLES = {settle_cycles};
  localparam CMP_HIGH_KEEPS_BIT = {int(high_keeps)};
  reg clk;
  reg rst_n;
  reg start;
  reg comparator;
  wire [RESOLUTION-1:0] dac_code;
  wire [RESOLUTION-1:0] adc_out;
  wire busy;
  wire done;
  reg [RESOLUTION-1:0] model_code;
  reg [RESOLUTION-1:0] trial_code;
  reg [RESOLUTION-1:0] expected_code;
  reg decision_keep;
  integer bit_index;
  integer settle_index;

  {design_name} dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .comparator(comparator),
    .dac_code(dac_code),
    .adc_out(adc_out),
    .busy(busy),
    .done(done)
  );

  always #5 clk = ~clk;

  task run_conversion;
    input [RESOLUTION-1:0] target_code;
    begin
      model_code = 0;
      start = 1;
      @(posedge clk); #1;
      start = 0;
      if (busy !== 1 || done !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: busy must rise on accepted start");
        $finish;
      end

      for (bit_index = RESOLUTION - 1; bit_index >= 0; bit_index = bit_index - 1) begin
        trial_code = model_code | (1 << bit_index);
        wait (dac_code === trial_code);
        @(negedge clk);
        if (dac_code !== trial_code || busy !== 1 || done !== 0) begin
          $display("FUNCTIONAL_FAILURE: FAIL: invalid trial/busy state at bit %0d", bit_index);
          $finish;
        end

        // Hold the opposite decision throughout settling. An early sample changes
        // DAC/result behavior and fails before the contract sample edge.
        comparator = {wrong_comparator};
        if (bit_index == RESOLUTION - 1) begin
          start = 1;
          @(posedge clk); #1;
          start = 0;
          if (busy !== 1 || done !== 0) begin
            $display("FUNCTIONAL_FAILURE: FAIL: start while busy restarted or completed conversion");
            $finish;
          end
        end
        for (settle_index = 0; settle_index < SETTLE_CYCLES; settle_index = settle_index + 1) begin
          @(negedge clk);
          if (dac_code !== trial_code || busy !== 1 || done !== 0) begin
            $display("FUNCTIONAL_FAILURE: FAIL: comparator decision consumed before %0d full settle cycles at bit %0d", SETTLE_CYCLES, bit_index);
            $finish;
          end
        end

        decision_keep = (trial_code <= target_code);
        comparator = CMP_HIGH_KEEPS_BIT ? decision_keep : !decision_keep;
        @(posedge clk); #1;
        expected_code = decision_keep ? trial_code : (trial_code & ~(1 << bit_index));
        if (dac_code !== expected_code || busy !== 1 || done !== 0) begin
          $display("FUNCTIONAL_FAILURE: FAIL: bit %0d decision/sample timing expected dac=%h got=%h busy=%b done=%b", bit_index, expected_code, dac_code, busy, done);
          $finish;
        end
        model_code = expected_code;
      end

      if (adc_out !== model_code || dac_code !== model_code) begin
        $display("FUNCTIONAL_FAILURE: FAIL: final result expected=%h adc=%h dac=%h", model_code, adc_out, dac_code);
        $finish;
      end
      if (busy !== 1 || done !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: busy/done timing at final decision is incorrect");
        $finish;
      end
      @(posedge clk); #1;
      if (done !== 1 || busy !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: done must assert exactly one cycle after final decision");
        $finish;
      end
      @(posedge clk); #1;
      if (done !== 0 || busy !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: done must be exactly one cycle and busy must remain low");
        $finish;
      end
    end
  endtask

  initial begin
    clk = 0;
    rst_n = 0;
    start = 0;
    comparator = {wrong_comparator};
    model_code = 0;
    trial_code = 0;
    expected_code = 0;
    decision_keep = 0;
    bit_index = 0;
    settle_index = 0;

    repeat (2) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    if ($bits(dut.dac_code) != RESOLUTION || $bits(dut.adc_out) != RESOLUTION) begin
      $display("CONTRACT_VIOLATION: SAR code ports must match resolution=%0d", RESOLUTION);
      $finish;
    end
    if (dac_code !== 0 || adc_out !== 0 || busy !== 0 || done !== 0) begin
      $display("FUNCTIONAL_FAILURE: FAIL: SAR reset outputs are not deterministic");
      $finish;
    end

    run_conversion({resolution}'d{target});
    run_conversion({resolution}'d{target2});
    $display("COVER_HIT:sar_msb_to_lsb_binary_search");
    $display("COVER_HIT:sar_settle_sample_timing");
    $display("COVER_HIT:sar_busy_start_and_repeat");
    $display("COVER_HIT:sar_done_one_cycle");
    $display("SIMULATION_SUCCESS");
    $finish;
  end

  initial begin
    #(1000 + RESOLUTION * (SETTLE_CYCLES + 12) * 20);
    $display("ORACLE_ERROR: SAR behavior did not complete within the contract timeout");
    $finish;
  end
endmodule"""


def build_spi_register_bank_testbench(prompt: str, design_name: str, port_interface: list) -> str:
    """Independent SPI mode-0 and bundled-data commit checker."""
    expected_ports = {
        "clk": "input", "rst_n": "input", "sclk": "input", "cs_n": "input",
        "mosi": "input", "miso": "output", "active_reg0": "output", "active_reg1": "output",
    }
    if not _has_contract_ports(port_interface, expected_ports):
        return ""

    return f"""`timescale 1ns/1ps

module tb_{design_name};
  reg clk;
  reg rst_n;
  reg sclk;
  reg cs_n;
  reg mosi;
  wire miso;
  wire [15:0] active_reg0;
  wire [15:0] active_reg1;
  reg [15:0] readback_value;
  reg [15:0] active_reg0_before_abort;
  integer bit_index;
  integer active_reg0_updates;
  reg [15:0] previous_active_reg0;

  {design_name} dut (
    .clk(clk),
    .rst_n(rst_n),
    .sclk(sclk),
    .cs_n(cs_n),
    .mosi(mosi),
    .miso(miso),
    .active_reg0(active_reg0),
    .active_reg1(active_reg1)
  );

  always #5 clk = ~clk;
  always #3 sclk = ~sclk;

  always @(negedge clk) begin
    if (!rst_n) begin
      previous_active_reg0 = active_reg0;
      active_reg0_updates = 0;
    end else if (active_reg0 !== previous_active_reg0) begin
      active_reg0_updates = active_reg0_updates + 1;
      previous_active_reg0 = active_reg0;
    end
  end

  task spi_send_frame;
    input [23:0] frame;
    begin
      @(negedge sclk);
      cs_n = 0;
      mosi = frame[23];
      for (bit_index = 23; bit_index >= 0; bit_index = bit_index - 1) begin
        @(posedge sclk);
        if (bit_index > 0) begin
          @(negedge sclk);
          mosi = frame[bit_index-1];
        end
      end
      @(negedge sclk);
      cs_n = 1;
      mosi = 0;
    end
  endtask

  task spi_readback_after_read_command;
    input [23:0] read_command;
    begin
      spi_send_frame(read_command);
      readback_value = 0;
      @(negedge sclk);
      cs_n = 0;
      mosi = 0;
      for (bit_index = 0; bit_index < 24; bit_index = bit_index + 1) begin
        @(posedge sclk);
        if (bit_index < 16) readback_value[15-bit_index] = miso;
        if (bit_index < 23) begin
          @(negedge sclk);
          mosi = 0;
        end
      end
      @(negedge sclk);
      cs_n = 1;
      mosi = 0;
    end
  endtask

  initial begin
    clk = 0;
    sclk = 0;
    rst_n = 0;
    cs_n = 1;
    mosi = 0;
    readback_value = 0;
    active_reg0_before_abort = 0;
    active_reg0_updates = 0;
    previous_active_reg0 = 0;

    repeat (3) @(negedge clk);
    rst_n = 1;
    repeat (3) @(negedge clk);
    if (active_reg0 !== 0 || active_reg1 !== 0) begin
      $display("FAIL: SPI active registers did not reset to zero");
      $finish;
    end

    // Mode 0 write: R/W=0, address 0, data A55A.
    spi_send_frame(24'h00A55A);
    repeat (10) @(negedge clk);
    if (active_reg0 !== 16'hA55A || active_reg0_updates !== 1) begin
      $display("FAIL: SPI committed write was not delivered exactly once, active=%h updates=%0d", active_reg0, active_reg0_updates);
      $finish;
    end
    $display("COVER_HIT:spi_atomic_commit");

    active_reg0_before_abort = active_reg0;
    @(negedge sclk);
    cs_n = 0;
    mosi = 1'b0;
    for (bit_index = 23; bit_index >= 16; bit_index = bit_index - 1) begin
      @(posedge sclk);
      if (bit_index > 16) @(negedge sclk);
    end
    @(negedge sclk);
    cs_n = 1;
    mosi = 0;
    repeat (10) @(negedge clk);
    if (active_reg0 !== active_reg0_before_abort || active_reg0_updates !== 1) begin
      $display("FAIL: CS_n abort changed active register state");
      $finish;
    end
    $display("COVER_HIT:spi_cs_abort");

    // Address 1 proves a second acknowledged event is not lost.
    spi_send_frame(24'h010F0F);
    repeat (10) @(negedge clk);
    if (active_reg1 !== 16'h0F0F) begin
      $display("FAIL: second SPI commit did not reach active_reg1, got=%h", active_reg1);
      $finish;
    end
    $display("COVER_HIT:spi_second_commit_delivered");

    // R/W=1 address 0; data is returned during the following frame, MSB first.
    spi_readback_after_read_command(24'h800000);
    if (readback_value !== 16'hA55A) begin
      $display("FAIL: SPI pipelined readback expected=A55A got=%h", readback_value);
      $finish;
    end
    $display("COVER_HIT:spi_pipelined_shadow_readback");
    $display("SIMULATION_SUCCESS");
    $finish;
  end
endmodule"""


def build_non_overlapping_clock_testbench(prompt: str, design_name: str, port_interface: list) -> str:
    """Cycle-level logical non-overlap oracle; it makes no physical timing claim."""
    expected_ports = {
        "clk": "input", "rst_n": "input", "enable": "input",
        "phi1": "output", "phi2": "output",
    }
    if not _has_contract_ports(port_interface, expected_ports):
        return ""

    return f"""`timescale 1ns/1ps

module tb_{design_name};
  reg clk;
  reg rst_n;
  reg enable;
  wire phi1;
  wire phi2;

  {design_name} #(.DEAD_CYCLES(2)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .phi1(phi1),
    .phi2(phi2)
  );

  always #5 clk = ~clk;

  always @(negedge clk) begin
    if (phi1 && phi2) begin
      $display("FAIL: logical overlap phi1=1 phi2=1");
      $finish;
    end
  end

  task expect_phase;
    input expected_phi1;
    input expected_phi2;
    begin
      @(negedge clk);
      if (phi1 !== expected_phi1 || phi2 !== expected_phi2) begin
        $display("FAIL: phase expected phi1=%b phi2=%b got phi1=%b phi2=%b", expected_phi1, expected_phi2, phi1, phi2);
        $finish;
      end
    end
  endtask

  initial begin
    clk = 0;
    rst_n = 0;
    enable = 0;
    repeat (2) @(negedge clk);
    if (phi1 !== 0 || phi2 !== 0) begin
      $display("FAIL: reset must force both phases low");
      $finish;
    end
    rst_n = 1;
    @(negedge clk);
    enable = 1;

    expect_phase(1'b1, 1'b0);
    expect_phase(1'b0, 1'b0);
    expect_phase(1'b0, 1'b0);
    expect_phase(1'b0, 1'b1);
    expect_phase(1'b0, 1'b0);
    expect_phase(1'b0, 1'b0);
    expect_phase(1'b1, 1'b0);
    $display("COVER_HIT:non_overlap_dead_cycles");

    enable = 0;
    expect_phase(1'b0, 1'b0);
    $display("COVER_HIT:non_overlap_disable");
    $display("SIMULATION_SUCCESS");
    $finish;
  end
endmodule"""


def validate_generated_testbench_structure(tb_code: str, cand_idx: int, port_interface: list = None) -> bool:
    """Reject common SystemVerilog or illegal Verilog-2001 constructs early."""
    stripped = tb_code.strip()
    checks = [
        (r'\bwhile\s*\(',
         "while loop emitted despite bounded-loop rule"),
        (r'\b(?:logic|bit|int|always_ff|always_comb|always_latch)\b',
         "SystemVerilog keyword emitted"),
        (r'\bfor\s*\(\s*(?:integer|int|reg)\b',
         "loop variable declared inside for()"),
        (r"(?:\([^;\n]+\)|\d+'\s*[hdbHDB][0-9a-fA-F_xzXZ]+)\s*\[",
         "expression or literal part-select emitted"),
    ]
    for pattern, msg in checks:
        if re.search(pattern, stripped):
            print(f"      Warning: TB {cand_idx+1}: {msg} - discarding.")
            return False

    first_proc = re.search(r'(?m)^\s*(?:initial|always|task|function)\b', stripped)
    if first_proc:
        tail = stripped[first_proc.start():]
        if re.search(r'(?m)^\s*(?:reg|wire|integer|localparam|parameter)\b', tail):
            print(
                f"      Warning: TB {cand_idx+1}: declaration appears after procedural code - discarding."
            )
            return False

    clock_names = [str(p.get("name", "")) for p in _clock_ports(port_interface or [])]
    if len(clock_names) > 1 and re.search(r'\b(?:reg|wire)\s+clk\b|\balways\s*#\d+\s+clk\s*=', stripped):
        print(f"      Warning: TB {cand_idx+1}: invented single clk for multi-clock DUT - discarding.")
        return False
    if not clock_names and re.search(r'\b(?:reg|wire)\s+clk\b|@\s*\(\s*(?:posedge|negedge)\s+clk\s*\)|\balways\s*#\d+\s+clk\s*=', stripped):
        print(f"      Warning: TB {cand_idx+1}: invented clk for combinational/no-clock DUT - discarding.")
        return False

    for p in _inout_ports(port_interface or []):
        name = str(p.get("name", ""))
        if not name:
            continue
        if re.search(rf'(?m)^\s*{re.escape(name)}\s*(?:<=|=)', stripped):
            print(f"      Warning: TB {cand_idx+1}: procedural drive of inout {name} - discarding.")
            return False
        if re.search(rf'\breg\b[^;\n]*\b{re.escape(name)}\b', stripped):
            print(f"      Warning: TB {cand_idx+1}: inout {name} declared as reg - discarding.")
            return False

    if not re.search(r'\bmodule\b', stripped):
        print(f"      Warning: TB {cand_idx+1}: Missing module declaration - discarding.")
        return False
    if not re.search(r'\bendmodule\b', stripped):
        print(f"      Warning: TB {cand_idx+1}: Missing endmodule - discarding.")
        return False
    return True


def generate_testbench(
    prompt:         str,
    provider:       str,
    design_name:    str,
    port_interface: list,   # architecture["port_interface"] — NOT the RTL code
    previous_error: str = None,
    previous_tb:    str = None,
    cand_idx:       int = 0,
    design_type_hint: str = None,
    round_id: int = 0,
    clarified_spec: dict = None,
) -> str:
    port_str  = json.dumps(port_interface, indent=2)
    design_type = design_type_hint or detect_design_type({}, prompt)
    tb_environment_guidance = build_tb_environment_guidance(prompt, port_interface)
    ams_contract = (clarified_spec or {}).get("ams_contract", {})
    ams_cache_identity = ams_contract.get("contract_hash", "") if isinstance(ams_contract, dict) else ""
    cache_key = get_cache_key(prompt, port_str, previous_error, previous_tb, cand_idx, design_type, ams_cache_identity)
    if cache_key in TB_CACHE:
        print(f"      ⚡ Cache Hit: Reusing Testbench for Thread {cand_idx+1}.")
        return TB_CACHE.get(cache_key)

    if design_type == "axi_stream_uart_tx":
        tb_code = build_axi_stream_uart_tx_testbench(prompt, design_name, port_interface)
        if tb_code:
            TB_CACHE.put(cache_key, tb_code)
            return tb_code

    if design_type == "uart_tx":
        tb_code = build_uart_tx_testbench(prompt, design_name, port_interface)
        if tb_code:
            TB_CACHE.put(cache_key, tb_code)
            return tb_code

    if design_type == "async_fifo":
        tb_code = build_async_fifo_testbench(prompt, design_name, port_interface)
        if tb_code:
            TB_CACHE.put(cache_key, tb_code)
            return tb_code

    if design_type == "sar_adc_controller":
        tb_code = build_sar_adc_controller_testbench(clarified_spec or {}, design_name, port_interface)
        if tb_code:
            valid_contract, contract_error = validate_sar_oracle_contract(tb_code, clarified_spec or {})
            if not valid_contract:
                print(f"      ❌ {contract_error}")
                return ""
            sar_config = _sar_contract_parameters(clarified_spec or {})
            print(
                f"      🧪 SAR oracle contract: resolution={sar_config['resolution']}, "
                f"settle_cycles={sar_config['settle_cycles']}, "
                f"comparator_high_keeps_bit={str(sar_config['comparator_high_keeps_bit']).lower()}, "
                f"hash={sar_config['contract_hash']}"
            )
            TB_CACHE.put(cache_key, tb_code)
            return tb_code

    if design_type == "spi_register_bank":
        tb_code = build_spi_register_bank_testbench(prompt, design_name, port_interface)
        if tb_code:
            TB_CACHE.put(cache_key, tb_code)
            return tb_code

    if design_type == "non_overlapping_clock":
        tb_code = build_non_overlapping_clock_testbench(prompt, design_name, port_interface)
        if tb_code:
            TB_CACHE.put(cache_key, tb_code)
            return tb_code

    error_context = ""
    if previous_error and previous_tb:
        failure_type  = classify_failure(previous_error)
        error_context = f"""
PREVIOUS TESTBENCH FAILED.
FAILURE CLASSIFICATION: {failure_type}
SIMULATION OUTPUT:
{previous_error}
FAILING TESTBENCH:
{previous_tb}
Fix only what caused the failure. Do not change passing test cases.
"""

    # Protocol assertion injection — Verilog-2001 compatible, no $past needed
    needs_protocol_assertions = has_ready_valid_ports(port_interface)
    protocol_rule = """
13. PROTOCOL ASSERTIONS (required for this ready/valid design):
    Add a manual history-register check for data stability under backpressure.
    Declare at top of module: reg [7:0] _prev_m_data; reg _prev_m_valid;
    Add this always block (Verilog-2001, no SystemVerilog):
      always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin _prev_m_data <= 0; _prev_m_valid <= 0; end
          else begin
              if (_prev_m_valid && !m_ready && m_valid && m_data !== _prev_m_data)
                  $display("ASSERTION FAILED: m_data mutated while m_valid=1 m_ready=0");
              _prev_m_data <= m_data; _prev_m_valid <= m_valid;
          end
      end
""" if needs_protocol_assertions else ""

    system_prompt = f"""
You are a hardware verification engineer. Write a Verilog-2001 testbench.
You MUST write a black-box testbench based purely on the spec and port interface.
You do NOT have access to the RTL implementation.

MASTER SPECIFICATION (authoritative; do not invent behavior outside this):
{prompt}

PORT INTERFACE (instantiate the DUT using exactly these signals):
{port_str}

{tb_environment_guidance}

TESTBENCH RULES — all mandatory:
1. First line of the file MUST be: `timescale 1ns/1ps
   (backtick, no space, exactly as shown — this prevents line-1 syntax errors)
2. Verilog-2001 STRICTLY. No SystemVerilog anywhere. Icarus will reject SV syntax.
   WRONG (SystemVerilog — never use):
     wire [7:0] rx_byte = 8'h00;    // initial value on wire — SV only, illegal in V2001
     logic tx_bit;                   // 'logic' type — SV only, use 'reg' or 'wire'
     int loop_var;                   // 'int' type — SV only, use 'integer'
     always_ff @(posedge clk) ...   // SV keyword, use always @(posedge clk)
   CORRECT:
     reg [7:0] rx_byte;             // declare as reg, assign value in initial block
     reg tx_bit;
     integer loop_var;
3. Declare ALL variables at top of module, never inside begin/end blocks.
   Assign initial values in initial begin blocks, NOT on wire/reg declarations.
   WRONG — Icarus rejects these:
     wire [7:0] rx_byte = 8'h00;        // initial value on wire — illegal in Verilog-2001
     initial begin rx_out = 1'b1; end   // driving a wire in initial — use reg instead
     begin reg [7:0] tmp; end           // variable in unnamed begin block — SystemVerilog only
     begin : unnamed begin reg tmp; end // same problem even with colon — must be at module top
     integer i = 0;                     // initializer on integer declaration — illegal
   Icarus error: "Variable declaration in unnamed block requires SystemVerilog"
   Fix: move ALL reg/wire/integer declarations to the TOP of the module, before any always/initial.
   CORRECT:
     reg [7:0] rx_byte;               // plain declaration
     initial begin rx_byte = 8'h00; end // assign in initial block
     integer i;                       // plain declaration
     initial begin i = 0; end         // assign in initial block
4. DUT INSTANTIATION — CRITICAL:
   You MUST instantiate the DUT module with its EXACT name: `{design_name}`
   CORRECT:   {design_name} dut ( .clk(clk), .rst_n(rst_n), ... );
   WRONG:     dut dut ( ... );          // 'dut' is NOT a module name
   WRONG:     my_module dut ( ... );    // use the exact name: {design_name}
   Connect ports by NAME using .portname(signal) — never by position.
5. Clock generation:
   Use ONLY the clock ports listed in DYNAMIC TESTBENCH ENVIRONMENT.
   For one clock, use always #5 <clock> = ~<clock>; with initial <clock> = 0.
   For multiple clocks (wr_clk/rd_clk, source/destination CDC clocks), generate
   independent clocks with different half-periods, e.g. #5 and #7.
   WRONG: inventing a bare clk when the port_interface has wr_clk and rd_clk.

6. CRITICAL TIMING RULE:
   If the DUT has a clock, it registers inputs on POSEDGE of that detected clock.
   Outputs are valid AFTER posedge. You MUST wait one full detected-clock cycle
   between applying registered inputs and reading registered outputs.

   If DYNAMIC TESTBENCH ENVIRONMENT says no clock-like port was found, this is a
   combinational testbench: DO NOT use @(posedge clk), @(negedge clk), or invent
   a clock. Drive inputs, wait #5 or #10 for combinational settle, then check
   outputs.

   MANDATORY PATTERN:
     @(negedge clk);          // apply inputs here (halfway before posedge)
     @(negedge clk);          // posedge at t+5 registered inputs; NOW read outputs
     if (out !== expected) ...  // check here, after posedge has settled

   WRONG — never do this:
     @(negedge clk); apply_inputs;
     @(posedge clk); check_outputs;  // RACE: posedge registering right now

   WRONG — never do this either:
     apply_inputs; #1; check_outputs;  // #1 does not guarantee register updated

   CORRECT WORKED EXAMPLE for a shift register (shifting in 1 then 0):
     // After reset: parallel_out = 8'b0
     @(negedge clk); shift_en=1; serial_in=1;  // apply at negedge t=5
     @(negedge clk);                             // posedge t=10 registered it
     // parallel_out is now 8'b00000001
     if (parallel_out !== 8'h01) $display("FAIL: expected 01 got %02h", parallel_out);
     @(negedge clk); shift_en=1; serial_in=0;  // apply at negedge t=15
     @(negedge clk);                             // posedge t=20 registered it
     // parallel_out is now 8'b00000010
     if (parallel_out !== 8'h02) $display("FAIL: expected 02 got %02h", parallel_out);

7. NO WHILE LOOPS. Use only bounded repeat() or for loops with fixed count.
   NO break OR continue — these are SystemVerilog-only keywords; Icarus rejects them.
   NO loop variable declaration inside the for() statement:
     WRONG:  for (integer i = 0; i < 8; i = i + 1)  // SV only — Icarus rejects
     CORRECT: integer i;  // declare at TOP of module
              ...
              for (i = 0; i < 8; i = i + 1)          // use pre-declared variable
   If you need early loop exit, use a named block with disable:
     begin : loop_name
       for (i = 0; i < MAX; i = i + 1) begin
         if (done) disable loop_name;
       end
     end
8. GOLDEN MODEL: Implement ONLY the master specification above.
   If it contains an operation_table/truth_table/opcode map, copy that exact
   mapping into the golden model. Do not invent alternate opcode meanings.
   Do not add wrap-around, saturation, or any behavior not in the spec.
   Combinational outputs (flags): check at sample negedge, same cycle as other outputs.
9. EVERY code path must call $finish.
10. On pass: $display("SIMULATION_SUCCESS"); $finish;
11. Timeout:
    Prefer cycle-based timeout using the primary detected clock and the derived
    timeout cycle budget above. For long-latency specs, do not use fixed #100000
    if the operation needs more simulated time.
12. COVERAGE BINS: When edge cases occur, emit:
    $display("COVER_HIT: <scenario_name>");
    e.g. COVER_HIT:overflow, COVER_HIT:reset_during_shift, COVER_HIT:shift_en_low
{protocol_rule}{error_context}

Return pure JSON:
{{
  "testbench_code": "..."
}}
"""
    result  = run_llm(prompt, system_prompt, provider, max_api_retries=2, cand_idx=cand_idx)
    tb_code = normalize_testbench_text(clean_code_string(result.get("testbench_code", "")))

    # Post-process: inject Verilog-2001 protocol assertions before endmodule
    # for ready/valid designs, if the LLM didn't already include them.
    if tb_code and needs_protocol_assertions and "ASSERTION FAILED" not in tb_code:
        assertion_block = build_protocol_assertions(port_interface, design_name)
        tb_code = tb_code.rstrip()
        if tb_code.endswith("endmodule"):
            tb_code = tb_code[:-len("endmodule")].rstrip()
            tb_code = tb_code + "\n\n" + assertion_block + "\n\nendmodule"
            tb_code = normalize_testbench_text(tb_code)

    # Structural validation — reject before it reaches Icarus and wastes a round-trip
    if not tb_code:
        return ""
    stripped = tb_code.strip()
    if not validate_generated_testbench_structure(stripped, cand_idx, port_interface):
        save_debug_artifact(
            design_name, "rejected_tb", round_id, cand_idx + 1,
            f"{design_name}_tb_rejected.v", stripped,
        )
        return ""
    if re.search(r'\bwhile\s*\(', stripped):
        print(f"      ⚠️  TB {cand_idx+1}: while loop emitted despite bounded-loop rule — discarding.")
        save_debug_artifact(
            design_name, "rejected_tb", round_id, cand_idx + 1,
            f"{design_name}_tb_rejected.v", stripped,
        )
        return ""
    if not re.search(r'\bmodule\b', stripped):
        print(f"      ⚠️  TB {cand_idx+1}: Missing module declaration — discarding.")
        save_debug_artifact(
            design_name, "rejected_tb", round_id, cand_idx + 1,
            f"{design_name}_tb_rejected.v", stripped,
        )
        return ""
    if not re.search(r'\bendmodule\b', stripped):
        print(f"      ⚠️  TB {cand_idx+1}: Missing endmodule — discarding.")
        save_debug_artifact(
            design_name, "rejected_tb", round_id, cand_idx + 1,
            f"{design_name}_tb_rejected.v", stripped,
        )
        return ""
    tb_code = normalize_testbench_text(tb_code)
    # Strip any stray markdown fences that slipped through clean_code_string
    tb_code = re.sub(r'```[a-z]*\n?', '', tb_code).strip()

    if tb_code:
        TB_CACHE.put(cache_key, tb_code)
    return tb_code

# ==============================================================================
# PORT MATCHING
# ==============================================================================
def _parse_ansi_module_ports(port_str: str) -> list:
    """Parse simple ANSI-style Verilog port declarations from a module header."""
    ports = []
    current_direction = None
    for raw_part in port_str.split(","):
        part = re.sub(r'//.*', '', raw_part).strip()
        if not part:
            continue
        direction_m = re.search(r'\b(input|output|inout)\b', part)
        if direction_m:
            current_direction = direction_m.group(1)
        if not current_direction:
            continue
        cleaned = re.sub(r'\b(input|output|inout|wire|reg|signed)\b', ' ', part)
        cleaned = re.sub(r'\[[^\]]+\]', ' ', cleaned)
        identifiers = re.findall(r'[A-Za-z_]\w*', cleaned)
        if identifiers:
            ports.append({"name": identifiers[-1], "direction": current_direction})
    return ports


def ports_match(v_code: str, locked_ports_json: str) -> bool:
    if not locked_ports_json:
        return True
    locked_ports = json.loads(locked_ports_json)

    # Handles both inline and multiline module declarations:
    #   module foo (input clk, ...);
    #   module foo\n(\n  input clk,\n  ...\n);
    m = re.search(r'module\s+\w+.*?\((.*?)\)\s*;', v_code, re.S)
    if not m:
        return False

    expected = {
        str(p.get("name", "")): str(p.get("direction", "")).lower()
        for p in locked_ports
    }
    actual = {
        p["name"]: p["direction"].lower()
        for p in _parse_ansi_module_ports(m.group(1))
    }
    return actual == expected

# ==============================================================================
# WORKSPACE + VERIFICATION
# ==============================================================================
def save_to_workspace(design_name: str, v_code: str, tb_code: str, candidate_id: int):
    workspace = f"./workspace/{design_name}/candidate_{candidate_id}"
    os.makedirs(f"{workspace}/src", exist_ok=True)
    os.makedirs(f"{workspace}/tb",  exist_ok=True)

    v_file  = f"{workspace}/src/{design_name}.v"
    tb_file = f"{workspace}/tb/{design_name}_tb.v"

    with open(v_file,  "w") as f: f.write(v_code)
    with open(tb_file, "w") as f: f.write(tb_code)
    return v_file, tb_file

def verify_with_verilator(v_file: str, design_name: str, v_code: str):
    if not re.search(r'\bmodule\s+' + re.escape(design_name) + r'\b', v_code):
        return False, f"%Error: Top module '{design_name}' not found."
    plan = resolve_link_plan(v_file, design_name)
    err = link_error(plan)
    if err:
        return False, err
    lint_flags = [
        "-Wno-DECLFILENAME", "-Wno-UNUSED", "-Wno-UNDRIVEN",
        "-Wno-PINCONNECTEMPTY",
    ]
    if plan.get("files"):
        # Verified child modules may use different reset styles. At hierarchy
        # boundaries this is a lint policy warning, not proof the top RTL is
        # functionally wrong. Keep width/implicit-net errors fatal.
        lint_flags.append("-Wno-SYNCASYNCNET")
    cmd = ["verilator", "--lint-only", "-Wall",
           *lint_flags,
           "-Werror-WIDTH", "-Werror-IMPLICIT",
           "--top-module", design_name] + plan["files"] + [v_file]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if "UNOPTFLAT" in r.stderr:
            return False, "COMBINATIONAL LOOP DETECTED (UNOPTFLAT):\n" + truncate_log(r.stderr, 25)
        if r.returncode == 0:
            return True, "Clean"
        return False, truncate_log(r.stderr)
    except subprocess.TimeoutExpired:
        return False, "Verilator timeout."

def verify_with_iverilog(v_file: str, tb_file: str, design_name: str, candidate_id: int):
    workspace  = f"./workspace/{design_name}/candidate_{candidate_id}"
    output_bin = f"{workspace}/{design_name}_sim.vvp"
    plan = resolve_link_plan(v_file, design_name)
    err = link_error(plan)
    if err:
        return False, err, 0

    cr = subprocess.run(
        ["iverilog", "-o", output_bin] + plan["files"] + [v_file, tb_file],
        capture_output=True, text=True
    )
    if cr.returncode != 0:
        return False, f"Compile failed:\n{truncate_log(cr.stderr)}", 0

    try:
        sr = subprocess.run(["vvp", output_bin],
                            capture_output=True, text=True, timeout=60)
        cover_hits = len(set(re.findall(r'COVER_HIT:\s*(\w+)', sr.stdout)))

        if sr.returncode != 0:
            return False, f"Runtime failed:\n{truncate_log(sr.stderr)}", cover_hits
        if any(kw in sr.stdout for kw in
               ("ERROR", "Error", "FAIL", "Fail", "ASSERTION FAILED")):
            return False, f"Verification failed:\n{truncate_log(sr.stdout, 15)}", cover_hits
        if "SIMULATION_SUCCESS" not in sr.stdout:
            return False, "Missing SIMULATION_SUCCESS — simulation exited prematurely.", cover_hits
        return True, "Simulation clean.", cover_hits
    except subprocess.TimeoutExpired:
        return False, "Simulation timeout (possible infinite loop).", 0

def verify_with_yosys(
    v_file: str,
    base_prompt: str,
    design_name: str,
    design_type: str = None,
    clarified_spec: dict = None,
):
    plan = resolve_link_plan(v_file, design_name)
    err = link_error(plan)
    if err:
        return False, err
    prompt_l = base_prompt.lower()
    top_is_hierarchy = (
        design_type in HIERARCHICAL_TYPES
        or looks_like_hierarchical_integration({}, base_prompt)
    )
    # Match hardware-memory terms as tokens. A raw substring check for "ram"
    # incorrectly classified ordinary words such as "frame" and "programmed".
    prompt_memoryish = bool(re.search(
        r"\b(?:memory|sram|bram|ram|byte_en|mem_depth)\b|byte[ -]enable",
        prompt_l,
        re.IGNORECASE,
    ))
    # Atomic memory primitives must preserve an inferable memory cell. Hierarchy
    # tops may merely instantiate verified memories, and output-less wrappers can
    # be optimized away entirely; do not re-apply the primitive SRAM contract here.
    memoryish = prompt_memoryish and not top_is_hierarchy
    preserve_memory_flow = memoryish or (
        top_is_hierarchy and plan.get("contains_memory", False)
    )
    read_cmds = yosys_read_commands(plan["files"] + [v_file])
    if preserve_memory_flow:
        memory_passes = "memory_collect" if design_type == "async_fifo" else "memory_dff; memory_collect"
        yosys_script = (
            f"{read_cmds} hierarchy -top {design_name}; proc; opt; {memory_passes}; "
            "check -assert; stat"
        )
    else:
        yosys_script = (
            f"{read_cmds} hierarchy -top {design_name}; proc; opt; fsm; memory; "
            "check -assert; stat"
        )
    cmd = ["yosys", "-p", yosys_script]
    try:
        r   = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        log = r.stderr + "\n" + r.stdout
        if r.returncode == 0 and "ERROR" not in log:
            dff_by_type = {}
            for cell_type, n in re.findall(
                r'^\s+(\$(?:a?dff|dffe|sdff))\s+(\d+)\s*$',
                log,
                re.IGNORECASE | re.MULTILINE,
            ):
                dff_by_type[cell_type.lower()] = int(n)
            dffs = sum(dff_by_type.values())
            cell_matches = re.findall(r'Number of cells:\s+(\d+)', log)
            mem_bit_matches = re.findall(r'Number of memory bits:\s+(\d+)', log)
            cells = cell_matches[-1] if cell_matches else "?"
            mem_bits = int(mem_bit_matches[-1]) if mem_bit_matches else 0
            mem_cells = sum(
                int(n) for n in re.findall(r'^\s+\$(?:mem|memrd|memwr)\s+(\d+)\s*$',
                                           log, re.IGNORECASE | re.MULTILINE)
            )
            if (not top_is_hierarchy) and dffs == 0 and expects_sequential_storage(
                base_prompt,
                v_file=v_file,
                design_type=design_type,
                clarified_spec=clarified_spec,
            ):
                return False, (
                    "SEQUENTIAL_OPTIMIZED_AWAY: Yosys synthesized 0 DFFs for a design "
                    "that should be sequential. The state/register logic was optimized "
                    "away, usually because outputs do not depend on registered state, "
                    "the reset/state transition is broken, or registers are unread."
                )
            if memoryish and ("Replacing memory" in log or mem_cells == 0):
                return False, (
                    "MEMORY_NOT_INFERRED: Yosys did not preserve an inferable memory cell. "
                    "Avoid async/combinational memory reads, memory reset loops, and "
                    "read-before-write merge logic. Use a synchronous SRAM template with "
                    "byte-lane writes inside always @(posedge clk) and registered rdata."
                )
            notes = []
            if "Replacing memory" in log:
                notes.append("memory array mapped to registers")
            if mem_cells > 0:
                notes.append(f"memory cells: {mem_cells}")
            if mem_bits > 0:
                notes.append(f"memory bits: {mem_bits}")
            if top_is_hierarchy and plan.get("contains_memory", False):
                notes.append("linked memory deps")
            note_text = f", Notes: {'; '.join(notes)}" if notes else ""
            link_note = f", {format_link_summary(plan)}" if plan.get("files") else ""
            return True, f"Cells: {cells}, DFF cells: {dffs}{note_text}{link_note}"
        return False, truncate_log(log, 20)
    except subprocess.TimeoutExpired:
        if memoryish:
            return False, (
                "YOSYS_TIMEOUT: synthesis timed out, likely because a memory was not "
                "inferable and exploded into registers/muxes. Use a synchronous SRAM/BRAM "
                "shape: reg [31:0] mem [0:MEM_DEPTH-1]; no memory reset loop; no "
                "combinational/asynchronous memory read such as assign word = mem[addr]; "
                "no read-before-write merge outside the clocked block; perform byte-lane "
                "writes under if (byte_en[i]) inside always @(posedge clk), and register "
                "rdata from mem[word_addr]."
            )
        return False, (
            "YOSYS_TIMEOUT: synthesis timed out. The RTL likely caused structural "
            "explosion, a very large inferred mux/register network, or an optimization "
            "loop. Reduce combinational fanout and ensure arrays/resources infer as "
            "intended primitives."
        )

# ==============================================================================
# MAIN EVOLUTIONARY BUILD LOOP
# ==============================================================================
def autonomous_build_loop(
    base_prompt:          str,
    design_name:          str,
    provider:             str,
    max_retries:          int = 10,
    candidates_per_round: int = 3,
) -> bool:

    print(f"\n🚀 ChipGPT | Design: '{design_name}' | Provider: {provider.upper()}")
    print(f"   🧠 Fast model  : {FAST_MODEL}")
    print(f"   🔥 Strong model: {STRONG_MODEL}\n")

    clarified_spec = generate_specification(base_prompt, provider)
    if not clarified_spec:
        print("   ❌ Clarifier returned empty spec.")
        return False
    clarified_spec = normalize_clarified_spec(clarified_spec, base_prompt)

    # Compute once — used for skeleton injection and invariant checking
    requested_port_names = extract_requested_port_names(base_prompt)
    if requested_port_names:
        clarified_spec["_user_requested_ports"] = requested_port_names

    design_type = detect_design_type(clarified_spec, base_prompt)
    clarified_spec["_detected_design_type"] = design_type
    spec_prompt = build_master_spec_prompt(clarified_spec, design_name)
    if design_type == "sar_adc_controller":
        sar_config = _sar_contract_parameters(clarified_spec)
        print(
            "   📑 SAR contract: "
            f"resolution={sar_config['resolution']}, "
            f"settle_cycles={sar_config['settle_cycles']}, "
            f"comparator high {'keeps' if sar_config['comparator_high_keeps_bit'] else 'clears'} trial bit, "
            f"sample={sar_config['sample_event']}, "
            f"contract_hash={sar_config['contract_hash']}"
        )
        if sar_config["sample_event"] != "posedge clk":
            print(
                "   ❌ CONTRACT_ERROR: this SAR controller contract currently supports "
                f"posedge clk sampling; requested sample_event={sar_config['sample_event']}"
            )
            return False
    if clarified_spec.get("assumptions_made"):
        preview = "; ".join(str(a) for a in clarified_spec["assumptions_made"][:2])
        print(f"   📌 Spec-Sync assumptions: {preview[:220]}")
    if design_type in SKELETON_LIBRARY:
        print(f"   🧩 Design type detected: {design_type} (skeleton available)")

    elif has_protocol_oracle(design_type):
        print(f"   Design type detected: {design_type} (protocol oracle available)")
    elif design_type in HIERARCHICAL_TYPES:
        print(f"   🧷 Design type detected: {design_type} (verified-module linker path)")

    arch_hashes          = set()
    code_hashes          = set()
    locked_ports         = None
    arch_failures        = {}   # arch_hash → set of failure_type strings
    error_log            = None
    architecture         = None
    architecture_flawed  = True
    invalid_arch_count   = 0
    duplicate_arch_count = 0

    # Tuple scoring: (pipeline_stage, secondary_quality)
    # stage: 0=no lint pass, 1=lint, 2=sim, 3=synth, 4=gate-level
    best_score           = (-1, -999999)
    best_v_code          = None
    best_tb_code         = None
    best_err_log         = None
    winning_v_file       = None
    winning_tb_file      = None
    winning_candidate_id = None
    winning_yosys_log    = None
    global_candidate_id  = 0
    consecutive_sim_fails = 0
    last_err_fingerprint  = None
    same_err_count        = 0
    pending_invariant_msgs = []
    empty_rounds           = 0    # consecutive rounds with zero valid RTL candidates
    reviewer_rejected_streak = 0  # consecutive rounds reviewer added/removed blocks without helping
    initial_failing_code   = None
    dataflow_override_announced = False

    for attempt in range(max_retries):
        print(f"\n{'─'*60}")
        print(f"🔄 Generation {attempt + 1}/{max_retries}")
        print(f"{'─'*60}")

        # ── Architecture redesign triggers ────────────────────────────────────
        trigger_redesign = False
        if consecutive_sim_fails >= 5:
            trigger_redesign = True
            print("   🔄 5 consecutive sim failures — forcing full architecture redesign...")
        elif architecture:
            current_arch_hash = code_hash(json.dumps(architecture, sort_keys=True))
            if len(arch_failures.get(current_arch_hash, set())) >= 3:
                trigger_redesign = True
                print("   🔄 3 distinct failure types in this architecture — redesigning...")

        if trigger_redesign:
            architecture_flawed   = True
            consecutive_sim_fails = 0
            best_score            = (-1, -999999)
            best_v_code           = None
            best_tb_code          = None
            locked_ports          = None
            code_hashes           = set()   # fresh arch → fresh candidate space
            fail_history_str = "\n".join(
                f"- Arch {h[:6]}: " + ", ".join(reasons)
                for h, reasons in arch_failures.items()
            )
            error_log = (
                f"CRITICAL: Previous architectures failed.\n"
                f"Failure History:\n{fail_history_str}\n"
                f"Produce a structurally DIFFERENT blueprint."
            )

        if architecture_flawed:
            enforce_strict_dataflow = _should_enforce_strict_dataflow(
                spec_prompt,
                clarified_spec=clarified_spec,
                requested_port_names=requested_port_names,
            )
            if (
                _requires_dataflow_modeling(spec_prompt)
                and not enforce_strict_dataflow
                and not dataflow_override_announced
            ):
                print("   🔧 Dataflow style disabled for clocked/sequential design.")
                dataflow_override_announced = True
            enforced_style = (
                "continuous_assign_dataflow"
                if enforce_strict_dataflow
                else ARCH_STYLES[attempt % len(ARCH_STYLES)]
            )
            architecture   = generate_architecture(
                clarified_spec, provider, error_log, attempt, enforced_style
            )
            if _override_clocked_dataflow_architecture(architecture):
                print("   🔧 Arch style overridden to sequential (clock ports detected).")
            arch_hash = code_hash(json.dumps(architecture, sort_keys=True))

            arch_ok, arch_invalid_reason = validate_architecture_with_reason(
                architecture,
                expected_port_names=requested_port_names,
                clarified_spec=clarified_spec,
                user_prompt=spec_prompt,
            )
            if not arch_ok:
                invalid_arch_count += 1
                if invalid_arch_count > 5:
                    print("   ❌ FATAL: Architect stuck in invalid schema loop.")
                    return False
                print(f"   ⚠️  Architect returned invalid blueprint ({arch_invalid_reason}). Retrying...")
                if requested_port_names:
                    error_log = (
                        "CRITICAL: Blueprint port_interface must contain exactly these "
                        f"user-requested port names, no renames: {requested_port_names}. "
                        f"Last rejection reason: {arch_invalid_reason}"
                    )
                else:
                    error_log = (
                        "CRITICAL: Blueprint missing required fields. "
                        f"Last rejection reason: {arch_invalid_reason}"
                    )
                continue

            if arch_hash in arch_hashes:
                duplicate_arch_count += 1
                if duplicate_arch_count > 5:
                    print("   ⚠️  Architect stuck in duplicate loop. Forcing acceptance.")
                    architecture_flawed  = False
                    duplicate_arch_count = 0
                else:
                    print("   ⚠️  Duplicate blueprint. Forcing variation...")
                    error_log = "CRITICAL: Do not repeat previous blueprints. Change structure."
                    continue
            else:
                arch_hashes.add(arch_hash)
                architecture_flawed  = False
                duplicate_arch_count = 0
                invalid_arch_count   = 0
                print(f"   📐 Architect: New blueprint (class={architecture.get('module_class','?')}, style={enforced_style})")
        else:
            print("   📐 Architect: Reusing stable blueprint.")

        # ── RTL generation (parallel) ─────────────────────────────────────────
        if best_v_code:
            print(f"   🧬 Evolution Engine: Mutating best RTL (score=Stage {best_score[0]})...")
        else:
            print(f"   🛠️  Generator: Spawning {candidates_per_round} parallel RTL candidates...")

        rtl_candidates = []
        _port_mismatch_count = 0
        rtl_error_context = best_err_log or error_log
        if rtl_error_context and classify_failure(rtl_error_context) in STRUCTURAL_TB_FAILURES:
            rtl_error_context = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidates_per_round) as executor:
            futures = {}
            for i in range(candidates_per_round):
                time.sleep(2)
                dynamic_idx = i + same_err_count  # shift temperature on repeated errors
                futures[executor.submit(
                    generate_rtl,
                    spec_prompt, provider, design_name, architecture,
                    clarified_spec,
                    locked_ports, rtl_error_context, best_v_code,
                    dynamic_idx,
                    design_type,
                )] = i

            try:
                for future in concurrent.futures.as_completed(futures, timeout=420):
                    try:
                        result = future.result(timeout=200)
                    except Exception as e:
                        print(f"      âš ï¸  RTL worker {futures[future]+1} crashed: {e}")
                        print(traceback.format_exc().rstrip())
                        continue

                    v_code = clean_code_string(result.get("verilog_code", ""))
                    if len(v_code) < 50:
                        continue
                    if design_type == "sar_adc_controller":
                        v_code = _attach_sar_contract_identity(v_code, clarified_spec)
                    if locked_ports and not ports_match(v_code, locked_ports):
                        print("   ⚠️  Port interface mismatch — candidate discarded.")
                        _port_mismatch_count += 1
                        continue

                    h = code_hash(v_code)
                    if h in code_hashes:
                        print("   ⚠️  Duplicate candidate — discarded.")
                        continue
                    code_hashes.add(h)

                    # Auto-patch narrow-register == PARAMETER width mismatches
                    v_code = auto_patch_param_width(v_code)

                    if design_type == "sar_adc_controller":
                        contract_ok, contract_error = validate_sar_contract_rtl(v_code, clarified_spec)
                        if not contract_ok:
                            print(f"   ⚠️  {contract_error} — candidate discarded.")
                            pending_invariant_msgs.append(contract_error)
                            continue

                    # Static BLKANDNBLK check: reject before Verilator wastes a call
                    blk_err = _check_blocking_mix(v_code)
                    if blk_err:
                        print(f"   ⚠️  BLKANDNBLK detected ({blk_err}) — candidate discarded.")
                        continue

                    if _should_enforce_strict_dataflow(
                        spec_prompt,
                        architecture=architecture,
                        clarified_spec=clarified_spec,
                        requested_port_names=requested_port_names,
                        v_code=v_code,
                    ):
                        _df_clean = re.sub(r'/\*.*?\*/', '', v_code, flags=re.DOTALL)
                        _df_clean = re.sub(r'//.*', '', _df_clean)
                        if re.search(r'\balways\b|\binitial\b|\breg\b|\btask\b|\bfunction\b', _df_clean):
                            msg = (
                                "Strict dataflow modeling violated: DUT contains always/initial/"
                                "reg/task/function. Use continuous assign statements only."
                            )
                            print(f"   ⚠️  {msg} Candidate discarded.")
                            pending_invariant_msgs.append(msg)
                            continue

                    state_err = _check_duplicate_state_encodings(v_code)
                    if state_err:
                        print(f"   Warning: {state_err} - candidate discarded.")
                        pending_invariant_msgs.append(state_err)
                        continue

                    if design_type in {"uart_tx", "axi_stream_uart_tx"} or "uart" in spec_prompt.lower():
                        phase_err = _check_ungated_uart_phase_transition(v_code)
                        if phase_err:
                            print(f"   Warning: {phase_err} - candidate discarded.")
                            pending_invariant_msgs.append(phase_err)
                            continue

                    # Skeleton invariant check: reject if frozen structural lines were removed
                    if design_type in SKELETON_LIBRARY:
                        inv_ok, violations = check_skeleton_invariants(v_code, design_type)
                        if not inv_ok:
                            msg = violations[0][:100]
                            print(f"   ⚠️  Skeleton invariant violated ({msg[:60]}) — discarding.")
                            pending_invariant_msgs.append(msg)
                            continue

                    has_memory_blocks = bool(architecture.get("memory_blocks"))
                    if (architecture.get("module_class") == "DATAPATH"
                            and has_memory_blocks
                            and design_type not in HIERARCHICAL_TYPES
                            and not re.search(r'reg\s*\[[^\]]+\]\s*\w+\s*\[[^\]]+\]', v_code)):
                        print("   ⚠️  DATAPATH with internal memory_blocks missing 2D reg. Discarding.")
                        continue

                    # All validations passed — safe to cache now
                    ck = result.get("_cache_key")
                    if ck:
                        RTL_CACHE.put(ck, result)

                    rtl_candidates.append(v_code)

            except concurrent.futures.TimeoutError:
                print("   ⚠️  RTL executor timeout. Restarting round.")

        if rtl_candidates:
            empty_rounds = 0  # reset streak when we get valid candidates
        else:
            empty_rounds += 1

        if not rtl_candidates:
            print("   ❌ No valid RTL candidates this round.")
            # After 3 empty rounds in a row, reset code_hashes and best_v_code
            # to force fresh generation instead of endlessly mutating broken code
            if empty_rounds >= 3:
                print(f"   🔄 {empty_rounds} empty rounds — clearing cache and regenerating fresh.")
                code_hashes.clear()
                # Also evict RTL_CACHE entries for this architecture so cache hits
                # don't re-serve the same broken RTL after the reset.
                # BoundedCache doesn't support selective eviction, so clear entirely.
                RTL_CACHE.cache.clear()
                best_v_code  = None
                empty_rounds = 0
                error_log = (best_err_log or
                             "All mutation attempts produced no valid candidates. Regenerating fresh.")
                continue
            if _port_mismatch_count > 0:
                print("   🔓 Port interface lock cleared (all candidates mismatched).");locked_ports = None;code_hashes.clear()
                error_log = "CRITICAL: All generated RTL had wrong port interface. Match port_interface array exactly — no added, removed, or renamed ports."
            elif pending_invariant_msgs:
                unique_msgs = list(dict.fromkeys(pending_invariant_msgs))
                inv_lines = "\n".join(f"  - {m}" for m in unique_msgs[:3])
                error_log = "STRUCTURAL INVARIANT VIOLATIONS:\n" + inv_lines + "\nFix these first."
                pending_invariant_msgs.clear()
            else:
                code_hashes.clear()
                error_log = "All RTL candidates were empty, had wrong ports, or were duplicates."
            continue

        # ── Reviewer pass ─────────────────────────────────────────────────────
        # Skip reviewer for skeleton-matched designs: the skeleton + invariant
        # checker already enforces structure. Reviewer adds always blocks ~90%
        # of the time on these designs, wasting API calls with no benefit.
        # Skip reviewer for skeleton designs, OR when reviewer has consistently
        # failed to improve candidates (adds/removes blocks every round = wasted API calls)
        dataflow_required = _should_enforce_strict_dataflow(
            spec_prompt,
            architecture=architecture,
            clarified_spec=clarified_spec,
            requested_port_names=requested_port_names,
        )
        skip_reviewer = (
            (design_type != "generic")
            or dataflow_required
            or (reviewer_rejected_streak >= 3)
        )
        if skip_reviewer:
            if dataflow_required:
                reason = "strict dataflow design"
            elif design_type in HIERARCHICAL_TYPES:
                reason = "hierarchical linker design"
            elif design_type != "generic":
                reason = "skeleton/protocol design"
            else:
                reason = f"reviewer unhelpful for {reviewer_rejected_streak} rounds"
            print(f"   🔎 Reviewer: Skipped ({reason}).")
            reviewed_candidates = list(rtl_candidates)
            reviewer_rejected_streak += 1  # keep counting; resets when reviewer helps
        else:
            print(f"   🔎 Reviewer: Scanning {len(rtl_candidates)} RTL candidate(s)...")
            reviewed_candidates = []
        for idx, v_code in enumerate(rtl_candidates if not skip_reviewer else []):
            review = review_hardware(v_code, provider)
            if review.get("status") == "REJECTED":
                fixed = clean_code_string(review.get("fixed_code", ""))
                if (fixed
                        and "module " in fixed
                        and "endmodule" in fixed
                        and design_name in fixed):
                    has_assign_before = bool(re.search(r'\bassign\b', v_code))
                    has_assign_after  = bool(re.search(r'\bassign\b', fixed))
                    always_before     = len(re.findall(r'\balways\b', v_code))
                    always_after      = len(re.findall(r'\balways\b', fixed))
                    if has_assign_before and not has_assign_after:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer removed assigns. Keeping original.")
                        reviewed_candidates.append(v_code)
                    elif always_after < always_before:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer removed always block. Keeping original.")
                        reviewed_candidates.append(v_code)
                    elif always_after > always_before:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer added always block (likely spurious latch fix). Keeping original.")
                        reviewed_candidates.append(v_code)
                    elif 0.75 < len(fixed) / len(v_code) < 1.3:
                        # Final BLKANDNBLK check on the patched code before accepting
                        blk_err = _check_blocking_mix(fixed)
                        if blk_err:
                            print(f"      ⚠️  Candidate {idx+1}: Reviewer introduced BLKANDNBLK ({blk_err}). Keeping original.")
                            reviewed_candidates.append(v_code)
                        else:
                            print(f"      ✅ Candidate {idx+1}: Patch applied.")
                            reviewed_candidates.append(fixed)
                            reviewer_rejected_streak = 0  # reviewer helped — reset streak
                    else:
                        print(f"      ⚠️  Candidate {idx+1}: Hallucinated rewrite. Keeping original.")
                        reviewed_candidates.append(v_code)
                else:
                    print(f"      ⚠️  Candidate {idx+1}: Reviewer broke module structure. Keeping original.")
                    reviewed_candidates.append(v_code)
            else:
                reviewed_candidates.append(v_code)

        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture["port_interface"])
            print(f"   🔒 Port interface locked ({len(architecture['port_interface'])} ports).")

        # ── Testbench generation ──────────────────────────────────────────────
        for idx, v_code in enumerate(reviewed_candidates):
            save_debug_artifact(
                design_name, "preverify_rtl", attempt + 1, idx + 1,
                f"{design_name}_preverify.v", v_code,
            )

        _structural_tb_failures = STRUCTURAL_TB_FAILURES
        tb_failure_type   = classify_failure(best_err_log) if best_err_log else None
        tb_is_logic_failure = (
            best_tb_code is not None
            and tb_failure_type is not None
            and tb_failure_type not in _structural_tb_failures
        )

        port_interface = architecture.get("port_interface", [])

        print(f"   📋 Testbench Agent: ", end="")
        if tb_is_logic_failure:
            print("Reusing locked testbench (logic failure — TB is correct fitness function).")
        else:
            print(f"Generating {len(reviewed_candidates)} testbench(es) in parallel...")

        def _gen_tb(args):
            idx, v_code = args
            if tb_is_logic_failure:
                return idx, v_code, best_tb_code
            # Only pass previous TB error context when the previous TB was structurally
            # broken (syntax error, timeout). For logic failures the TB is locked above.
            # Passing a logic-failure error as "fix the testbench" context to a fresh
            # generation would tell the model the RTL is correct and corrupt a good TB.
            tb_error = best_err_log if (
                best_tb_code is not None
                and tb_failure_type in _structural_tb_failures
            ) else None
            prev_tb = best_tb_code if tb_error else None
            tb = generate_testbench(
                spec_prompt, provider, design_name,
                port_interface,
                tb_error,
                prev_tb,
                idx,
                design_type,
                attempt + 1,
                clarified_spec,
            )
            return idx, v_code, tb

        candidates = []
        tb_workers = max(1, min(2, len(reviewed_candidates)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=tb_workers) as tb_executor:
            tb_futures = [
                tb_executor.submit(_gen_tb, (idx, v_code))
                for idx, v_code in enumerate(reviewed_candidates)
            ]
            try:
                for tb_future in concurrent.futures.as_completed(tb_futures, timeout=300):
                    try:
                        idx, v_code, tb_code = tb_future.result(timeout=200)
                    except Exception as e:
                        print(f"      ⚠️  Testbench generation failed: {e}")
                        print(traceback.format_exc().rstrip())
                        continue
                    if not tb_code or len(tb_code) < 50:
                        print(f"      ⚠️  Testbench {idx+1}: Empty. Discarding.")
                        continue
                    status = "🔒 Locked" if tb_is_logic_failure else "✅ Generated"
                    print(f"      {status} Testbench {idx+1}.")
                    candidates.append((v_code, tb_code))
            except concurrent.futures.TimeoutError:
                for f in tb_futures:
                    f.cancel()
                print(f"   ⚠️  Testbench timeout — {len(candidates)} collected before cutoff.")

        if not candidates:
            print("   ❌ No valid candidates after testbench generation.")
            TB_CACHE.cache.clear()
            RTL_CACHE.cache.clear()
            code_hashes.clear()
            best_tb_code = None
            best_err_log = (
                "TB_STRUCTURAL: all generated testbenches were empty or rejected "
                "by Verilog-2001 preflight. Regenerate the testbench/oracle; do "
                "not mutate RTL from this failure."
            )
            error_log = best_err_log
            print("   🔄 Cleared RTL/TB caches after total TB failure to avoid duplicate replay.")
            continue

        # ── Verification ──────────────────────────────────────────────────────
        round_passed     = False
        round_sim_passed = False

        for v_code, tb_code in candidates:
            global_candidate_id += 1
            print(f"\n   🔬 Testing Candidate #{global_candidate_id}...")

            v_file, tb_file  = save_to_workspace(design_name, v_code, tb_code, global_candidate_id)
            current_score    = (0, 0)

            passed, log = verify_with_verilator(v_file, design_name, v_code)
            if not passed:
                err = extract_primary_error(log)
                if design_type == "sar_adc_controller":
                    category = "IMPLEMENTATION_SYNTAX_ERROR" if "syntax" in log.lower() else "STRUCTURAL_ERROR"
                    err = f"{category}: {err}"
                print(f"      ❌ Verilator: {err}")
                current_score = (0, -len(err))
                if current_score >= best_score:
                    best_score   = current_score
                    best_err_log = f"Syntax/Lint Error:\n{err}"
                    # Save v_code even at stage 0 so mutation engine has something
                    # to mutate from next round. Remove its hash from code_hashes so
                    # mutations of this base are not blocked as duplicates next round.
                    best_v_code  = v_code
                    code_hashes.discard(code_hash(v_code))
                continue

            print("      ✅ Verilator passed.")
            passed, log, cover_hits = verify_with_iverilog(
                v_file, tb_file, design_name, global_candidate_id
            )
            if not passed:
                err              = extract_primary_error(log)
                full_sim_context = truncate_log(log, 30)
                failure_type     = classify_failure(full_sim_context)
                print(f"      ❌ Icarus [{failure_type}]: {err}")

                if failure_type in STRUCTURAL_TB_FAILURES:
                    print("      ⚠️  Treating this as a testbench/tooling failure; RTL will not be mutated from it.")
                    if best_err_log is None or classify_failure(best_err_log) in STRUCTURAL_TB_FAILURES:
                        best_tb_code = tb_code
                        best_err_log = (
                            f"FAILURE TYPE: {failure_type}\n"
                            f"Testbench/tooling Error (full output):\n{full_sim_context}"
                        )
                    continue

                # Same-error detection → unlock TB if stuck
                err_fp = get_error_fingerprint(failure_type, full_sim_context)
                if err_fp == last_err_fingerprint:
                    same_err_count += 1
                else:
                    last_err_fingerprint = err_fp
                    same_err_count       = 1

                if same_err_count >= 4:
                    print(f"      🔁 Same error repeated {same_err_count}x — unlocking TB.")
                    best_tb_code   = None
                    same_err_count = 0

                mismatches    = len(re.findall(
                    r'(fail|error|mismatch|expected|assert)', log, re.IGNORECASE
                ))
                current_score = (1, (cover_hits * 10) - mismatches)

                if current_score >= best_score:
                    best_score   = current_score
                    best_v_code  = v_code
                    best_tb_code = tb_code
                    if initial_failing_code is None:
                        initial_failing_code = v_code

                    # Extract the specific FAIL assertion lines — these are the most
                    # actionable feedback for the mutation engine, more useful than
                    # the full truncated log which buries them.
                    fail_lines = "\n".join(
                        line for line in log.splitlines()
                        if re.search(r'\bFAIL\b|\bfail\b|expected|mismatch|assertion', line, re.I)
                    )
                    fail_summary = (
                        f"SPECIFIC FAILING ASSERTIONS:\n{fail_lines}\n\n"
                        if fail_lines else ""
                    )

                    best_err_log = (
                        f"FAILURE TYPE: {failure_type}\n"
                        f"{fail_summary}"
                        f"Simulation Error (full output):\n{full_sim_context}"
                    )

                if architecture:
                    ah = code_hash(json.dumps(architecture, sort_keys=True))
                    if ah not in arch_failures:
                        arch_failures[ah] = set()
                    arch_failures[ah].add(failure_type)
                continue

            round_sim_passed = True
            print(f"      ✅ Icarus simulation passed. (Coverage Hits: {cover_hits})")

            passed, log = verify_with_yosys(
                v_file,
                spec_prompt,
                design_name,
                design_type,
                clarified_spec,
            )
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Yosys: {err}")
                current_score = (2, 0)
                if current_score >= best_score:
                    best_score   = current_score
                    best_v_code  = v_code
                    best_tb_code = tb_code
                    best_err_log = f"Synthesis Error:\n{err}"
                    if initial_failing_code is None:
                        initial_failing_code = v_code
                continue

            print(f"      ✅ Yosys passed: {log}")
            current_score = (3, 0)
            if current_score >= best_score:
                best_score   = current_score
                best_v_code  = v_code
                best_tb_code = tb_code

            round_passed = True
            winning_v_file       = v_file
            winning_tb_file      = tb_file
            winning_candidate_id = global_candidate_id
            winning_yosys_log    = log
            break

        if round_passed:
            try:
                entry = register_verified_module(
                    design_name,
                    winning_v_file,
                    candidate_id=winning_candidate_id,
                    tb_path=winning_tb_file,
                    design_type=design_type,
                    prompt=spec_prompt,
                    yosys_summary=winning_yosys_log,
                    verification_contract=(
                        clarified_spec.get("ams_contract")
                        if design_type == "sar_adc_controller"
                        else None
                    ),
                )
                print(f"🧾 Verified-module registry updated: {entry['src']}")
            except Exception as e:
                print(f"⚠️  Registry update failed: {e}")
            if initial_failing_code is not None and best_err_log is not None and winning_v_file:
                try:
                    with open(winning_v_file, "r", encoding="utf-8", errors="replace") as f:
                        winning_code = f.read()
                    print("   🧠 Memory Agent: Extracting lesson from resolution...")
                    commit_tooling_insight(
                        design_name,
                        design_type,
                        best_err_log,
                        initial_failing_code,
                        winning_code,
                        provider,
                    )
                except Exception as e:
                    print(f"   ⚠️  Memory Agent skipped: {e}")
            print(f"\n{'═'*60}")
            print("🎉 PIPELINE COMPLETE — Structural + Functional + Synthesis verification passed.")
            print(f"📁 Winning design: ./workspace/{design_name}/candidate_{winning_candidate_id}/")
            print(f"{'═'*60}\n")
            return True

        if pending_invariant_msgs:
            unique_inv = list(dict.fromkeys(pending_invariant_msgs))
            inv_note = "\nSTRUCTURAL INVARIANTS VIOLATED (fix alongside sim failure):\n" + "\n".join(f"  - {m}" for m in unique_inv[:3])
            best_err_log = (best_err_log + inv_note) if best_err_log else inv_note.strip()
            pending_invariant_msgs.clear()

        # Clear code_hashes when round produces no simulation pass.
        # This allows mutation to produce variants without being blocked by duplicates.
        # Stage 0 (Verilator fail): fresh generation needed
        # Stage 1 (sim fail): mutation needed — clear so mutated variants aren't
        #   blocked by the hash of the original candidate they're derived from.
        if not round_sim_passed and global_candidate_id > 0:
            code_hashes.clear()

        if best_score[0] <= 1 and not round_sim_passed:
            consecutive_sim_fails += 1
        else:
            consecutive_sim_fails = 0

        print(f"\n   📊 Round summary: best_score=Stage {best_score[0]} | "
              f"sim_fails_streak={consecutive_sim_fails} | "
              f"candidates_tested={global_candidate_id}")

    print("\n🚨 Max retries reached without a passing design.")
    return False

def autonomous_soc_loop(
    base_prompt,
    design_name,
    provider,
    max_retries,
    candidates_per_round,
) -> bool:
    """Build missing SoC dependencies first, then build the hierarchical top."""
    print("\n============================================================")
    print("SoC Decomposition Mode")
    print("============================================================")

    plan = decompose_soc_architecture(base_prompt, provider)
    missing_modules = plan.get("missing_modules", [])
    top_level_prompt = plan.get("top_level_prompt", "")

    if not top_level_prompt:
        print("FATAL: SoC architect did not produce a top_level_prompt.")
        return False

    if missing_modules:
        print(f"System Architect: {len(missing_modules)} missing module(s) planned.")
    else:
        print("System Architect: all requested dependencies appear to be registered.")

    for mod in missing_modules:
        mod_name = str(mod.get("name", "")).strip()
        mod_prompt = str(mod.get("prompt", "")).strip()
        if not mod_name or not mod_prompt:
            print(f"FATAL: malformed missing module entry: {mod}")
            return False

        registered_now = set(get_registry_module_names())
        if mod_name in registered_now:
            print(f"\n--- Sub-module already verified: {mod_name} ---")
            continue

        print("\n============================================================")
        print(f"Building missing sub-module: {mod_name}")
        print("============================================================")
        ok = autonomous_build_loop(
            base_prompt=mod_prompt,
            design_name=mod_name,
            provider=provider,
            max_retries=max_retries,
            candidates_per_round=candidates_per_round,
        )
        if not ok:
            print(f"FATAL: required sub-module build failed: {mod_name}")
            return False

    print("\n============================================================")
    print(f"Top-Level Integration: {design_name}")
    print("============================================================")
    available_modules = get_registry_module_names()
    try:
        sigs = format_registered_module_signatures(" ".join(available_modules))
    except TypeError:
        try:
            sigs = format_registered_module_signatures(available_modules)
        except Exception:
            sigs = ""
    except Exception:
        try:
            sigs = format_registered_module_signatures()
        except Exception:
            sigs = ""
    if not sigs:
        sigs = "(no verified module signatures were available)"

    enriched_top_prompt = (
        f"{top_level_prompt}\n\n"
        "CRITICAL PORT CONTRACTS:\n"
        "You are building a hierarchical wrapper. You MUST instantiate the "
        "submodules using EXACTLY these port signatures. Do not invent or guess "
        "port names. If a port is not in this list, it does not exist. If widths "
        "differ, add explicit slices, zero/sign extension, or adapter glue in the "
        "wrapper; never connect mismatched widths directly.\n\n"
        f"{sigs}"
    )
    return autonomous_build_loop(
        base_prompt=enriched_top_prompt,
        design_name=design_name,
        provider=provider,
        max_retries=max_retries,
        candidates_per_round=candidates_per_round,
    )

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChipGPT: Autonomous RTL Generator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prompt",       type=str)
    parser.add_argument("--name",       type=str, default="my_module")
    parser.add_argument("--provider",   type=str, choices=["openai", "groq"], default="openai")
    parser.add_argument("--retries",    type=int, default=10)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--soc", action="store_true", help="Enable autonomous SoC decomposition")
    args = parser.parse_args()

    if args.soc:
        success = autonomous_soc_loop(
            base_prompt=args.prompt,
            design_name=args.name,
            provider=args.provider,
            max_retries=args.retries,
            candidates_per_round=args.candidates,
        )
    else:
        success = autonomous_build_loop(
            base_prompt=args.prompt,
            design_name=args.name,
            provider=args.provider,
            max_retries=args.retries,
            candidates_per_round=args.candidates,
        )
    exit(0 if success else 1)
