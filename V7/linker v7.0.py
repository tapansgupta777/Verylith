"""
linker.py - ChipGPT verified-module registry and hierarchy linker.

The workspace may contain many stale candidates. This module treats only the
registry's VERIFIED winner entries as linkable dependencies.
"""

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone


REGISTRY_VERSION = 1
DEFAULT_WORKSPACE = "./workspace"
DEFAULT_REGISTRY = os.path.join(DEFAULT_WORKSPACE, "registry.json")

_LANGUAGE_WORDS = {
    "always", "and", "assign", "begin", "buf", "bufif0", "bufif1", "case",
    "casex", "casez", "cmos", "deassign", "default", "defparam", "disable",
    "edge", "else", "end", "endcase", "endfunction", "endmodule",
    "endprimitive", "endspecify", "endtable", "endtask", "event", "for",
    "force", "forever", "fork", "function", "highz0", "highz1", "if",
    "initial", "inout", "input", "integer", "join", "large", "macromodule",
    "medium", "module", "nand", "negedge", "nmos", "nor", "not", "notif0",
    "notif1", "or", "output", "parameter", "pmos", "posedge", "primitive",
    "pull0", "pull1", "pulldown", "pullup", "rcmos", "real", "realtime",
    "reg", "release", "repeat", "rnmos", "rpmos", "rtran", "rtranif0",
    "rtranif1", "scalared", "small", "specify", "specparam", "strong0",
    "strong1", "supply0", "supply1", "table", "task", "time", "tran",
    "tranif0", "tranif1", "tri", "tri0", "tri1", "triand", "trior",
    "trireg", "vectored", "wait", "wand", "weak0", "weak1", "while",
    "wire", "wor", "xnor", "xor",
}


def _now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm_path(path):
    return os.path.normpath(path).replace("\\", "/")


def _abs_norm(path):
    return _norm_path(os.path.abspath(path))


def _read(path):
    with open(path, "r") as f:
        return f.read()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_comments(code):
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return re.sub(r"//.*", "", code)


def find_defined_modules(code):
    return re.findall(r"\bmodule\s+([A-Za-z_]\w*)\b", strip_comments(code))


def extract_module_header(code, module_name):
    text = strip_comments(code)
    pattern = re.compile(
        r"\bmodule\s+" + re.escape(module_name) +
        r"\b\s*(?:#\s*\([^;]*?\)\s*)?\((.*?)\)\s*;",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return ""
    port_blob = re.sub(r"\s+", " ", m.group(1)).strip()
    return f"module {module_name}({port_blob});"


def _uses_module(code, module_name):
    text = strip_comments(code)
    pattern = (
        r"(?m)(?:^|;)\s*" + re.escape(module_name) +
        r"\s+(?:#\s*\([^;]*?\)\s*)?[A-Za-z_]\w*\s*\("
    )
    return bool(re.search(pattern, text, re.DOTALL))


def find_instantiated_modules(code):
    """Best-effort instantiation finder for unresolved-dependency diagnostics."""
    text = strip_comments(code)
    found = []
    pattern = re.compile(
        r"(?m)(?:^|;)\s*([A-Za-z_]\w*)\s+(?:#\s*\([^;]*?\)\s*)?([A-Za-z_]\w*)\s*\(",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        mod = m.group(1)
        inst = m.group(2)
        if mod in _LANGUAGE_WORDS:
            continue
        if inst in {"if", "for", "while", "case"}:
            continue
        if mod not in found:
            found.append(mod)
    return found


def empty_registry():
    return {
        "version": REGISTRY_VERSION,
        "updated_at": _now_iso(),
        "modules": {},
        "history": [],
    }


def load_registry(registry_path=DEFAULT_REGISTRY):
    if not os.path.exists(registry_path):
        return empty_registry()
    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return empty_registry()
    data.setdefault("version", REGISTRY_VERSION)
    data.setdefault("modules", {})
    data.setdefault("history", [])
    return data


def save_registry(data, registry_path=DEFAULT_REGISTRY):
    registry_dir = os.path.dirname(registry_path)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)
    data["version"] = REGISTRY_VERSION
    data["updated_at"] = _now_iso()
    tmp = registry_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, registry_path)


def register_verified_module(
    module_name,
    src_path,
    candidate_id=None,
    tb_path=None,
    design_type=None,
    prompt=None,
    yosys_summary=None,
    registry_path=DEFAULT_REGISTRY,
):
    """Record the currently verified winner for a module."""
    src_path = _norm_path(src_path)
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)
    code = _read(src_path)
    defined = find_defined_modules(code)
    if module_name not in defined:
        raise ValueError(f"module {module_name!r} not found in {src_path}; defined={defined}")

    entry = {
        "module": module_name,
        "status": "VERIFIED",
        "src": src_path,
        "src_abs": _abs_norm(src_path),
        "src_sha256": _sha256_file(src_path),
        "candidate_id": candidate_id,
        "tb": _norm_path(tb_path) if tb_path else None,
        "design_type": design_type,
        "prompt_sha256": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest() if prompt else None,
        "yosys_summary": yosys_summary,
        "registered_at": _now_iso(),
    }

    data = load_registry(registry_path)
    previous = data["modules"].get(module_name)
    data["modules"][module_name] = entry
    hist = dict(entry)
    hist["replaced_src"] = previous.get("src") if previous else None
    data["history"].append(hist)
    save_registry(data, registry_path)
    return entry


def _entry_src(entry):
    path = entry.get("src")
    if path and os.path.exists(path):
        return path
    abs_path = entry.get("src_abs")
    if abs_path and os.path.exists(abs_path):
        return abs_path
    return None


def resolve_link_plan(top_file, top_module, registry_path=DEFAULT_REGISTRY):
    """Resolve verified dependency files recursively for a top RTL file."""
    data = load_registry(registry_path)
    modules = {
        name: entry
        for name, entry in data.get("modules", {}).items()
        if entry.get("status") == "VERIFIED" and name != top_module
    }
    dep_files = []
    dep_modules = []
    unresolved = []
    seen = set()
    visiting = set()

    def walk_file(path, owner_module):
        try:
            code = _read(path)
        except OSError:
            unresolved.append({"module": owner_module, "reason": f"cannot read {path}"})
            return

        for dep_name, entry in sorted(modules.items()):
            if dep_name in seen or dep_name in visiting:
                continue
            if not _uses_module(code, dep_name):
                continue
            dep_src = _entry_src(entry)
            if not dep_src:
                unresolved.append({"module": dep_name, "reason": "registered source missing"})
                continue
            visiting.add(dep_name)
            walk_file(dep_src, dep_name)
            visiting.discard(dep_name)
            if dep_name not in seen:
                seen.add(dep_name)
                dep_modules.append(dep_name)
                dep_files.append(dep_src)

        known = set(modules) | set(find_defined_modules(code)) | {top_module}
        for inst_mod in find_instantiated_modules(code):
            if inst_mod in known or inst_mod in _LANGUAGE_WORDS:
                continue
            if inst_mod not in [u.get("module") for u in unresolved]:
                unresolved.append({"module": inst_mod, "reason": "not in verified registry"})

    walk_file(top_file, top_module)
    current_abs = _abs_norm(top_file)
    dedup_files = []
    for path in dep_files:
        if _abs_norm(path) == current_abs:
            continue
        if path not in dedup_files:
            dedup_files.append(path)

    contains_memory = any(
        (data["modules"].get(name, {}).get("design_type") == "sram_byte_en")
        or ("memory" in name.lower())
        or ("sram" in name.lower())
        for name in dep_modules
    )
    return {
        "registry_path": registry_path,
        "top_module": top_module,
        "top_file": top_file,
        "files": dedup_files,
        "modules": dep_modules,
        "unresolved": unresolved,
        "contains_memory": contains_memory,
    }


def yosys_read_commands(files):
    cmds = []
    for path in files:
        q = _norm_path(path).replace('"', '\\"')
        cmds.append(f'read_verilog "{q}";')
    return " ".join(cmds)


def link_error(plan):
    if not plan.get("unresolved"):
        return ""
    bits = [f"{u.get('module')} ({u.get('reason')})" for u in plan["unresolved"]]
    return "LINK_ERROR: unresolved verified module dependencies: " + ", ".join(bits)


def format_link_summary(plan):
    if plan.get("files"):
        mods = ", ".join(plan.get("modules", []))
        return f"Linked verified deps: {mods}"
    return "Linked verified deps: none"


def format_registered_module_signatures(text, registry_path=DEFAULT_REGISTRY):
    """Return compact module headers for registered modules mentioned in text."""
    data = load_registry(registry_path)
    text_l = (text or "").lower()
    lines = []
    for name, entry in sorted(data.get("modules", {}).items()):
        if not re.search(r"\b" + re.escape(name.lower()) + r"\b", text_l):
            continue
        src = _entry_src(entry)
        if not src:
            continue
        header = extract_module_header(_read(src), name)
        if header:
            lines.append(header)
    if not lines:
        return ""
    return (
        "VERIFIED SUBMODULE INTERFACES FROM workspace/registry.json:\n"
        + "\n".join(lines)
    )


def main():
    parser = argparse.ArgumentParser(description="ChipGPT verified module linker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="register a verified module winner")
    reg.add_argument("--module", required=True)
    reg.add_argument("--src", required=True)
    reg.add_argument("--candidate", type=int)
    reg.add_argument("--tb")
    reg.add_argument("--design-type")
    reg.add_argument("--registry", default=DEFAULT_REGISTRY)

    plan = sub.add_parser("plan", help="resolve dependencies for a top module")
    plan.add_argument("--top", required=True)
    plan.add_argument("--module", required=True)
    plan.add_argument("--registry", default=DEFAULT_REGISTRY)

    args = parser.parse_args()
    if args.cmd == "register":
        entry = register_verified_module(
            args.module,
            args.src,
            candidate_id=args.candidate,
            tb_path=args.tb,
            design_type=args.design_type,
            registry_path=args.registry,
        )
        print(json.dumps(entry, indent=2, sort_keys=True))
    elif args.cmd == "plan":
        print(json.dumps(resolve_link_plan(args.top, args.module, args.registry), indent=2))


if __name__ == "__main__":
    main()
