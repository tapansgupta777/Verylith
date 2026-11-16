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
import shutil
import tempfile
from datetime import datetime, timezone


REGISTRY_VERSION = 1
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKSPACE = os.path.join(PROJECT_ROOT, "workspace")
DEFAULT_REGISTRY = os.path.join(DEFAULT_WORKSPACE, "registry.json")
EVIDENCE_SCHEMA_VERSION = "verylith-verification-evidence-v2"
VERIFICATION_POLICY_ID = "verylith-verification-policy-v1"
SAR_V2_CONTRACT = "sar-controller-contract-v2"
SAR_V2_ORACLE = "sar-blackbox-oracle-v2.1"
SAR_V2_ORACLE_SCHEMA = "sar-oracle-schema-v2"

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


def _canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RegistryLoadError(RuntimeError):
    pass


def _quarantine_registry(registry_path):
    """Copy a damaged registry to a unique path without replacing any prior copy."""
    base = f"{registry_path}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    for suffix in range(1000):
        target = base if suffix == 0 else f"{base}-{suffix}"
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "wb") as out, open(registry_path, "rb") as src:
                shutil.copyfileobj(src, out)
            return target
        except OSError:
            try:
                os.unlink(target)
            except OSError:
                pass
            raise
    raise RegistryLoadError(f"Could not allocate a unique quarantine path for {registry_path}")


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
    try:
        os.stat(registry_path)
    except FileNotFoundError:
        return empty_registry()
    except OSError as exc:
        raise RegistryLoadError(
            f"Registry metadata could not be read at {registry_path}; writes are refused: {exc}"
        ) from exc
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        try:
            quarantine = _quarantine_registry(registry_path)
        except OSError as copy_exc:
            quarantine = f"quarantine failed: {copy_exc}"
        raise RegistryLoadError(
            f"Registry JSON is corrupt at {registry_path}; original preserved; {quarantine}: {exc}"
        ) from exc
    except OSError as exc:
        raise RegistryLoadError(
            f"Registry could not be read at {registry_path}; writes are refused: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("modules", {}), dict):
        raise RegistryLoadError(f"Registry has invalid structure at {registry_path}; writes are refused")
    data.setdefault("version", REGISTRY_VERSION)
    data.setdefault("modules", {})
    data.setdefault("history", [])
    return data


def save_registry(data, registry_path=DEFAULT_REGISTRY):
    # Never let a direct save bypass fail-closed loading of existing state.
    load_registry(registry_path)
    registry_dir = os.path.dirname(registry_path)
    if registry_dir:
        os.makedirs(registry_dir, exist_ok=True)
    data["version"] = REGISTRY_VERSION
    data["updated_at"] = _now_iso()
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(registry_path) + ".", suffix=".tmp", dir=registry_dir or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, registry_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_evidence_record(
    evidence, module_name, src_path, tb_path,
    registry_path=DEFAULT_REGISTRY, project_root=None,
):
    """Validate integrity and promotion prerequisites for a strict evidence record."""
    if not isinstance(evidence, dict):
        return False, "structured evidence missing"
    unsigned = dict(evidence)
    recorded_digest = unsigned.pop("evidence_digest", None)
    if not recorded_digest or _canonical_digest(unsigned) != recorded_digest:
        return False, "evidence digest missing or invalid"
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        if evidence.get("schema_version") == "verylith-verification-evidence-v1":
            return False, "older evidence schema lacks required success_marker_seen field"
        return False, "unsupported evidence schema"
    if evidence.get("module") != module_name:
        return False, "evidence module identity mismatch"
    if evidence.get("verification_status") != "VERIFICATION_PASSED":
        return False, "evidence does not record a passed verification result"
    if evidence.get("verification_policy_id") != VERIFICATION_POLICY_ID:
        return False, "verification policy identity mismatch"
    if not evidence.get("test_vector_identity"):
        return False, "test vector/oracle identity is missing"
    expected_rtl = evidence.get("rtl_sha256")
    expected_tb = evidence.get("testbench_sha256")
    if not src_path or not os.path.isfile(src_path) or _sha256_file(src_path) != expected_rtl:
        return False, "RTL missing or SHA-256 mismatch"
    try:
        if module_name not in find_defined_modules(_read(src_path)):
            return False, "module identity is not defined by immutable RTL"
    except OSError:
        return False, "immutable RTL is unreadable"
    if not tb_path or not os.path.isfile(tb_path) or _sha256_file(tb_path) != expected_tb:
        return False, "testbench missing or SHA-256 mismatch"
    tools = evidence.get("tools", {})
    for tool_name in ("verilator", "iverilog_compile", "iverilog_simulation", "yosys"):
        item = tools.get(tool_name, {})
        if (not item.get("version") or not item.get("command")
                or item.get("exit_code") != 0 or item.get("result") != "PASS"):
            return False, f"missing or failed tool evidence: {tool_name}"
    sim = tools["iverilog_simulation"]
    marker_seen = sim.get("success_marker_seen")
    if not isinstance(marker_seen, bool):
        return False, "Icarus simulation success_marker_seen evidence is missing or invalid"
    if sim.get("exit_code") != 0 or not marker_seen:
        return False, "Icarus simulation did not exit successfully with SIMULATION_SUCCESS"
    if "output" in sim and marker_seen != ("SIMULATION_SUCCESS" in sim.get("output", "")):
        return False, "Icarus simulation success_marker_seen contradicts captured simulation output"
    contract = evidence.get("verification_contract")
    if contract is not None:
        version = contract.get("version")
        if evidence.get("contract_version") != version or evidence.get("contract_hash") != contract.get("contract_hash"):
            return False, "evidence contract identity is inconsistent"
        if version != SAR_V2_CONTRACT or version not in evidence.get("oracle_supported_contract_versions", []):
            return False, "oracle does not declare support for the evidence contract version"
        if version == SAR_V2_CONTRACT and (
            evidence.get("oracle_kind") != "deterministic_generator"
            or evidence.get("oracle_implementation_version") != SAR_V2_ORACLE
            or evidence.get("oracle_schema_version") != SAR_V2_ORACLE_SCHEMA
        ):
            return False, "SAR oracle provenance is not the trusted deterministic implementation/schema"
        if version == SAR_V2_CONTRACT:
            parameters = contract.get("parameters", {})
            configured = evidence.get("configuration", {}).get("sar_parameters", {})
            expected = {
                "resolution": parameters.get("resolution"),
                "settle_cycles": parameters.get("settle_cycles"),
                "comparator_high_keeps_bit": parameters.get("comparator_high_keeps_bit"),
            }
            if any(configured.get(key) != value for key, value in expected.items()):
                return False, "SAR evidence configuration differs from its canonical contract"
            try:
                with open(tb_path, "r", encoding="utf-8", errors="replace") as f:
                    tb_text = f.read()
            except OSError:
                return False, "SAR oracle testbench is unreadable"
            localparams = {
                "RESOLUTION": expected["resolution"],
                "SETTLE_CYCLES": expected["settle_cycles"],
                "CMP_HIGH_KEEPS_BIT": int(bool(expected["comparator_high_keeps_bit"])),
            }
            for key, value in localparams.items():
                match = re.search(rf"\blocalparam\s+{key}\s*=\s*(\d+)\b", tb_text, re.I)
                if not match or int(match.group(1)) != int(value):
                    return False, f"SAR oracle configuration mismatch for {key}"
            if (contract.get("contract_hash", "") not in tb_text
                    or f"contract_version={version}" not in tb_text):
                return False, "SAR oracle contract identity marker mismatch"
    elif (evidence.get("oracle_kind") not in {"deterministic_generator", "llm_generated"}
          or not evidence.get("oracle_implementation_version")
          or not evidence.get("oracle_schema_version")):
        return False, "non-SAR oracle provenance is missing or invalid"
    bundle = evidence.get("bundle", {})
    if not bundle.get("path") or not bundle.get("bundle_digest"):
        return False, "immutable bundle reference missing"
    bundle_path = _resolve_registry_path(bundle.get("path"), registry_path, project_root)
    if not bundle_path:
        return False, "immutable bundle path is invalid"
    expected_src = os.path.join(bundle_path, f"{module_name}.v")
    expected_tb = os.path.join(bundle_path, f"{module_name}_tb.v")
    if _canonical_path_identity(src_path) != _canonical_path_identity(expected_src):
        return False, "registry source does not point to immutable bundle"
    if _canonical_path_identity(tb_path) != _canonical_path_identity(expected_tb):
        return False, "registry testbench does not point to immutable bundle"
    if _bundle_digest(evidence) != bundle.get("bundle_digest"):
        return False, "bundle digest mismatch"
    manifest_path = os.path.join(bundle_path, "evidence.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False, "immutable bundle manifest missing or unreadable"
    if manifest != evidence:
        return False, "immutable bundle manifest differs from registry evidence"
    return True, "strict evidence valid"


def _bundle_digest(evidence):
    tools = {
        name: {
            "version": item.get("version"),
            "command": item.get("command"),
            "exit_code": item.get("exit_code"),
            "result": item.get("result"),
            "success_marker_seen": item.get("success_marker_seen"),
            "output_sha256": hashlib.sha256(item.get("output", "").encode("utf-8")).hexdigest()
            if "output" in item else None,
        }
        for name, item in sorted(evidence.get("tools", {}).items())
    }
    return _canonical_digest({
        "module": evidence.get("module"),
        "candidate_id": evidence.get("candidate_id"),
        "rtl_sha256": evidence["rtl_sha256"],
        "testbench_sha256": evidence["testbench_sha256"],
        "contract_version": evidence.get("contract_version"),
        "contract_hash": evidence.get("contract_hash"),
        "oracle_implementation_version": evidence.get("oracle_implementation_version"),
        "oracle_schema_version": evidence.get("oracle_schema_version"),
        "verification_policy_id": evidence.get("verification_policy_id"),
        "evidence_schema_version": evidence.get("schema_version"),
        "test_vector_identity": evidence.get("test_vector_identity"),
        "configuration": evidence.get("configuration"),
        "tools": tools,
    })


def _promotion_preflight(evidence, module_name):
    if not isinstance(evidence, dict) or evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        if isinstance(evidence, dict) and evidence.get("schema_version") == "verylith-verification-evidence-v1":
            raise ValueError("promotion rejected: older evidence schema lacks required success_marker_seen field")
        raise ValueError("promotion rejected: unsupported or missing evidence schema")
    if evidence.get("module") != module_name or evidence.get("verification_status") != "VERIFICATION_PASSED":
        raise ValueError("promotion rejected: module identity or verification result is invalid")
    if evidence.get("verification_policy_id") != VERIFICATION_POLICY_ID:
        raise ValueError("promotion rejected: verification policy identity mismatch")
    if not evidence.get("test_vector_identity"):
        raise ValueError("promotion rejected: test vector/oracle identity is missing")
    for key in ("verilator", "iverilog_compile", "iverilog_simulation", "yosys"):
        item = evidence.get("tools", {}).get(key, {})
        if (not item.get("version") or not item.get("command")
                or item.get("exit_code") != 0 or item.get("result") != "PASS"):
            raise ValueError(f"promotion rejected: incomplete or failed tool evidence for {key}")
    sim = evidence["tools"]["iverilog_simulation"]
    if not isinstance(sim.get("success_marker_seen"), bool):
        raise ValueError("promotion rejected: Icarus simulation success_marker_seen evidence is missing or invalid")
    if sim.get("exit_code") != 0 or not sim["success_marker_seen"]:
        raise ValueError("promotion rejected: Icarus simulation did not exit successfully with SIMULATION_SUCCESS")
    if "output" in sim and sim["success_marker_seen"] != ("SIMULATION_SUCCESS" in sim.get("output", "")):
        raise ValueError("promotion rejected: Icarus success marker evidence contradicts captured output")
    contract = evidence.get("verification_contract")
    if contract is not None:
        version = contract.get("version")
        if evidence.get("contract_version") != version or evidence.get("contract_hash") != contract.get("contract_hash"):
            raise ValueError("promotion rejected: contract identity mismatch")
        if version != SAR_V2_CONTRACT or version not in evidence.get("oracle_supported_contract_versions", []):
            raise ValueError("promotion rejected: oracle does not declare support for the contract version")
        if version == SAR_V2_CONTRACT and (
            evidence.get("oracle_kind") != "deterministic_generator"
            or evidence.get("oracle_implementation_version") != SAR_V2_ORACLE
            or evidence.get("oracle_schema_version") != SAR_V2_ORACLE_SCHEMA
        ):
            raise ValueError("promotion rejected: SAR oracle provenance is not the trusted deterministic implementation/schema")
    elif (evidence.get("oracle_kind") not in {"deterministic_generator", "llm_generated"}
          or not evidence.get("oracle_implementation_version")
          or not evidence.get("oracle_schema_version")):
        raise ValueError("promotion rejected: non-SAR oracle provenance is invalid")


def validate_promotion_evidence(evidence, module_name):
    try:
        _promotion_preflight(evidence, module_name)
        return True, "promotion evidence complete"
    except ValueError as exc:
        return False, str(exc)


def publish_verified_bundle(
    module_name, src_path, tb_path, evidence, workspace="workspace",
    project_root=None,
):
    """Copy verified files to a content-addressed store and validate every published byte."""
    evidence = json.loads(json.dumps(evidence))
    _promotion_preflight(evidence, module_name)
    actual_rtl_hash = _sha256_file(src_path)
    actual_tb_hash = _sha256_file(tb_path)
    if evidence.get("rtl_sha256") not in (None, actual_rtl_hash):
        raise ValueError("promotion rejected: RTL changed after verification")
    if evidence.get("testbench_sha256") not in (None, actual_tb_hash):
        raise ValueError("promotion rejected: testbench changed after verification")
    evidence["rtl_sha256"] = actual_rtl_hash
    evidence["testbench_sha256"] = actual_tb_hash
    contract = evidence.get("verification_contract")
    evidence["contract_version"] = contract.get("version") if contract else None
    evidence["contract_hash"] = contract.get("contract_hash") if contract else None
    digest = _bundle_digest(evidence)
    root_context = os.path.abspath(project_root) if project_root else PROJECT_ROOT
    workspace_input = workspace
    workspace_abs = _resolve_registry_path(workspace, project_root=root_context)
    root = os.path.join(workspace_abs, "verified", module_name)
    os.makedirs(root, exist_ok=True)
    final_path = os.path.join(root, digest[:16])
    bundle_record_path = (
        _norm_path(os.path.relpath(final_path, root_context))
        if not os.path.isabs(workspace_input)
        else _norm_path(final_path)
    )
    evidence["bundle"] = {"path": bundle_record_path, "bundle_digest": digest}
    evidence.pop("evidence_digest", None)
    evidence["evidence_digest"] = _canonical_digest(evidence)

    def validate_bundle(path):
        manifest_path = os.path.join(path, "evidence.json")
        rtl_path = os.path.join(path, f"{module_name}.v")
        tb_bundle_path = os.path.join(path, f"{module_name}_tb.v")
        if not (os.path.isfile(manifest_path) and os.path.isfile(rtl_path) and os.path.isfile(tb_bundle_path)):
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return (
                saved == evidence
                and _sha256_file(rtl_path) == evidence["rtl_sha256"]
                and _sha256_file(tb_bundle_path) == evidence["testbench_sha256"]
                and _bundle_digest(saved) == digest
            )
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    if os.path.exists(final_path):
        if not validate_bundle(final_path):
            raise ValueError(f"existing immutable bundle is inconsistent: {final_path}")
        return evidence, final_path

    stage = tempfile.mkdtemp(prefix=".stage-", dir=root)
    fallback_created = False
    try:
        shutil.copyfile(src_path, os.path.join(stage, f"{module_name}.v"))
        shutil.copyfile(tb_path, os.path.join(stage, f"{module_name}_tb.v"))
        if _sha256_file(os.path.join(stage, f"{module_name}.v")) != evidence["rtl_sha256"]:
            raise ValueError("staged RTL SHA-256 mismatch")
        if _sha256_file(os.path.join(stage, f"{module_name}_tb.v")) != evidence["testbench_sha256"]:
            raise ValueError("staged testbench SHA-256 mismatch")
        with open(os.path.join(stage, "evidence.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(evidence, f, indent=2, sort_keys=True)
            f.write("\n")
        if not validate_bundle(stage):
            raise ValueError("staged verification bundle failed validation")
        try:
            os.replace(stage, final_path)
            stage = None
        except OSError:
            if os.path.exists(final_path):
                if not validate_bundle(final_path):
                    raise ValueError(f"bundle publish collision at {final_path}")
            else:
                os.makedirs(final_path, exist_ok=False)
                fallback_created = True
                for filename in os.listdir(stage):
                    shutil.copyfile(os.path.join(stage, filename), os.path.join(final_path, filename))
                if not validate_bundle(final_path):
                    raise ValueError("copy-published verification bundle failed validation")
        return evidence, final_path
    except Exception:
        if fallback_created and os.path.exists(final_path):
            shutil.rmtree(final_path, ignore_errors=True)
        raise
    finally:
        if stage and os.path.exists(stage):
            shutil.rmtree(stage, ignore_errors=True)


def register_verified_module(
    module_name,
    src_path,
    candidate_id=None,
    tb_path=None,
    design_type=None,
    prompt=None,
    yosys_summary=None,
    registry_path=DEFAULT_REGISTRY,
    verification_contract=None,
    verification_evidence=None,
    legacy_import=False,
    project_root=None,
):
    """Record the currently verified winner for a module."""
    if verification_evidence is None and not legacy_import:
        raise ValueError("registration rejected: verification evidence is required; use legacy_import=True only for an explicit legacy import")
    if verification_evidence is not None and legacy_import:
        raise ValueError("registration rejected: strict evidence and legacy_import are mutually exclusive")
    registry_path = _absolute_registry_path(registry_path)
    stored_src_path = _norm_path(os.fspath(src_path))
    resolved_src_path = _resolve_registry_path(stored_src_path, registry_path, project_root)
    if not resolved_src_path or not os.path.isfile(resolved_src_path):
        raise FileNotFoundError(resolved_src_path or stored_src_path)
    code = _read(resolved_src_path)
    defined = find_defined_modules(code)
    if module_name not in defined:
        raise ValueError(f"module {module_name!r} not found in {resolved_src_path}; defined={defined}")

    entry = {
        "module": module_name,
        "record_schema": 2,
        "status": "LEGACY" if legacy_import else "VERIFIED",
        "src": stored_src_path,
        "src_abs": _abs_norm(resolved_src_path),
        "src_sha256": _sha256_file(resolved_src_path),
        "candidate_id": candidate_id,
        "tb": _norm_path(tb_path) if tb_path else None,
        "design_type": design_type,
        "prompt_sha256": hashlib.sha256((prompt or "").encode("utf-8")).hexdigest() if prompt else None,
        "yosys_summary": yosys_summary,
        "registered_at": _now_iso(),
    }
    if legacy_import:
        entry["compat"] = "legacy_import"
    if verification_contract is not None:
        entry["verification_contract"] = verification_contract
    if verification_evidence is not None:
        if verification_contract != verification_evidence.get("verification_contract"):
            raise ValueError("strict promotion rejected: registry/evidence contract identity mismatch")
        bundle_path = _resolve_registry_path(
            verification_evidence.get("bundle", {}).get("path"), registry_path, project_root
        )
        valid, reason = validate_evidence_record(
            verification_evidence, module_name, resolved_src_path,
            os.path.join(bundle_path, f"{module_name}_tb.v") if bundle_path else None,
            registry_path=registry_path, project_root=project_root,
        )
        if not valid:
            raise ValueError(f"strict promotion rejected: {reason}")
        entry["verification_evidence"] = verification_evidence
        entry["artifact_store"] = verification_evidence["bundle"]["path"]

    data = load_registry(registry_path)
    previous = data["modules"].get(module_name)
    if previous is not None and legacy_import:
        previous_kind, _, _ = registry_entry_linkability(module_name, previous, registry_path, project_root)
        if previous_kind == "STRICT":
            raise ValueError(
                f"registration rejected: refusing to downgrade strict VERIFIED module {module_name} to LEGACY"
            )
    data["modules"][module_name] = entry
    hist = dict(entry)
    hist["replaced_src"] = previous.get("src") if previous else None
    data["history"].append(hist)
    save_registry(data, registry_path)
    return entry


def _registry_project_root(registry_path=DEFAULT_REGISTRY):
    registry_abs = _absolute_registry_path(registry_path)
    registry_dir = os.path.dirname(registry_abs)
    if os.path.basename(registry_dir).lower() == "workspace":
        return os.path.dirname(registry_dir)
    return PROJECT_ROOT


def _absolute_registry_path(registry_path):
    return os.path.normpath(os.path.abspath(os.fspath(registry_path)))


def _resolve_registry_path(path, registry_path=DEFAULT_REGISTRY, project_root=None):
    if not path:
        return None
    path = os.fspath(path)
    if os.path.isabs(path):
        return os.path.normpath(path)
    if re.match(r"^[A-Za-z]:[\\/]", path) and os.name != "nt":
        drive = path[0].lower()
        path = os.path.join("/mnt", drive, path[3:].replace("\\", "/"))
        return os.path.normpath(path)
    root = os.path.abspath(project_root) if project_root else _registry_project_root(registry_path)
    return os.path.normpath(os.path.join(root, path))


def _canonical_path_identity(path):
    if not path:
        return None
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _entry_src(entry, registry_path=DEFAULT_REGISTRY, project_root=None):
    path = entry.get("src")
    resolved = _resolve_registry_path(path, registry_path, project_root)
    if resolved and os.path.isfile(resolved):
        return resolved
    abs_path = entry.get("src_abs")
    resolved_abs = _resolve_registry_path(abs_path, registry_path, project_root)
    if resolved_abs and os.path.isfile(resolved_abs):
        return resolved_abs
    return None


def registry_entry_linkability(
    module_name, entry, registry_path=DEFAULT_REGISTRY, project_root=None,
):
    """Return (kind, reason, source) using the single registry link policy."""
    registry_path = _absolute_registry_path(registry_path)
    src = _entry_src(entry, registry_path, project_root)
    evidence = entry.get("verification_evidence")
    if evidence is not None:
        recorded_hash = entry.get("src_sha256")
        if not src:
            return "STALE", "registered source file is missing", None
        if not recorded_hash:
            return "STALE", "recorded src_sha256 is missing", src
        try:
            if _sha256_file(src) != recorded_hash:
                return "STALE", "registered source SHA-256 differs from registry src_sha256", src
        except OSError as exc:
            return "STALE", f"registered source cannot be hashed: {exc}", src
        artifact_store = _resolve_registry_path(entry.get("artifact_store"), registry_path, project_root)
        evidence_bundle = evidence.get("bundle", {}).get("path")
        evidence_bundle_path = _resolve_registry_path(evidence_bundle, registry_path, project_root)
        if not artifact_store or not evidence_bundle_path:
            return "INVALID", "strict bundle path is missing or invalid", src
        if _canonical_path_identity(artifact_store) != _canonical_path_identity(evidence_bundle_path):
            return "INVALID", "artifact_store and evidence bundle path identify different directories", src

        expected_src = os.path.join(evidence_bundle_path, f"{module_name}.v")
        expected_tb = os.path.join(evidence_bundle_path, f"{module_name}_tb.v")
        registry_src = _resolve_registry_path(entry.get("src"), registry_path, project_root)
        if not registry_src or _canonical_path_identity(registry_src) != _canonical_path_identity(expected_src):
            return "INVALID", "registered source does not identify the expected verified RTL artifact", src
        if entry.get("tb"):
            registry_tb = _resolve_registry_path(entry.get("tb"), registry_path, project_root)
            if not registry_tb or _canonical_path_identity(registry_tb) != _canonical_path_identity(expected_tb):
                return "INVALID", "registered testbench does not identify the expected verified testbench artifact", src

        valid, reason = validate_evidence_record(
            evidence, module_name, expected_src, expected_tb,
            registry_path=registry_path, project_root=project_root,
        )
        if not valid:
            return "INVALID", reason, src
        if entry.get("record_schema", 1) == 2 and entry.get("status") != "VERIFIED":
            return "INVALID", "evidence-backed schema-2 record must have VERIFIED status", src
        if entry.get("verification_contract") != evidence.get("verification_contract"):
            return "INVALID", "registry/evidence contract identity mismatch", src
        return "STRICT", "strict evidence valid", src

    schema = entry.get("record_schema")
    explicit_import = schema == 2 and entry.get("compat") == "legacy_import" and entry.get("status") == "LEGACY"
    historical = schema is None
    if schema == 2 and not explicit_import:
        return "INVALID", "schema-2 record lacks valid verification evidence or explicit legacy_import designation", src
    if not historical and not explicit_import:
        return "INVALID", "unsupported registry record schema/status", src
    if entry.get("status") not in ({"LEGACY"} if explicit_import else {"VERIFIED", "LEGACY"}):
        return "INVALID", f"legacy record has non-legacy status {entry.get('status')!r}", src
    if not src:
        return "STALE", "registered source file is missing", None
    try:
        if module_name not in find_defined_modules(_read(src)):
            return "INVALID", f"registered source does not define module {module_name}", src
    except OSError as exc:
        return "STALE", f"registered source is unreadable: {exc}", src
    recorded_hash = entry.get("src_sha256")
    if not recorded_hash:
        return "STALE", "recorded src_sha256 is missing", src
    try:
        actual_hash = _sha256_file(src)
    except OSError as exc:
        return "STALE", f"registered source cannot be hashed: {exc}", src
    if actual_hash != recorded_hash:
        return "STALE", f"source SHA-256 mismatch (recorded {recorded_hash}, actual {actual_hash})", src
    return ("LEGACY_IMPORT" if explicit_import else "LEGACY"), "source hash matches; strict verification evidence unavailable", src


def resolve_link_plan(top_file, top_module, registry_path=DEFAULT_REGISTRY, project_root=None):
    """Resolve verified dependency files recursively for a top RTL file."""
    registry_path = _absolute_registry_path(registry_path)
    data = load_registry(registry_path)
    modules = {}
    warnings = []
    invalid_modules = {}
    for name, entry in data.get("modules", {}).items():
        if name == top_module:
            continue
        kind, reason, _ = registry_entry_linkability(name, entry, registry_path, project_root)
        if kind in {"STRICT", "LEGACY", "LEGACY_IMPORT"}:
            modules[name] = entry
            if kind != "STRICT":
                warnings.append(f"LEGACY: {name} is linked through hash-checked compatibility policy; strict verification evidence is unavailable.")
        else:
            invalid_modules[name] = reason
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
            dep_src = _entry_src(entry, registry_path, project_root)
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
            if inst_mod in invalid_modules:
                unresolved.append({"module": inst_mod, "reason": invalid_modules[inst_mod]})
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
        "warnings": warnings,
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
    return "LINK_ERROR: unresolved registered module dependencies: " + ", ".join(bits)


def format_link_summary(plan):
    for warning in plan.get("warnings", []):
        print(f"WARNING: {warning}")
    if plan.get("files"):
        mods = ", ".join(plan.get("modules", []))
        strict = [name for name in plan.get("modules", []) if name not in {
            warning.split(":", 1)[1].split(" is linked", 1)[0].strip()
            for warning in plan.get("warnings", []) if warning.startswith("LEGACY:")
        }]
        legacy = [name for name in plan.get("modules", []) if name not in strict]
        labels = []
        if strict:
            labels.append("strict=" + ",".join(strict))
        if legacy:
            labels.append("LEGACY=" + ",".join(legacy))
        return "Linked dependencies: " + "; ".join(labels or [mods])
    return "Linked verified deps: none"


def format_registered_module_signatures(text, registry_path=DEFAULT_REGISTRY, project_root=None):
    """Return compact module headers for registered modules mentioned in text."""
    registry_path = _absolute_registry_path(registry_path)
    data = load_registry(registry_path)
    text_l = (text or "").lower()
    lines = []
    for name, entry in sorted(data.get("modules", {}).items()):
        if not re.search(r"\b" + re.escape(name.lower()) + r"\b", text_l):
            continue
        kind, reason, src = registry_entry_linkability(name, entry, registry_path, project_root)
        if kind not in {"STRICT", "LEGACY", "LEGACY_IMPORT"}:
            print(f"WARNING: module signature omitted for {name}: {kind} - {reason}")
            continue
        if kind != "STRICT":
            print(f"WARNING: LEGACY: {name} signature comes from hash-checked legacy source; strict verification evidence is unavailable.")
        try:
            header = extract_module_header(_read(src), name)
        except OSError as exc:
            print(f"WARNING: module signature omitted for {name}: source unreadable: {exc}")
            continue
        if header:
            lines.append(header)
    if not lines:
        return ""
    return (
        "REGISTERED SUBMODULE INTERFACES FROM workspace/registry.json:\n"
        + "\n".join(lines)
    )


def registry_health_report(registry_path=DEFAULT_REGISTRY, output_path=None, project_root=None):
    """Inspect registry records and referenced artifacts without modifying them."""
    registry_path = _absolute_registry_path(registry_path)
    data = load_registry(registry_path)
    modules = data.get("modules", {})
    rows = []
    counts = {
        "strictly evidence-valid": 0,
        "historical legacy-linkable": 0,
        "explicit legacy-import linkable": 0,
        "stale or missing source": 0,
        "incomplete or mismatched evidence": 0,
        "other validation failures": 0,
    }
    for name, entry in sorted(modules.items()):
        reasons = []
        src = _entry_src(entry, registry_path, project_root)
        src_exists = bool(src)
        src_hash = _sha256_file(src) if src else None
        recorded_hash = entry.get("src_sha256")
        src_matches = bool(src and recorded_hash and src_hash == recorded_hash)
        if not src_exists:
            reasons.append("source missing")
            counts["stale or missing source"] += 1
        elif recorded_hash and not src_matches:
            reasons.append("source SHA-256 mismatch")
            counts["stale or missing source"] += 1
        elif not recorded_hash:
            reasons.append("source hash not recorded")

        tb_path = _resolve_registry_path(entry.get("tb"), registry_path, project_root)
        tb_exists = bool(tb_path and os.path.isfile(tb_path))
        tb_text = ""
        if tb_exists:
            try:
                tb_text = _read(tb_path)
            except OSError as exc:
                tb_exists = False
                reasons.append(f"testbench unreadable: {exc}")
        elif entry.get("tb"):
            reasons.append("testbench missing")
        else:
            reasons.append("testbench path not recorded")

        code = ""
        if src:
            try:
                code = _read(src)
            except OSError as exc:
                reasons.append(f"source unreadable: {exc}")
        identity_ok = bool(code and name in find_defined_modules(code))
        if src and not identity_ok:
            reasons.append("module identity not found in source")
            counts["other validation failures"] += 1

        evidence = entry.get("verification_evidence") or entry.get("evidence")
        evidence_present = isinstance(evidence, dict)
        evidence_schema = evidence.get("schema_version") if evidence_present else None
        contract = entry.get("verification_contract")
        contract_identity = None
        scope_matches = None
        if isinstance(contract, dict):
            contract_identity = f"{contract.get('version', 'unknown')} / {contract.get('contract_hash', 'no hash')}"
        if entry.get("design_type") == "sar_adc_controller":
            width = None
            settle_default = None
            tb_resolution = None
            tb_settle = None
            if code:
                match = re.search(r"\bparameter\s+(?:integer\s+)?RESOLUTION\s*=\s*(\d+)", code, re.I)
                if match:
                    width = int(match.group(1))
                match = re.search(r"\bparameter\s+(?:integer\s+)?SETTLE_CYCLES\s*=\s*(\d+)", code, re.I)
                if match:
                    settle_default = int(match.group(1))
            if tb_text:
                for key, target in (("RESOLUTION", "resolution"), ("SETTLE_CYCLES", "settle")):
                    match = re.search(rf"\blocalparam\s+{key}\s*=\s*(\d+)", tb_text, re.I)
                    if match:
                        if target == "resolution":
                            tb_resolution = int(match.group(1))
                        else:
                            tb_settle = int(match.group(1))
            effective_resolution = tb_resolution if tb_resolution is not None else width
            effective_settle = tb_settle if tb_settle is not None else settle_default
            if width is not None and effective_resolution is not None:
                scope_matches = width == effective_resolution
                if not scope_matches:
                    reasons.append(f"SAR width/simulation scope mismatch: RTL default {width}, testbench {effective_resolution}")
            elif name == "sar_adc_12b" and entry.get("candidate_id") == 18:
                scope_matches = False
                reasons.append("historical candidate 18 labelled 12-bit; testbench RESOLUTION=4, SETTLE_CYCLES=2")
            else:
                scope_matches = False
                reasons.append("SAR verification scope could not be established from source and testbench")
            if effective_settle is not None and isinstance(contract, dict):
                contract_settle = contract.get("parameters", {}).get("settle_cycles")
                if contract_settle is not None and int(contract_settle) != effective_settle:
                    scope_matches = False
                    reasons.append(f"SAR contract/testbench settling mismatch: contract {contract_settle}, testbench {effective_settle}")
            contract_hash = (contract or {}).get("contract_hash") if isinstance(contract, dict) else None
            if evidence_present and evidence.get("contract_hash") != contract_hash:
                reasons.append("evidence contract identity missing or mismatched")

        candidate_path = bool(src and re.search(r"[/\\]candidate_[^/\\]+[/\\]", src))
        immutable_ref = entry.get("artifact_store") or (evidence or {}).get("artifact_store") if evidence_present else entry.get("artifact_store")
        evidence_valid = False
        evidence_reason = "structured verification evidence is absent"
        if evidence_present:
            evidence_valid, evidence_reason = validate_evidence_record(
                evidence, name, src, tb_path,
                registry_path=registry_path, project_root=project_root,
            )
            if entry.get("verification_contract") != evidence.get("verification_contract"):
                evidence_valid = False
                evidence_reason = "registry/evidence contract identity mismatch"
        strict_ok = bool(
            evidence_present and evidence_schema and not candidate_path and immutable_ref
            and src_matches and tb_exists and identity_ok and evidence_valid
            and (entry.get("design_type") != "sar_adc_controller" or scope_matches is True)
        )
        kind, link_reason, _ = registry_entry_linkability(name, entry, registry_path, project_root)
        if strict_ok and kind == "STRICT":
            counts["strictly evidence-valid"] += 1
        else:
            if kind == "LEGACY" and src_exists and src_matches and identity_ok:
                counts["historical legacy-linkable"] += 1
                reasons.append("historical LEGACY link allowed after source hash validation")
            elif kind == "LEGACY_IMPORT" and src_exists and src_matches and identity_ok:
                counts["explicit legacy-import linkable"] += 1
                reasons.append("explicit legacy import; hash-checked LEGACY link only")
            elif kind == "STALE":
                if not (not src_exists or (recorded_hash and not src_matches) or not recorded_hash):
                    counts["other validation failures"] += 1
                reasons.append(link_reason)
            else:
                counts["incomplete or mismatched evidence"] += 1
                reasons.append(evidence_reason if evidence_present else link_reason)
        rows.append({
            "module": name,
            "status": entry.get("status"),
            "src": src or entry.get("src"),
            "src_exists": src_exists,
            "src_sha256_matches": src_matches,
            "tb": tb_path or entry.get("tb"),
            "tb_exists": tb_exists,
            "evidence_present": evidence_present,
            "evidence_schema": evidence_schema,
            "contract": contract_identity,
            "scope_matches": scope_matches,
            "candidate_path": candidate_path,
            "strict_link_ready": strict_ok,
            "reasons": reasons,
        })

    lines = ["# Registry Health Report", "", f"Registry: `{registry_path}`", "", "## Counts", ""]
    for label, count in counts.items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Entries", ""])
    for row in rows:
        lines.append(f"### {row['module']} ({row['status'] or 'no status'})")
        lines.append(f"- Source: `{row['src']}`; exists={row['src_exists']}; recorded hash matches={row['src_sha256_matches']}")
        lines.append(f"- Testbench: `{row['tb']}`; exists={row['tb_exists']}")
        lines.append(f"- Evidence: present={row['evidence_present']}; schema={row['evidence_schema']}; contract={row['contract']}")
        lines.append(f"- Scope matches={row['scope_matches']}; candidate path={row['candidate_path']}; strict-link-ready={row['strict_link_ready']}")
        if row["reasons"]:
            lines.append(f"- Findings: {'; '.join(row['reasons'])}")
        lines.append("")
    report = "\n".join(lines)
    if output_path:
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
            f.write("\n")
    return {"counts": counts, "entries": rows, "report": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description="ChipGPT verified module linker")
    sub = parser.add_subparsers(dest="cmd", required=True)

    reg = sub.add_parser("register", help="register a verified module winner")
    reg.add_argument("--module", required=True)
    reg.add_argument("--src", required=True)
    reg.add_argument("--candidate", type=int)
    reg.add_argument("--tb")
    reg.add_argument("--design-type")
    reg.add_argument("--registry", default=DEFAULT_REGISTRY)
    reg.add_argument("--project-root", help="project root for registry-relative paths in a custom layout")
    reg.add_argument("--legacy-import", action="store_true", help="explicitly import without strict verification evidence")
    reg.add_argument("--evidence", help="strict verification evidence JSON")

    plan = sub.add_parser("plan", help="resolve dependencies for a top module")
    plan.add_argument("--top", required=True)
    plan.add_argument("--module", required=True)
    plan.add_argument("--registry", default=DEFAULT_REGISTRY)
    plan.add_argument("--project-root", help="project root for registry-relative paths in a custom layout")

    health = sub.add_parser("health", help="read-only registry and evidence health report")
    health.add_argument("--registry", default=DEFAULT_REGISTRY)
    health.add_argument("--project-root", help="project root for registry-relative paths in a custom layout")
    health.add_argument("--output", default="REGISTRY_HEALTH_REPORT.md")

    args = parser.parse_args(argv)
    args.registry = _absolute_registry_path(args.registry)
    if args.cmd == "register":
        evidence = None
        contract = None
        if args.evidence:
            with open(args.evidence, "r", encoding="utf-8") as f:
                evidence = json.load(f)
            contract = evidence.get("verification_contract")
            if not args.tb and evidence.get("bundle", {}).get("path"):
                args.tb = os.path.join(evidence["bundle"]["path"], f"{args.module}_tb.v")
        entry = register_verified_module(
            args.module,
            args.src,
            candidate_id=args.candidate,
            tb_path=args.tb,
            design_type=args.design_type,
            registry_path=args.registry,
            verification_contract=contract,
            verification_evidence=evidence,
            legacy_import=args.legacy_import,
            project_root=args.project_root,
        )
        if args.legacy_import:
            print("WARNING: imported as LEGACY; no strict verification evidence is available.")
        print(json.dumps(entry, indent=2, sort_keys=True))
    elif args.cmd == "plan":
        print(json.dumps(resolve_link_plan(args.top, args.module, args.registry, args.project_root), indent=2))
    elif args.cmd == "health":
        result = registry_health_report(args.registry, args.output, args.project_root)
        print(result["report"])


if __name__ == "__main__":
    main()
