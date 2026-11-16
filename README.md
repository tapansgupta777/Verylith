# Verylith 3.0: AMS Digital Companion Compiler

Verylith (ChipGPT) translates hardware descriptions into candidate Verilog RTL, then tests and synthesizes candidates before permitting registry promotion. It targets the **digital control side** of analog/mixed-signal systems. It also supports generic primitives, selected protocols, and hierarchical designs assembled from registered modules. A passing build is evidence about the tested RTL configuration and stimuli, not formal equivalence, physical timing closure, CDC sign-off, or analog performance.

## Architecture

```text
Prompt -> Clarifier -> Spec-Sync -> Micro-Architect -> parallel RTL candidates
       -> structural preflight -> black-box testbench association
       -> Verilator lint -> Icarus compile + simulate -> Yosys synthesis and checks
       -> immutable evidence bundle -> registry promotion

SAR: raw settling preflight before Clarifier; canonical interface and trusted
     deterministic oracle established before candidate generation.
Failure: classify -> targeted RTL mutation, tool/TB handling, or new architecture.
```

`chipgpt.py` drives generation, tests, and promotion. `skeletons.py` provides design detection, templates, invariants, and protocol assertions. `linker.py` validates reusable modules, computes dependency plans, and owns registry persistence and evidence integrity.

Supported templates include counters, shift registers, skid buffers, FSMs, UART TX, synchronous and asynchronous FIFOs, SAR control, SPI register banks, and non-overlapping clocks. Known AXI-stream-to-UART composition has a deterministic testbench. Unknown bridges can use the generic path instead of inheriting a mismatched peripheral template. The asynchronous FIFO template separates reset-free synchronous RAM ports from Gray-pointer CDC logic; its oracle includes full, overflow, and readback checks. The SPI template uses a bundled-data toggle/ack commit. Those templates do not guarantee physical CDC or analog behavior.

## Getting Started

Run from the project root in Linux or WSL. The offline suite has been exercised with Python 3.13.9, Verilator 4.038, Icarus Verilog 11.0, and Yosys 0.9. Make `verilator`, `iverilog`, `vvp`, and `yosys` available on PATH. For Debian/Ubuntu:

```bash
sudo apt-get install python3-venv verilator iverilog yosys
python3 -m venv venv
./venv/bin/python -m pip install openai groq
./venv/bin/python -B chipgpt.py --help
./venv/bin/python -B linker.py --help
```

Live generation uses the selected SDK's environment credential: `OPENAI_API_KEY` for `--provider openai` or `GROQ_API_KEY` for `--provider groq`. Configure it locally, not in source control. Offline tests do not require provider credentials.

```bash
./venv/bin/python -B chipgpt.py 'Design an 8-bit synchronous up counter with reset and enable.' \
  --name counter8 --provider openai --candidates 3 --retries 10

./venv/bin/python -B chipgpt.py 'Design a 12-bit SAR ADC controller. SETTLE_CYCLES=2. Comparator high keeps the trial bit.' \
  --name my_sar --provider openai --candidates 3

./venv/bin/python -B chipgpt.py 'Design a two-module counter and status-register hierarchy.' \
  --name counter_system --provider openai --soc
```

These examples invoke paid generation. `--name` defaults to `my_module`; `--provider` defaults to `openai`. The SoC mode builds missing dependencies and then the top; it aborts if a dependency does not reach promotion. Use a fresh name to avoid overwriting older candidate/debug scratch paths.

## Design Generation

Spec-Sync reconciles explicit user fields with Clarifier output, records assumptions/provenance, and hands a common specification to generation and verification. AMS specifications distinguish digital structural, CDC, protocol, and numerical invariants from physical assumptions. The architect returns a structured blueprint; interface locking, static checks (including mixed blocking/nonblocking and template invariants where applicable), and a limited generic reviewer filter candidate RTL. For SAR the reference layout is advisory, not a source-pattern requirement.

The build defaults to three candidate slots and ten rounds. For OpenAI reasoning calls, the slot profile defaults to **low, low, medium** and stays fixed across build rounds and transient retries. The previous implementation chose effort from a temperature index clamped to 3; repeated-error offsets could map all candidates to low. Temperature selection is now independent of reasoning effort and still applies to non-reasoning calls. To opt into a high-effort slot:

```bash
export VERYLITH_REASONING_EFFORTS=low,medium,high
```

The environment variable accepts exactly three comma-separated values from `low`, `medium`, and `high`. Additional slots cycle through the profile; a candidate slot can fail or be filtered without producing usable RTL. OpenAI routes fast calls to `gpt-4o-mini` and strong calls to `gpt-5.4-mini` (4096 and 16384 token budgets); Groq calls `llama-3.3-70b-versatile` rather than applying the OpenAI effort profile. OpenAI response diagnostics include finish reason, completion and reasoning usage when present, effort, and budget. A confirmed `length` response is rejected without an identical-budget retry, even when partial JSON appears parseable.

Failure classification distinguishes preflight, DUT lint/syntax, functional simulation, testbench/tooling, and synthesis problems. The bounded loop can preserve a better prior candidate, mutate on actionable failures, or redesign the architecture. Experience memory can feed prior lessons back into later builds. In-process caches include the applicable spec/architecture/interface, candidate and policy identity; SAR identities include contract and oracle metadata, and fresh OpenAI reasoning RTL keys include the effort profile.

## SAR ADC Controller

The public interface is fixed by Spec-Sync, not by the architect:

| Direction | Signals |
| --- | --- |
| Input, scalar | `clk`, `rst_n`, `start`, `comparator` |
| Output, resolution bits | `dac_code`, `adc_out` |
| Output, scalar | `busy`, `done` |

`sar-controller-contract-v2` is the live default. It uses an active-low reset, rising-edge comparator decisions, MSB-to-LSB trials and an explicit comparator convention. Spec-Sync accepts positive integer settling counts such as `SETTLE_CYCLES=2`, spelled counts, and equivalent DAC/CDAC stability phrases. Equivalent raw values agree; conflicting, invalid, or ambiguous raw counts fail closed. Raw requirements take precedence over a Clarifier assumption. Where raw-only design detection identifies SAR, these settling checks happen before the Clarifier call. The pinned 12-bit/two-cycle/high-keeps contract hash is:

```text
d7af02003bf7d54aedb13c674860e0c1262076be9138e5ecc815328fbae7567f
```

The built-in v2 oracle (`sar-blackbox-oracle-v2.1`, schema `sar-oracle-schema-v2`) is constructed and provenance-checked before architecture generation. There is no LLM testbench fallback. It checks reset outputs, immediate busy assertion, trial progression, rejecting stimulus during settling, comparator decisions, start while busy, result commit, one-cycle done, and a second conversion. The digital comparator model keeps a trial when it is no greater than the target. High- and low-keeps conventions encode that keep decision with opposite logic levels. RTL verification does not establish the physical DAC or comparator settling time.

### Reference Timing

Let **C** be the accepted start posedge and **U** a trial publication posedge. The independent test reference raises busy at C, initializes at C+1, and publishes the first trial at C+2. It holds the public trial through U+N for N settling cycles, drives the resolution decision on the intervening falling edge, and updates the public DAC at **U+N+1** using the current comparator. The reference publishes the next trial on the following edge. At the LSB decision it commits `adc_out` and resolved DAC together. One edge later it drops busy and asserts done; done clears on the next edge. For 12 bits and N=2, this reference commits at C+49, asserts done at C+50 and clears done at C+51.

The existing v2 bench waits for each published trial rather than enforcing C+2 or an exact next-trial cadence; the reference schedule is not an additional v2 acceptance constraint. Its settling stimulus is a fixed reject level until the decision, so not all early samples are independently distinguishable on rejecting bits. This is bounded, vector-based black-box testing, not an exhaustive timing proof.

### Live Verification and Strict Promotion

On November 16, 2026, a fresh live 12-bit/two-cycle SAR v2 candidate passed Verilator, Icarus (four coverage hits), and Yosys (49 cells, eight DFF cells). **The live run did not promote it.** The evidence producer stamped `verilator-iverilog-yosys-policy-v1`, whereas the linker required `verylith-verification-policy-v1`. The first `validate_promotion_evidence()` call rejected the draft in `_promotion_preflight()`, before publishing a bundle or writing the registry. This was not a failure of the RTL, simulator, or oracle.

The winning pair was saved under `workspace/live_sar_v2_2cycle_02/candidate_1` at 00:48:49 IST. Its RTL matches the round-1 slot-1 preverification file. It is a different candidate from the failed candidate 1 of the previous run, which reused the same design name.

| Saved artifact | SHA-256 |
| --- | --- |
| RTL | `d670978615038f8ee218ad8563f7f965519d6b4da5125a85245c850897866e5c` |
| Testbench | `9dfd4766809c2ed88b79f4408f0e08e5c6bf3caf6f1f838973ed214aade90cea` |

The offline regression matches both hashes and requires the saved testbench to be byte-identical to the current trusted v2 generator output before using that provenance. It recompiles and reruns the pair using actual local tools, feeds the results directly into the production evidence producer, and reaches `VERIFIED_AND_PROMOTED` in an isolated registry. The reloaded record is `STRICT` and links from the temporary project root and unrelated directories. It did not publish to the production registry.

### Earlier Live Failure and Offline Replay

The earlier `workspace/live_sar_v2_2cycle_02` run reached lint and simulation but did not reach Yosys. Before candidate 1 was overwritten by the later run, Icarus replay reproduced the busy failure for candidate 1 and the bit-10 failure for candidates 2–10. In the earlier candidate 1, `busy <= busy_flag` observes the old flag on the accepted start edge. In candidates 2–10 the decision path drives the public DAC from the previous trial shadow `dac_output`; clearing the internal trial does not clear that shadow. In candidate 2, sampling and polarity are correct at the first rejection. The output projection is wrong, not the settling counter or a testbench scheduling race.

Archived candidate 2 source and testbench were both timestamped November 15, 2026, 23:56:52 local time. Candidate 2's source SHA-256 is `fdde933cca8552205d11c09b635f8fafbae4b71dc11a14b6cf4ba839ff555605`, matching the round-2 preverify slot and candidate 3 despite a different timestamp. Before the rerun, all ten benches from the earlier run matched SHA-256 `be8000f32c27585d5efeab3a7402280039d3440545ca34acc9594f561a87d031`. Round-5 preverify slot order maps to saved candidates 10, 9, 8 by hash, not by numeric ID. Preverify artifacts from November 12 are not evidence for this replay. The replay used source and testbench bytes, not the saved simulation binaries or unverified assumptions about candidate IDs.

For target `A00`, an independent drive schedule captures candidate 2 after nonblocking updates. ADC remains `000`, busy is 1 and done is 0 throughout the table.

| Edge | Trial bit and event | Comparator | Expected DAC | Observed DAC | Internal trial |
| --- | --- | --- | --- | --- | --- |
| C+0 | Accept start | 0 | 000 | 000 | 000 |
| C+1 | Initialize | 0 | 000 | 000 | 000 |
| C+2 | 11: publish trial | 0 | 800 | 800 | 800 |
| C+3, C+4 | 11: settle | 0 | 800 | 800 | 800 |
| C+5 | 11: sample and retain | 1 | 800 | 800 | 800 |
| C+6 | 10: publish trial | 0 | c00 | c00 | c00 |
| C+7, C+8 | 10: settle | 0 | c00 | c00 | c00 |
| **C+9** | **10: sample and reject** | **0** | **800** | **c00** | **800** |

The offline test repairs a temporary copy: update both DAC shadow and public DAC from the current decision, then remove the separate delayed public done/busy behavior. The original oracle passes both complete conversions on that copy. A distinct countdown-based Verilog-2001 reference passes both comparator conventions at 12 bits/two cycles and 4 bits/one cycle and synthesizes with Yosys. Early, late, inverted, missing-reject, bad-busy, and stretched-done variants are rejected. Neither the temporary repair nor the reference was promoted.

### Functional Feedback

A decision failure prints the bit, trial, expected and observed DAC, comparator level, and expected/checked edge relative to accepted start. An eight-edge observational trace includes DAC, ADC, comparator, busy, and done. The full runtime log, candidate fingerprint and RTL hash go under `workspace/<name>/debug/round_<N>/sar_functional/`; mutation gets the concise assertion and timing context. A DUT that never completes is classified as a DUT functional timeout, not automatically as a broken oracle. The v2 stimulus and assertion acceptance conditions are unchanged.

The trusted SAR oracle remains locked across DUT functional failures and architecture redesigns. The loop skips RTL already rejected functionally by the same oracle (hash normalized for comments/whitespace) and requests a new architecture after three distinct RTL hashes hit the same assertion under one blueprint. Oracle construction/provenance and tooling errors are handled separately; the pipeline does not substitute a weaker oracle.

### Contract v3

`--experimental-sar-contract-v3` opts into independent v3 verification and strict promotion. V2 remains the default. V3 defaults to minimum cadence, a one-cycle done pulse (explicit longer pulses remain supported), and maximum bit latency of settling cycles plus two. The canonical hash uses the same existing sorted semantic serialization, which excludes provenance and unresolved-field bookkeeping. Use a new module name for v3: a STRICT v2 registration cannot be replaced by a v3 registration of the same name.

```bash
# Live generation uses provider credits; this example was not run for validation.
./venv/bin/python -B chipgpt.py '12-bit SAR ADC. SETTLE_CYCLES=2; cadence_mode=exact; comparator high keeps the trial bit.' \
  --name my_sar_v3 --provider openai --experimental-sar-contract-v3
```

#### V3 edge interpretation

`C` is the rising edge accepting start, `U_i` publishes bit i's trial,
and `D_i` samples the comparator and resolves that trial. `S` is settling
cycles, `L` the maximum bit latency, and `P` the configured done length.

| Event | Exact cadence | Minimum cadence |
| --- | --- | --- |
| Accepted start; busy rises, old result holds | C | C |
| First registered trial | C+2 | C through C+L (oracle startup watchdog) |
| Settling begins | U_i | U_i |
| Decision, rejected trial clears on DAC | U_i+S+1 | U_i+S through U_i+L-1 for non-final bits; through U_i+L for LSB |
| Next trial, non-final bits | D_i+1 | At least D_i+1, no later than U_i+L |
| LSB result commits, done rises, busy falls | D_0 | D_0 |
| Done clears | D_0+P | D_0+P |

The normalized v3 phrase "done asserts after final result commit; busy falls
on the same edge" is interpreted as ordering within the commit edge, not an
extra clock, unlike v2. The minimum first-trial deadline is an explicit
oracle liveness bound: the contract's per-bit counter otherwise starts only
when a trial is published. Before initial publication the old DAC may be held
or cleared; the old ADC result holds until commit. Asserting reset between
edges must clear all four outputs before the next rising edge and abort the
conversion. Deassertion alone does not start a conversion. A held-high start
cannot retrigger; the checker exercises rearming by observing low on an idle
rising edge, then accepting a later high. It does not require overlapping a
previous done pulse with acceptance of a new transaction.

#### Independent oracle and stimulus

[`sar_v3.py`](sar_v3.py) emits an Icarus black-box scoreboard
(`sar-edge-scoreboard-v3.1`, schema `sar-oracle-schema-v3`). It derives the
trial and result from the target and its independent committed prefix, never
from DUT registers or observed ADC results. Equality keeps. At falling edges
the oracle drives stable or adversarial comparator levels; at rising edges
it checks public outputs after nonblocking updates. It bounds trial publication,
rejection, progress, completion, and the run itself. The modular countdown
reference in [the test suite](tests/test_sar_v3.py) is not used by the oracle.

The deterministic vectors include zero/all ones, midscale and neighbors,
alternating codes, walking ones/zeros, odd/even endpoints, and up to 12
additional draws with fixed xorshift32 seed `0x51A7C0DE`; duplicates are removed
at small widths. The suite tests 1-, 4-, 8-, and 12-bit positive controls, both
comparator conventions, and two distinct legal 8-bit minimum schedules. It
also exercises a 257-cycle settle count and an explicit three-cycle done pulse.
Transaction checks include held-high start, busy-start pulses, idle-low rearm,
successive different targets without extra idle padding, retention of the
previous ADC result, and atomic commit. Reset is asserted between edges while
idle, during initial settling, immediately after a middle-bit rejection,
before the LSB decision window, and during done. A one-bit design naturally
has no middle-bit phase.

| Targeted mutants | Outcome |
| --- | --- |
| Reversed polarity; missing reject; wrong LSB | Rejected: illegal decision or non-atomic final result |
| Early exact/minimum sample; absent settling; narrow settling counter | Rejected: DAC changes before legal decision |
| Late exact sample; early initial trial; skipped/repeated bit; excessive minimum latency | Rejected: publication, decision or progress window |
| Premature commit; result changes in busy; late/early busy; early, missing or stretched done | Rejected: transaction boundary or result checks |
| Busy start accepted; held start retriggers | Rejected: shifted trial or unexpected busy |
| Synchronous-only reset; aborted state resumes | Rejected: outputs before the next edge or after release |

All 23 tested mutants were rejected. **This is not internal edge proof.** A
non-final keep can leave DAC unchanged; absent decision-valid, a hidden sample
that produces the same visible outputs cannot be timed precisely. In minimum
mode the correct comparator level must be valid across the entire legal
decision window or the oracle could falsely reject a slower compliant DUT.
Exact mode inverts the level outside the single decision edge for adversarial
vectors; minimum mode inverts it outside the legal window. Separate stable
vectors exercise consecutive keeps/rejects. A minimum-mode vector boundary
with an independently expected previous result equal to the MSB trial is
reset to avoid mistaking a stationary DAC for a new publication edge. The
checker does not prove subcycle glitch freedom, exhaustive stimulus coverage
or analog behavior. Configurations exceeding a conservative four-million-edge
verification budget fail at oracle construction rather than silently reducing
the vector matrix.

## Verification and Evidence

Verilator checks lint and structural wiring. Icarus compiles RTL and the bench, then `vvp` runs the test. Passing requires runtime exit zero, no detected failure, and an actual `SIMULATION_SUCCESS` marker; compile output cannot supply the marker. Yosys runs synthesis passes and structural checks, with design-dependent sequential/memory policy. These stages test the exact artifact bytes and configuration, not all parameter values or possible stimuli.

Promotion stages and verifies a copy of the exact RTL/testbench into `workspace/verified/<module>/<bundle-prefix>/`, then publishes the validated bundle (atomic rename where possible, verified copy fallback). Store collisions or mismatches are rejected. `verylith-verification-evidence-v2` records tool versions, commands, statuses and output, module/candidate/configuration, hashes, oracle provenance and contract support, verification policy, vector identity, and the bundle reference. Non-SAR records can have a null contract and record the actual deterministic or LLM testbench path. Strict SAR requires the trusted oracle for the exact contract version. Schema v1 lacks required simulation marker evidence and is not accepted as strict.

V3 adds the cadence, sample event, latency, pulse length, seeded vector
plan/targets and SHA-256 of captured tool output in evidence configuration.
Promotion and strict linking recompute the canonical contract hash and rebuild
the deterministic v3 oracle, checking its byte identity and runtime success
and coverage markers. A version label cannot turn a v2 bench into v3 evidence.
Evidence and bundle digest algorithms and the existing v2 rules are unchanged.
Retain this exact generator when introducing a new oracle identity if older
strict v3 bundles need continued verification.

`linker.VERIFICATION_POLICY_ID` is the single authority for `verylith-verification-policy-v1`; ChipGPT imports it for evidence production and policy-keyed caches. It is separate from the evidence schema, contract version/hash, oracle implementation/schema and artifact hashes. The promotion preflight checks policy and tool evidence, including the runtime success marker. Bundle publication checks artifact bytes and seals the manifest. Registration validates the sealed record before writing, then ChipGPT reloads and requires `STRICT` classification. The linker rechecks the same policy during dependency planning. No old evidence needs to be edited or accepted under an alias.

The evidence digest hashes canonical JSON with sorted keys and compact separators, excluding the digest itself. The bundle digest includes artifact hashes, module/candidate and contract identity, oracle/policy/schema, vectors/configuration, and tool result metadata including runtime marker/output digest. Full digests are verified, not just the directory prefix. These checks detect staleness and inconsistency, not malicious forgery or incorrect verifier logic.

| Exit | Outcome |
| --- | --- |
| 0 | `VERIFIED_AND_PROMOTED` |
| 1 | `FAILED` |
| 3 | `VERIFIED_NOT_PROMOTED`: verification passed, persistence/post-write validation did not |

`autonomous_build_loop()` returns true only for promotion; `get_last_build_result()` exposes the structured outcome.

## Registry and Path Safety

Strict linking rechecks module identity, source and testbench hashes, evidence manifest and digest, bundle identity, contract/oracle compatibility and tool results. Relative stored paths resolve from the project root associated with `<root>/workspace/registry.json`, not the invoking directory. Custom layouts can specify `--project-root`. Paths for `artifact_store` and the evidence bundle must canonicalize to the same directory, and source/testbench must refer to its expected files. Absolute paths also work, including mapped Windows drive paths under WSL. Stored strings, manifests and hashes are not reserialized during resolution. Tests cover root, unrelated, and decoy working directories and invalid bundle paths.

Corrupt JSON fails closed and is quarantined without replacing the original registry; unreadable registries stop discovery and SoC startup. Saves are atomic. Historical records may link as explicitly marked **LEGACY** only if the stored source hash matches and it defines the named module. Invalid evidence is not downgraded, and legacy import cannot replace a currently strict record. Strict-to-strict and legacy-to-strict replacements are allowed; registry history stores the new entry and replaced source path, not a full prior-evidence snapshot.

```bash
./venv/bin/python -B linker.py health --registry workspace/registry.json --output /tmp/verylith-health.md
./venv/bin/python -B linker.py plan --top path/to/wrapper.v --module wrapper
```

Manual registration requires validated evidence (`register --evidence <file>`) or an explicit `--legacy-import` for compatibility only. Import stores schema 2, LEGACY status and the actual source hash; it does not verify RTL. The [historical registry health report](REGISTRY_HEALTH_REPORT.md) predates the first STRICT promotion and documents the 4-bit bench versus 12-bit RTL scope mismatch in `sar_adc_12b` candidate 18. The separate `sar_adc_12b_v2` entry has matching historical scope but lacks strict evidence. Do not regenerate or overwrite historical source and bench paths to repair a registry entry.

Current production health is **1 STRICT, 14 historical legacy-linkable, zero integrity failures**. The first production STRICT module, `live_sar_v2_2cycle_03`, passed live v2 verification with 12 bits, two settling cycles, high-keeps, posedge sampling, four coverage markers, 46 Yosys cells and seven DFF cells. Its pinned v2 contract hash is unchanged. This v3 work did not write the production registry or its bundle.

| Protected production identity | SHA-256 / digest |
| --- | --- |
| Registry file | `61b1cac2326f0c1c22f97038487d18d3a00a339fb324f95a9491dc10c7d6e6f1` |
| Published RTL | `b8ce25d32e334c395685db5cfa2c7683e35c64acf54e2162cc80009173f52ce8` |
| Published testbench | `ad75c96b05ba8ac9c47195fb10abb4c42e2943f151e062a271c10eddce706031` |
| Evidence manifest bytes | `451c3f3ce26004296e73179d5270abd895a7cdab390e087fc4785a454601a084` |
| Evidence digest | `936a0be0be6884cd0d2919637633ca24d393ebdf026d668d62c2810911186b0c` |
| Bundle digest | `e7f52e21b2d7680f3c57e0e3a0581a85a9290831769d5df033802210090ec9e2` |

## Testing

```bash
./venv/bin/python -B -m unittest discover -s tests -v
./venv/bin/python -B -m unittest discover -s tests -p 'test_sar_*.py' -v
./venv/bin/python -B -m unittest discover -s tests -p 'test_sar_strict_promotion.py' -v
./venv/bin/python -B -m unittest discover -s tests -p 'test_sar_v3.py' -v
```

Before v3 work the suite ran 79 tests with one failed assertion expecting zero production STRICT entries. The updated test now verifies the actual first STRICT v2 record, including its published RTL hash. Milestone A passed **84 tests in 39.233 seconds**, exit 0, with v3 promotion still disabled.

Final offline results: **87 full-suite tests passed** in 41.795 seconds and
**41 focused SAR tests passed** in 5.607 seconds; both exited 0, with zero
skips and failures. Seven positive reference configurations passed Verilator,
Icarus and Yosys, and the exact and minimum integration fixtures reached
`VERIFIED_AND_PROMOTED` in temporary registries. The strict gate rejected
the wrong contract/policy/oracle identifiers, contractless v3 claims, a v2
testbench hash relabeled as v3, altered vector seed or tool output, changed
RTL/testbench bytes, and corrupted evidence and bundle digests. A separate
production link-plan check resolved the original v2 bundle with no unresolved
dependencies. The before/after path and SHA-256 inventories matched for all
780 production workspace files. Tools: Python 3.13.9, Verilator 4.038,
Icarus Verilog 11.0, Yosys 0.9.

The reference and archived replay use `iverilog -g2001` and `vvp` in isolated temporary directories; the reference synthesis check runs `yosys -Q -p 'read_verilog rtl.v; synth -top sar_reference; check -assert'`. The archive-specific regression skips with a reason when its saved inputs are missing or do not match the pinned source/TB hashes. Mocked tests cover effort and truncation behavior, stable candidate slots, canonical oracle reuse, repeated-failure redesign and duplicate avoidance, without invoking a provider. The wider suite includes actual UART promotion/link testing with local EDA tools in a temporary workspace, strict path integrity tests, and SAR contract/RTL regressions.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Spec-Sync contract violation | Read the quoted raw phrases and parsed values; use consistent positive settling counts and a supported sample event. |
| SAR port mismatch | Check the eight canonical names/directions and code widths. The preflight handles simple ANSI packed ranges, aliases and basic arithmetic, not full Verilog elaboration (for example macros or arbitrary constant functions). |
| Unsupported contract / oracle construction failure | Use v2 for live verification. Check the contract hash, widths, sample event and built-in provenance rather than generating a replacement bench. |
| Structural preflight rejects RTL | Inspect the preverify candidate and the actual invariant, width, latch, or mixed-assignment diagnostic. |
| Reasoning-token exhaustion | Check `finish_reason`, completion and reasoning usage, effort and budget. Try the default profile before raising the budget. A length-truncated slot is not usable RTL. |
| SAR decision expected `800`, observed `c00` | Compare expected edge, comparator input and registered DAC before/after it. Check for stale shadow/NBA dependencies and verify done/busy alignment after the last bit. |
| Functional failure vs. tooling error | DUT assertion failures guide mutation; tool, compile-TB and oracle errors require fixing the environment or harness. A simulator process returning zero alone is not a pass. |
| Registry evidence rejected | Run health/plan, check project-root resolution and both artifact hashes. Do not rewrite historical hashes or downgrade strict entries. |

## Development History

| Milestone | Durable finding and recorded verification |
| --- | --- |
| Protocol and hierarchy foundations (May 2026) | Exact-port and deterministic AXI/UART support; verified-module linking, Spec-Sync, memory-inference controls, and mutation feedback. |
| Milestone 1A | V3 provenance, hash/cache isolation and an explicit v2-only gate; 17 focused tests. V3 is not an oracle implementation. |
| Milestone 1B | Immutable evidence promotion, atomic registry safety, structured exit states, legacy compatibility, and real temporary UART promotion; 29 tests. |
| Post-Milestone 1 patch | Require runtime marker in evidence schema v2; explicit hash-checked legacy imports and no implicit schema-2 fallback; 37 tests. |
| Pre-oracle integrity | Registry-relative sources, strict downgrade refusal, fail-closed SoC discovery, trusted SAR cache/provenance and no fallback; 44 tests, health reports identical from root and another CWD. |
| Strict bundle paths | Registry-relative RTL/TB and manifests, no decoy CWD shadowing, explicit custom project root; 46 tests, root and /tmp health report SHA-256 both `9fc7a76a19912ca7aafa195bf492ea451fd03005e78e2080471e3a826dffed22`. |
| SAR live pipeline | Canonical ports replace incomplete architect interfaces before validation and locking. Trusted oracle errors abort immediately; width/range diagnostics and truncation handling; 66 tests. |
| Settling false rejection | Original splitter failed to split after `SETTLE_CYCLES=2`; the unsupported DAC clock-period clause tripped the unmatched-text guard. Equivalent phrases now reconcile and contradictory/invalid raw SAR requests fail before clarification; 69 tests. |
| SAR functional investigation | All ten saved failures replayed; public DAC shadow bug confirmed at C+9, temporary repair and independent reference passed, oracle retained and reasoning slots decoupled. |
| SAR verification and promotion integration | New live candidate passed all three EDA stages; mismatch at evidence preflight fixed with one authoritative policy ID. Actual offline replay produced a STRICT entry in a temporary registry. Explicit `Sample the comparator on posedge clk` now records `explicit_user` without changing v2 timing or the pinned hash. |
| First production STRICT SAR v2 | `live_sar_v2_2cycle_03` passed live verification and strict promotion; the v3 work preserves the registry and bundle. |
| Independent SAR v3 | Independent scoreboard and mutation gate passed before version-specific promotion was activated. Exact and minimum fixtures achieved strict promotion offline in temporary registries only. |

The seven completed milestone and checkpoint notes in the table were consolidated here; the separate registry health record was retained. Earlier integrity investigations used the pre-promotion registry SHA-256 `2657474f9943263ec20ea01d273ceb4cc3919af61b0e735fe1141f38f4a47a91`; the first production STRICT registration legitimately changed it. Earlier external Windows backups include:

- Source/test snapshots: `D:/Verylith/chipgpt_pre_milestone1_20261104` and `D:/Verylith/pre_post_m1_patch_backups`.
- Registry copies under `D:/Verylith`: `chipgpt_registry_backup_20261107.json`, `chipgpt_registry_pre_oracle_20261107.json`, `chipgpt_registry_strict_bundle_20261112.json`, and `registry_sar_live_20261115.json`.

## Current Status and Roadmap

SAR v2 has a live-verified, production STRICT baseline. V3 is opt-in and has passed independent positive controls, mutation tests, and real EDA and promotion tests in isolated workspaces. The autonomous v3 test mocks model responses but executes the real verification and promotion pipeline. No v3 live model generation or production promotion has been attempted here. Formal property proving, analog co-verification, and physical implementation closure are not provided.
