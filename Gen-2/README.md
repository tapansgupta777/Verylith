# Verylith 2.0 — AMS Digital Companion Compiler

> **Goal:** Engineering intent for analog/mixed-signal support logic -> verified synthesizable Verilog digital companion IP. Physical analog closure remains outside this RTL compiler.

---

## What It Does

Verylith takes a plain-English hardware description and autonomously generates, verifies, and synthesizes Verilog RTL through a multi-agent evolutionary pipeline. The current focus is digital companions for analog/mixed-signal blocks, while retaining its generic primitive, protocol, registry, and hierarchical-linking flows. It targets Verilator (lint), Icarus Verilog (simulation), and Yosys (synthesis).

```
User Prompt
    ↓
Clarifier Agent        — disambiguate spec, extract widths/behavior
    ↓
Micro-Architect Agent  — produce structural blueprint (FSM/DATAPATH)
    ↓
RTL Generator (×N)     — parallel candidate generation with skeleton injection
    ↓
Static Preflight       — BLKANDNBLK check, skeleton invariant check, port match
    ↓
Reviewer Agent         — latch/loop patch (generic designs only)
    ↓
Testbench Generator    — black-box testbench from spec + port interface
    ↓
Verilator → Icarus → Yosys
    ↓
If fail → classify → mutate → repeat (up to 10 rounds)
```

---

## Design Decisions

### Model Tier
| Role | Model | Reason |
|---|---|---|
| Clarifier, Architect, Reviewer | `gpt-4o-mini` | Fast, cheap, sufficient for structured JSON tasks |
| RTL Generator, Testbench Generator | `gpt-5.4-mini` | Reasoning model needed for correct hardware logic |
| Token budget (fast) | 4096 | Sufficient for structured JSON output |
| Token budget (strong) | 16384 | RTL + testbench can be large |

### Candidate Diversity
- 3 parallel candidates per round by default (configurable via `--candidates`)
- Temperature cycling: `[0.2, 0.5, 0.8]` indexed by `(cand_idx + attempt) % 3`
- Reasoning effort cycling: `["low", "medium", "high"]` same index
- Variation salt in architect prompt prevents identical blueprint caching

### Failure-Classification-Based Mutation
`classify_failure(log)` maps simulation output to a typed failure before mutating:

| Type | Trigger | Mutation Target |
|---|---|---|
| `BIT_EXTRACTION` | "requires N bit index" | Counter width / index slicing |
| `WIDTH_MISMATCH` | "Operator EQ expects" | Literal sizing, reg width |
| `HANDSHAKE_LOGIC` | s_ready/m_ready keywords | Skid buffer drain/push paths |
| `PROTOCOL_ASSERTION_FAILED` | "tready not deasserted" | AXI handshake timing |
| `FSM_OUTPUT_WRONG` | tx_out/baud/uart keywords | FSM state output assignments |
| `RESET_AFTER_TX` | "tx_ready.*got 0 after reset" | Reset block completeness |
| `SHIFT_LOGIC` | shift/serial keywords | Concatenation direction |
| `OFF_BY_ONE` | abs(exp-got)==1 AND values>1 | Boundary conditions |
| `BLKANDNBLK` | (pre-flight static check) | Blocking/non-blocking discipline |
| `RTL_SYNTAX_ERROR` | Icarus/Verilator syntax points at `src/` | Fix DUT RTL, not the TB |
| `TB_SYNTAX_ERROR` | Icarus syntax points at `tb/` | Regenerate/repair TB only |
| `YOSYS_TIMEOUT` | Synthesis timeout | Fix synthesis shape / primitive inference |
| `MEMORY_NOT_INFERRED` | Yosys replaces or fails to preserve memory | Use synchronous SRAM/BRAM idiom |

### Spec-Sync Master Specification
The Clarifier is now the contract authority. It resolves ambiguity once, records assumptions, and emits a master specification. RTL and testbench agents receive that canonical document instead of the raw fuzzy prompt, preventing them from inventing different truth tables or opcode maps.

- Ambiguous opcode/truth-table prompts must be expanded by the Clarifier before generation.
- A 16-operation ALU prompt without an opcode map receives a deterministic 0..15 operation table and flag semantics.
- Strict modeling-style requests such as dataflow are preserved as hard structural constraints.
- Downstream agents are told to implement/test the master spec only; the raw prompt is used only for front-door parsing, exact port extraction, and human traceability.
- AMS contracts are categorized as `RTL_INVARIANT`, `CDC_INVARIANT`, `PROTOCOL_INVARIANT`, `NUMERICAL_INVARIANT`, or `PHYSICAL_ASSUMPTION`; physical settling, skew, and PVT claims are not represented as RTL proofs.

### Delta-Based Patching
`build_delta_mutation()` + `localize_bug()` extract the failing lines from the RTL rather than rewriting from scratch. The mutation prompt includes the diff context and targeted fix rules. Full rewrites are avoided — they lose correct logic.

### Skeleton Library (`skeletons.py`)
Ten structural templates with frozen invariants:
- `counter` — binary counter with load/enable/overflow
- `shift_register` — SIPO with configurable direction
- `skid_buffer` — 1-entry ready/valid pipeline register (frozen `s_ready` equation)
- `fsm` — two-always Moore FSM
- `uart_tx` — 4-state UART transmitter with baud counter
- `fifo_sync` — circular-buffer FIFO with registered pointers and full/empty flags
- `async_fifo` — dual-clock FIFO physical contract with Gray pointers, 2-flop synchronizers, and reset-free synchronous RAM access blocks
- `sar_adc_controller` — deterministic registered MSB-to-LSB conversion controller with explicit comparator sampling and settle-cycle semantics
- `spi_register_bank` — SPI mode-0 shadow/active register bank with a bundled-data toggle/ack commit protocol
- `non_overlapping_clock` — registered break-before-make phase controller with logical cycle-level non-overlap

Skeletons are injected into the RTL generator prompt for matching designs. Invariant checker verifies structural properties post-generation.

### Bridge/Adapter Detection
`detect_design_type()` now distinguishes known protocol compositions from unknown bridges. `axi_stream_uart_tx` is recognized as AXI4-Stream sink + UART transmitter and routed to a deterministic composed oracle. Unknown bridges still return `"generic"` so a single peripheral skeleton is not forced onto a multi-protocol design.

### Caching Strategy
- `RTL_CACHE` — keyed on `(prompt, architecture, cand_idx)`, written only after all preflight checks pass
- `TB_CACHE` — keyed on `(prompt, port_interface, error_context, cand_idx)`
- Mutation calls bypass cache (unique key includes error context)
- Cache is cleared on `empty_rounds >= 3` reset to prevent stale RTL replay
- Cache is also cleared after a total testbench-generation failure, so rejected TBs cannot trap the next round in duplicate RTL replay

### Experience Memory
Resolved runs can now append one distilled structural lesson to `workspace/tooling_insights.md`.

- `get_tooling_insights(design_type)` loads universal rules and rules tagged for the current design type.
- Mutation prompts inject those lessons immediately after the mutation banner under `[LESSONS LEARNED FROM PAST RUNS]`.
- The Memory Agent compares the first real simulation/synthesis failing RTL against the final passing RTL and appends one JSON-extracted rule.
- Verilator-only lint failures and rejected testbenches are not used as the first failing example, so the memory stays focused on functional/synthesis fixes.

### Verified Module Registry (`linker.py`)
Candidate folders are append-only and may contain stale losers from earlier runs. The source of truth is now `workspace/registry.json`, which is updated only after a candidate passes Verilator, Icarus, and Yosys.

- `linker.py register` records the verified winner for a module name, including source path, candidate id, testbench path, design type, prompt hash, RTL hash, and Yosys summary.
- Verilator, Icarus, and Yosys resolve dependencies only from `workspace/registry.json`; they do not glob every old candidate folder.
- Hierarchical tops can instantiate previously verified modules. The linker recursively includes registered dependencies and reports `LINK_ERROR` for unknown modules.
- If a top file defines helper modules in the same RTL file, those helpers are treated as local definitions and do not need registry entries.

### Testbench Agent
Black-box: receives spec + port interface JSON, never sees the RTL. Prefer deterministic protocol oracles when available, then fall back to LLM-generated Verilog-2001 testbenches with:
- Correct negedge→negedge timing discipline
- Protocol assertions for ready/valid designs
- Coverage bins for edge cases
- Safe normalization of trivial formatting issues such as bare `timescale` without a backtick
- Pre-verification debug artifacts under `workspace/<design>/debug/round_<N>/`

### Reviewer Agent
Runs post-generation on generic designs only (skipped for skeleton-matched designs). Guards:
- Rejects if reviewer adds `always` blocks
- Rejects if reviewer removes `assign` statements
- Rejects if patched code has BLKANDNBLK
- Disabled after 3 consecutive unhelpful rounds (`reviewer_rejected_streak`)

---

## Current Architecture — Key Functions

```
autonomous_build_loop()       — main evolutionary loop (10 rounds, N candidates)
  ├── generate_specification()    — Clarifier: NL → JSON spec
  ├── generate_architecture()     — Architect: spec → blueprint with algorithm description
  ├── validate_architecture()     — port count, clk/rst presence, FSM completeness
  ├── generate_rtl()              — RTL Generator: blueprint → Verilog + mutation context
  │     ├── detect_design_type()  — choose skeleton and dsrules
  │     ├── get_skeleton()        — inject structural template
  │     ├── build_delta_mutation()— extract failing lines for targeted fix
  │     └── _check_blocking_mix() — static BLKANDNBLK preflight
  ├── review_hardware()           — Reviewer: latch/loop patch (generic only)
  ├── generate_testbench()        — TB Agent: spec → Verilog-2001 testbench
  ├── verify_with_verilator()     — lint + width check, with verified dependency linking
  ├── verify_with_iverilog()      — functional simulation, with verified dependency linking
  ├── verify_with_yosys()         — synthesis + DFF/memory-shape count, with verified dependency linking
  └── register_verified_module()  — persist the passing candidate in workspace/registry.json

classify_failure(log)         — map error text to typed failure category
check_skeleton_invariants()   — regex + custom checkers on generated RTL
resolve_link_plan()           — linker.py dependency resolver for hierarchical designs
auto_patch_param_width()      — DISABLED (was widening regs, breaking literals)
```

### Loop State Variables
| Variable | Purpose |
|---|---|
| `best_score` | `(stage, tiebreak)` — 0=lint fail, 1=sim fail, 2=sim pass, 3=synth pass |
| `best_v_code` | Best RTL seen so far (used as mutation base) |
| `best_tb_code` | Best testbench (locked when it catches real logic failures) |
| `best_err_log` | Error context fed to next mutation |
| `consecutive_sim_fails` | Triggers arch redesign at 5 |
| `empty_rounds` | Triggers cache clear + fresh generation at 3 |
| `same_err_count` | Triggers TB unlock at 4 identical errors |
| `arch_failures` | Per-arch failure type set — triggers redesign at 3 distinct types |
| `reviewer_rejected_streak` | Skips reviewer after 3 consecutive unhelpful rounds |
| `pending_invariant_msgs` | Skeleton violations buffered → fed into next mutation prompt |
| `code_hashes` | Prevents duplicate candidates; cleared each round to allow mutation flow |

---

## Progress Log

### 2026-05-17 - Protocol composition hardening
- Protected the simple UART success path: known `uart_tx` prompts continue to use a deterministic UART oracle instead of an LLM-generated testbench.
- Fixed the explicit-interface parser so descriptive phrases like `the serial output tx` resolve to the real signal name `tx`, not the article `the`.
- Added semantic detection for the first composed protocol type: `axi_stream_uart_tx`.
- Added a deterministic AXI4-Stream-to-UART testbench oracle composed from two reusable obligations: AXI ready/valid capture/backpressure and UART start/data/stop frame timing.
- Added architecture and RTL guidance for AXI-stream UART bridges: IDLE/START/DATA/STOP FSM, combinational `s_axis_tready = (state == IDLE)`, 32-bit baud counter, 3-bit data index, UART idle-high behavior.
- Strengthened generic testbench preflight to reject common Verilog-2001 violations before Icarus: declarations after procedural code, `for (integer i=...)`, SystemVerilog keywords, `while`, and expression/literal part-selects.
- Verified locally that the new parser returns `["aclk", "aresetn", "s_axis_tvalid", "s_axis_tdata", "s_axis_tready", "tx"]` for the failed AXI-UART prompt.
- Verified locally that the new AXI-UART oracle compiles and passes against a known-good bridge implementation.

### 2026-05-17 - AXI-UART full-run result
- Fresh `axi_stream_uart_tx` run passed in generation 2, candidate 6: Verilator clean, deterministic Icarus oracle passed with 2 coverage hits, Yosys passed with 38 cells and 10 reported DFF cells.
- Candidate 1-3 failed because architect/model emitted duplicate state encodings (`IDLE` and `DATA` both equal `2`), so reset/idle decoded as DATA and `tx` idled low from `data_buffer[0]`.
- Candidate 4-5 fixed state encodings but transitioned START->DATA without waiting for `TICKS_PER_BAUD - 1`, so the UART start bit lasted one clock instead of one baud cell.
- Candidate 6 fixed both: unique states, combinational ready from IDLE, explicit idle-high tx, baud-gated START/DATA/STOP transitions, 32-bit baud counter, 3-bit data index.
- Next hardening target: add static preflight checks for duplicate `localparam` state values and ungated UART START/DATA/STOP transitions before simulation.

### 2026-05-17 - General silicon-rule and skeleton hardening
- Added global RTL rules for open-drain/tri-state `inout` ports, signed arithmetic/SRA with `$signed()` and `>>>`, and mandatory `default:` handling in combinational case statements.
- Softened reset guidance so control/state/valid/ready/output regs must reset, while payload memories/register arrays may be reset-less when the spec or RAM/register-file structure requires it.
- Added static preflight checks for duplicate FSM state encodings and UART phase transitions that skip baud-counter gating.
- Upgraded generic testbench prompting with detected clock ports, multi-clock guidance, derived cycle-based timeout budgets, and inout pullup/pulldown patterns.
- Added testbench preflight rejection for invented single `clk` in multi-clock DUTs and procedural/reg driving of `inout` pins.
- Added skeleton detection/templates for `async_fifo`, `axi_stream_arbiter`, and `rv32i_regfile`.
- Improved UNOPTFLAT feedback by preserving Verilator loop output and teaching `localize_bug()` to look for the reported unoptimizable signal.
- Improved the Yosys zero-register failure into `SEQUENTIAL_OPTIMIZED_AWAY` with an actionable message.

### 2026-05-17 - RV32I regfile/ALU run analysis
- `rv32i_regfile` passed in generation 2 after the skeleton invariant rejected malformed memory declarations in generation 1. The winner implements `reg [31:0] regs [0:31]`, x0 read-as-zero, ignored writes to x0, and two asynchronous read ports.
- The original regfile invariant was useful but too syntactic: `regs\s*\[0:31\]` rejected legal variants such as a different memory name or `[31:0]` address ordering. It has since been replaced with a semantic invariant: one 32-entry x 32-bit memory, x0 read-as-zero, and writes to x0 ignored.
- This is a functional/synthesis pass, but not proof of BRAM inference. A direct Yosys run reports `Warning: Replacing memory \regs with list of registers` and maps the design to `$adff` register cells. The async reset loop over the full array is hostile to true RAM/BRAM inference.
- The pipeline's reported `DFFs` metric was a log-occurrence count. It now reports final Yosys stat-table DFF cells and annotates when a memory array was mapped to registers.
- `rv32i_alu` failed before RTL generation because `validate_architecture()` still assumes every valid design has clock and reset ports. A purely combinational ALU has neither, so the architect loop was rejected repeatedly.
- General fix applied: architecture validation is now conditional on design sequentiality, not on a universal clock/reset rule. If the clarified spec or port list indicates a combinational module, `clock_and_reset` may be absent/null and testbenches must use delay-based stimulus rather than `@(posedge clk)`.

### 2026-05-17 - Combinational architecture fixtures
- `extract_requested_port_names()` now parses labeled `Inputs:` / `Outputs:` clauses in addition to `Interface:`, so exact-port locking works for clockless combinational prompts such as ALUs.
- `validate_architecture()` now allows `clock_and_reset: null` for pure combinational modules and requires clock/reset ports only when the spec, exact ports, or architecture actually imply sequential logic.
- The architect prompt now explicitly says pure combinational modules must not invent clock/reset ports and should use datapath/flag/mux descriptions instead of registers.
- Generic testbench prompting now states that no-clock DUTs must use `#delay` stimulus/checks, and testbench preflight rejects invented `clk` usage when the port interface has no clock.
- The RV32I regfile invariant is now semantic rather than string-exact: it accepts any 32-entry x 32-bit reg-array name/order while still requiring x0 read-zero behavior and ignored writes to x0.
- Yosys reporting now counts final stat-table DFF cells instead of every log occurrence and reports when a memory array was mapped to registers.

### 2026-05-17 - RV32I decoder and TB-failure recovery fixtures
- Diagnosed the `rv32i_decoder` run as a testbench-generation/orchestration failure, not a proven RTL failure: all generated TBs were discarded for a bare `timescale` directive before any RTL reached Verilator/Icarus/Yosys.
- Added safe TB normalization: a leading `timescale` directive is repaired to `` `timescale`` before structural validation, while real Verilog-2001 violations are still rejected.
- Added total-TB-failure recovery: when every generated testbench is empty or structurally rejected, `TB_CACHE`, `RTL_CACHE`, and per-run code hashes are cleared to avoid replaying duplicate RTL in the next generation.
- Added pre-verification debug artifact saving so generated RTL remains inspectable even when no valid TB is produced.
- Added `rv32i_decoder` skeleton detection, a structural semantic invariant, and a deterministic combinational oracle covering R-type, I-type, load, store, branch, U-type, and J-type immediate generation.
- Locally verified the decoder oracle against the skeleton with Icarus: all seven coverage bins hit and `SIMULATION_SUCCESS` emitted.

### 2026-05-17 - SRAM physical contract and synthesis taxonomy
- Split syntax failures by source path: Icarus errors under `src/` now classify as `RTL_SYNTAX_ERROR`, while errors under `tb/` classify as `TB_SYNTAX_ERROR`. DUT syntax errors are no longer protected as testbench/tooling failures.
- Removed `COMBINATIONAL_LOOP` from structural-TB failures so UNOPTFLAT feedback mutates RTL rather than regenerating testbenches.
- Added explicit `YOSYS_TIMEOUT` feedback that warns about synthesis-shape explosion, especially failed memory inference causing register/mux blow-up.
- Made the Yosys flow memory-aware: memory-like prompts use `memory_dff; memory_collect` and require a preserved `$mem` cell. If Yosys replaces the array or no `$mem` remains, the run fails as `MEMORY_NOT_INFERRED`.
- Relaxed counter and shift-register reset invariants to accept sized zero literals such as `32'h0` while still requiring an explicit reset assignment.
- Added `sram_byte_en`, a JIT skeleton for synchronous byte-enable SRAM/data memories. It freezes the inferable shape: no memory reset loop, no asynchronous/combinational memory read, direct byte-lane writes inside `always @(posedge clk)`, and registered `rdata`.
- Verified locally that the `sram_byte_en` skeleton passes Verilator and memory-aware Yosys with one `$mem` cell, while the previous async-read data-memory candidate fails as `MEMORY_NOT_INFERRED`.

### 2026-05-17 - SRAM invariant relaxation and deterministic oracle
- The first `sram_byte_en` full run proved the physical-contract detection worked, but exposed an over-strict custom invariant: it only accepted `if (byte_en[0])` style guards and rejected semantically equivalent byte-enable conditions.
- Relaxed the SRAM custom invariant to accept explicit boolean checks such as `byte_en[0] == 1'b1`, combined guards such as `mem_write && byte_en[0]`, and static Verilog-2001 byte-lane loops using `byte_en[i]` with indexed part-selects.
- Kept the physical guardrails: old candidates with combinational memory reads or wire-initializer async reads still fail the invariant.
- Added a deterministic `sram_byte_en` testbench oracle so known SRAM/data-memory prompts no longer depend on the fallback LLM TB generator.
- Locally verified the SRAM oracle against the skeleton with Icarus: reset/no-read zero, full-word write, two partial byte masks, zero-mask no-change, and word-addressing coverage all hit `SIMULATION_SUCCESS`.

### 2026-05-17 - Verified-module registry and hierarchical linker
- Added `linker.py`, a separate registry/linker utility so stale candidate folders are no longer treated as reusable modules.
- Successful pipeline runs now register the exact winning candidate in `workspace/registry.json` after Verilator, Icarus, and Yosys all pass.
- Verilator, Icarus, and Yosys now call `resolve_link_plan()` and include only registered verified dependencies, recursively.
- Yosys now receives an explicit `hierarchy -top <design_name>` command, which is essential for multi-file top-level integration.
- Verified the resolver with an isolated temp registry: a top module instantiating `leaf` linked only the registered `leaf.v` and ignored unregistered stale files.
- Manual backfill is still possible from terminal history: register only candidates that the terminal identifies as winners.
- Backfilled the current workspace registry from terminal-confirmed winners: `uart_tx_axi` candidate 6, `rv32i_regfile` candidate 1, `rv32i_alu` candidate 1, `rv32i_decoder` candidate 2, `rv32i_pc` candidate 1, `rv32i_bcu` candidate 1, `rv32i_lsu` candidate 1, and `data_memory` candidate 1.
- Tightened dependency detection after a dry-run top exposed a false unresolved module named `i` from `if (...)` splitting; instantiation detection now requires a real module/instance word boundary.

### 2026-05-17 - Model routing update
- Switched the strong RTL/testbench generation tier from `gpt-5-mini` to `gpt-5.4-mini`.
- Removed bare `gpt-5` from the reasoning-model allowlist; the allowlist now includes `gpt-5.4-mini` for the strong model path.

### 2026-05-17 - Hierarchical top-level classification
- Fixed a semantic misclassification where a top-level CPU prompt mentioning `rv32i_decoder` as a dependency was classified as the decoder primitive itself.
- Added `hierarchical_top` detection for prompts that instantiate, wire, integrate, or wrap verified/existing submodules, plus CPU core / SoC style prompts.
- The architect now treats `hierarchical_top` as a wrapper/integration design, not as a pure combinational decoder/ALU just because those submodules are named.
- The RTL generator now receives compact registered submodule interfaces from `workspace/registry.json` and is instructed to instantiate dependencies with named port connections, not inline or redefine their logic.
- Reviewer is skipped for hierarchical linker designs; structural correctness is handled by Verilator/Icarus/Yosys plus the verified-module linker.
- Verilator now allows intentionally unused output pins such as `.zero()` by suppressing `PINCONNECTEMPTY`; hierarchy prompting still prefers dummy wires like `alu_zero_unused`.
- Yosys memory-inference checks are disabled for hierarchy prompts. Wrapper ports such as `data_byte_en`, or linked verified memory dependencies, no longer imply the wrapper itself must synthesize to a `$mem` cell.
- WSL smoke-tested both policy changes: a `data_byte_en` wrapper around `rv32i_alu` passes Yosys without `MEMORY_NOT_INFERRED`, and an empty `zero` output connection passes Verilator.

### 2026-05-18 - Spec-Sync and ambiguous ALU hardening
- Upgraded the Clarifier prompt into a master-spec producer: it must resolve missing opcode maps, truth tables, reset semantics, flag behavior, FSM priorities, and modeling style constraints instead of passing ambiguity downstream.
- Added a canonical downstream spec builder. RTL, testbench, and synthesis classification now consume the Clarifier's master specification instead of the raw user prompt.
- Added a deterministic fallback contract for ambiguous 16-operation 4-bit ALU prompts: opcode map 0..15, result wrapping, and exact `ov` semantics for carry/borrow/range/divide errors.
- Preserved strict dataflow modeling as a structural rule: the architect is forced to `continuous_assign_dataflow`, the RTL prompt bans procedural constructs, and preflight rejects `always`, `initial`, `reg`, `task`, and `function` in the DUT.
- The pipeline now prints short `Spec-Sync assumptions` so users can see when ChipGPT had to choose missing behavior.

### 2026-05-19 - Sequential-intent synthesis guard
- Fixed an autoimmune Yosys failure where Spec-Sync section names such as `sequential_or_combinational` and `clocking_scheme` made pure combinational/dataflow designs look sequential.
- Replaced the naive keyword scan with `expects_sequential_storage()`, which checks structured Clarifier fields, exact requested ports, and edge-sensitive RTL before treating 0 DFFs as `SEQUENTIAL_OPTIMIZED_AWAY`.
- Strict dataflow designs now skip the generic reviewer, preventing reviewer patches from reintroducing procedural style after dataflow preflight.

### 2026-05-21 - Hierarchical memory-instance guard
- Fixed an SoC-wrapper preflight false reject: `memory_blocks` in a hierarchical architecture no longer forces the top module to declare a local 2D `reg` array.
- The atomic-memory guard still applies to non-hierarchical DATAPATH designs, where a RAM/FIFO primitive must own its storage shape.
- Architect and RTL prompts now distinguish local memory arrays from verified child memory instances, so wrappers should instantiate `data_memory`/SRAM dependencies instead of redeclaring their internals.
- Exact-port extraction now treats clauses such as `Outputs: none` as an empty port list instead of inventing a fake descriptive output name.
- Architecture validation now lets exact user port lists override the old three-port minimum, so wrappers with only `clk`/`rst_n` can pass. Invalid blueprint retries now print the concrete rejection reason.
- Hierarchical Verilator lint now suppresses `SYNCASYNCNET` only when verified dependency files are linked, because independently verified child modules may legally mix synchronous and asynchronous reset styles on the same top-level reset net.
- Hierarchy prompting now requires every registered submodule port to be connected and discourages local edge-triggered wrapper flops unless the user explicitly asks for new top-level state.
- Yosys uses a memory-preserving flow for hierarchies that link verified memories, but memory-cell and zero-DFF enforcement still applies only to atomic primitives. A wrapper that instantiates registered memories may be optimized away if it has no observable outputs; the child memory contract was already proven when the module was registered.
- Sanitized ambiguous 16-op ALU assumptions so vague divide-by-zero language from the Clarifier is removed when the deterministic fallback opcode table is installed.
- WSL smoke-tested the latest `basic_alu` candidate through Yosys: it passes with `DFF cells: 0`, as a combinational ALU should.

### 2026-05-22 - Experience Memory
- Added `workspace/tooling_insights.md` as an append-only learned-rule store.
- Added `get_tooling_insights(design_type)` to inject universal and design-type-specific lessons into mutation prompts.
- Added `commit_tooling_insight()` Memory Agent to distill one structural rule from a failing RTL and its later passing version.
- The main loop captures the first real simulation/Yosys failing RTL and commits a lesson only after the run later reaches a verified winner.

### 2026-05-22 - Evolution feedback unblocking
- Fixed TB amnesia: structural testbench/tooling failures no longer overwrite an existing real RTL/Yosys failure context.
- Allowed lateral candidate replacement by accepting equal-score updates (`>=`) in the verification loop, so stage-2 synthesis failures can rotate through newer candidates instead of mutating the same stale local minimum forever.
- Experience Memory is now visible to first-generation RTL attempts as well as mutations by injecting learned rules directly into the RTL generator system prompt.

### Current projection
- Move from skeleton-only recognition to protocol/component recognition: UART, AXI-stream, APB/AXI-Lite, SPI, FIFO, and ready/valid should become reusable protocol oracles.
- Treat bridge designs as compositions of known protocol obligations rather than generic monolithic modules.
- Keep LLM-generated testbenches as fallback only when no deterministic oracle exists.
- Add cache versioning by parser/oracle/design-type hash so stale bad assumptions cannot be replayed after infrastructure fixes.
- Eventually emit a run manifest containing parsed ports, detected protocols, selected oracle, architecture hash, cache keys, and failure classifications.
- Later add interactive prompt-back mode for high-impact assumptions; for now assumptions are logged and users can refine the prompt if they disagree.

## Known Problems

### Active / Unresolved

**1. AXI-UART bridge generalization**
The failure is now understood as a composition/oracle problem, not only a `tready` bug. The system now detects `axi_stream_uart_tx` and uses a deterministic composed checker. The next full pipeline run should validate whether the generator follows the new architecture guidance reliably.

**2. Mutation rounds producing zero candidates**
After a stage-1 failure, mutation attempts often produce zero valid candidates for 2-4 rounds. Root cause: mutation of a close variant produces the same hash as the base candidate (discarded as duplicate). Fix applied (code_hashes cleared each non-passing round) but the model also sometimes produces genuinely empty content under API load.

**3. Testbench SV syntax leaks**
Despite explicit rules, the testbench agent occasionally generates:
- Variables declared in unnamed begin blocks
- `for (integer i = 0; ...)` C-style loop declarations
- Wire initial value assignments (`wire x = 0`)
These produce Icarus compile errors. Each caught pattern is added to the anti-pattern list.

**4. Architect returning invalid blueprint repeatedly**
For complex designs (FSM with datapath), gpt-4o-mini frequently returns blueprints that fail `validate_architecture()`. The validation was relaxed for FSMs (only rejects if all three fsm_specific fields are absent) but still fires ~30% of rounds on novel designs.

**5. Width mismatch: localparam explicit sizing**
After fixing the "use unsized literals" issue, the model overcorrected by writing `parameter [15:0] TICKS_PER_BAUD = 434`, making the parameter 16-bit. This then mismatches against 32-bit counters. New rule added: never give numeric parameters explicit bit widths.

### Fixed / Resolved

| Problem | Fix Applied |
|---|---|
| `has_ready_valid_ports` import missing | Added to `from skeletons import` |
| TB generation crash killing entire process | `try/except` on `as_completed` |
| RTL cache deadlock when TB crashes | `code_hashes.clear()` on TB-failure path |
| UART skeleton killing AXI bridge designs | Bridge detection → `generic` type |
| `UnboundLocalError: reviewed_candidates` | Init in both branches of reviewer skip |
| Stage-0 candidates blocked as duplicates next round | `code_hashes.discard()` on Verilator fail |
| Auto-patcher breaking internally consistent RTL | Patcher disabled; prompt rules added instead |
| Reviewer adding always blocks on skeleton designs | Reviewer skipped for non-generic designs |
| `UNKNOWN` failure locking testbench when it shouldn't | `LOGIC_MISMATCH` + `TB_STRUCTURAL` classification |
| COVER_HIT labels triggering wrong failure class | Strip COVER_HIT lines before classification |
| OFF_BY_ONE misfiring on 1-bit handshake signals | Guard: only fires when `max(exp, got) > 1` |
| `break`/`continue` in testbenches | Banned in TB prompt with `disable` alternative |
| Expression part-selects `(a-b)[2:0]` | Rule 7 expanded with wrong/correct examples |
| Bare `timescale` directive in generated TBs | Normalized to backtick-prefixed `timescale` before validation |
| All TBs rejected causing duplicate replay | Clear TB/RTL caches and code hashes after total TB-generation failure |
| Known RV32I decoder relying on LLM TBs | Added deterministic decoder oracle plus skeleton invariants |
| DUT syntax errors treated as TB failures | Split into `RTL_SYNTAX_ERROR` vs `TB_SYNTAX_ERROR` by source path |
| Yosys timeout gave no synthesis-shape guidance | Added `YOSYS_TIMEOUT` and `MEMORY_NOT_INFERRED` feedback |
| Valid sized zero resets rejected | Counter/shift reset invariant now accepts `32'h0` style literals |
| SRAM simulated but exploded in synthesis | Added `sram_byte_en` physical-contract skeleton and memory-aware Yosys check |
| SRAM invariant overfit to exact `if (byte_en[0])` syntax | Relaxed semantic byte-lane checks while preserving async-read rejection |
| SRAM relied on LLM-generated testbench | Added deterministic SRAM byte-enable oracle |
| Stale candidate folders made winners ambiguous | Added `workspace/registry.json` and `linker.py`; only verified registered winners are linkable |
| Linker mistook `if (...)` as module `i` | Tightened instantiation regex to require separate module and instance identifiers |
| Top-level CPU prompt misclassified as `rv32i_decoder` | Added `hierarchical_top` detection, architect guidance, registry-derived submodule signatures, and reviewer skip for linker designs |
| Hierarchical wrapper failed on unused `.zero()` output | Suppressed Verilator `PINCONNECTEMPTY` and prompted dummy wires for unused submodule outputs |
| `data_byte_en` wrapper port triggered SRAM `$mem` requirement | Memory-aware Yosys check now ignores prompt memory keywords for hierarchy tops unless linked deps contain memory |
| RTL and TB hallucinated different ALU opcode maps | Added Spec-Sync master spec routing; RTL/TB now consume the same Clarifier-resolved contract |
| Strict dataflow prompt produced behavioral candidates | Added dataflow architecture forcing, RTL prompt rules, and static preflight rejection for procedural DUT constructs |
| Combinational ALU rejected for 0 DFFs | Replaced Yosys sequential keyword scan with structured sequential-intent detection |
| SoC wrapper rejected for missing local 2D RAM array | `memory_blocks` 2D-reg preflight now skips `hierarchical_top`; prompts clarify verified memories are instantiated dependencies |
| `Outputs: none` could become a fake port | Exact-port parser now recognizes empty input/output clauses |
| Two-port wrappers rejected as invalid architecture | Exact requested ports now override the generic three-port minimum; invalid blueprint logs include rejection reasons |
| Linked SoC rejected for mixed child reset styles | Verilator suppresses `SYNCASYNCNET` only for linked verified hierarchies; width and implicit-net errors remain fatal |
| Wrapper candidates omitted extra submodule outputs | Hierarchy prompt now says to connect every registered submodule port, using dummy wires for unused outputs |
| Output-less SoC wrapper failed `MEMORY_NOT_INFERRED` after linking SRAM deps | Yosys primitive memory/DFF shape checks are skipped for `hierarchical_top`; registry proves child primitive contracts |
| Fixed lessons were lost between runs | Added Experience Memory: resolved runs append one structural insight and future mutations read matching lessons |
| TB tooling failures erased real RTL/Yosys feedback | Structural TB failures now update `best_err_log` only when the current best error is also TB/tooling |
| Evolution stuck on equal stage-2 candidates | Verification loop now accepts equal-score candidate replacement to permit lateral mutation |
| Experience Memory only appeared after a failure | Learned insights are injected into every RTL generator system prompt, including generation 1 |
| Dataflow candidates patched by generic reviewer | Reviewer is now skipped for strict dataflow designs |
| Fallback ALU spec kept vague divide-by-zero assumptions | Ambiguous assumptions are sanitized when deterministic opcode tables are installed |
| Async FIFO TB expected `empty=0` after reset | Added deterministic async FIFO oracle; reset now requires `full=0, empty=1` |
| Async FIFO RAM passed simulation but failed `$mem` preservation | Async FIFO skeleton now separates reset-free memory read/write blocks from resettable CDC pointer blocks |
| Async FIFO invariant allowed async `assign rd_data = mem[...]` | Added semantic invariant rejecting assign/wire/combinational memory reads and reset-bearing memory access blocks |
| Async FIFO smoke test missed wrap/full bugs | Oracle now fills all 16 entries, checks `full`, attempts an overflow write, and reads back all data |
| Async FIFO pointers hallucinated as N bits | Semantic invariant now requires N+1 write/read pointer declarations |
| Clocked divider was forced into strict dataflow | Dataflow enforcement now disables itself for clocked/sequential designs and overrides impossible clocked-dataflow blueprints |

---

## Improvements Tried (with outcomes)

| Approach | Outcome |
|---|---|
| Auto-patch `reg [N:0]` → `[31:0]` for param comparisons | Removed. Created new mismatches (localparam vs parameter, ADD literal width, wire mismatch). Half-compiler worse than none. |
| Skeleton for every design type | Works for well-defined primitives (skid buffer, UART). Fails for bridge/adapter designs — no single skeleton fits. |
| Reviewer for all designs | Disabled for skeleton designs. Was adding `always` blocks 90% of the time. Net negative. |
| `pending_invariant_msgs` feedback | Working. Skeleton violations now flow into next mutation prompt. |
| Algorithm description in architect prompt | Working. Architect produces pseudocode-level behavior, reducing RTL hallucination. |
| Semantic s_ready invariant checker | Working. Accepts `||`, `|`, De Morgan equivalent — not just exact string. |
| TB locking on logic failures | Working. Testbench locked when it catches a real logic error, preventing mutation of a valid fitness function. |
| Same-error escape hatch | Working. Unlocks testbench and forces fresh generation after 4 identical errors. |

---

## Next Goals

### Short Term (current sprint)
- **AMS full-run validation** - Exercise SAR, SPI register-bank, and non-overlapping-clock prompts against live LLM candidates, then assess which failures are model quality versus missing contract coverage.
- **Companion hierarchy validation** - Build a small mixed-signal control hierarchy from verified digital companions, using the registry and exact-linker signatures.
- **Mutation reliability** — understand why mutation rounds produce empty content under API load. May need longer timeouts or a fallback to fresh generation sooner.

### Medium Term
- **AMS assertions and formal properties** — add property-level checks for SAR progression, SPI event delivery, and phase exclusivity, particularly where simulation only samples a subset of legal traces.
- **Additional companion contracts** — prioritize comparator calibration sequencing, PLL/DLL lock control, sensor bias-register interfaces, and reset/clock-control glue according to actual use cases.
- **CDC constraint integration** — export or validate the bundled-data timing assumptions that cannot be established by two-flop event synchronization alone.

### Long Term (roadmap)
- **Hierarchical composition** — registry/linking now exists; next step is composition planning so the architect intentionally instantiates verified modules instead of regenerating monoliths.
- **Mixed-signal co-verification** — connect digital companion contracts to behavioral analog models; keep numerical and physical assumptions explicit at the model boundary.
- **Formal verification** — SVA/proof flows for protocol and CDC compliance. RTL simulation alone cannot prove all asynchronous or analog-boundary behavior.
- **FPGA backend** — synthesis → place-and-route via Nextpnr/VPR. WNS/TNS feedback to mutation engine.
- **ASIC backend** — OpenLane flow, real post-layout metrics feeding back into generator. This is the GDS/OAS goal.

---

## Usage

```bash
python3 chipgpt.py "<natural language hardware description>" \
    --name <module_name> \
    --provider openai \
    --candidates 3 \
    --retries 10
```

For autonomous SoC decomposition, add `--soc`. ChipGPT asks the System
Architect agent for a dependency graph, builds any missing modules into the
verified registry, then generates the final hierarchical wrapper:

```bash
python3 chipgpt.py "<SoC-level hardware description>" \
    --name <top_module_name> \
    --provider openai \
    --candidates 3 \
    --retries 10 \
    --soc
```

**Dependencies:** `openai`, `groq`, Verilator, Icarus Verilog (`iverilog`/`vvp`), Yosys

**Outputs:** `./workspace/<name>/candidate_<N>/src/<name>.v` — winning RTL for the current run  
**Registry:** `./workspace/registry.json` — authoritative map of verified winners used by the linker  
**Companion files:** `skeletons.py`, `linker.py` — must be in same directory

Manual winner backfill from terminal history:

```bash
python3 linker.py register \
    --module <module_name> \
    --src ./workspace/<module_name>/candidate_<winning_N>/src/<module_name>.v \
    --candidate <winning_N>
```

---

*Last updated: Verylith 2.0 AMS digital-companion contract and deterministic-oracle update.*

*2026-05-17 update: Added exact-port parser hardening, `axi_stream_uart_tx` composition detection, deterministic AXI-stream/UART oracle, and stronger generic testbench preflight.*
