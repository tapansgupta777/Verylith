# ChipGPT — Autonomous RTL Generator

> **Goal:** Natural language prompt → verified synthesizable Verilog RTL → *(future)* GDS/OAS layout

---

## What It Does

ChipGPT takes a plain-English hardware description and autonomously generates, verifies, and synthesizes Verilog RTL through a multi-agent evolutionary pipeline. It targets Verilator (lint), Icarus Verilog (simulation), and Yosys (synthesis) as the verification stack.

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
| RTL Generator, Testbench Generator | `gpt-5-mini` | Reasoning model needed for correct hardware logic |
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

### Delta-Based Patching
`build_delta_mutation()` + `localize_bug()` extract the failing lines from the RTL rather than rewriting from scratch. The mutation prompt includes the diff context and targeted fix rules. Full rewrites are avoided — they lose correct logic.

### Skeleton Library (`skeletons.py`)
Six structural templates with frozen invariants:
- `counter` — binary counter with load/enable/overflow
- `shift_register` — SIPO with configurable direction
- `skid_buffer` — 1-entry ready/valid pipeline register (frozen `s_ready` equation)
- `fsm` — two-always Moore FSM
- `uart_tx` — 4-state UART transmitter with baud counter
- `fifo_sync` — circular buffer FIFO with gray-pointer full/empty flags

Skeletons are injected into the RTL generator prompt for matching designs. Invariant checker verifies structural properties post-generation.

### Bridge/Adapter Detection
`detect_design_type()` returns `"generic"` when the prompt mixes bus-side signals (`axi`, `aclk`, `tvalid`) with peripheral signals (`uart`, `spi`). This prevents the wrong skeleton from being forced onto a bridge design. Peripheral-specific mutation hints are still injected via `_dsrules` when peripheral keywords appear in the prompt.

### Caching Strategy
- `RTL_CACHE` — keyed on `(prompt, architecture, cand_idx)`, written only after all preflight checks pass
- `TB_CACHE` — keyed on `(prompt, port_interface, error_context, cand_idx)`
- Mutation calls bypass cache (unique key includes error context)
- Cache is cleared on `empty_rounds >= 3` reset to prevent stale RTL replay

### Testbench Agent
Black-box: receives spec + port interface JSON, never sees the RTL. Generates Verilog-2001 testbench with:
- Correct negedge→negedge timing discipline
- Protocol assertions for ready/valid designs
- Coverage bins for edge cases

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
  ├── verify_with_verilator()     — lint + width check
  ├── verify_with_iverilog()      — functional simulation
  └── verify_with_yosys()         — synthesis + DFF count

classify_failure(log)         — map error text to typed failure category
check_skeleton_invariants()   — regex + custom checkers on generated RTL
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

## Known Problems

### Active / Unresolved

**1. AXI-UART bridge `tready` deassertion**
The bridge design has been failing for ~15 runs. The RTL correctly implements UART transmission but drives `s_axis_tready` from a registered `busy` flag rather than combinationally from state. This means ready deasserts one cycle late. The testbench catches this correctly. The fix (use `assign s_axis_tready = (state == IDLE)`) is now in the mutation rules and `_dsrules`, but gpt-5-mini hasn't reliably applied it yet.

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
- **AXI-UART bridge passing** — the `tready` issue is the immediate blocker. The fix is architecturally clear (combinational `assign` from state). Need the model to reliably apply it.
- **Mutation reliability** — understand why mutation rounds produce empty content under API load. May need longer timeouts or a fallback to fresh generation sooner.

### Medium Term
- **UART RX skeleton** — oversampling clock, majority-voting noise filter, shift register capture. Completes the UART peripheral pair.
- **SPI master skeleton** — clock divider, MOSI/MISO shift register, CS_n state machine.
- **Async FIFO skeleton** — gray-code pointer crossing, the hardest common peripheral. Simulation can't prove correctness; needs formal property specification alongside.
- **APB/AXI-Lite slave skeleton** — address decoder, register file, read/write mux. Opens path to memory-mapped peripherals.

### Long Term (roadmap)
- **Hierarchical composition** — generate modules that instantiate other verified modules. Currently each run generates one monolithic module.
- **RISC-V RV32I core** — decomposed generation: ALU → register file → decoder → datapath → top-level integration. Requires hierarchical composition first.
- **Formal verification** — SVA properties for protocol compliance. Simulation proves nothing for async FIFO, handshake protocols, or CPU instruction decode.
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

**Dependencies:** `openai`, `groq`, Verilator, Icarus Verilog (`iverilog`/`vvp`), Yosys

**Outputs:** `./workspace/<name>/candidate_<N>/src/<name>.v` — winning RTL  
**Companion file:** `skeletons.py` — must be in same directory

---

*Last updated: Session covering AXI4-Stream UART bridge failures and width normalization. Core pipeline stable for single-module DATAPATH and simple FSM designs. Bridge/adapter designs remain the active challenge.*
