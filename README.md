# Symbolic Resource Planning Environment (`symplan`)

English | [简体中文](README_zh.md)

## Background and Motivation

Post-training of large language models commonly mixes reasoning trajectories collected from multiple teachers, prompting strategies, or search procedures. Because the same problem can be solved through structurally distinct reasoning paths—such as forward construction versus backward subgoal regression, or breadth-first exploration versus depth-first search—combining multiple reasoning modes under identical problem inputs introduces conditional multi-modality in the training target.

Evaluating how such mode diversity interacts with model capacity, data scale, and training budget on natural-language benchmarks faces three obstacles:
1. **Unobserved teacher modes**: The underlying search strategy of each trajectory is rarely labeled, and surface-level stylistic variation is entangled with structural search choices.
2. **Coarse-grained verification**: Checking only the final answer cannot distinguish a coherent reasoning trace from a valid final answer preceded by an invalid search process (`hack`), nor can it pinpoint the exact step where two modes diverge or where an error renders the goal unreachable.
3. **Uncontrolled problem distribution**: Pretraining contamination and uneven structural difficulty prevent clean isolation of capacity scaling, token exposure, and per-problem mode coverage.

`symplan` is built to make these factors explicitly controllable and measurable. It pairs parameterized symbolic resource-planning tasks with four deterministic search teachers (`forward-bfs`, `forward-dfs`, `backward-bfs`, `backward-dfs`) and an independent rule-replay verifier, enabling controlled investigation of three core questions:
1. **Capacity, data, and mode scaling**: Under bounded model parameters, unique problems, and token budgets, how does mixing multiple reasoning modes affect learning efficiency and per-problem multi-mode coverage compared to single-mode training?
2. **Divergence and step criticality**: Where do different search modes diverge along a trajectory, and how do errors at critical steps (which irreversibly consume required resources) versus non-critical steps determine task success?
3. **Process-level failure diagnosis**: How do local search-graph construction errors (e.g., `NEW`/`DUP` state tracking or frontier scheduling violations) propagate to final plan executability, and how can targeted interventions or post-goal path verification (`RECHECK`) repair them?

## Why Symbolic Resource Planning?

Each problem specifies initial resources, transformation rules, and goal resources. The model must select a sequence of rules that transforms the initial resources into a state meeting the goal. Controlling the problem graph structure and search algorithms makes every reasoning choice and its downstream consequence directly computable:

| Design Feature | Purpose |
|---|---|
| Multiple valid paths per problem | Provides distinct valid reasoning processes and final plans for the same problem |
| Four search modes (Forward/Backward × BFS/DFS) | Controls teacher search direction, scheduling order, and mode mixture ratios |
| Resource-consuming rules and irreversible dead ends | Identifies critical steps and measures reachability and repair cost after deviations |
| Configurable resource scale, dependency depth, merges, and distractors | Constructs stratified training, validation, and test splits across difficulty and structure |
| Rule-defined state transitions and termination | Enables line-by-line replay verification, search cost accounting, mode compatibility, and divergence localization |

---

## Environment Mechanics and Task Protocol

### 1. Resource States and Transformation Rules

A world (`World`) consists of four sections:
- `RESOURCES`: Ordered list of resource symbols (e.g., `A B C D G`).
- `INITIAL`: Non-zero resource counts in the initial state.
- `GOAL`: Minimum resource counts required to reach the goal.
- `RULES`: Directed transformation rules written as `r0: A:1 B:1 -> D:1`.

**Forward State Transition**: A rule is applicable in state $s$ when every resource count in $s$ is at least the rule's input consumption. Applying the rule subtracts its inputs and adds its outputs. Any executable rule sequence whose final state has at least the `GOAL` count for every resource is a valid plan (extra resources are allowed; shortest length is not required). Because no rule produces more total units than it consumes, the reachable state space is finite.

**Backward Demand Regression**: Starting from the `GOAL` demand vector $d$, regressing $d$ through rule $r$ yields a new demand vector $\text{inputs}(r) + \max(d - \text{outputs}(r), 0)$ componentwise. Any regression whose total demand exceeds the total initial units is pruned. Termination is reached when every component of the demand vector is met by `INITIAL`; reversing the rule sequence along the regression path yields a forward-executable plan.

### 2. Four Programmatic Search Teachers

`symplan` includes four deterministic search teachers that deduplicate visited states and stop upon expanding the first state that meets the termination condition:

| Mode | Direction | Root Node (`n0`) | Frontier Schedule | Vector Keyword |
|---|---|---|---|---|
| `forward-bfs` | Forward (`INITIAL` $\to$ `GOAL`) | `INITIAL` | First-In-First-Out (FIFO, layer-by-layer) | `STATE` |
| `forward-dfs` | Forward (`INITIAL` $\to$ `GOAL`) | `INITIAL` | Last-In-First-Out (LIFO, earlier-discovered child expanded first) | `STATE` |
| `backward-bfs` | Backward (`GOAL` $\to$ `INITIAL`) | `GOAL` | First-In-First-Out (FIFO, layer-by-layer) | `NEED` |
| `backward-dfs` | Backward (`GOAL` $\to$ `INITIAL`) | `GOAL` | Last-In-First-Out (LIFO, earlier-discovered child expanded first) | `NEED` |

### 3. Search Event Grammar and Vector Formats

In the `search` view, the reasoning trace records one search event per line inside `<think>...</think>`, followed by the final plan outside `</think>`:

| Event Syntax | Meaning |
|---|---|
| `EXPAND <node> STATE\|NEED <counts>` | Pop `<node>` from the frontier and inspect its resource state or demand vector |
| `NEW <node> VIA <rule> STATE\|NEED <counts>` | Applying `<rule>` to the expanded node discovers a new state, assigned sequential node ID `n1, n2, ...` |
| `DUP <node> VIA <rule>` | Applying `<rule>` to the expanded node reproduces the state of existing node `<node>` |
| `CLOSE <node>` | All applicable rules for the expanded node have been enumerated in increasing rule ID order |
| `GOAL <node>` | The expanded node satisfies the termination condition |
| `PLAN <rule_ids...>` | Forward-executable sequence of rule IDs (placed after `</think>`) |
| `END` | End-of-response marker |

Resource vectors `<counts>` support two canonical encodings:
- **Sparse (`sparse`)**: Lists only non-zero entries as `<symbol>:<count>` in `RESOURCES` order (e.g., `A:1 C:2`), or `-` if all counts are zero.
- **Dense (`dense`)**: Lists integer counts for all symbols in `RESOURCES` order (e.g., `1 0 2 0`).

### 4. Walkthrough Example

Consider a world with initial resources `A:1 B:1 C:1` and goal `G:1`:

```text
RESOURCES A B C D E X G
INITIAL A:1 B:1 C:1
GOAL G:1
RULES
r0: A:1 B:1 -> D:1
r1: A:1 C:1 -> X:1
r2: C:1 D:1 -> G:1
r3: D:1 -> E:1
r4: C:1 E:1 -> G:1
```

At the initial state `A:1 B:1 C:1`, both `r0` and `r1` are applicable: applying `r1` consumes `A` and `C` to produce `B:1 X:1`, leaving no applicable rules and making the goal unreachable (so step 1 is a critical step). Applying `r0` produces `C:1 D:1`, from which the goal can be reached either via `r2` (2-step plan `PLAN r0 r2`) or via `r3 -> r4` (3-step plan `PLAN r0 r3 r4`).

The `forward-bfs` teacher response (using `sparse` vectors and `search/on` protocol) is:

```text
<think>
EXPAND n0 STATE A:1 B:1 C:1
NEW n1 VIA r0 STATE C:1 D:1
NEW n2 VIA r1 STATE B:1 X:1
CLOSE n0
EXPAND n1 STATE C:1 D:1
NEW n3 VIA r2 STATE G:1
NEW n4 VIA r3 STATE C:1 E:1
CLOSE n1
EXPAND n2 STATE B:1 X:1
CLOSE n2
EXPAND n3 STATE G:1
GOAL n3
</think>

PLAN r0 r2
END
```

On the same problem, `forward-dfs` expands `n3` immediately after closing `n1` and triggers `GOAL n3` without expanding the dead-end node `n2`.

---

## Repository Structure

```text
symplan/
  cli.py        CLI entry point for the four subcommands: generate / solve / export / verify
  env.py        World validation, forward step & backward regression, plan check, step criticality, and graph isomorphism
  generate.py   Problem generator: families, structural skeletons, difficulty levels, advantage filtering, and deduplication
  search.py     Four deterministic search teachers, event emission, and pairwise mode divergence analysis
  qa.py         Prompt and assistant rendering, think/nonthink views, target_hash, and token length measurement
  verify.py     Event parser, independent replay verifier, mode compatibility, hack classification, and tagged reports
  recheck.py    Opt-in post-GOAL path verification protocol (search_recheck_v1)
scripts/
  prepare_training_data.py  Shared multi-mode corpus builder, tokenizer length filtering, and nested subset extractor
tests/          Unit and acceptance test suite
```

---

## Installation and Quick Start

Requires Python 3.10 or later. The core environment uses only the Python standard library.

```bash
# Core installation
python -m pip install -e .

# Optional: install transformers for --tokenizer length auditing and corpus building
python -m pip install -e '.[tokenizer]'

# Optional: install pytest to run the test suite
python -m pip install -e '.[dev]'
```

Once installed, run `symplan` (or `python -m symplan`) from the repository root to execute a complete CPU pipeline—generation, four-mode solving, QA export, and verification:

```bash
# 1. Generate 5 easy problems
symplan generate --out data/demo --family foundation --level easy --count 5 --seed 42

# 2. Solve with all four search modes using sparse vector encoding
symplan solve --worlds data/demo/worlds.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --vector-format sparse --out data/demo/solutions.jsonl

# 3. Export Qwen-formatted search/on QA records
symplan export --worlds data/demo/worlds.jsonl --solutions data/demo/solutions.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --view search --thinking on --require-all-modes --out data/demo/qa.jsonl

# 4. Independently verify the exported QA records and write a summary report
symplan verify --worlds data/demo/worlds.jsonl --input data/demo/qa.jsonl \
  --kind search --tag demo --strict
```

`data/demo/qa.jsonl` contains the model inputs and teacher responses (`messages` field), and `result/demo/summary.md` contains the verification summary report.

---

## CLI Reference and Usage

### 1. `symplan generate`: Problem Generation

Constructs problems parameterized by family, structural skeleton, and complexity, plants a witness plan, verifies executability, and deduplicates isomorphic structures.

```bash
symplan generate \
  --out data/pool \
  --family general --structure mixed --level medium \
  --count 1000 --seed 42 --workers 8
```

**Key Arguments:**
- `--out DIR` (required): Output directory containing `worlds.jsonl` (problem definitions), `manifest.json` (configuration, rejection counts, and quantiles), and `fingerprints.json` (structural fingerprint counts).
- `--count INT` (required): Number of accepted, deduplicated problems to produce. Exits with code `3` after saving partial results if candidate attempts are exhausted before reaching `--count`.
- `--family {foundation,general,deep-trap,wide-trap}`: Problem family. `foundation` caps witness depth and halves distractor ratio; `deep-trap` and `wide-trap` construct chained deep dead ends and parallel wide distractor branches, respectively.
- `--structure {chain,merge,alternative,competition,trap,divergence,mixed}`: Structural skeleton:
  - `chain`: Single-path dependency chain (supports `--forced-prefix` to enforce a unique applicable rule along the first $k$ steps).
  - `merge`: Tree dependency structure containing multi-input merge rules.
  - `alternative`: Includes an independent valid plan using a distinct rule set in addition to the witness plan.
  - `competition`: Contains rules competing for the same scarce resource along the witness path.
  - `trap`: Contains distractor rules that consume available resources into dead-end decoy symbols.
  - `divergence`: Two valid plans share a common prefix before branching into separate suffixes.
  - `mixed` (default): Allocates slots across `chain` (15%), `merge` (25%), `alternative` (25%), `competition` (20%), and `trap` (15%).
- `--level {easy,medium,hard}`: Complexity presets:
  - `easy`: resources `8:12`, initial units `3:5`, rules `10:18`, steps `2:4`, `arity=2`, `distractors=0.25`.
  - `medium`: resources `16:24`, initial units `4:8`, rules `24:48`, steps `4:8`, `arity=2`, `distractors=0.40`.
  - `hard`: resources `24:40`, initial units `6:10`, rules `48:80`, steps `8:12`, `arity=3`, `distractors=0.55`.
- Explicit complexity overrides: `--resources`, `--initial-units`, `--rules`, `--steps` (accepting integers or closed ranges like `16:24`), `--arity`, and `--distractors` override the `--level` preset.
- Mode advantage filtering: `--pair MODE1,MODE2`, `--budget B`, and `--class {both,first-only,second-only,neither,any}` filter for problems where specific modes succeed or fail within expansion budget `B`.
- `--workers INT` and `--resume`: Parallel generation and checkpoint resumption. Candidate seeds are deterministically derived from `(seed, slot, attempt)` and deduplicated in slot order, making output identical across any `--workers` count for a fixed `--seed`.

---

### 2. `symplan solve`: Multi-Mode Search and Trace Analysis

Runs one or more deterministic search teachers over a problem set to produce event traces and final plans.

```bash
symplan solve \
  --worlds data/pool/worlds.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --vector-format sparse \
  --max-expansions 4096 --max-states 100000 \
  --criticality \
  --out data/pool/solutions.jsonl --workers 8
```

**Key Arguments:**
- `--worlds PATH` / `--out PATH` (required): Input worlds JSONL and output solutions JSONL.
- `--modes LIST` (required): Comma-separated list of modes from `forward-bfs,forward-dfs,backward-bfs,backward-dfs`.
- `--vector-format {dense,sparse}`: Vector encoding in `EXPAND` and `NEW` lines (default `dense`; `sparse` is recommended for long-context training).
- `--max-expansions INT` (default `4096`), `--max-states INT` (default `100000`), `--max-trace-lines INT` (default `0` for unlimited): Expansion budget, visited state limit, and trace line cap. Each record is marked with status `solved`, `exhausted`, `budget_exceeded`, `state_limit`, `trace_limit`, or `error`.
- `--criticality`: Evaluates all applicable rules at each step along every solved plan, classifying each step into `forced`, `free`, `wasteful`, or `critical` and computing the repair cost of deviations (stored in the `criticality` field).
- `--workers INT` and `--resume`: Parallel solving and resumption from existing `(world_id, mode)` records.

**Companion Outputs:**
- `<stem>.stats.json`: Summary statistics including status counts, quantiles for plan length, checked nodes, max frontier size, and pairwise mode comparisons.
- `<stem>.divergence.jsonl` (written automatically when two or more modes are given): Per-world, per-mode-pair structural divergence metrics including `shared_lines`, `shared_expansions`, and `plan_shared_prefix`.

---

### 3. `symplan export`: Training and Evaluation QA Export

Renders worlds and solved traces into chat `messages` format and re-verifies every selected trace prior to export.

```bash
symplan export \
  --worlds data/pool/worlds.jsonl \
  --solutions data/pool/solutions.jsonl \
  --modes forward-bfs,forward-dfs \
  --view search --thinking on \
  --prompt-style structured \
  --selection all --require-all-modes \
  --tokenizer /path/to/Qwen3-tokenizer --max-tokens 65536 \
  --out data/pool/train_qa.jsonl
```

**Key Arguments:**
- `--view {search,plan}` and `--thinking {on,off}` (required): Protocol pair. Supported combinations:
  - `--view search --thinking on`: Places the full search event trace inside `<think>...</think>` followed by `PLAN ...` and `END`.
  - `--view plan --thinking off`: Emits an empty `<think>\n\n</think>\n\n` prefix followed directly by `PLAN ...` and `END`.
- `--prompt-style {structured,detailed}`:
  - `structured` (default): User prompt contains only the `RESOURCES`, `INITIAL`, `GOAL`, and `RULES` header blocks without prose instructions or mode labels, allowing different modes to share identical user prompts.
  - `detailed`: Includes English task descriptions and direction-specific event syntax instructions.
- `--prompt-mode {none,teacher}`: When set to `teacher`, appends `SEARCH_MODE <mode>` to the problem header.
- `--selection {all,uniform,budget-aware}`:
  - `all` (default): Exports all available modes per problem and assigns equal `target_weight` ($1/K$) across records sharing the same question.
  - `uniform`: Samples one mode uniformly at random per problem (requires `--require-all-modes`).
  - `budget-aware`: Samples one mode uniformly among those solved within `--budget B`.
- `--require-all-modes`: Retains only problems solved by all requested `--modes`.
- `--eligible-worlds PATH`: Restricts export to a newline-delimited list of `world_id`s.
- `--limit-worlds INT`: Stratified sampling across primary structures up to the specified world count.
- `--tokenizer PATH` and `--max-tokens INT`: Measures `input`, `think`, `plan`, and `total` token lengths using a local HuggingFace tokenizer and filters out records exceeding `--max-tokens`.

**Companion Outputs:**
- `<stem>.jsonl`: Exported QA records (`messages`, `assistant_prefill`, `target_hash`, `target_weight`, `char_lengths`, `token_lengths`).
- `<stem>.world_ids.txt`: Retained `world_id` list.
- `<stem>.manifest.json`: Export configuration, per-structure retention rates, rejection counts, and length quantiles.

---

### 4. `symplan verify`: Independent Response Verification

The verifier replays every search event and final plan directly against the world's rules without relying on string matching against a reference trace.

#### Input Format

Save model predictions as JSONL. Each line must provide `world_id` and the assistant output (via `text`, `model_answer`, assistant `messages`, or `trace`). Multiple samples for the same problem should use distinct `record_id`s (auto-numbered if omitted):

```json
{"record_id": "sample_001", "world_id": "<world_id from worlds.jsonl>", "text": "<complete assistant response>"}
```

#### Running Verification

**Option A: Tagged Report Directory (`--tag`, Recommended)**

```bash
symplan verify \
  --worlds data/demo/worlds.jsonl \
  --input predictions.jsonl \
  --kind search \
  --tag eval_run_v1
```

Writes three files to `result/<tag>/` (refuses to overwrite existing tags):

| File | Contents |
|---|---|
| `summary.md` | Joint success rate, problem coverage, truncation rate, outcome breakdown (success / hack / invalid plan), difficulty tier table, and top-3 first-error types |
| `summary.json` | Complete metrics, error distribution, input file SHA-256 hashes, verification settings, and completion status |
| `qa.jsonl` | Per-record question, full model answer, optional reference answer (via `--reference`), and `verification` diagnostics |

**Option B: Direct JSONL Output (`--out`)**

```bash
symplan verify \
  --worlds data/demo/worlds.jsonl \
  --input predictions.jsonl \
  --kind search --modes forward-bfs,forward-dfs \
  --out reports/verify.jsonl --strict
```

Writes per-record verdicts to `reports/verify.jsonl` and summary statistics to `reports/verify.stats.json`.

#### Verification Options and Metrics

- `--kind {search,plan,prefix}` (required):
  - `search`: Verifies both the complete search trace and the final plan. **The primary metric is Joint Success Rate** (`plan_valid=true` AND `search_valid=true`). Responses with a valid final plan but an invalid or incomplete search trace are flagged as `hack=true` (subcategorized into `process_error`, `incomplete_search`, or `plan_trace_mismatch`) and counted as failures in the primary metric.
  - `plan`: Verifies only the executability and goal satisfaction of the final `PLAN ... END` block.
  - `prefix`: Verifies a partial search event prefix (an unfinished trailing event line is marked `pending_event`).
- `--protocol {standard,search_recheck_v1}`: Defaults to `standard`. Setting `search_recheck_v1` allows optional `RECHECK_BEGIN STATE ...` / `RECHECK_STEP <rule> STATE ...` / `RECHECK_END` blocks between `GOAL` and `</think>` and independently validates every recheck step.
- `--prefill {think,nonthink}`: Restores `<think>\n` or `<think>\n\n</think>\n\n` when the inference engine pre-filled the prefix and saved only the generated suffix.
- `--modes LIST`: Checks compatibility against specific deterministic teacher schedules (recorded in `compatible_modes` and `mode_errors`). Generic search validity (`search_valid`) only requires expanded nodes to be present in the current frontier.
- `--budget INT`: Checks whether `checked_nodes` stays within the given node expansion budget (`within_budget`).
- `--divergence PATH` and `--criticality`: Computes first-error offset relative to pairwise mode divergence (`first_error_offset_to_divergence`), frontier size at first error (`first_error_frontier_size`), whether the first error occurred at a critical step (`first_error_at_critical_step`), and the first step where the goal becomes unreachable (`first_unrecoverable_step`).
- `--strict`: Exits with code `6` if any record fails verification.

---

## Building the Multi-Mode Training Corpus

`scripts/prepare_training_data.py` builds a stratified (`easy` / `mid` 50% each, preserving the five structural quotas) four-teacher shared corpus. A candidate problem is accepted only when all four teachers solve it, pass independent replay verification, and produce tokenized training texts within `--max-tokens`. Underlying construction groups (`group_id`) are strictly isolated across `train`, `validation`, and `test`.

```bash
# 1. Build the full dataset (supports checkpoint resumption via --resume)
python scripts/prepare_training_data.py --phase build \
  --out data/corpus \
  --tokenizer /path/to/Qwen3-8B-Base \
  --train 20000 --validation 1000 --test 1000 \
  --seed 20261113 --max-tokens 65536 --max-trace-lines 16000 \
  --max-candidates 250000 --workers 12

# 2. Materialize a nested training subset from world_id indexes (e.g., 5,000 worlds = 10,000 QA records for forward_mixed2)
python scripts/prepare_training_data.py --phase subset \
  --input data/corpus/datasets/train/search_forward_mixed2.jsonl \
  --world-ids data/corpus/datasets/subsets/train_5000.world_ids.txt \
  --out data/corpus/train_5000_forward_mixed2.jsonl
```

Upon completion, `data/corpus/datasets/{train,validation,test}/` contains 6 dataset variants per split:
- Single-mode: `search_forward-bfs.jsonl`, `search_forward-dfs.jsonl`, `search_backward-bfs.jsonl`, `search_backward-dfs.jsonl` (1 record per problem)
- Forward two-mode mixture: `search_forward_mixed2.jsonl` (2 records per problem: forward BFS + forward DFS)
- Four-mode mixture: `search_mixed4.jsonl` (4 records per problem: all four modes)

`data/corpus/datasets/subsets/` provides nested `world_id` index lists for `1000`, `2500`, `5000`, `10000`, `15000`, and `20000` training problems (smaller subsets are strict subsets of larger ones while preserving difficulty and structure proportions). A `--phase pilot` mode (e.g., `--pilot 500`) is also available for small-scale trial runs and rejection auditing.

---

## Testing

```bash
python -m pytest -q
```

The test suite covers forward resource transitions and backward demand regression, line-by-line trace consistency across all four search teachers, expansion budget accounting on deep dead ends and full binary trees, detection of tampered events and syntax errors, mode compatibility and first-divergence localization, `think`/`nonthink` export pairing and target weight aggregation, deterministic multiprocess generation and `--resume` consistency, and `--tag` report generation.
