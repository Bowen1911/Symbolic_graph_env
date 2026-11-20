# 符号资源推理环境（symplan）

[English](README.md) | 简体中文

同一道题可以有多种正确的推理过程。`symplan` 提供可控的符号资源规划任务、四种确定性搜索教师和基于规则重放的验证器，用于研究多种推理模式在有限模型容量、数据规模和训练预算下的学习效率。

## 为什么采用符号资源规划任务

每道题给出初始资源、转换规则和目标资源。模型需要选择一串规则，把初始资源转换成满足目标的状态。通过控制题目的图结构与搜索算法，可以直接计算推理过程中的每一步选择及其后果：

| 设计特性 | 作用 |
|---|---|
| 同一道题允许多条有效路径 | 为同一问题提供不同的正确推理过程与最终方案 |
| 正向/反向与广度优先/深度优先组合成四种模式 | 控制教师轨迹的搜索方向、调度顺序及其混合比例 |
| 规则消耗输入资源，部分选择使目标不可达 | 定位关键步骤，计算偏离后的可达性与修复代价 |
| 资源规模、依赖深度、合并与干扰结构可调 | 按难度和结构分层构造训练、验证与测试集 |
| 状态转移与目标判定完全由规则定义 | 逐行重放验证搜索轨迹，统计搜索成本、模式相容性与首次分歧位置 |

---

## 环境机制与任务协议

### 1. 资源状态与转换规则

一个世界（`World`）由四部分组成：
- `RESOURCES`：资源符号列表（如 `A B C D G`）。
- `INITIAL`：初始状态下的非零资源计数。
- `GOAL`：目标状态要求的最低资源计数。
- `RULES`：有向转换规则集合，形如 `r0: A:1 B:1 -> D:1`。

**正向状态转移**：当当前状态中每项资源的数量都不低于规则的输入需求时，该规则可执行；执行后扣除输入资源并增加输出资源。只要最终状态中各项资源数量均不低于 `GOAL`（允许存在多余资源），规则序列即构成有效方案（不要求最短）。每条规则的输出总量不超过输入总量，因此可达状态空间有限。

**反向需求回归**：从 `GOAL` 的需求向量 $d$ 出发，沿规则 $r$ 反向回归得到新需求向量 $\text{inputs}(r) + \max(d - \text{outputs}(r), 0)$。若回归后的总需求量超过初始资源总量则剪枝；当需求向量中各项均被 `INITIAL` 满足时到达终止条件，将回归路径反转即可得到正向可执行方案。

### 2. 四种程序化搜索教师

环境内置四种确定性搜索教师，对重复状态去重并在找到首个满足条件的状态时停止：

| 模式 | 搜索方向 | 根节点 | 节点调度顺序 | 状态关键字 |
|---|---|---|---|---|
| `forward-bfs` | 正向（从 `INITIAL` 到 `GOAL`） | `n0 = INITIAL` | 先进先出（FIFO，按层展开） | `STATE` |
| `forward-dfs` | 正向（从 `INITIAL` 到 `GOAL`） | `n0 = INITIAL` | 后进先出（LIFO，同父后继按发现顺序优先） | `STATE` |
| `backward-bfs` | 反向（从 `GOAL` 回归到 `INITIAL`） | `n0 = GOAL` | 先进先出（FIFO，按层回归） | `NEED` |
| `backward-dfs` | 反向（从 `GOAL` 回归到 `INITIAL`） | `n0 = GOAL` | 后进先出（LIFO，同父后继按发现顺序优先） | `NEED` |

### 3. 搜索事件语法与向量编码

在 `search` 视图下，轨迹按行记录搜索事件，并在 `<think>` 块结束后输出最终方案：

| 事件语法 | 含义 |
|---|---|
| `EXPAND <node> STATE\|NEED <counts>` | 从待展开队列（frontier）取出节点并检查其状态或需求向量 |
| `NEW <node> VIA <rule> STATE\|NEED <counts>` | 对当前展开节点应用规则，首次发现新状态并分配递增节点编号 `n1, n2, ...` |
| `DUP <node> VIA <rule>` | 对当前展开节点应用规则，得到已有节点 `<node>` 的状态 |
| `CLOSE <node>` | 当前展开节点的全部可执行规则已按规则编号升序枚举完毕 |
| `GOAL <node>` | 当前展开节点满足终止条件 |
| `PLAN <rule_ids...>` | 给出最终正向可执行的规则序列（位于 `</think>` 之后） |
| `END` | 回答结束标记 |

资源向量 `<counts>` 支持两种规范编码：
- **稀疏编码（`sparse`）**：仅按 `RESOURCES` 顺序写出非零项 `资源名:数量`（如 `A:1 C:2`），全零向量写作 `-`。
- **稠密编码（`dense`）**：按 `RESOURCES` 顺序写出全部资源的整数计数（如 `1 0 2 0`）。

### 4. 示例

以下题目初始有 `A:1 B:1 C:1`，目标是得到 `G:1`：

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

在初始状态 `A:1 B:1 C:1` 下，`r0` 和 `r1` 均可执行：若执行 `r1` 会消耗 `A` 和 `C` 得到 `B:1 X:1`，此后无规则可用，目标不可达（因此第 1 步是关键步）；若执行 `r0` 得到 `C:1 D:1`，后续可通过 `r2`（2 步方案 `PLAN r0 r2`）或 `r3 -> r4`（3 步方案 `PLAN r0 r3 r4`）到达目标。

对应的 `forward-bfs`（稀疏编码、`search/on` 协议）assistant 回答为：

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

同一题在 `forward-dfs` 下展开完 `n1` 后会优先展开 `n3` 并直接触发 `GOAL n3`，跳过对死路节点 `n2` 的展开。

---

## 仓库结构与功能概览

```text
symplan/
  cli.py        命令行入口：generate / solve / export / verify 四个子命令
  env.py        世界定义与校验、正向转移与反向回归、方案验证、步骤分类（criticality）、结构指纹与同构判定
  generate.py   问题生成器：问题族、结构骨架、难度预设、策略优势筛选与按槽位确定性去重
  search.py     四种确定性搜索教师、事件发射与模式对分歧计算（divergence）
  qa.py         Prompt 与 Assistant 模板渲染、think/nonthink 视图、target_hash 与 token 长度计量
  verify.py     事件解析、独立重放验证、模式相容判别、hack 分类与按 tag 汇总报告
  recheck.py    可选的 GOAL 后路径核验协议（search_recheck_v1）
scripts/
  prepare_training_data.py  多模式共用问题池构建、分词长度过滤与嵌套子集提取脚本
tests/          单元测试与验收测试
```

---

## 安装与快速开始

环境要求 Python 3.10 或更高版本。核心功能仅依赖 Python 标准库。

```bash
# 基础安装
python -m pip install -e .

# 可选：安装 transformers 以支持 --tokenizer 长度审计与语料构建
python -m pip install -e '.[tokenizer]'

# 可选：安装 pytest 运行测试套件
python -m pip install -e '.[dev]'
```

安装后可直接调用 `symplan`（或使用 `python -m symplan`）。以下命令在 CPU 上完成“生成问题 → 四模式求解 → 导出 QA → 验证”的完整闭环：

```bash
# 1. 生成 5 道简单难度的题目
symplan generate --out data/demo --family foundation --level easy --count 5 --seed 42

# 2. 用四种搜索模式求解，使用稀疏向量编码
symplan solve --worlds data/demo/worlds.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --vector-format sparse --out data/demo/solutions.jsonl

# 3. 导出为 Qwen 格式的 search/on 训练问答
symplan export --worlds data/demo/worlds.jsonl --solutions data/demo/solutions.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --view search --thinking on --require-all-modes --out data/demo/qa.jsonl

# 4. 独立验证导出的 QA 记录并生成汇总报告
symplan verify --worlds data/demo/worlds.jsonl --input data/demo/qa.jsonl \
  --kind search --tag demo --strict
```

运行后，`data/demo/qa.jsonl` 包含模型输入与教师回答（`messages` 字段），`result/demo/summary.md` 包含验证汇总报告。

---

## 命令行功能与用法详解

### 1. `symplan generate`：生成问题

根据问题族、结构骨架和复杂度参数构造题目，植入见证方案并通过独立执行器验证与结构同构去重。

```bash
symplan generate \
  --out data/pool \
  --family general --structure mixed --level medium \
  --count 1000 --seed 42 --workers 8
```

**主要参数说明：**
- `--out DIR`（必填）：输出目录，写出 `worlds.jsonl`（题目定义）、`manifest.json`（生成配置、拒绝原因与统计分位数）和 `fingerprints.json`（结构指纹分布）。
- `--count INT`（必填）：最终通过验证和去重的题目数量。候选不足时保存已有结果并以退出码 `3` 结束。
- `--family {foundation,general,deep-trap,wide-trap}`：问题族。`foundation` 限制步数上限并降低干扰比例；`deep-trap` 与 `wide-trap` 分别构造链式深死路与并列宽干扰分支。
- `--structure {chain,merge,alternative,competition,trap,divergence,mixed}`：题目结构骨架：
  - `chain`：单链依赖结构（可配合 `--forced-prefix` 控制前段唯一可执行步数）。
  - `merge`：包含多输入汇合规则的树状依赖结构。
  - `alternative`：除见证方案外存在使用不同规则集合的独立可行路径。
  - `competition`：见证路径状态上存在争夺同一稀缺资源的竞争规则。
  - `trap`：包含消耗可用资源并通向无关符号的干扰分支。
  - `divergence`：两条有效方案共享前段步骤并在后段分叉。
  - `mixed`（默认）：按 `chain` 15%、`merge` 25%、`alternative` 25%、`competition` 20%、`trap` 15% 的配额分配结构槽位。
- `--level {easy,medium,hard}`：复杂度预设档位：
  - `easy`：资源数 `8:12`，初始资源总量 `3:5`，规则数 `10:18`，见证步数 `2:4`，规则输入上限 `arity=2`，干扰比例 `0.25`。
  - `medium`：资源数 `16:24`，初始资源总量 `4:8`，规则数 `24:48`，见证步数 `4:8`，`arity=2`，干扰比例 `0.40`。
  - `hard`：资源数 `24:40`，初始资源总量 `6:10`，规则数 `48:80`，见证步数 `8:12`，`arity=3`，干扰比例 `0.55`。
- 显式复杂度覆盖：`--resources`、`--initial-units`、`--rules`、`--steps`（支持整数或闭区间如 `16:24`）以及 `--arity`、`--distractors`，显式传入时覆盖 `--level` 的对应预设。
- 策略优势筛选：`--pair MODE1,MODE2`、`--budget B`、`--class {both,first-only,second-only,neither,any}`，用于筛选在节点展开预算 `B` 下特定模式可解或占优的题目。
- `--workers INT` 与 `--resume`：支持多进程并行生成与断点续跑。槽位种子由 `(seed, slot, attempt)` 确定性派生，主进程按槽位顺序执行指纹比对与回溯同构去重，因此同一 `--seed` 的输出与 `--workers` 数量无关。

---

### 2. `symplan solve`：多模式搜索求解与轨迹分析

对题目集合运行一个或多个确定性搜索教师，生成搜索事件轨迹与最终方案。

```bash
symplan solve \
  --worlds data/pool/worlds.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --vector-format sparse \
  --max-expansions 4096 --max-states 100000 \
  --criticality \
  --out data/pool/solutions.jsonl --workers 8
```

**主要参数说明：**
- `--worlds PATH` / `--out PATH`（必填）：输入题目 JSONL 路径与输出求解结果 JSONL 路径。
- `--modes LIST`（必填）：逗号分隔的模式列表，可选 `forward-bfs,forward-dfs,backward-bfs,backward-dfs`。
- `--vector-format {dense,sparse}`：事件行中资源向量的编码格式（默认 `dense`；长上下文或多资源场景推荐使用 `sparse`）。
- `--max-expansions INT`（默认 `4096`）、`--max-states INT`（默认 `100000`）、`--max-trace-lines INT`（默认 `0` 不限）：搜索展开节点预算、访问状态上限与轨迹事件行数上限。每条记录的状态记为 `solved`、`exhausted`、`budget_exceeded`、`state_limit`、`trace_limit` 或 `error`。
- `--criticality`：沿每个已解方案逐步分析当前状态下的全部可执行规则，计算每步的类别（`forced` 强制步、`free` 自由步、`wasteful` 浪费步、`critical` 关键步）及偏离后的修复步数代价，写入记录的 `criticality` 字段。
- `--workers INT` 与 `--resume`：支持多进程求解与基于已有 `(world_id, mode)` 记录的断点续跑。

**伴随输出文件：**
- `<stem>.stats.json`：记录各模式求解状态计数、方案长度分位数、展开节点数与最大队列规模分位数、模式对比较等统计。
- `<stem>.divergence.jsonl`（当 `--modes` 含两个及以上模式时自动生成）：逐题逐模式对记录 `shared_lines`（轨迹公共事件前缀行数）、`shared_expansions`（公共展开节点数）与 `plan_shared_prefix`（方案公共规则前缀长度）。

---

### 3. `symplan export`：导出训练与评测 QA

将题目与求解结果渲染为对话格式（`messages`），在导出前对每条选中轨迹重新执行独立验证，确保导出的教师数据 100% 合法。

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

**主要参数说明：**
- `--view {search,plan}` 与 `--thinking {on,off}`（必填）：协议组合，支持两种配置：
  - `--view search --thinking on`：`<think>` 块内输出完整搜索事件序列，块外输出 `PLAN ...` 与 `END`。
  - `--view plan --thinking off`：输出空思考块 `<think>\n\n</think>\n\n` 后直接给出 `PLAN ...` 与 `END`。
- `--prompt-style {structured,detailed}`：
  - `structured`（默认）：用户消息仅包含 `RESOURCES`、`INITIAL`、`GOAL`、`RULES` 四段结构化题目定义，不含自然语言教程或模式标签，适合多模式共用同一输入。
  - `detailed`：附带英文任务规则说明与对应方向的事件语法说明。
- `--prompt-mode {none,teacher}`：设为 `teacher` 时在题面末尾附加 `SEARCH_MODE <mode>` 提示。
- `--selection {all,uniform,budget-aware}`：
  - `all`（默认）：导出每道题下所有可用模式的 QA 记录，并对同一问题下的记录按可用模式数等权分配 `target_weight`。
  - `uniform`：每道题从可用模式中均匀随机抽取一条（须同时开启 `--require-all-modes`）。
  - `budget-aware`：在 `--budget B` 内可解的模式中随机抽取一条。
- `--require-all-modes`：仅保留指定 `--modes` 全部成功求解的题目。
- `--eligible-worlds PATH`：仅从给定 world ID 列表中导出（可用于对齐不同视图或不同模式组的问题集合）。
- `--limit-worlds INT`：按主结构分层等比例抽样指定数量的题目。
- `--tokenizer PATH` 与 `--max-tokens INT`：使用本地 HuggingFace Tokenizer 计量 `input`、`think`、`plan` 和 `total` token 长度，并过滤超出 `--max-tokens` 的题目。

**伴随输出文件：**
- `<stem>.jsonl`：导出的 QA 记录（含 `messages`、`assistant_prefill`、`target_hash`、`target_weight`、`char_lengths`、`token_lengths` 等字段）。
- `<stem>.world_ids.txt`：最终保留的 `world_id` 列表。
- `<stem>.manifest.json`：导出配置、各结构留存率、拒绝原因计数与长度分位数统计。

---

### 4. `symplan verify`：独立验证模型回答

验证器不依赖参考文本比对，而是根据题目规则独立重放模型输出的每一个搜索事件与最终方案，支持按 `tag` 生成结构化报告或按路径输出逐条判定。

#### 输入格式

预测文件为 JSONL，每行至少包含 `world_id` 和模型生成的文本（支持 `text`、`model_answer`、包含 assistant 的 `messages` 或 `trace` 字段）。同一题目的多次采样回答需提供不同的 `record_id`（省略时按行号自动生成）：

```json
{"record_id": "sample_001", "world_id": "<题目文件中的 world_id>", "text": "<完整的 assistant 文本>"}
```

#### 运行方式

**方式一：使用 `--tag` 输出汇总目录（推荐）**

```bash
symplan verify \
  --worlds data/demo/worlds.jsonl \
  --input predictions.jsonl \
  --kind search \
  --tag eval_run_v1
```

结果保存在 `result/<tag>/` 下（同名 tag 拒绝覆盖）：

| 文件 | 内容 |
|---|---|
| `summary.md` | 联合成功率、问题覆盖率、截断率、回答分类表（成功 / hack / 方案错误）、难度分层表现与前三类首错分布 |
| `summary.json` | 完整指标、错误码分布、输入文件 SHA-256、验证参数设置与运行状态 |
| `qa.jsonl` | 逐条合并题目、模型完整回答、可选参考答案（通过 `--reference` 传入）及 `verification` 判定详情 |

**方式二：使用 `--out` 输出判定文件**

```bash
symplan verify \
  --worlds data/demo/worlds.jsonl \
  --input predictions.jsonl \
  --kind search --modes forward-bfs,forward-dfs \
  --out reports/verify.jsonl --strict
```

写出逐条判定 `reports/verify.jsonl` 与统计文件 `reports/verify.stats.json`。

#### 验证选项与判定口径

- `--kind {search,plan,prefix}`（必填）：
  - `search`：验证完整搜索过程与最终方案。**主指标为联合成功率**（`plan_valid=true` 且 `search_valid=true`）。最终方案可执行且达到目标、但搜索过程存在错误或未完成的回答标记为 `hack=true`（细分 `process_error`、`incomplete_search`、`plan_trace_mismatch`），在主指标中计为失败。
  - `plan`：仅验证 `PLAN ... END` 最终方案的可执行性与目标达成情况。
  - `prefix`：验证已生成的搜索事件前缀（末尾未写完的事件行标记为 `pending_event`）。
- `--protocol {standard,search_recheck_v1}`：默认 `standard`。设为 `search_recheck_v1` 时允许在 `GOAL` 之后、`</think>` 之前插入 `RECHECK_BEGIN STATE ...` / `RECHECK_STEP <rule> STATE ...` / `RECHECK_END` 路径核验块，并对核验块内的状态转移进行独立校验。
- `--prefill {think,nonthink}`：当推理端使用 `<think>\n` 或 `<think>\n\n</think>\n\n` 预填且输出仅保存生成后缀时，恢复完整前缀后再验证。
- `--modes LIST`：用于检验模型搜索轨迹与哪些确定性教师模式完全相容（写入 `compatible_modes` 与 `mode_errors`）。通用搜索合法性（`search_valid`）只要求被展开节点位于当前 frontier 中，不强制绑定单一教师调度。
- `--budget INT`：检查展开节点数 `checked_nodes` 是否在给定预算内（写入 `within_budget`）。
- `--divergence PATH` 与 `--criticality`：传入 `solve` 生成的 `.divergence.jsonl` 或启用 `--criticality` 时，额外计算首错位置相对模式分歧行的偏移量（`first_error_offset_to_divergence`）、首错时 frontier 规模（`first_error_frontier_size`）、首错步是否为关键步（`first_error_at_critical_step`）以及首个使目标不可达的步序号（`first_unrecoverable_step`）。
- `--strict`：若存在未通过验证的记录，命令以退出码 `6` 结束。

---

## 构建多模式实验语料

`scripts/prepare_training_data.py` 使用 `symplan` 核心模块构建按难度（`easy` / `mid` 各半）与五类结构配额分层的大规模四模式共用语料。仅当四种教师模式全部求解成功、通过独立重放验证且完整训练文本不超过 `--max-tokens` 时才接纳该题目，并保证跨 `train` / `validation` / `test` 的底层构造组（`group_id`）完全隔离。

```bash
# 1. 构建完整数据集（支持中断后加 --resume 续跑）
python scripts/prepare_training_data.py --phase build \
  --out data/corpus \
  --tokenizer /path/to/Qwen3-8B-Base \
  --train 20000 --validation 1000 --test 1000 \
  --seed 20261113 --max-tokens 65536 --max-trace-lines 16000 \
  --max-candidates 250000 --workers 12

# 2. 按嵌套子集 ID 索引提取指定题量的数据集（如 5,000 题的正向双模式混合集，共 10,000 条 QA）
python scripts/prepare_training_data.py --phase subset \
  --input data/corpus/datasets/train/search_forward_mixed2.jsonl \
  --world-ids data/corpus/datasets/subsets/train_5000.world_ids.txt \
  --out data/corpus/train_5000_forward_mixed2.jsonl
```

构建完成后，`data/corpus/datasets/{train,validation,test}/` 下每个切分包含 6 个数据集文件：
- 单模式：`search_forward-bfs.jsonl`、`search_forward-dfs.jsonl`、`search_backward-bfs.jsonl`、`search_backward-dfs.jsonl`（每题 1 条）
- 正向双模式混合：`search_forward_mixed2.jsonl`（每题含正向 BFS 与正向 DFS 各 1 条）
- 四模式混合：`search_mixed4.jsonl`（每题含四个模式各 1 条）

`data/corpus/datasets/subsets/` 下提供 `1000`、`2500`、`5000`、`10000`、`15000`、`20000` 题的嵌套子集 ID 名单（小规模子集严格包含于大规模子集中，且保持难度与结构比例不变）。此外还支持 `--phase pilot`（如 `--pilot 500`）用于快速小规模试跑与分布审计。

---

## 测试

```bash
python -m pytest -q
```

测试套件覆盖：资源状态转移与反向需求回归语义、四种搜索教师轨迹一致性、长死路与完全二叉树的展开预算计数、事件篡改与语法错误的检出、模式相容与首次分歧定位、`think`/`nonthink` 视图导出与目标权重合并、多进程生成与 `--resume` 的确定性一致性，以及按 `--tag` 汇总报告的生成。
