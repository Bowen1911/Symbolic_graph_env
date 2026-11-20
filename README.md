# 符号资源推理环境（symplan）

同一道题可以有多种正确的推理过程。这个环境用可控的符号规划任务研究，多种搜索模式如何影响模型在有限容量、数据和训练预算下的学习。

每道题给出初始资源、转换规则和目标资源。规则在各项输入充足时可执行，扣除输入并增加输出。只要最终各项资源达到目标数量，方案就有效。例如，初始有两个 `B`，目标是得到一个 `D`，按顺序应用 `r0: 2B → H` 和 `r1: H → D`，即可得到方案 `PLAN r0 r1`。

正向和反向搜索各支持 BFS 与 DFS，因此可以为同一题生成四种教师轨迹。验证器按规则重放搜索事件与最终方案。搜索任务以联合成功率为主指标，同时要求最终方案正确、搜索过程合法。最终方案正确而过程无效的回答记为 `hack`，按失败计入主指标。

## 快速开始

使用 Python 3.10 或更高版本。从仓库根目录运行，可在 CPU 上完成生成、求解、QA 导出和验证。

```bash
python -m pip install -e .
symplan generate --out data/demo --family foundation --level easy --count 5 --seed 42
symplan solve --worlds data/demo/worlds.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --vector-format sparse --out data/demo/solutions.jsonl
symplan export --worlds data/demo/worlds.jsonl --solutions data/demo/solutions.jsonl \
  --modes forward-bfs,forward-dfs,backward-bfs,backward-dfs \
  --view search --thinking on --require-all-modes --out data/demo/qa.jsonl
symplan verify --worlds data/demo/worlds.jsonl --input data/demo/qa.jsonl \
  --kind search --tag demo --strict
```

`data/demo/qa.jsonl` 的 `messages` 字段包含模型输入和教师回答，验证汇总见 `result/demo/summary.md`。模型回答格式、逐条判定和语料构建命令见[使用参考](doc/usage.md)。

## 测试

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```
