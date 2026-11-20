# 使用参考

[项目首页](../README.md)提供安装与最小示例。以下命令从仓库根目录运行。

## 验证模型回答

将模型回答保存为 JSONL。每行包含题目文件中的 `world_id` 和完整的 assistant 文本 `text`。同一道题的多次回答应使用不同的 `record_id`，省略时按输入顺序生成。

```json
{"record_id":"sample_001","world_id":"<题目文件中的 world_id>","text":"<完整的 assistant 文本>"}
```

```bash
symplan verify --worlds data/demo/worlds.jsonl --input predictions.jsonl \
  --kind search --tag model_demo
```

- `--kind search` 验证完整的搜索事件与最终方案。`--kind plan` 验证最终方案，`--kind prefix` 检查已经生成的搜索前缀。
- 如果推理使用了 `<think>` 预填，且输入只保存生成的后缀，可加 `--prefill think`。使用 `--budget` 可指定按 `EXPAND` 事件计数的节点预算。
- `--tag` 将 `qa.jsonl`、`summary.json` 和 `summary.md` 写入当前目录下的 `result/<tag>/`。每次运行使用新的 tag。若只需逐条判定，可用 `--out` 指定 JSONL 路径。

搜索任务的联合成功率要求最终方案正确且搜索过程合法。题目覆盖率按收到回答的题目数除以题目文件的题目总数计算。模式相容性及最终方案与搜索恢复路径的一致性可在逐条判定中查看。

## 构建实验语料

指定本地 Qwen3 tokenizer 目录。脚本采用 `structured-v2` 输入和 `search/on` 协议，保留四种模式都能求解且满足长度限制的问题，并按难度和结构分层划分训练、验证和测试集。下面使用原实验的语料构建设定。

```bash
python -m pip install -e '.[tokenizer]'
python scripts/prepare_training_data.py --phase build --out data/corpus \
  --tokenizer /path/to/Qwen3-8B-Base \
  --train 20000 --validation 1000 --test 1000 --seed 20261113 \
  --max-tokens 65536 --max-trace-lines 16000 --max-candidates 250000 --workers 12
```

`data/corpus/datasets/train/search_forward_mixed2.jsonl` 每题各含一条正向 BFS 和一条正向 DFS 轨迹。`datasets/subsets/` 下的问题 ID 可用于提取嵌套子集，例如 5,000 道题对应的 10,000 条 QA。

```bash
python scripts/prepare_training_data.py --phase subset \
  --input data/corpus/datasets/train/search_forward_mixed2.jsonl \
  --world-ids data/corpus/datasets/subsets/train_5000.world_ids.txt \
  --out data/corpus/train_5000_forward_mixed2.jsonl
```
