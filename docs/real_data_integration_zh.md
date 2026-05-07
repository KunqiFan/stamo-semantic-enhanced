# 真实数据接入说明

当前仓库已经把下面这条最小实验链跑通了：

`StaMo delta_z -> compact semantics -> process chain`

如果你接下来要从 toy 数据迁移到真实数据，建议按下面顺序做。

## 1. 准备原始图像对和标签

在：

- [data/raw/README_real_data_template.md](/C:/Users/ryanf/Desktop/stamo_pro/data/raw/README_real_data_template.md)

我已经写了推荐格式。

你至少需要准备三份 jsonl：

- `train_pairs.jsonl`
- `val_pairs.jsonl`
- `test_pairs.jsonl`

每一行需要包含：

- `image_t`
- `image_tp`
- `labels`

推荐额外包含：

- `caption`
- `process_text`
- `action`
- `ee_delta`
- `trajectory_id`
- `start_step`
- `end_step`

## 2. 用通用 manifest 生成器转成实验格式

配置模板在：

- [configs/real_data_manifest_template.json](/C:/Users/ryanf/Desktop/stamo_pro/configs/real_data_manifest_template.json)

脚本在：

- [scripts/build_real_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_real_pair_manifest.py)

运行方式：

```bash
py scripts/build_real_pair_manifest.py --config configs/real_data_manifest_template.json
```

输出会写到：

- `data/processed/real_process_chain/train.jsonl`
- `data/processed/real_process_chain/val.jsonl`
- `data/processed/real_process_chain/test.jsonl`

## 3. 提取真实 StaMo delta_z

沿用现有脚本：

- [scripts/extract_delta_z.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/extract_delta_z.py)

示例：

```bash
py scripts/extract_delta_z.py --manifest data/processed/real_process_chain/train.jsonl --out_dir data/interim/real_process_chain/latents
```

如果后面你换成真实 StaMo 大权重，只需要把：

- `--stamo_config`
- `--checkpoint_dir`

改成对应路径即可。

## 4. 训练 compact semantics

脚本：

- [scripts/train_compact_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_compact_semantics.py)

当前它默认用 `delta_z` 做特征、对四个语义属性分别训练轻量分类器。

这一步在真实数据上往往是你最值得重点升级的部分。

## 5. 跑 process chain 融合

脚本：

- [scripts/train_process_chain.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_process_chain.py)

你可以比较：

- `text_only`
- `semantics_only`
- `text_plus_semantics`

## 6. 当前最建议你下一步提供给我的东西

如果你想让我直接继续把真实数据版本接好，最有帮助的是你给我下面任意一种：

1. 一个真实数据目录路径和几张样例图
2. 一份你已有的 `pair jsonl`
3. 一份标签字段说明

只要有其中一种，我就可以继续把当前脚本从“通用模板”推进到“你的真实数据版本”。

