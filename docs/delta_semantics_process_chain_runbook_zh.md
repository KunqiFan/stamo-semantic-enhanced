# delta_z -> compact semantics -> process chain 运行说明

这份说明对应当前本地已经接好的最小实验接口。

目标链路：

1. 构造图像对 manifest
2. 用 StaMo 提取 `delta_z`
3. 训练 `delta_z -> compact semantics`
4. 把 semantics 喂给 text/process chain 基线

## 1. 构造 toy process-chain 数据

```bash
py scripts/build_toy_pair_manifest.py
```

输出目录：

- [data/processed/toy_process_chain](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/toy_process_chain)

生成文件：

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

每条样本包含：

- `image_t`
- `image_tp`
- `caption`
- `process_text`
- `labels`

## 2. 提取 StaMo delta_z

训练集：

```bash
py scripts/extract_delta_z.py --manifest data/processed/toy_process_chain/train.jsonl --out_dir data/interim/toy_process_chain/latents
```

验证集：

```bash
py scripts/extract_delta_z.py --manifest data/processed/toy_process_chain/val.jsonl --out_dir data/interim/toy_process_chain/latents
```

测试集：

```bash
py scripts/extract_delta_z.py --manifest data/processed/toy_process_chain/test.jsonl --out_dir data/interim/toy_process_chain/latents
```

默认使用：

- StaMo 配置: [StaMo/configs/toy.yaml](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/configs/toy.yaml)
- StaMo checkpoint: [StaMo/ckpts/toy_debug/4](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/ckpts/toy_debug/4)

每个样本会输出一个 `.npz`，里面有：

- `z_t`
- `z_tp`
- `delta_z`
- `pooled_t`
- `pooled_tp`
- `delta_pooled`

## 3. 训练 compact semantics 预测器

```bash
py scripts/train_compact_semantics.py --train_manifest data/processed/toy_process_chain/train.jsonl --eval_manifest data/processed/toy_process_chain/val.jsonl --latent_dir data/interim/toy_process_chain/latents --feature_key delta_z --out_path data/interim/toy_process_chain/semantics/val_predicted_semantics.jsonl
```

输出：

- 控制台打印四个属性的分类报告
- 预测文件写到：
  [data/interim/toy_process_chain/semantics/val_predicted_semantics.jsonl](/C:/Users/ryanf/Desktop/stamo_pro/data/interim/toy_process_chain/semantics/val_predicted_semantics.jsonl)

当前默认预测的四个属性是：

- `contact_state`
- `gripper_state`
- `object_motion`
- `target_relation`

## 4. 跑 process chain 基线

### text-only

```bash
py scripts/train_process_chain.py --train_manifest data/processed/toy_process_chain/train.jsonl --eval_manifest data/processed/toy_process_chain/val.jsonl --setting text_only --target stage_label --text_key caption
```

### text + predicted semantics

```bash
py scripts/train_process_chain.py --train_manifest data/processed/toy_process_chain/train.jsonl --eval_manifest data/processed/toy_process_chain/val.jsonl --setting text_plus_semantics --target stage_label --text_key caption --predicted_semantics data/interim/toy_process_chain/semantics/val_predicted_semantics.jsonl
```

可选 setting：

- `text_only`
- `semantics_only`
- `text_plus_semantics`

## 5. 当前这套接口的定位

这不是论文正式复现版，而是一个已经真实接通的研究原型。

它的意义是：

1. 你已经可以从 StaMo checkpoint 提取真实 `delta_z`
2. 你已经可以训练 `delta_z -> compact semantics`
3. 你已经可以把 compact semantics 接到文本过程链基线里

## 6. 你下一步最值得改的地方

如果要从 toy 版本迁移到你的正式实验，优先改下面几项：

1. 用真实数据替换 [scripts/build_toy_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_toy_pair_manifest.py)
2. 用真实标签替换 toy 规则标签
3. 在 [scripts/train_compact_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_compact_semantics.py) 里换成更强的语义头
4. 在 [scripts/train_process_chain.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_process_chain.py) 里换成你要的过程链模型

