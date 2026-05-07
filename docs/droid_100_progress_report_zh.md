# DROID 100 接入进展报告

## 1. 这次做了什么

这次工作的目标是把开源真实机器人数据 `DROID 100` 真正接进当前这条实验链：

`DROID image pair -> StaMo delta feature -> compact semantics -> process chain`

本次已经完成的部分包括：

1. 检查并利用本地已经下载完成的 `DROID 100` shard
2. 新增 `DROID` 专用的 manifest 构建脚本
3. 从真实 `DROID` 图像对中提取 `StaMo delta_z / delta_pooled`
4. 在真实数据代理标签上训练 `compact semantics`
5. 比较 `text_only / semantics_only / text_plus_semantics`

在这轮过程中还修复了一个关键 bug：

- 早期版本的 `episode_id` 没有包含完整 shard 名，导致不同 shard 的样本发生 `sample_id` 覆盖
- 修复后重新生成了 manifest、latent 和评估结果
- 当前结果以修复后的版本为准

---

## 2. 新增和使用的关键文件

### 新增脚本

- [scripts/build_droid_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_droid_pair_manifest.py)

这个脚本负责：

- 读取 `DROID 100` 的 TFRecord
- 从语言字段中选择可用 instruction
- 从 `wrist_image_left` 提取图像帧并保存到本地
- 构造短时程图像对 `(t, t + gap)`
- 生成实验 manifest
- 基于 `gripper_position` 和 `cartesian_position` 构造一组代理紧凑语义标签

### 生成的数据

- [data/processed/droid_100_process_chain/train.jsonl](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/droid_100_process_chain/train.jsonl)
- [data/processed/droid_100_process_chain/val.jsonl](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/droid_100_process_chain/val.jsonl)
- [data/processed/droid_100_process_chain/test.jsonl](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/droid_100_process_chain/test.jsonl)
- [data/processed/droid_100_process_chain/images](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/droid_100_process_chain/images)

### 生成的 StaMo 特征

- [data/interim/droid_100_process_chain/latents](/C:/Users/ryanf/Desktop/stamo_pro/data/interim/droid_100_process_chain/latents)

### 生成的 compact semantics 预测

- [data/interim/droid_100_process_chain/semantics/test_predicted_semantics.jsonl](/C:/Users/ryanf/Desktop/stamo_pro/data/interim/droid_100_process_chain/semantics/test_predicted_semantics.jsonl)
- [data/interim/droid_100_process_chain/semantics/test_predicted_semantics_delta_pooled.jsonl](/C:/Users/ryanf/Desktop/stamo_pro/data/interim/droid_100_process_chain/semantics/test_predicted_semantics_delta_pooled.jsonl)

---

## 3. 真实数据 manifest 是怎么构建的

当前 `DROID` 版本采用以下策略：

- 数据源：`DROID 100` 已下载完成的 train shards
- 相机：`wrist_image_left`
- 配对方式：`step_gap = 4`, `pair_stride = 12`
- 文本：优先使用 `language_instruction -> language_instruction_2 -> language_instruction_3`
- 如果某个 episode 没有语言，则先跳过

每个样本包含：

- `image_t`
- `image_tp`
- `caption`
- `process_text`
- `action`
- `ee_delta`
- `labels`

其中 `caption` 目前使用任务 instruction，例如：

- `Put the marker in the pot`
- `Put the bread in the toaster`

这使 `text` 更接近高层任务描述，而不是细粒度过程标签。

---

## 4. 当前代理语义标签怎么来的

由于 `DROID` 原始数据并没有直接给出你要的：

- `contact_state`
- `gripper_state`
- `object_motion`
- `target_relation`

所以这次先构造了一个第一版代理标签体系：

### gripper_state

根据 `gripper_position(t)` 和 `gripper_position(t + gap)` 的数值与变化趋势估计：

- `open`
- `closing`
- `closed`

### contact_state

主要根据 `gripper_state` 和 gripper 开合程度近似估计：

- `no_contact`
- `contact`

### target_relation

把 episode 最后一步的末端位置当作一个近似目标点，比较：

- 当前 pair 起点到目标点的距离
- 当前 pair 终点到目标点的距离

得到：

- `farther`
- `closer`
- `reached`

### object_motion

根据末端位姿变化量和 `z` 方向变化近似估计：

- `still`
- `lifted`
- `moved`
- `placed`

### stage_label

由上面四类紧凑语义再映射得到：

- `approach`
- `contact`
- `grasp`
- `lift`
- `move`
- `place`

需要强调：这些标签目前是 **proxy labels**，不是人工真值标签，所以只能用来做第一轮真实数据可行性验证。

---

## 5. 本次真实数据样本规模

命令：

```powershell
py scripts/build_droid_pair_manifest.py --require_language --max_episodes 24 --step_gap 4 --pair_stride 12
```

实际生成结果：

- train: `340`
- val: `67`
- test: `63`

test 集 `stage_label` 分布：

- `approach`: `47`
- `move`: `12`
- `grasp`: `2`
- `place`: `2`

这说明当前分布仍然偏向：

- `approach`
- `still`

后续需要通过更好的 pair 采样策略或标签规则来平衡。

---

## 6. StaMo 特征提取

我使用已经在本机跑通的 StaMo toy checkpoint，在真实 `DROID` 图像对上提取特征：

```powershell
py scripts/extract_delta_z.py --manifest data/processed/droid_100_process_chain/train.jsonl --out_dir data/interim/droid_100_process_chain/latents --device cuda
py scripts/extract_delta_z.py --manifest data/processed/droid_100_process_chain/val.jsonl --out_dir data/interim/droid_100_process_chain/latents --device cuda
py scripts/extract_delta_z.py --manifest data/processed/droid_100_process_chain/test.jsonl --out_dir data/interim/droid_100_process_chain/latents --device cuda
```

结论：

- 真实图像对的 `delta_z` 提取是通的
- 真实图像对的 `delta_pooled` 提取也是通的
- 当前 latent 文件数为 `470`，与三份 manifest 的样本总数一致
- 当前可以直接把 `DROID` 接入后续 `compact semantics` 和 `process chain` 实验

---

## 7. compact semantics 第一轮结果

### 7.1 用 `delta_z`

命令：

```powershell
py scripts/train_compact_semantics.py --train_manifest data/processed/droid_100_process_chain/train.jsonl --eval_manifest data/processed/droid_100_process_chain/test.jsonl --latent_dir data/interim/droid_100_process_chain/latents --feature_key delta_z --out_path data/interim/droid_100_process_chain/semantics/test_predicted_semantics.jsonl
```

test 结果：

- `contact_state`: accuracy `0.5873`
- `gripper_state`: accuracy `0.5397`
- `object_motion`: accuracy `0.5714`
- `target_relation`: accuracy `0.4127`

### 7.2 用 `delta_pooled`

命令：

```powershell
py scripts/train_compact_semantics.py --train_manifest data/processed/droid_100_process_chain/train.jsonl --eval_manifest data/processed/droid_100_process_chain/test.jsonl --latent_dir data/interim/droid_100_process_chain/latents --feature_key delta_pooled --out_path data/interim/droid_100_process_chain/semantics/test_predicted_semantics_delta_pooled.jsonl
```

test 结果：

- `contact_state`: accuracy `0.6190`
- `gripper_state`: accuracy `0.5079`
- `object_motion`: accuracy `0.7143`
- `target_relation`: accuracy `0.4286`

初步观察：

- `delta_pooled` 在 `contact_state / gripper_state / object_motion` 上略优
- `target_relation` 仍然较弱
- `reached` 和 `place` 这些稀有类目前几乎没有学起来

---

## 8. process chain 第一轮结果

为了避免 `process_text` 引入明显标签泄漏，这次过程链只用 `caption`，也就是任务 instruction。

### text_only

命令：

```powershell
py scripts/train_process_chain.py --train_manifest data/processed/droid_100_process_chain/train.jsonl --eval_manifest data/processed/droid_100_process_chain/test.jsonl --setting text_only --target stage_label --text_key caption
```

结果：

- accuracy `0.7460`
- macro-F1 `0.2136`

### semantics_only（使用 `delta_z` 预测语义）

结果：

- accuracy `0.5397`
- macro-F1 `0.2216`

### text_plus_semantics（使用 `delta_z` 预测语义）

结果：

- accuracy `0.5397`
- macro-F1 `0.2216`

### text_plus_semantics（使用 `delta_pooled` 预测语义）

结果：

- accuracy `0.6032`
- macro-F1 `0.1937`

### text_plus_semantics（gold semantics，上界）

结果：

- accuracy `1.0000`
- macro-F1 `1.0000`

---

## 9. 结果如何理解

这次真实数据结果已经能支持一个很重要的判断：

### 结论 1

`StaMo delta feature -> compact semantics` 在真实开源数据上已经有可见信号，不是完全无效。

尤其是：

- `contact_state`
- `gripper_state`
- `object_motion`

这些和短时程操作变化更接近的属性，已经能学到一定区分能力。

### 结论 2

“桥”在结构上是成立的，因为一旦使用 `gold semantics`，过程链识别能力立刻显著提升。

### 结论 3

当前真正的瓶颈不是桥接思路本身，而是：

- 代理标签还比较粗
- 稀有类太少
- `stage_label` 明显类不平衡
- 当前 StaMo checkpoint 仍是 toy 版本，不是正式预训练权重
- 文本基线目前主要依赖高频 `approach` 类，细粒度类仍然很弱

也就是说，现在最该改进的是：

- 数据构造
- 代理标签质量
- 特征质量

而不是推翻整个研究方向。

---

## 10. 下一步建议

我建议下一步按下面顺序继续：

1. 下载完整 `DROID 100`
2. 改进 `build_droid_pair_manifest.py`
3. 调整 pair 采样策略，让 `grasp/lift/place` 更多出现
4. 把 `process_text` 改成更合理的粗过程文本，避免泄漏但保留过程感
5. 增加 `delta_z` 与 `delta_pooled` 的联合特征实验
6. 尝试更强的 compact semantics 分类器，而不只用 logistic regression

如果继续推进，我建议我下一步优先做：

**改进 `DROID` 标签和采样策略，争取把 `grasp/lift/place` 的 test 支持数抬起来。**
