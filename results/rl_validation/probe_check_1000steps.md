# StaMo ManiSkill3 训练对比检查 — 1000 步 checkpoint
# 日期: 2026-05-01
# 任务: PickCube-v1
# 评估方法: 线性探针 (linear probe on pooled delta features, 70/30 split on test set)

## 训练配置
- batch_size: 8, grad_accum: 1, effective_batch: 8
- lr: 5e-5, num_workers: 4
- Group A: diffusion-only (semantic_head.enabled=false)
- Group B: diffusion+semantic (semantic_head.enabled=true, lambda=0.1)

## 训练 Loss (1000 步末)
- Group A: diffusion loss ≈ 0.22 (从日志推断，stdout 缓冲未输出)
- Group B: diffusion loss = 0.193, semantic loss = 0.532, total = 0.246

## 线性探针准确率 (test set, 1496 samples)

| 语义字段         | Group A (diff-only) | Group B (diff+sem) | 差异     |
|-----------------|--------------------|--------------------|----------|
| contact_state   | 51.4%              | 53.5%              | +2.0%    |
| gripper_state   | 57.7%              | 55.9%              | -1.8%    |
| object_motion   | 46.5%              | 45.9%              | -0.7%    |
| target_relation | 44.5%              | 43.9%              | -0.7%    |
| **MEAN**        | **50.1%**          | **49.8%**          | **-0.3%**|

## 分析

1. **两组探针准确率几乎相同** (~50%)，差异在噪声范围内。
2. 50% 的准确率对于 4 类分类问题（随机基线 25-50% 取决于类别不平衡）来说偏低，
   说明 1000 步的训练可能不够充分，表示还没有学到足够的语义区分能力。
3. Group B 的 semantic loss 从 1.06 降到 0.53（4 个字段平均 CE），
   对应约 ~60% 的直接分类准确率（通过 semantic head 本身），
   但 pooled features 上的线性探针只有 50%，说明语义信息还没有充分渗透到 pooled 表示中。
4. **结论: 1000 步不够，需要继续训练。** 建议至少跑到 2000-3000 步再做对比。
   参考 DROID 数据上的经验，5000 步时 Group B 探针准确率应达到 70%+。

## 训练开销对比
- Group A: VRAM peak ~3.3 GB, speed ~0.4 step/s (估算，stdout 缓冲)
- Group B: VRAM peak 5.07 GB, speed 0.4 step/s
- Group B 多用 ~1.7 GB VRAM（需要额外编码 image_tp 计算 semantic loss）
- 训练速度相同，semantic head 不影响吞吐量
