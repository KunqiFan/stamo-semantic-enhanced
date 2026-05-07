# StaMo ManiSkill3 训练对比检查 — 5000 步 checkpoint
# 日期: 2026-05-02
# 任务: PickCube-v1
# 评估方法: 线性探针 (logistic regression on delta_pooled, 70/30 split)

## 训练配置
- batch_size: 8, grad_accum: 1, effective_batch: 8
- lr: 5e-5 (cosine decay), num_workers: 4
- Group A: diffusion-only (semantic_head.enabled=false)
- Group B: diffusion+semantic (semantic_head.enabled=true, lambda=0.1)

## 训练 Loss (5000 步末)
- Group A: diffusion loss ≈ 0.20
- Group B: diffusion loss ≈ 0.20, semantic loss ≈ 0.29, total ≈ 0.23

## 线性探针准确率 (test set)

| 语义字段         | Group A (diff-only) | Group B (diff+sem) | 差异     |
|-----------------|--------------------|--------------------|----------|
| contact_state   | 50.8%               | 50.3%               | -0.4%     |
| gripper_state   | 47.2%               | 56.6%               | +9.4%     |
| object_motion   | 53.2%               | 58.4%               | +5.1%     |
| target_relation | 49.7%               | 56.1%               | +6.5%     |
| **MEAN**        | **50.2%**           | **55.3%**           | **+5.1%**|
