# RL 验证实验进度报告

**日期**: 2026-05-02  
**目标**: 验证语义增强的 StaMo 表示是否有助于下游策略学习（更高 success rate / 更快收敛）

---

## 实验设计

| 组别 | StaMo 训练方式 | 含义 |
|------|--------------|------|
| Group A | 纯扩散 loss | 语义信息被压缩丢弃的表示 |
| Group B | 扩散 + 语义 loss (λ=0.1) | 语义信息被主动编码的表示 |

核心逻辑：两组 encoder 冻结后接入相同的 PPO 策略网络，对比 success rate 和 sample efficiency。如果 Group B 显著优于 Group A，则证明语义增强的表示对策略学习有实际价值。

---

## Phase 1: 数据采集 ✅

从 ManiSkill3 预录演示中回放轨迹，渲染 224×224 RGB 图像对，自动构造语义标注。

| 任务 | 图像对数 | 训练集 | 测试集 |
|------|---------|--------|--------|
| PickCube-v1 | 7,476 | 5,980 | 1,496 |
| StackCube-v1 | 10,412 | 8,329 | 2,083 |

语义标注字段：contact_state, gripper_state, object_motion, target_relation  
图像对参数：step_gap=4, pair_stride=2

---

## Phase 2: StaMo 表示训练 ✅

两组使用相同超参，仅 semantic_head 开关不同。

### 训练配置
- Vision backbone: DINOv2 ViT-B/14 (冻结)
- Projector: 2 token, 4 层 cross-attention, output 1024D
- DiT: toy mode, 6 层, 8 heads
- batch_size=8, lr=5e-5 (warmup + cosine decay), 5000 步
- GPU: RTX 5060 8GB

### 训练结果

| 指标 | Group A | Group B |
|------|---------|---------|
| Diffusion loss (5000步) | 0.20 | 0.20 |
| Semantic loss (5000步) | — | 0.29 |
| VRAM 峰值 | 4.05 GB | 5.07 GB |
| 训练速度 | 0.4 step/s | 0.4 step/s |
| 总训练时间 | ~3.5h | ~3.5h |

Group B 的 semantic loss 从 1.06 (初始) → 0.53 (1000步) → 0.29 (5000步)，持续下降。

### 线性探针验证（5000 步 checkpoint）

在测试集上用 logistic regression 对 delta_pooled 特征做分类，验证表示中的语义信息含量。

| 语义字段 | Group A | Group B | 差异 |
|----------|---------|---------|------|
| contact_state | 50.8% | 50.3% | -0.4% |
| gripper_state | 47.2% | 56.6% | +9.4% |
| object_motion | 53.2% | 58.4% | +5.1% |
| target_relation | 49.7% | 56.1% | +6.5% |
| **MEAN** | **50.2%** | **55.3%** | **+5.1%** |

**结论**: Group A 的 pooled 表示接近随机水平（~50%），确认纯扩散目标不编码语义。Group B 在 gripper_state、object_motion、target_relation 上均有显著提升，确认语义辅助训练成功将语义信息注入了 pooled 表示。

这一步是前提验证——确认两组表示确实存在可测量的语义差异，为 Phase 3 的 RL 对比提供基础。

---

## Phase 3: RL 策略学习对比 🔄 进行中

### Phase 3a: State-only 基线验证

在接入 StaMo encoder 之前，先验证 PPO 能否在纯状态观测下学会任务。这一步排除 RL 算法本身的问题，确保后续 A vs B 对比有意义。

#### 控制模式探索

| 控制模式 | 动作空间 | 需要 pinocchio | 结果 |
|----------|---------|---------------|------|
| `pd_joint_pos` | 8D 绝对关节位置 | 否 | ❌ 500K 步 success=0%，return<1 |
| `pd_ee_delta_pos` | 4D 末端增量 | 是 | ❌ Windows 无法安装 pinocchio |
| `pd_joint_delta_pos` | 8D 关节增量 | 否 | ✅ 采用此模式 |

`pd_joint_delta_pos` 动作范围 [-1, 1]，策略只需输出小增量而非学习绝对目标位置，对 RL 友好得多。

#### 环境并行化

ManiSkill3 支持 GPU 并行仿真（多环境同时跑在 GPU 上）。过程中解决了两个关键问题：

1. **sapien `cuda.dll` bug**：sapien 在 Windows 上错误地尝试加载 `cuda.dll`（Linux 命名），实际应为 `nvcuda.dll`。手动修复了 `sapien/physx/__init__.py`。
2. **GPU 模式不自动重置**：ManiSkill3 GPU 模式下 episode 结束后环境不会自动 reset，`truncated` 信号持续为 True 但 `elapsed_steps` 继续递增。需要手动调用 `env.reset(options={'env_idx': done_idx})` 重置已完成的环境。

修复后实现了 64 环境 GPU 并行，采样速度 ~1100 steps/s。

#### PickCube-v1 基线结果

| 配置 | 步数 | Success Rate | Episode Return | 备注 |
|------|------|-------------|----------------|------|
| 单环境, `pd_joint_pos`, 500K | 500K | 0% | 0.05-0.9 | 策略几乎没学到东西 |
| 单环境, `pd_joint_delta_pos`, 100K | 100K | 0% | 7-11 | return 有上升但不够 |
| 单环境, `pd_joint_delta_pos`, 250K, max_ep=200 | 250K | 0% | 20-24 | return 平台期 |
| **64 并行, `pd_joint_delta_pos`, 1M** | **1M** | **0.1%** | **~22** | **8/10880 episodes 成功** |

**结论**：PickCube-v1 对 vanilla PPO 来说太难。成功需要完成 approach → grasp → lift → place → static 的多阶段序列，即使 1M 步（64 并行）也只有 0.1% success rate，且已进入平台期。不适合作为 A vs B 对比的任务。

#### PushCube-v1 初步测试

PushCube-v1 只需将方块推到目标位置，不需要抓取和放置，难度显著低于 PickCube。

| 配置 | 步数 | Success Rate | 备注 |
|------|------|-------------|------|
| 64 并行, `pd_joint_delta_pos` | 64K | 0.2% (1/640) | 仅 64K 步即出现 success |

PushCube 在 64K 步就出现了 success，而 PickCube 需要 256K 步。PushCube 更适合做 A vs B 对比实验。

### Phase 3b: StaMo Encoder + PPO 对比 ⬅️ 下一步

#### 修订后的实验方案

基于 Phase 3a 的发现，调整实验设计：

```
RGB 观测 → StaMo encoder (冻结) → 2560D 特征 + 35D 本体感知 → MLP(256,256) → 动作
```

| 配置项 | 值 |
|--------|---|
| **任务** | **PushCube-v1**（替代 PickCube-v1） |
| 控制模式 | `pd_joint_delta_pos` |
| 算法 | PPO (clipped objective, GAE) |
| 策略网络 | Actor-Critic MLP(256, 256) |
| 并行环境 | 单环境（受限于 StaMo encoder 推理） |
| 总步数 | 500K-1M timesteps |
| 种子 | 42, 123, 456 |
| 评估频率 | 每 10K 步，50 episodes |
| PPO 超参 | lr=3e-4, clip=0.2, γ=0.99, λ_GAE=0.95, ent_coef=0.0 |

注意：接入 StaMo encoder 后无法使用 GPU 并行环境（encoder 需要渲染 RGB 图像，GPU 并行模式的渲染 API 不同），因此回退到单环境模式，训练时间会更长。

#### 评估指标

1. **Success rate**（主指标）：最终成功率对比
2. **Episode return**（辅助指标）：即使 success rate 差异不显著，return 差异也能反映表示质量
3. **Sample efficiency**：达到特定 success rate 所需步数
4. **训练开销**：wall-clock time, encoder forward time per step
5. **统计显著性**：3 seed 均值 ± std

#### 预计耗时

- PushCube-v1 单环境 + StaMo encoder: 6 runs × ~4-8h = 24-48h

---

## Phase 4: 分析与结论 ⬜

- 学习曲线图（success rate vs timesteps，两组 ± std）
- Episode return 曲线对比
- Sample efficiency 表
- 统计检验
- 最终结论：语义增强表示对策略学习的价值

---

## 关键技术发现

1. **控制模式选择至关重要**：`pd_joint_pos`（绝对位置）对 RL 极不友好，`pd_joint_delta_pos`（增量）效果好得多
2. **ManiSkill3 GPU 并行需要手动重置**：与文档描述不同，GPU 模式下 episode 结束后不会自动 reset
3. **sapien Windows 兼容性 bug**：`cuda.dll` 应为 `nvcuda.dll`
4. **PickCube 对 vanilla PPO 过难**：多阶段操作任务需要更强的算法（SAC、demo-augmented RL）或更多训练步数
5. **PushCube 是更合适的验证任务**：单阶段推动任务，PPO 可学，且语义信息（物体运动方向、与目标关系）仍然相关

---

## 文件结构

```
rl_validation/
├── configs/
│   └── stamo_maniskill.yaml
├── data/
│   ├── pickcube_v1/  (train.jsonl, test.jsonl, images/)
│   └── stackcube_v1/ (train.jsonl, test.jsonl, images/)
├── scripts/
│   ├── collect_maniskill_data.py   # Phase 1
│   ├── train_stamo_maniskill.py    # Phase 2
│   ├── probe_5000steps.py          # Phase 2 验证
│   ├── ppo_stamo.py                # Phase 3
│   ├── run_experiment.py           # Phase 3 编排
│   └── analyze_results.py          # Phase 4
├── wrappers/
│   └── stamo_encoder_wrapper.py    # Gym wrapper
└── results/
    ├── probe_check_1000steps.md
    ├── probe_check_5000steps.md
    └── progress_report.md
```
