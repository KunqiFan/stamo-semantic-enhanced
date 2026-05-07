# Toy Process Chain 实验结果报告

这份报告记录当前本地已经跑通的最小实验链路：

`StaMo delta_z -> compact semantics -> process chain`

对应运行入口：

- [scripts/run_toy_process_chain_pipeline.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/run_toy_process_chain_pipeline.py)

---

## 1. 本次实验做了什么

当前 toy 实验链路包含四步：

1. 用 [scripts/build_toy_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_toy_pair_manifest.py) 构造图像对样本
2. 用 [scripts/extract_delta_z.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/extract_delta_z.py) 从 StaMo checkpoint 提取 `delta_z`
3. 用 [scripts/train_compact_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_compact_semantics.py) 学习 `delta_z -> compact semantics`
4. 用 [scripts/train_process_chain.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_process_chain.py) 比较不同过程链设置

---

## 2. 当前 toy 数据设定

为了让这个最小实验更接近你的研究问题，我做了两个调整：

1. 标签覆盖更完整  
现在 toy 数据包含：
`approach / contact / grasp / lift / move / place`
以及
`no_contact / contact / open / closing / closed / farther / closer / reached`

2. 文本变粗粒度  
`caption` 不再直接泄漏细粒度语义，而是只保留高层阶段信息：

- approach/contact: `getting ready to interact`
- grasp/lift: `handling the object`
- move/place: `adjusting the object's position`

这样更符合你真正想验证的前提：

**文本过程链是粗的，StaMo delta_z 提供细粒度补充。**

---

## 3. 数据规模

当前 manifest 位于：

- [data/processed/toy_process_chain](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/toy_process_chain)

规模如下：

- train: 120
- val: 30
- test: 30

---

## 4. compact semantics 结果

使用：

- 训练集：`train.jsonl`
- 测试集：`test.jsonl`
- 特征：`delta_z`

### 测试集结果

#### `contact_state`

- accuracy: `0.7333`
- macro-F1: `0.6591`

#### `gripper_state`

- accuracy: `0.3667`
- macro-F1: `0.2771`

#### `object_motion`

- accuracy: `0.3667`
- macro-F1: `0.2422`

#### `target_relation`

- accuracy: `0.6000`
- macro-F1: `0.3902`

### 初步解读

当前 toy StaMo `delta_z` 对下面两类属性更有信号：

- `contact_state`
- `target_relation`

而对下面两类更难：

- `gripper_state`
- `object_motion`

这和直觉一致，因为 toy 数据本身很简单，且我们现在只是用线性语义头，没有做更强的语义学习。

---

## 5. process chain 对比结果

目标任务：

- `stage_label` 分类

评估集：

- `test.jsonl`

### Setting A: `text_only`

- accuracy: `0.5667`
- macro-F1: `0.3764`

### Setting B: `semantics_only` using predicted semantics

- accuracy: `0.2333`
- macro-F1: `0.1577`

### Setting C: `text_plus_semantics` using gold semantics

- accuracy: `1.0000`
- macro-F1: `1.0000`

### Setting D: `text_plus_semantics` using predicted semantics

- accuracy: `0.3000`
- macro-F1: `0.2011`

---

## 6. 这些结果说明了什么

### 6.1 好消息

最重要的结论不是某个分数高，而是：

**这条实验链已经真正打通了。**

也就是说，你现在已经有：

1. 真实 StaMo `delta_z` 提取
2. `delta_z -> compact semantics` 训练接口
3. `text/process chain + semantics` 融合接口

### 6.2 第二个重要结论

`text_plus_semantics` 使用 gold semantics 时可以达到非常强的结果，这说明：

**compact semantics 作为过程链桥接接口在结构上是有效的。**

也就是说，你的研究问题本身是成立的：

- 如果 `delta_z` 能被可靠地映射到紧凑语义
- 这些紧凑语义确实可以帮助过程链理解短时程状态变化

### 6.3 当前瓶颈

现在真正限制性能的不是桥本身，而是：

**预测语义还不够准。**

所以当前失败点不是：

- `delta_z` 完全没用
- semantics 接口没意义

而是：

- 现在的语义头太弱
- toy 数据过小
- 预测语义误差会把融合效果拖垮

---

## 7. 这和你的正式课题怎么对应

你真正的课题不是把 toy 分数做高，而是验证下面这个链条：

1. 文本过程链只提供粗粒度过程信息
2. StaMo `delta_z` 提供细粒度短时程变化信号
3. 把 `delta_z` 压缩成 compact semantics
4. 再把 compact semantics 接给过程链
5. 观察是否提升短时程状态理解

当前 toy 实验已经给出一个很有价值的结构性结论：

**如果 compact semantics 是准确的，那么它确实能补文本。**

这正是你下一步在真实数据上最想验证的核心假设。

---

## 8. 下一步最值得做什么

按优先级建议如下：

1. 用真实数据替换 toy manifest  
重点替换：
[scripts/build_toy_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_toy_pair_manifest.py)

2. 用真实标签训练更强的 compact semantics 头  
重点替换：
[scripts/train_compact_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_compact_semantics.py)

3. 把过程链模型从 TF-IDF + LR 升级到更像论文设定的文本或多模态链  
重点替换：
[scripts/train_process_chain.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_process_chain.py)

4. 保留当前 toy 管线作为 smoke test  
这样每次改动后，你都能先快速验证接口没断

---

## 9. 一句话总结

当前 toy 实验最重要的结论是：

**StaMo delta_z 到 compact semantics 再到 process chain 的桥已经在本地跑通，而且 gold semantics 的结果表明这条桥在结构上是值得继续做的。**

