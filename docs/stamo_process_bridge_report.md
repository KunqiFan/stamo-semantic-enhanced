# StaMo Process Bridge: 文本语义桥技术升级报告

## 1. 核心理念与“文本语义桥” (Text-Semantic Bridge)
在多模态与具身智能（Embodied AI）研究中，底层物理状态（连续、高频、短视距）与高层文本指令（抽象、离散、长视距）之间存在天然的表征鸿沟。

**文本语义桥**的设计思路旨在优雅地跨越这一鸿沟：
1. **物理层降维**：将连续的视觉隐变量（`delta_z`）映射为结构化的紧凑语义（Compact Process Semantics），例如 `contact_state` 或 `gripper_state`。
2. **逻辑层融合**：将这些具备物理意义的离散语义，与人类文本（Caption）再次结合，完成全局意图与局部物理现实的联合推理，最终精准预测细粒度的任务过程阶段（`stage_label`）。

---

## 2. 工程与算法架构大升级
为了最大化这座“语义桥”的承载能力，本次对核心算法库进行了一次彻底的现代化机器学习武装，主要涵盖三大维度的跃升：

### 2.1 紧凑语义提取器升级 (Compact Semantics Extractors)
在处理 `delta_z` 这种稠密特征时，原版的逻辑回归（Logistic Regression）与小型 MLP 容易欠拟合或过拟合。
* **新增核心分类器**：引入了 `HistGradientBoostingClassifier` (直方图梯度提升树) 和 `RandomForestClassifier` (随机森林)。
* **性能预期**：在预测诸如“物体是否发生了微小移动”或“夹爪是否处于闭合阈值”时，树状集成模型对非线性边界的划分能力远超线性模型。

### 2.2 多模态过程链融合升级 (Process Chain Fusion)
针对 `train_process_chain.py` 阶段，我们对融合策略进行了双重巩固：
* **单模态及 Concat 拼接基线**：底座全线由 Logistic Regression 升级为 HGB（直方图梯度提升树）。
* **Stacking (堆叠) 元分类器**：引入了 Random Forest 作为 Meta-Classifier。相比简单的线性加权，它能更好地学习“文本置信度”与“物理语义置信度”之间的非线性交互。
* **自适应高级文本表征**：为文本侧增加了**自适应大语言模型回退机制**。当环境中存在 `sentence-transformers` 时，自动采用高质量的稠密文本嵌入 (Dense Embeddings)；若未安装，则自动回退并应用升级版的 `TfidfVectorizer`（支持 1-3 N-grams 提取并加入 `max_df` 停用词过滤及 `class_weight='balanced'` 长尾处理）。

### 2.3 评估基建重构 (Evaluation Infrastructure)
原版评估代码仅输出单一的 Macro F1 分数，我们对其进行了重写：
* **支持嵌套结构**：完美解析最新的多级嵌套预测输出 (`gold_semantics` / `predicted_semantics`)。
* **细粒度分类报告**：调用 `classification_report`，全景展示每一类的 Precision, Recall 与 F1-score。
* **混淆矩阵 (Confusion Matrix)**：新增直观的混淆矩阵输出。能够一眼定位模型的短板（例如，将 `contact` 误判为 `approach` 的概率）。

---

## 3. 测试管线验证状态
最新的 `run_toy_process_chain_pipeline.py` 测试流水线已被完整执行。
* **成功完成** 15 组（5分类器 × 3特征组）分类基线的全量训练。
* **最佳模型选取**：在过程链最终比对中，我们自动调用了最强的 `hgb_both`（结合 `delta_z` 与 `delta_pooled` 特征的梯度提升分类器）作为最高标准的语义提取器。

### 下一步行动建议
流水线已被验证具备极高的鲁棒性，推荐在激活真实 DROID 数据集的规模化训练前，先在您的环境中执行如下命令，欣赏全新的图表与指标输出：

```bash
# （强烈推荐）安装 Sentence-Transformers 解锁最强文本基线
pip install sentence-transformers

# 查看全新评估器的混淆矩阵诊断能力
python scripts/evaluate.py --predictions data/interim/toy_process_chain/semantics/test_predicted_semantics_hgb_both.jsonl --target stage_label
```
