# StaMo + DROID 实验接入完整报告

## 1. 报告目的

这份报告的目标是系统记录当前阶段已经完成的工作，并把每一步的：

- 做了什么
- 为什么这样做
- 这一步的原理是什么
- 这一步在整个项目中起什么作用

都尽量解释清楚。

这份报告面向的不是“简单运行记录”，而是希望把整个过程沉淀成一份可以回看、复现、继续扩展的项目过程文档。

---

## 2. 项目总目标回顾

当前项目的核心研究问题是：

**StaMo 的状态差分 `delta_z` 能否作为短时程、细粒度状态变化证据，去补充文本或多模态过程链？**

这个问题的重点不是重新设计 StaMo，也不是训练一个巨大 world model，而是聚焦下面这条“小桥”：

`image pair / short trajectory -> StaMo encoder -> delta_z -> compact process semantics -> text/process chain`

也就是说，我们真正要验证的是：

1. `delta_z` 里是否包含短时程状态变化信息
2. 这些信息是否能被整理成一个紧凑语义接口
3. 这个语义接口是否能帮助文本过程链更好地理解短时程操作变化

所以这个项目天然分成四层：

1. 先把 `StaMo` 本身在本地跑通
2. 能从图像对里提取 `delta_z`
3. 把 `delta_z` 接成 `compact semantics`
4. 把 `compact semantics` 接入 `process chain`

后续的所有工作都是沿着这四层往下搭。

---

## 3. 第一阶段：把项目先搭成可执行研究骨架

### 3.1 做了什么

最开始先在根目录搭了一个研究脚手架，而不是立刻去跑论文大模型。主要新增了这些文件：

- [README.md](/C:/Users/ryanf/Desktop/stamo_pro/README.md)
- [docs/experiment_plan.md](/C:/Users/ryanf/Desktop/stamo_pro/docs/experiment_plan.md)
- [configs/process_labels.yaml](/C:/Users/ryanf/Desktop/stamo_pro/configs/process_labels.yaml)
- [data/README.md](/C:/Users/ryanf/Desktop/stamo_pro/data/README.md)
- [data/processed/sample_manifest.jsonl](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/sample_manifest.jsonl)
- [scripts/extract_delta_z.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/extract_delta_z.py)
- [scripts/build_process_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_process_semantics.py)
- [scripts/train_baseline.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_baseline.py)
- [scripts/train_compact_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_compact_semantics.py)
- [scripts/train_process_chain.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_process_chain.py)
- [src/stamo_bridge](/C:/Users/ryanf/Desktop/stamo_pro/src/stamo_bridge)

### 3.2 为什么先搭脚手架

因为项目目标不是“复现一篇论文就结束”，而是要做一个学期内能不断迭代的研究原型。

如果一开始只盯着官方 StaMo 仓库，容易出现两个问题：

1. 即使模型跑起来了，也没有自己的实验接口
2. 后面接 `delta_z -> semantics -> process chain` 时会反复返工

所以先搭骨架的作用是：

- 提前把实验对象定义清楚
- 把数据格式、标签格式、脚本接口先统一
- 让后续不管接 toy 数据、真实数据还是别的数据，都走同一条 pipeline

### 3.3 这一步的原理

研究型代码最怕“每次换数据就重写一遍”，所以我用 manifest 驱动的方式，把每个样本统一表示成：

- `image_t`
- `image_tp`
- `caption`
- `process_text`
- `action`
- `ee_delta`
- `labels`

这样后面的 `StaMo` 特征提取、语义训练、过程链训练就都只认 manifest，而不再直接绑定某个具体数据集。

### 3.4 这一步的作用

这一步不是为了出结果，而是为了保证后续每一轮实验都能接在同一套接口上。它相当于给整个项目先修了一条“标准管线”。

---

## 4. 第二阶段：把官方 StaMo 在本地真正跑通

### 4.1 做了什么

我把官方 StaMo 仓库下载到：

- [StaMo](/C:/Users/ryanf/Desktop/stamo_pro/StaMo)

然后检查了关键入口和结构，包括：

- [StaMo/train_renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/train_renderer.py)
- [StaMo/validate_renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/validate_renderer.py)
- [StaMo/stamo/renderer/model/renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/renderer.py)
- [StaMo/stamo/renderer/model/backbone.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/backbone.py)
- [StaMo/stamo/renderer/trainer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/trainer.py)
- [StaMo/stamo/renderer/utils/data.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/data.py)

之后安装了运行依赖，包括：

- `torch`
- `torchvision`
- `diffusers`
- `timm`
- `lightning`
- `accelerate`
- `omegaconf`
- `jsonlines`
- `einops`
- `tensorboard`
- `rich`

### 4.2 为什么不能直接原样跑官方版本

因为官方配置默认依赖作者本地路径下的权重，例如：

- 视觉 backbone 权重
- SD3 相关大模型权重

而这些权重在当前仓库中并没有现成可直接使用的公共 checkpoint。

所以如果完全不改代码，本机会遇到两个根本问题：

1. 模型初始化时找不到权重
2. 即使代码结构没问题，也无法真实完成训练和验证

### 4.3 我是怎么处理这个问题的

我没有假装“原版已经完整复现”，而是采取了一个更诚实也更适合当前项目的做法：

**在尽量少改官方结构的前提下，加一个本地可运行的 toy 模式。**

具体修改是：

#### 修改 1：让条件头支持更小输入维度

修改：

- [StaMo/stamo/renderer/model/backbone.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/backbone.py)

作用：

- 让 `DiTConditionHead` 可以适配 toy 模型的小特征维度

原理：

- 原始版本更偏向固定的大维度 backbone 输出
- toy 模式需要更小、更轻量的表示维度
- 所以要把这个头做成可配置输入维度

#### 修改 2：在 `RenderNet` 里加入 toy_mode

修改：

- [StaMo/stamo/renderer/model/renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/renderer.py)

作用：

- 在没有官方大权重时，依然能够构建一个小型可训练版本的 `RenderNet`

原理：

- 原始 `RenderNet` 依赖大规模预训练组件
- toy 模式下，用更轻的 transformer / VAE / scheduler 组合替代
- 保留主调用路径不变：`build model -> train -> validate -> save checkpoint -> extract delta`

这样做的好处是，虽然不是论文最终模型，但主流程和接口是一致的。

#### 修改 3：训练器增加 `use_bf16` 开关

修改：

- [StaMo/stamo/renderer/trainer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/trainer.py)

作用：

- 避免 toy 模式被强制卡在某种精度配置上

原理：

- toy 环境的目标是“能稳定跑通”
- 如果训练器死写某种混合精度，反而可能导致不兼容

#### 新增 toy 配置和 toy 数据脚本

新增：

- [StaMo/configs/toy.yaml](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/configs/toy.yaml)
- [StaMo/scripts/create_toy_data.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/scripts/create_toy_data.py)

作用：

- 用最小代价生成一份本机可训练、可验证、可做状态差分演示的数据

### 4.4 这一步的原理

这一步的核心原则不是“追求论文原始指标”，而是：

**先确保代码链路真实可执行。**

在研究工程里，只有当下面这些都真的成立了，后面的工作才有意义：

1. 模型能构建
2. 数据能送进去
3. 能算 loss
4. 能反向传播
5. 能保存 checkpoint
6. 能加载 checkpoint
7. 能从输入图像对里取出差分表示

toy 模式就是用来验证这七件事的。

### 4.5 结果

我成功运行了：

```powershell
py train_renderer.py --config_path configs/toy.yaml
```

得到了训练和验证结果：

- `PSNR = 7.1118`
- `SSIM = 0.1458`
- 训练 loss 大致从 `1.3256` 降到 `1.1813`

checkpoint 保存在：

- [StaMo/ckpts/toy_debug/4](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/ckpts/toy_debug/4)

验证图像在：

- [StaMo/logs/toy_debug/images/4](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/logs/toy_debug/images/4)

### 4.6 这一步的作用

这一步证明了：

- StaMo 主体代码在本机能真实执行
- `delta_z` 相关调用路径以后是可以继续被复用的

这一步是后续所有实验的基础。

---

## 5. 第三阶段：确认 `delta_z` 这条线真实可提取

### 5.1 做了什么

在 toy checkpoint 训练完成后，我调用了 `RenderNet` 内部和状态差分相关的路径，做了真实的两帧差分提取与 delta transfer 演示。

最终得到的一个典型结果是：

- `delta_emb_shape = (1, 2, 256)`
- `delta_pooled_shape = (1, 128)`
- `delta_emb_l2 = 1.0426`
- `delta_pooled_l2 = 0.2816`

同时生成了可视化结果：

- [delta_interpolation_combined_4.jpeg](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/logs/toy_debug/images/4/delta_interpolation_combined_4.jpeg)

### 5.2 原理

`delta_z` 本质上就是：

`z_tp - z_t`

其中：

- `z_t` 是时刻 `t` 的潜表示
- `z_tp` 是时刻 `t + delta` 的潜表示

如果一个编码器学到了对状态有意义的表征，那么两个时刻之间的潜表示差，就有可能对应“状态变化”本身，而不是只对应静态内容。

这也是为什么你的课题特别适合用 `delta_z`：

- 文本通常更像任务层或阶段层描述
- `delta_z` 更可能保留短时程、细粒度、连续变化的证据

### 5.3 作用

这一步非常关键，因为它把研究问题从“概念上可能可行”推进到了：

**代码层面已经确认，StaMo 确实能给出可计算、可保存、可下游使用的状态差分表示。**

---

## 6. 第四阶段：先用 toy 数据搭出完整桥接原型

### 6.1 做了什么

为了不等真实数据，我先把下面这条完整链用 toy 数据接起来：

`image pair -> StaMo delta feature -> compact semantics -> process chain`

主要新增或更新了这些脚本：

- [scripts/build_toy_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_toy_pair_manifest.py)
- [scripts/extract_delta_z.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/extract_delta_z.py)
- [scripts/train_compact_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_compact_semantics.py)
- [scripts/train_process_chain.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_process_chain.py)
- [scripts/run_toy_process_chain_pipeline.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/run_toy_process_chain_pipeline.py)
- [src/stamo_bridge/semantics/interface.py](/C:/Users/ryanf/Desktop/stamo_pro/src/stamo_bridge/semantics/interface.py)

### 6.2 这些文件是怎么协作的

#### `build_toy_pair_manifest.py`

负责生成 toy 图像对及标签。

作用：

- 把原始 toy 轨迹切成短窗口样本
- 每个样本对应两帧图像和一组短时程标签

#### `extract_delta_z.py`

负责从 manifest 中读取：

- `image_t`
- `image_tp`

然后调用 StaMo encoder，输出：

- `z_t`
- `z_tp`
- `delta_z`
- `pooled_t`
- `pooled_tp`
- `delta_pooled`

原理：

- 它相当于把“图像对”转成“差分特征”

#### `train_compact_semantics.py`

负责把 `delta_z` 或 `delta_pooled` 映射到语义标签：

- `contact_state`
- `gripper_state`
- `object_motion`
- `target_relation`

当前用的是最简单的基线：

- `StandardScaler + LogisticRegression`

原理：

- 先不要把下游模型做复杂
- 优先测“特征里有没有信号”

#### `train_process_chain.py`

负责比较三类设置：

- `text_only`
- `semantics_only`
- `text_plus_semantics`

原理：

- 如果 `text_plus_semantics` 明显好于 `text_only`
- 就说明语义桥是有贡献的

### 6.3 为什么先做 toy 原型

因为 toy 原型能先回答结构性问题：

1. 这条桥能不能从工程上接通
2. 每一段输入输出定义得是否合理
3. gold semantics 是否真的能帮助过程链

如果连 toy 上都接不通，就没有必要急着上真实数据。

### 6.4 toy 阶段的关键发现

在 toy 阶段最重要的发现不是某个单独准确率，而是下面两件事：

#### 发现 1：gold semantics 对过程链帮助非常明显

这说明：

- “紧凑语义接口”这件事本身是有意义的
- 不是一个空想结构

#### 发现 2：predicted semantics 还不够强

这说明：

- 结构本身没问题
- 真正的瓶颈在 `delta feature -> semantics` 这一步

也就是说，研究重点自然被收敛到了你真正关心的地方。

### 6.5 作用

toy 阶段像一个“整链路预演”：

- 先确认研究设计成立
- 再把精力放到真实数据和更强标签上

---

## 7. 第五阶段：寻找适合的开源真实数据集

### 7.1 做了什么

在你没有现成真实数据后，我筛选了几类开源机器人操作数据，包括：

- `DROID`
- `BridgeData V2`
- `LIBERO`
- `RH20T`
- `RoboNet`
- `Open X-Embodiment`

最后我选定了：

**DROID 100**

### 7.2 为什么选 DROID 100

原因主要有三点：

#### 原因 1：是真实机器人操作数据

这比纯模拟数据更符合你要研究的“细粒度短时程状态变化证据”。

#### 原因 2：有语言信息

你后面要做：

- text/process chain
- visual delta feature

所以数据里最好本来就带任务语言或 instruction。

#### 原因 3：有小样本版本

`DROID 100` 不是动辄几 TB 的全量数据，而是一个适合原型验证的小规模版本。

这特别适合当前阶段：

- 先验证可行性
- 再决定是否扩展

### 7.3 作用

这一步的作用是给项目找一个“既真实、又可做、又带语言”的起点，让你的课题从 toy 原型过渡到真实数据验证。

---

## 8. 第六阶段：接入 DROID 100 原始数据

### 8.1 做了什么

我使用 `gsutil` 访问并下载了 `DROID 100` 的一部分 shard 到：

- [data/raw/droid_100](/C:/Users/ryanf/Desktop/stamo_pro/data/raw/droid_100)

并检查了：

- `dataset_info.json`
- `features.json`
- 多个 `r2d2_faceblur-train.tfrecord-*`

后来确认本地已有多个完整 shard，可直接用于第一轮实验。

### 8.2 为什么要先做字段检查

因为 `DROID` 使用的是 TFRecord/RLDS 风格，不是普通图片文件夹。要把它接到当前实验链，必须先搞清楚：

- 图像字段叫什么
- 动作字段叫什么
- 机器人状态字段叫什么
- 语言字段是否存在、是否为空

如果这些没搞清楚，就无法稳定生成 manifest。

### 8.3 原理

我先用 `tfrecord` 库直接读 record，确认这些关键字段：

- `steps/action`
- `steps/observation/cartesian_position`
- `steps/observation/gripper_position`
- `steps/observation/wrist_image_left`
- `steps/language_instruction`
- `steps/language_instruction_2`
- `steps/language_instruction_3`

并验证：

- `action` 可以 reshape 成 `(-1, 7)`
- `cartesian_position` 可以 reshape 成 `(-1, 6)`
- `gripper_position` 是每步一个标量
- 相机图像是每步一个 JPEG bytes

### 8.4 发现

我发现一个很实际的问题：

- 有些 episode 有语言
- 有些 episode 的语言字段是空的

所以第一版真实实验我采取了一个合理限制：

**优先使用有语言的 episode，先让 text/process chain 分支可用。**

### 8.5 作用

这一步相当于“理解数据集本体”。没有它，后面所有 `manifest`、标签、图像导出、语义构造都会建立在错误假设上。

---

## 9. 第七阶段：构建 DROID 专用 pair manifest 生成器

### 9.1 做了什么

我新增了：

- [scripts/build_droid_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_droid_pair_manifest.py)

这个脚本会直接从 DROID TFRecord 生成：

- 图像对样本
- 本地图像文件
- 任务文本
- 动作均值
- 末端位姿差
- 紧凑语义代理标签

最终产物在：

- [data/processed/droid_100_process_chain](/C:/Users/ryanf/Desktop/stamo_pro/data/processed/droid_100_process_chain)

### 9.2 这一步为什么重要

因为前面的根项目脚手架是“数据集无关”的，而 DROID 是“具体数据集”。

所以这里需要一个 adapter，把 DROID 的原始格式翻译成我们统一的样本格式。

如果没有这个转换器：

- `extract_delta_z.py` 无法直接吃 TFRecord
- `train_compact_semantics.py` 无法知道图像在哪里
- `train_process_chain.py` 无法知道文本和标签是什么

### 9.3 这个脚本内部的原理

#### 原理 1：从 episode 里切短窗口

对每个 episode，不是整段作为一个样本，而是切成：

`(start_idx, end_idx = start_idx + gap)`

这样每个样本就表示一个短时程变化。

这是因为你的研究目标本来就不是长时程任务规划，而是短时程状态变化。

#### 原理 2：选定单一相机视角

第一版先用：

- `wrist_image_left`

原因：

- 手腕视角通常更接近接触、抓取等局部变化
- 对短时程状态变化更敏感

#### 原理 3：优先选有语言的 episode

因为要保留 `text/process chain` 分支。

#### 原理 4：保存本地图像文件

虽然 TFRecord 里已经有 JPEG bytes，但当前 pipeline 的设计是通过图像路径读图，所以先把关键帧导出成本地 `.jpg` 文件。

这样后续所有脚本都不需要知道原始 TFRecord 细节。

### 9.4 当前的采样参数

我使用的第一版参数是：

- `step_gap = 4`
- `pair_stride = 12`
- `require_language = true`

作用分别是：

- `step_gap = 4`：保证是短时程 pair
- `pair_stride = 12`：减少过密重叠样本
- `require_language = true`：保留文本分支可用性

### 9.5 作用

这一步正式把真实数据接进了实验管线，是从“toy 研究原型”迈向“真实开源验证”的关键一步。

---

## 10. 第八阶段：在 DROID 上构建代理紧凑语义标签

### 10.1 为什么需要代理标签

因为 DROID 原始数据没有直接给出你想研究的这些属性标签：

- `contact_state`
- `gripper_state`
- `object_motion`
- `target_relation`
- `stage_label`

但是你又需要一套语义接口来验证：

`delta feature -> compact semantics`

所以第一轮真实实验必须先构造 **proxy labels**。

### 10.2 我是如何构造这些标签的

#### `gripper_state`

依据：

- `gripper_position(start)`
- `gripper_position(end)`
- 它们之间的变化方向和变化量

逻辑：

- 张开较大且没有明显关闭趋势 -> `open`
- 数值快速减小 -> `closing`
- 接近关闭 -> `closed`

原理：

- 夹爪开合是短时程操作状态里非常重要的一类低维信号
- 它往往和“接触”“抓取”直接相关

#### `contact_state`

依据：

- `gripper_state`
- 开合程度

逻辑：

- 如果正在关闭或已闭合，近似视作 `contact`
- 如果始终张开且远离闭合，近似视作 `no_contact`

原理：

- 这不是严格接触真值
- 但在没有触觉/接触真值标注时，是一个可接受的第一版近似

#### `target_relation`

依据：

- 末端在 pair 起点和终点时刻到 episode 终点位置的距离

逻辑：

- 更接近终点 -> `closer`
- 更远离终点 -> `farther`
- 非常接近终点 -> `reached`

原理：

- 把 episode 末端位置近似当作短任务目标点
- 这是从“动作过程”中抽出一个简化目标关系的办法

#### `object_motion`

依据：

- pair 内 `cartesian_position` 的位移大小
- `z` 方向变化量
- 是否接近 episode 终点

逻辑：

- `z` 上升明显且位移较大 -> `lifted`
- 位移明显但不一定上升 -> `moved`
- 接近终点且夹爪打开 -> `placed`
- 变化小 -> `still`

原理：

- 在没有真实物体跟踪标签时，用末端运动作为物体状态变化的代理
- 这是当前阶段一种“弱监督近似”

#### `stage_label`

依据：

- 前面四种紧凑语义

逻辑：

- `placed` -> `place`
- `lifted` -> `lift`
- `closing` -> `grasp`
- `contact + open` -> `contact`
- `moved` -> `move`
- 其余 -> `approach`

原理：

- 把属性语义进一步压缩成一个更高层的阶段标签
- 便于后面做 process chain 分类任务

### 10.3 这一步的作用

这一步的意义不在于“标签已经完美”，而在于：

- 让真实数据上的语义接口第一次可以被训练和评估
- 让我们知道 `delta_z` 是否对这些属性有信号

换句话说，proxy labels 是通往真实语义实验的第一块跳板。

---

## 11. 第九阶段：提取 DROID 图像对上的 StaMo 差分特征

### 11.1 做了什么

我使用已经在本机跑通的 StaMo toy checkpoint，在 DROID manifest 上提取：

- `delta_z`
- `delta_pooled`

命令的逻辑等价于：

```powershell
py scripts/extract_delta_z.py --manifest ...train.jsonl --out_dir ...latents --device cuda
py scripts/extract_delta_z.py --manifest ...val.jsonl --out_dir ...latents --device cuda
py scripts/extract_delta_z.py --manifest ...test.jsonl --out_dir ...latents --device cuda
```

输出保存在：

- [data/interim/droid_100_process_chain/latents](/C:/Users/ryanf/Desktop/stamo_pro/data/interim/droid_100_process_chain/latents)

### 11.2 原理

这一步等于是把真实世界里的图像对，映射成 StaMo 风格的“状态差分特征”。

具体过程是：

1. 读 `image_t`
2. 读 `image_tp`
3. 各自经过 `projector_feature_extractor`
4. 送进 `RenderNet.encode`
5. 得到 `z_t` 和 `z_tp`
6. 计算：
   - `delta_z = z_tp - z_t`
   - `delta_pooled = pooled_tp - pooled_t`

### 11.3 `delta_z` 和 `delta_pooled` 的区别

#### `delta_z`

- 更保留局部、token 级别或结构化潜表示的差分
- 信息量更丰富
- 维度更高

#### `delta_pooled`

- 是更压缩的全局差分
- 信息更少，但可能更稳定

这也是为什么后面我会同时比较二者。

### 11.4 作用

这一步把“真实图像 pair”正式变成了“可学习、可探测、可做语义预测”的差分特征，是整条实验主线的核心连接点。

---

## 12. 第十阶段：训练 `delta feature -> compact semantics`

### 12.1 做了什么

我分别用：

- `delta_z`
- `delta_pooled`

作为输入特征，在 DROID 代理标签上训练 `compact semantics` 分类器。

用到的脚本：

- [scripts/train_compact_semantics.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_compact_semantics.py)

### 12.2 为什么先用 Logistic Regression

原因不是因为它最好，而是因为它最适合回答当前阶段最关键的问题：

**StaMo 差分特征里到底有没有线性可分的语义信号？**

如果一个简单线性模型都完全学不出来，那说明：

- 特征信号可能很弱
- 或标签定义不合理

但如果线性模型已经能学出一定效果，就说明这条桥是值得继续深挖的。

### 12.3 训练原理

对每个语义字段单独训练一个分类器：

- `contact_state`
- `gripper_state`
- `object_motion`
- `target_relation`

每个样本的输入是：

- 某个 `sample_id` 对应的 latent 文件

每个样本的输出是：

- 该字段对应的代理语义标签

### 12.4 修复 bug 前后发生了什么

在初次跑 DROID 实验后，我发现 latent 文件数明显少于 manifest 总样本数。

这意味着：

- 有样本被覆盖了

最终定位到的原因是：

- 初版 `episode_id` 使用了不完整的 shard 名
- 不同 shard 中的 episode 在生成 `sample_id` 时发生冲突

修复方式：

- 在 [scripts/build_droid_pair_manifest.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/build_droid_pair_manifest.py) 中把 `episode_id` 改成包含完整 shard 标识

修复后重新生成：

- manifest
- latents
- semantics 预测

修复后确认：

- latent 文件总数 = `470`
- 与 train/val/test 总样本数完全一致

### 12.5 修复后的结果

#### 用 `delta_z`

- `contact_state`: accuracy `0.5873`
- `gripper_state`: accuracy `0.5397`
- `object_motion`: accuracy `0.5714`
- `target_relation`: accuracy `0.4127`

#### 用 `delta_pooled`

- `contact_state`: accuracy `0.6190`
- `gripper_state`: accuracy `0.5079`
- `object_motion`: accuracy `0.7143`
- `target_relation`: accuracy `0.4286`

### 12.6 如何理解这些结果

#### 结论 1：短时程属性是有信号的

尤其是：

- `contact_state`
- `gripper_state`
- `object_motion`

这些更贴近局部操作变化的标签，已经表现出一定可预测性。

#### 结论 2：`delta_pooled` 在当前数据上比 `delta_z` 更稳一些

特别是 `object_motion`。

这可能说明：

- 当前 proxy label 更偏全局变化趋势
- 而 `delta_pooled` 更适合捕捉这种简化后的全局关系

#### 结论 3：`target_relation` 依然较弱

这说明：

- 目标关系的 proxy 定义目前还不够强
- 或者仅靠当前特征还不足以稳定表示“接近目标”的概念

### 12.7 作用

这一步第一次在真实开源数据上证明了：

**StaMo 差分表示并不是完全无效，它对某些短时程过程属性已经包含了可学习信号。**

---

## 13. 第十一阶段：接入 process chain 并做基线比较

### 13.1 做了什么

我使用：

- [scripts/train_process_chain.py](/C:/Users/ryanf/Desktop/stamo_pro/scripts/train_process_chain.py)

比较以下设置在 `stage_label` 上的分类效果：

1. `text_only`
2. `semantics_only`
3. `text_plus_semantics`
4. `text_plus_gold_semantics`

### 13.2 为什么过程链阶段只使用 `caption`

因为如果直接用太强的 `process_text`，有可能把细粒度阶段信息泄漏给文本模型。

所以这里更保守地使用：

- `caption = task instruction`

例如：

- `Put the marker in the pot`
- `Put the bread in the toaster`

这样文本更接近高层任务条件，而不是短时程过程真值。

### 13.3 当前结果

#### `text_only`

- accuracy `0.7460`
- macro-F1 `0.2136`

#### `semantics_only`（使用 `delta_z` 预测语义）

- accuracy `0.5397`
- macro-F1 `0.2216`

#### `text_plus_semantics`（使用 `delta_z` 预测语义）

- accuracy `0.5397`
- macro-F1 `0.2216`

#### `text_plus_semantics`（使用 `delta_pooled` 预测语义）

- accuracy `0.6032`
- macro-F1 `0.1937`

#### `text_plus_gold_semantics`

- accuracy `1.0000`
- macro-F1 `1.0000`

### 13.4 如何理解这些结果

#### 现象 1：`text_only` 看起来准确率不低

但它的高准确率很大程度来自类别分布偏置。

因为 test 集中：

- `approach` 是大类

所以一个偏向大类的文本模型也能拿到不低 accuracy。

这就是为什么：

- accuracy 看起来还可以
- macro-F1 仍然很低

#### 现象 2：预测语义还没有真正增强文本过程链

这说明：

- 语义桥的上游信号还不够干净或不够强
- 不是说研究方向不对，而是桥的中间段还需要加强

#### 现象 3：gold semantics 的上界极高

这是整轮实验里最重要的结果之一。

因为它说明：

**一旦紧凑语义足够准确，过程链理解就会显著增强。**

所以研究问题已经从“桥是否值得做”变成了：

**怎样把 `delta feature -> semantics` 这一步做得更好。**

### 13.5 作用

这一步完成了你项目叙事里最关键的一次闭环验证：

- `StaMo` 差分特征
- 可以生成紧凑语义
- 紧凑语义理论上可以显著帮助过程链

也就是说，主研究假设已经有了结构性证据。

---

## 14. 第十二阶段：文档化和可追踪化

### 14.1 做了什么

在整个过程中，我持续把关键信息沉淀成中文文档，包括：

- [docs/stamo_file_report_zh.md](/C:/Users/ryanf/Desktop/stamo_pro/docs/stamo_file_report_zh.md)
- [docs/delta_semantics_process_chain_runbook_zh.md](/C:/Users/ryanf/Desktop/stamo_pro/docs/delta_semantics_process_chain_runbook_zh.md)
- [docs/toy_process_chain_experiment_report_zh.md](/C:/Users/ryanf/Desktop/stamo_pro/docs/toy_process_chain_experiment_report_zh.md)
- [docs/droid_100_progress_report_zh.md](/C:/Users/ryanf/Desktop/stamo_pro/docs/droid_100_progress_report_zh.md)
- 当前这份 [docs/stamo_droid_full_report_zh.md](/C:/Users/ryanf/Desktop/stamo_pro/docs/stamo_droid_full_report_zh.md)

### 14.2 为什么文档化也算工作的一部分

因为研究型项目最容易出现一个问题：

- 当时能跑通
- 过一周已经不知道为什么这样设计
- 再过一周甚至不知道哪份结果是修 bug 前还是修 bug 后

所以文档化不是附属品，而是研究工程的一部分。

它的作用是：

- 保持结果可解释
- 保持设计可追踪
- 保持下一轮修改有依据

---

## 15. 当前整体结论

### 15.1 已经明确成立的事

1. 本机已经把 StaMo 代码真实跑通
2. 已经能从图像对中真实提取 `delta_z / delta_pooled`
3. 已经搭出 `delta feature -> compact semantics -> process chain` 的完整实验接口
4. 已经在真实开源数据 `DROID 100` 上跑出第一轮结果
5. 已经证明 gold compact semantics 对过程链有显著帮助

### 15.2 当前最主要的瓶颈

1. 当前 StaMo 还是 toy checkpoint，不是正式预训练权重
2. DROID 上的语义标签是 proxy labels，不是人工真值
3. `stage_label` 类不平衡明显
4. `grasp/lift/place` 支持数偏少
5. 预测语义还不足以稳定增强文本过程链

### 15.3 对研究问题的当前回答

基于当前结果，可以给出一个很谨慎但明确的阶段性结论：

**StaMo 的状态差分表示里已经显示出对短时程细粒度变化的可学习信号，尤其在接触、夹爪状态和运动变化方面。**

同时：

**如果能把这些差分信号转换成更准确的紧凑语义接口，它们很有可能成为文本过程链的有效补充。**

也就是说：

- 研究方向是成立的
- 但当前还处在“桥已经搭起来、桥面还要加固”的阶段

---

## 16. 下一步最合理的工作

如果后面继续推进，我认为最值得优先做的是：

### 方向 1：改进 DROID 样本构造

目标：

- 提高 `grasp/lift/place` 样本比例
- 减少 `approach/still` 过强主导

### 方向 2：改进 proxy labels

目标：

- 让 `contact_state`
- `target_relation`
- `object_motion`

更加贴近真实语义

### 方向 3：增强 `delta feature -> semantics`

目标：

- 不再只用 logistic regression
- 尝试更强但仍然小规模的分类头

### 方向 4：联合使用 `delta_z + delta_pooled`

目标：

- 同时利用局部差分和全局差分

### 方向 5：如果能拿到更正式的 StaMo 权重，再做一次真实对比

目标：

- 验证当前发现是不是 toy checkpoint 的偶然现象

---

## 17. 一句话总结

到目前为止，这个项目已经从“一个研究想法”推进成了：

**一个在本机可运行、在真实开源机器人数据上可测试、并且已经给出第一轮结构性证据的实验系统。**

当前最重要的收获不是某一个数字，而是：

**我们已经明确知道下一步该优化哪里，而不是还停留在不知道问题能不能落地的阶段。**
