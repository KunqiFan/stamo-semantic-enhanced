# StaMo 文件运作说明报告

这份报告的目标不是复述论文，而是帮助你直接对照本地代码理解：

1. 这些文件各自负责什么
2. 它们之间是怎么串起来的
3. 我为了让它在本机跑通，具体改了哪些地方
4. 你后面如果要接入 `delta_z -> compact semantics`，应该从哪里下手

报告基于当前本地目录：

- [StaMo](/C:/Users/ryanf/Desktop/stamo_pro/StaMo)

---

## 1. 先给结论：StaMo 这套代码在做什么

从代码实现看，当前仓库的核心不是一个完整 VLA 系统，而是一个 **renderer / diffusion autoencoder 风格的训练框架**。

它的大致流程是：

1. 输入一张图像
2. 用视觉 backbone 提取 patch 特征
3. 用 projector 把大量 patch 特征压缩成少量 token
4. 把这些 token 作为条件，喂给一个 DiT / SD3 风格的扩散 transformer
5. 让模型学会重建图像 latent
6. 在这个过程中，压缩 token 就被当成紧凑状态表示
7. 再通过图像之间 token 的差，形成类似 `delta_z` 的状态差分

所以如果你从研究角度看，这个仓库里最值得关注的主线其实是：

`image -> vision backbone -> projector -> compact tokens -> delta between tokens`

而不是只盯着扩散生成本身。

---

## 2. 代码主入口

### 2.1 [StaMo/train_renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/train_renderer.py)

这是训练入口。

它做的事情很简单：

1. 调用 `init_args()` 读取配置
2. 构建 `RenderNet`
3. 构建 optimizer / scheduler / loss
4. 调用数据加载器
5. 创建 `Trainer`
6. 执行 `trainer.train_eval_by_iter(...)`

你可以把它理解成：

- `train_renderer.py` 不负责模型细节
- 它只是把“模型、数据、训练器”拼起来

所以当你以后想改实验入口时，优先改配置，而不是先改这个文件。

### 2.2 [StaMo/validate_renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/validate_renderer.py)

这是验证和演示入口。

它主要提供三类能力：

1. 普通重建评估
2. 插值评估 `interpolation_eval`
3. 基于起点和终点差分的 `delta_interpolation`

从你的课题角度，最关键的是第三个，因为它直接对应：

- 从 `start` 和 `end` 提取差分
- 把这个差分迁移到另一张图像上

这就是当前代码里最接近“`delta_z` 表达状态变化”的现成实现。

---

## 3. 配置文件怎么控制运行

### 3.1 [StaMo/configs/debug.yaml](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/configs/debug.yaml)

这是官方给的调试配置。

它定义了这些模块：

- `vision_backbone`
- `projector`
- `render_net`
- `data`
- `train`

其中最重要的几个字段是：

- `vision_backbone.model_name`
- `vision_backbone.local_ckpt`
- `render_net.sd3.local_ckpt`
- `data.train_json_path`
- `data.eval_json_path`

官方版本的问题在于：

- `local_ckpt` 指向作者自己机器上的绝对路径
- 本地没有这些大权重时，模型无法构建

### 3.2 [StaMo/configs/toy.yaml](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/configs/toy.yaml)

这是我新增的最小可运行配置。

它的目的不是复现论文指标，而是：

1. 让代码在本机先完整走通
2. 保留官方主流程结构
3. 产出真实的训练、验证、checkpoint、delta 演示结果

这个配置做了几件事：

- 用更小的 `img_size=64`
- 用更轻的视觉 backbone：`resnet18`
- 打开 `render_net.toy_mode: True`
- 把 DiT 和 VAE 都缩成小模型
- 把训练步数压到 8 step

你可以把它理解为一个“代码通路验证配置”。

---

## 4. 参数读取层

### [StaMo/stamo/renderer/utils/args.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/args.py)

这个文件负责：

1. 解析命令行参数
2. 读取 yaml 配置
3. 把分布式相关字段补进配置

入口函数是：

- `init_args()`

它做了两件关键事：

- 读取 `--config_path`
- 给配置对象补上 `world_size / local_rank / deepspeed`

所以如果你以后手工在 Python 里直接 `OmegaConf.load(...)`，要注意有些字段可能还没自动补上。

我这次做 checkpoint 加载 demo 时就踩到了这个点，后来手工加了：

- `args.deepspeed = False`

---

## 5. 数据层怎么工作

### 5.1 [StaMo/stamo/renderer/utils/data.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/data.py)

这是数据层核心文件。

最重要的类和函数有：

- `ImageData`
- `load_multi_datasets_form_json`
- `load_unsampler_datasets_from_json`
- `collate_fn`

### `ImageData`

它非常简单：

1. 读取一个 `.jsonl`
2. 每一行只取 `{"image": 路径}`
3. 打开图像
4. resize + tensor 化
5. 返回 `{"image": image_tensor}`

这说明当前官方训练根本不要求复杂轨迹格式，它只需要“图像路径列表”。

### `collate_fn`

它把多个样本拼成：

```python
{"images": [B, C, H, W]}
```

### `load_multi_datasets_form_json`

这个函数支持两种模式：

1. 多数据集混合采样
2. 退化成单数据集加载

官方训练里用的是：

- `make_single_dataset=True`

所以实际走的是“单数据集加载”那条分支。

### 5.2 [StaMo/scripts/create_jsons.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/scripts/create_jsons.py)

这个脚本负责把图像目录转换成官方训练需要的 `json/jsonl` 格式。

它的工作是：

1. 扫描图像目录
2. 生成训练 jsonl
3. 生成评估 jsonl
4. 生成指向这些 jsonl 的总配置 json

它不是训练核心，但它告诉我们一个非常重要的事实：

**当前 StaMo 仓库训练的是图像集，而不是显式的动作序列文件。**

这也是你后面要从“图像重建 token”走向“状态差分语义”的原因。

### 5.3 [StaMo/scripts/create_toy_data.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/scripts/create_toy_data.py)

这是我新增的 toy 数据脚本。

它会自动生成：

- `toy_data/train/*.png`
- `toy_data/eval/*.png`
- `jsons/train_toy.json`
- `jsons/eval_toy.json`

作用是：

- 在没有真实机器人数据时，先验证训练流程本身能不能跑

---

## 6. 模型层怎么工作

### 6.1 [StaMo/stamo/renderer/model/backbone.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/backbone.py)

这个文件里有三个关键模块：

- `VisionBackbone`
- `SD3TransformerBackbone`
- `DiTConditionHead`

### `VisionBackbone`

负责：

1. 用 `timm.create_model(...)` 构建视觉 backbone
2. 提取最后一层特征图
3. reshape 成 patch token 序列

输出形状大致是：

- `[B, num_patches, channels]`

这一步相当于：

`图像 -> patch 级表征`

### `SD3TransformerBackbone`

这是对 `diffusers` 里 `SD3Transformer2DModel` 的扩展封装。

它负责扩散 transformer 的主体前向。

如果你只关心 `delta_z`，不用一开始就把这里全啃完，因为它更偏生成器。

### `DiTConditionHead`

它做的事情比较简单：

1. 对压缩 token 做平均池化
2. 再线性映射成 pooled condition

这个 pooled 向量会和 token 一起送给 DiT。

我这次改了这里的一个点：

- 原版固定写死输入维度是 `4096`
- 我改成了可配置输入维度

原因是 toy 模式下 projector 输出维度不再是 4096。

### 6.2 [StaMo/stamo/renderer/model/projector.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/projector.py)

这是非常重要的文件。

它的核心作用是：

**把视觉 backbone 产生的大量 patch token，压缩成少量更紧凑的 token。**

流程可以理解成：

1. 输入 `image_embeddings`
2. 先经过几层 attention，让 patch token 互相交互
3. 再通过一套压缩层逐步减少 token 数
4. 最后输出少量压缩 token

输出大致形状是：

- `[B, num_token, output_align_dim]`

这就是当前代码里最接近“状态表示 z”的地方。

如果你要做 `delta_z`，最自然的选择就是从这里的输出出发：

- `z_t = projector(backbone(image_t))`
- `z_t2 = projector(backbone(image_t2))`
- `delta_z = z_t2 - z_t`

也就是说，从研究角度看，**projector 的输出比扩散图像更重要**。

### 6.3 [StaMo/stamo/renderer/model/renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/renderer.py)

这是整个模型的核心文件。

`RenderNet` 基本把所有模块都包起来了：

- `vision_backbone`
- `projector`
- `dit_condition_head`
- `DiT`
- `VAE`
- `scheduler`

你可以把 `RenderNet` 看成真正的系统主体。

里面最关键的方法有：

- `encode`
- `train_step`
- `eval_step`
- `get_delta_action`
- `delta_interpolation`

#### `encode(images)`

它做的事情是：

1. 用 `vision_backbone` 提取图像特征
2. 用 `projector` 压缩成少量 token
3. 用 `dit_condition_head` 得到 pooled embedding

返回：

- `image_embeds`
- `pooled_embeds`

这一步就是最值得你后面单独拿出来的编码接口。

#### `train_step(...)`

训练逻辑是：

1. 输入图像
2. 一路编码得到条件 token
3. 图像经过 VAE 编码成 latent
4. 给 latent 加噪声
5. 用 DiT 预测噪声/残差
6. 用 diffusion loss 优化

这就是一个条件扩散重建训练。

#### `eval_step(...)`

评估时会：

1. 从编码 token 出发
2. 通过 scheduler 逐步去噪
3. 生成重建图像
4. 最后算 PSNR / SSIM

#### `get_delta_action(start, end)`

这是和你课题最相关的函数之一。

它做的事情非常直接：

1. 编码 `start`
2. 编码 `end`
3. 直接做差

返回：

- `delta_emb = emb_end - emb_start`
- `delta_pooled = pooled_end - pooled_start`

这基本就是“代码版本的 `delta_z`”。

#### `delta_interpolation(image, start, end, generator)`

这是另一个关键函数。

它的逻辑是：

1. 先把当前图像编码成 `emb`
2. 再从 `(start, end)` 提取 `delta_emb`
3. 把差分加到当前图像编码上
4. 用扩散解码生成一张“施加了该变化后的图像”

这一步很重要，因为它说明：

**在作者的实现里，状态差分不是抽象概念，而是可以真实作用于图像生成条件的。**

这也是你后面把 `delta_z` 和语义桥接的最好切入口。

### 6.4 我在 `renderer.py` 里做了什么改动

为了让它在本机跑通，我新增了一个 `toy_mode`。

这个改动的作用是：

1. 如果没有官方 SD3 权重，就不走 `from_pretrained(...)`
2. 直接构造一个小型 DiT
3. 直接构造一个小型 VAE
4. 用小 scheduler 完成最小训练和推理链路

另外我还补了：

- `vae` 的 `shift_factor/scaling_factor` 默认值保护

因为 toy VAE 配置里这两个字段可能是 `None`。

---

## 7. 训练器层怎么工作

### [StaMo/stamo/renderer/trainer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/trainer.py)

这个文件负责训练和评估流程控制。

你可以把它理解成：

- `RenderNet` 负责“模型怎么前向”
- `Trainer` 负责“什么时候训练、什么时候评估、什么时候存盘”

最重要的方法有：

- `setup_model_for_training`
- `train_eval_by_iter`
- `eval_fn`
- `manually_eval`
- `interpolation_eval`
- `delta_interpolation`

### `train_eval_by_iter`

这是正式训练循环。

它会做：

1. 取 batch
2. 准备数据到 GPU
3. 前向
4. 反向
5. optimizer step
6. scheduler step
7. 定期保存
8. 定期评估

### `eval_fn`

评估时它会：

1. 调模型生成重建图像
2. 调 `calculate_psnr`
3. 调 `calculate_ssim`
4. 把预测图和 GT 图保存到日志目录

这就是为什么你能在：

- [StaMo/logs/toy_debug/images/4](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/logs/toy_debug/images/4)

看到重建结果图。

### `delta_interpolation`

这是 trainer 对模型里 `delta_interpolation` 的一层包装：

1. 负责把 PIL 图像转 tensor
2. 调模型做 delta transfer
3. 把生成结果保存到文件

我这次跑出来的结果图就是这里保存的：

- [delta_interpolation_combined_4.jpeg](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/logs/toy_debug/images/4/delta_interpolation_combined_4.jpeg)

### 我在 `trainer.py` 里做了什么改动

我加了一个很小的开关：

- `use_bf16`

原因是 toy 模式下没必要强制转 `bf16`，否则某些小模型和 Windows 环境下可能更容易出兼容问题。

---

## 8. 优化与工具文件

### [StaMo/stamo/renderer/utils/optim.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/optim.py)

这个文件负责：

- optimizer 构建
- scheduler 构建
- loss 构建

重要内容有：

- `WarmupLinearLR`
- `WarmupLinearConstantLR`
- `DiffusionLoss`
- `get_optimizer`
- `get_criterion`

其中最重要的是 `DiffusionLoss`：

它对模型预测的噪声残差和目标残差做加权平方误差。

这说明训练目标本质上是扩散模型风格的噪声预测。

### [StaMo/stamo/renderer/utils/metrics.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/metrics.py)

这个文件负责：

- `PSNR`
- `SSIM`
- 参数统计
- 训练 meter 和 timer

你这次看到的：

- `PSNR = 7.1118`
- `SSIM = 0.1458`

就是这里算出来的。

### [StaMo/stamo/renderer/utils/files.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/files.py)

这个文件是目录工具函数：

- 建目录
- 保证日志目录存在
- 保证 checkpoint 目录存在

它不复杂，但训练和评估保存都依赖它。

### [StaMo/stamo/renderer/utils/overwatch.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/overwatch.py)

这个文件负责统一日志接口。

作用是：

- 兼容单卡和分布式
- 控制不同 rank 的日志输出

你终端里看到那些 `INFO / WARNING` 日志就是从这里统一出来的。

---

## 9. 文件之间的调用关系

最核心的调用链可以简化成下面这样：

```text
train_renderer.py
  -> utils/args.py:init_args()
  -> model/renderer.py:RenderNet
       -> model/backbone.py:VisionBackbone
       -> model/projector.py:Projector
       -> model/backbone.py:DiTConditionHead
       -> diffusers DiT / VAE / scheduler
  -> utils/data.py:load_multi_datasets_form_json()
  -> trainer.py:Trainer
  -> trainer.py:train_eval_by_iter()
```

而你最关心的 `delta_z` 路径则是：

```text
image_start, image_end
  -> RenderNet.encode()
  -> get_delta_action()
  -> delta_emb / delta_pooled
  -> delta_interpolation()
```

---

## 10. 你应该按什么顺序读代码

如果你是为了做研究，不建议一上来从 diffusion block 细节啃起。

推荐顺序：

1. [StaMo/train_renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/train_renderer.py)  
看训练入口是怎么拼起来的

2. [StaMo/stamo/renderer/model/renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/renderer.py)  
重点看 `encode`、`get_delta_action`、`delta_interpolation`

3. [StaMo/stamo/renderer/model/projector.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/projector.py)  
理解紧凑 token 是怎么来的

4. [StaMo/stamo/renderer/utils/data.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/utils/data.py)  
理解数据输入格式到底需要什么

5. [StaMo/stamo/renderer/trainer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/trainer.py)  
理解训练、评估、存图、存 checkpoint

6. 最后再看 [StaMo/stamo/renderer/model/backbone.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/backbone.py) 里的 DiT 细节  
这一步属于“加深理解”，不是最先必须做的

---

## 11. 结合你的课题，最该盯住哪些位置

从你的研究问题出发，最关键的不是“如何生成图像更清晰”，而是：

1. `encode()` 输出的 token 到底是不是稳定状态表示
2. `get_delta_action()` 里的差分能不能作为短时程状态变化证据
3. 这些差分能不能进一步映射为紧凑语义标签

所以你后面真正要扩展的地方，优先级应该是：

### 第一优先级

- [StaMo/stamo/renderer/model/renderer.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/renderer.py)
- [StaMo/stamo/renderer/model/projector.py](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/stamo/renderer/model/projector.py)

这里是 `z` 和 `delta_z` 的来源。

### 第二优先级

- 新增一个你自己的 `delta_z extraction` 脚本
- 新增一个 `delta_z -> compact semantics` 头

### 第三优先级

- 再去接文本过程链

如果一开始就直接把文本链和大模型融合塞进来，反而会把核心问题冲淡。

---

## 12. 这次本地跑通版和官方原版的关系

你要特别区分两件事：

### 官方原版想做的

- 用较大的视觉 backbone
- 用 SD3 transformer + VAE
- 做更真实的重建和插值

### 我这次帮你补的

- 一个最小可运行版本
- 让你本机上先有“能跑、能存图、能存 checkpoint、能做 delta 差分”的版本

所以这次运行结果说明的是：

- 代码通路已打通
- 关键接口已可用

而不是：

- 论文结果已经被完全复现

---

## 13. 你现在应该怎么用这份代码

如果你下一步是继续研究 `delta_z`，建议这样做：

1. 先保留 [StaMo/configs/toy.yaml](/C:/Users/ryanf/Desktop/stamo_pro/StaMo/configs/toy.yaml) 作为本地 smoke test 配置
2. 用真实数据再准备一版新的 `config`
3. 单独写一个脚本，只调用 `RenderNet.encode()` 和 `get_delta_action()`
4. 把输出保存成你自己的特征文件
5. 在这个基础上再做 compact semantics 预测

这会比直接在 `train_renderer.py` 里硬改实验逻辑更干净。

---

## 14. 一句话总结

如果只用一句话概括当前这些文件的工作机制，那就是：

**StaMo 当前仓库本质上是在学习一个“由图像压缩 token 构成的紧凑状态空间”，而你要研究的 `delta_z`，正是这个状态空间中前后状态之差。**

所以你后面最值得抓住的代码，不是生成器最后画出了什么，而是：

- 图像怎样被编码成 token
- token 怎样被压缩
- 前后 token 怎样做差
- 这个差如何转成你要的过程语义

