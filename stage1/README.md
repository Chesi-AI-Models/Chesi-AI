# 澈思 Stage-1 训练器（从零预训练基座 · 设备无关）

> 最后更新：2026-10-21（冒烟验收全过）｜配套计划：`../docs/OPENI_MULTIMODAL_PLAN.md`
> 前置条件：tokenizer v1.1 双档（Stage 1 = 7,276 字符）+ 预训练语料 16.9 亿 token——均已在 2026-10-21 就绪

## 1. 这是什么

一个**设备无关**的字符级小 GPT 训练器 + 独立验收脚本。目标不是刷榜，是把
「从零预训练一条基座」的**工程链路**（数据流 → 模型 → 训练循环 → 验收判据 → 多卡续训）
在本地 CPU/4080 与启智社区三档国产卡（910B / 沐曦 C500 / 天数 BI-V100）上**用同一份代码跑通**。

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | ChesiGPT，6 层 / d=384 / 8 头 / seq 512 | 唯一参数 **13,639,680**（tied head 只算一次；标称「10M 级」为诚实近似） |
| 词表 | Stage 1 档 **7,280** = 7,276 高频字符 + 4 特殊 token | 10b 语料 100% 覆盖；Stage 2+ 可 `--full-vocab` 切 11,387 |
| 精度 | **FP32 锁定** | 违反 DECISION_LOG 2026-10-06 纪律；`--dtype bf16` 需用户单独授权并打 WARNING |
| 优化器 | AdamW（矩阵 wd 0.01 / 标量 0，betas 0.9/0.95）+ cosine LR（warmup + min 0.1）+ clip 1.0 | |
| 多卡 | torchrun 启动自动进 DDP（nccl/gloo）；单卡直接 python，行为一致 | 启智平台即 torchrun DDP 范式 |

## 2. 文件

| 文件 | 职责 |
|---|---|
| `model.py` | ChesiGPT：Pre-LN + 朴素显式因果注意力 + tied head + GPT-2 风格 scaled init |
| `data.py` | `load_tokenizer`（按 id 截断取 Stage 1 档）+ `PackedCorpus`（流式 jsonl → token 流 → 定长块）+ `make_batch` |
| `train.py` | 训练核心 `train_run`（设备解析 / DDP / 断点续训 / 历史记录）+ CLI |
| `verify_stage1.py` | 验收四判据（独立朴素参考实现对照，不 import 主模型前向） |

## 3. 快速开始

### 本地冒烟（已实测）

```bash
# 环境：miniconda3/envs/chesi-train（torch 2.5.1）
cd stage1

# CUDA（RTX 4080 Laptop 实测）
python verify_stage1.py --mode smoke --device cuda --steps 120 --batch-size 4

# CPU 通道（设备无关性证据；判据④内部也强制 CPU 跑 2 次）
python verify_stage1.py --mode smoke --device cpu --steps 120 --batch-size 4

# 只训练不验收
python train.py --steps 120 --batch-size 4 --device cuda --out checkpoints/smoke
```

### 启智社区（910B / C500 / BI-V100 同一份代码）

```bash
# 单卡（BI-V100 1 积分/时档，适合先冒烟）
python train.py --corpus /tmp/dataset/<subset>.jsonl --steps 20000 --device auto

# 多卡（torchrun DDP；C500/910B 档）
torchrun --nproc_per_node=4 train.py --corpus /tmp/dataset/<subset>.jsonl --steps 20000 --device auto

# 断点续训（checkpoint 含 model/opt/step/cfg）
python train.py --resume checkpoints/run_x/ckpt_latest.pt --steps 20000 --device auto

# 正式验收（对 full run 产物跑四判据）
python verify_stage1.py --mode full --ckpt checkpoints/run_x/ckpt_final.pt
```

**设备无关原理**：`resolve_device("auto")` 按 `cuda → npu → xpu → musa → cpu` 链探测。
沐曦 MXMACA / 天数等 CUDA 兼容栈只要其 torch 构建注册了 `cuda` 设备即自动命中；
昇腾走 `torch_npu` 的 `npu` 设备。模型/数据/损失不含任何后端专属调用。

**语料路径**：启智上数据集放 `/tmp/dataset/`（平台持久化）；代码放 `/tmp/code/`。
10b 语料 7.7GB 整包超平台 100GiB 存储上限，须**本地子集化后上传**
（如 Stage 1 取 50-100MB 子集；完整子集化口径见 OPENI_MULTIMODAL_PLAN.md §6）。

## 4. 验收四判据与实测结果（2026-10-21）

| 判据 | 口径 | CUDA bs=4 | CPU bs=4 | 结论 |
|---|---|---|---|---|
| ① eval loss < 1.8 | full 模式强校验；smoke 只报趋势 | 6.3707 | 6.4009 | 冒烟 120 步不可能达标，属预期；需 full run |
| ② 窗口均值非递增 + 无 NaN | 容差 1e-4 相对 | **PASS** 7.3401→6.2108 | **PASS** 7.3425→6.2144 | ✅ |
| ③ 独立朴素参考实现相对差 < 15% | 手写循环注意力/LN/GELU/CE，期望 ~1e-6 | **PASS** 6.85e-08 | **PASS** **0.00e+00**（bitwise） | ✅ |
| ④ 同种子 2 次逐步损失 bitwise 相等 | CPU 通道 | **PASS** | **PASS** | ✅ |

**冒烟总结论：硬判据（②③④）全过 + ① 趋势正常（loss 8.39 → 6.2 持续下降）。**

### 两个诚实的实测发现

1. **batch size 对判据②敏感（对照实验，非缺陷）**：CPU 通道 `--batch-size 2` 时 ② FAIL
   （窗口均值 6.5527→6.6310）。同一通道把 batch 提到 4 即 PASS。原因是 120 步 × 2 block 的
   梯度噪声让末段 ±0.3 波动超过了 1e-4 容差——**小 batch 短训练的固有声噪，不是代码 bug**
   （③④ 在 bs=2 下同样全过；CUDA bs=4 亦全过）。full run（万步级）② 不受此影响。
2. **CUDA vs CPU 曲线几乎重合**：bs=4 下首窗 7.3401 vs 7.3425、末窗 6.2108 vs 6.2144
   （差 <0.4%）——这是「设备无关」最硬的实证，也预示启智三卡上行为一致。

## 5. ⚠️ 语料纪律（C+ 决策，DECISION_LOG 2026-10-21）

- **冒烟/调试**可默认用 `chesi_10b_corpus.jsonl`（1,174,481 行 / 16.9 亿 token）。
- **正式基座预训练必须换 100% 开源人类语料**（10b 语料 98% 由 DeepSeek 生成，按 C+ 方案
  降级为「继续预训练/学科增强语料」并永久标注来源，不作基座）。
- 换语料只需改 `--corpus`，tokenizer/模型/训练循环不动。
- SFT / 人格 / 七步思维链阶段永远纯人工 + STaR 自产（蒸馏禁令维持）。

## 6. 验收通过后的下一步

1. Stage-1 正式语料子集（开源人类语料，≥50MB，许可审计后）→ 本地 full run 试跑
2. 启智三级冒烟：本地 → BI-V100（1 积分/时）→ C500（2 积分/时）
3. 澈思-1B 预训练（P1 阶段，积分预算 ~1,440，见 OPENI_MULTIMODAL_PLAN.md §4）
4. 图像模态 0.4B（P2）→ 七步链种子扩充 100→500 → 融资叙事 M2（澈思-1B 开源发布）
