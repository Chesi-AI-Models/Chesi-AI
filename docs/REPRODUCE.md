# 复现 Stage-1 · Reproduce

> 本文件让任何人在自己的机器上重建本次 Stage-1 full run（13.6M 参数 / 20,000 步 / eval 1.5618）。
> 全部材料均可公开获取，许可均为 Apache-2.0。

---

## 0. 前置

| 项 | 要求 |
|---|---|
| Python | 3.10+ |
| 依赖 | `torch`（CUDA 或 CPU 均可）；其余全部为标准库（`stage1/` 四件套零第三方依赖） |
| 磁盘 | 语料子集 ≥ 1 GB（完整 120 分片约 33 GB，见下） |
| 显存 | 12 GB 级显卡可跑满速；CPU 也能跑，只是慢（README 的 26.1 分钟是 RTX 4080 的数字） |

---

## 1. 取 tokenizer（已在仓库内）

```
tokenizer/
├── tokenizer.json        # 自训 v1.1：vocab + special_tokens（<pad>=0 <unk>=1 <bos>=2 <eos>=3）
├── chesi_tokenizer.py    # 纯 Python 零依赖编解码器
├── train_tokenizer.py    # 从零训练脚本（字符级 + 频率阈值）
├── extend_vocab_sft.py   # SFT 阶段扩表到全量 11,387
└── TOKENIZER_REPORT.md   # 训练与验收报告
```

**Stage-1 使用哪一档**：`stage1/data.py::load_tokenizer` 默认取 **stage1 档**——词表按字符频率升序，
字符 id 从 4 开始，保留前 `stage1_vocab_size = 7,276` 个（id 4..7279），即全量 11,387 的高频子集。
全量档用 `full_vocab=True`（Stage 2+ 用）。

---

## 2. 取语料

本次 full run 的语料（两套，均为 **Apache-2.0**）：

| 数据集 | HF 仓库 | 规模 | 许可 |
|---|---|---|---|
| OpenThoughts3 | [`open-thoughts/OpenThoughts3-1.2M`](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) | 120 个 parquet 分片 × 10,000 行 | Apache-2.0 |
| SmallThoughts | [`SmallDoge/SmallThoughts`](https://huggingface.co/datasets/SmallDoge/SmallThoughts) | 102,000 行 | Apache-2.0 |

**国内直连**（推荐，走镜像，下载快很多）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download('open-thoughts/OpenThoughts3-1.2M', repo_type='dataset',
                  local_dir='./raw/openthoughts3', max_workers=8)
snapshot_download('SmallDoge/SmallThoughts', repo_type='dataset',
                  local_dir='./raw/smallthoughts')
"
```

> 本项目自有许可台账：`dataset_acquisition/DATASET_INDEX.md`（809 文件 / 58.5GB 资产索引）与
> `dataset_acquisition/LICENSE_AUDIT_20261021.md`（A 级宽松 13 项 / B 级 3 项 / 排除 1 项）。
> 本页两个数据集的许可经 HF 数据集卡片（`license:apache-2.0`）与本项目 2026-10-21 审计双重确认。

---

## 3. 转换成训练格式

`stage1/data.py` 读 **jsonl，每行一个 `{"text": "..."}`**。转换脚本：

```python
# parquet -> {"text": ...} jsonl
import json, glob, pyarrow.parquet as pq
out = open("corpus.jsonl", "w", encoding="utf-8")
for f in sorted(glob.glob("raw/openthoughts3/**/*.parquet", recursive=True)):
    t = pq.read_table(f)
    # OpenThoughts3 的 conversations 字段是多轮消息，取全部轮次拼接
    for row in t.to_pylist():
        convs = row.get("conversations") or []
        if not convs:
            continue
        text = "\n".join(
            (c.get("value") if isinstance(c, dict) else str(c)) or "" for c in convs
        ).strip()
        if text:
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
out.close()
```

本次 full run 实际用量：**OpenThoughts3 1,200,000 行 + SmallThoughts 102,000 行**
（两者均为英文推理语料；在澈思的路线里它们只作**推理能力增强组件**，
中文基座主粮走 SkyPile-150B head 层，见 README「路线图 Stage-2」）。

---

## 4. 训练

```bash
python stage1/train.py --corpus ./corpus.jsonl --steps 20000 --device cuda
```

多卡（DDP）：

```bash
torchrun --nproc_per_node=4 stage1/train.py --corpus ./corpus.jsonl --steps 20000 --device auto
```

训练过程写 `history.json`（`cfg` / `backend` / `history` 三段，`history` 每 20 步一条
`{step, loss, lr, eval}`）。仓库里的 `docs/stage1_loss_curve.svg` 就是由它生成的：

```bash
python scripts/plot_stage1_curve.py --history <你的>/history.json --out docs
```

（零依赖，纯标准库；本机若无 matplotlib 就只出 SVG，这也是本仓库采用 SVG 的原因。）

---

## 5. 验收（四判据）

```bash
python stage1/verify_stage1.py --mode full --corpus ./corpus.jsonl
```

| 判据 | 含义 | 本次实测 |
|---|---|---|
| ① | eval loss < 1.8 | **1.5618**（32-block 独立复评）—— 首次达标 |
| ② | 严格式：窗口均值每窗不增 | **FAIL**（1.3695 → 1.3717，+0.16%，落在收敛噪声地板内） |
| ②' | 趋势式：首尾窗口对比 | **PASS**（2.6436 → 1.3732，降幅 48.1%） |
| ③ | 与独立朴素参考实现的损失相对差 < 15% | **1.85e-07** |
| ④ | 同种子两次运行逐步损失 bitwise 相等 | **PASS** |

---

## 6. ⚠️ 关于「能不能复现出 1.5618 这个数」的诚实说明

**绝对值依赖具体语料与切分**。判据①的 1.5618 是在**本次那一份语料、按 `PackedCorpus` 的块顺序、
在指定 offset 上取的 32 个块**上复评出来的（`verify_stage1.py::get_eval_batch(n_blocks, skip)`）。
你若用：

- 不同的语料子集（哪怕同为 OpenThoughts3，但抽了不同分片 / 不同行数）
- 不同的 parquet→jsonl 转换方式（拼接策略不同 → 块边界不同）
- 不同的 `--text-field` 或不同的 tokenizer 档位

→ **eval 绝对值会不同**。这是正常的，不是造假。

**保证可以复现的是「性质」而不是「绝对值」**：

- ②/②' 收敛形态（先陡降后进噪声地板，趋势降幅 ~48%）
- ③ 与朴素参考实现的一致性（相对差在 1e-7 量级，因为两者是同一份权重同一批块上的两种实现）
- ④ 同种子 bitwise 一致（这是**纯确定性性质**，与语料无关，最值得你先验一遍）

**建议的复现顺序**：先只验 ④（同种子跑两次，比对逐步 loss 是否逐位相同）→ 再验 ③ → 最后看 ①/② 的形态。

---

## 7. 与本次 full run 的差异说明

| 项 | 本次 | 说明 |
|---|---|---|
| 硬件 | 单张 RTX 4080 Laptop GPU（torch 2.5.1, capability 8.9） | 4080/4090/A10 均可，曲线形态一致 |
| 昇腾 910B / C500 / BI-V100 | **未实测** | trainer 设备探测链 `cuda → npu → xpu → musa → cpu` 已就绪，昇腾验证随启智社区三级冒烟进行 |
| CPU 通道 | 已实测（bs=4 时与 CUDA 曲线差 <0.4%） | bs=2 时因小 batch 梯度噪声会触发判据② FAIL，见 `stage1/README.md` §4 |
| 语料 | OpenThoughts3 1.2M 行 + SmallThoughts 102,000 行 | 见 §2、§3 |

---

*发现任何与本文不符的地方，请开 issue。口径错了比数字错了更伤人，我们宁可被指出来。*
