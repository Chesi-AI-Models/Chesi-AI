# -*- coding: utf-8 -*-
"""
澈思自训 tokenizer v1 训练器（从零，纯 Python 标准库，零第三方依赖）
- 全语料流式字符频率扫描（chesi_10b_corpus.jsonl，117 万行 / 23.8 亿字符）
- 字符级词表 + 自适应频率阈值（目标 4k-8k，对齐训练计划 Stage 1 规格）
- 特殊 token：<pad>=0 <unk>=1 <bos>=2 <eos>=3
- 输出 tokenizer.json + 训练报告
"""
import json
import collections
import time
import os

T0 = time.time()
CORPUS = r"H:\ChesiAI-Data\chesi_pretrain_corpus\chesi_10b_corpus.jsonl"
OUTDIR = r"G:\LianAIdamoxinkaifwj\tokenizer"
os.makedirs(OUTDIR, exist_ok=True)

SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

print("[1/3] 全语料字符频率扫描（流式）...", flush=True)
freq = collections.Counter()
n_docs = 0
n_chars = 0
bad = 0
with open(CORPUS, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        try:
            o = json.loads(s)
        except Exception:
            bad += 1
            continue
        t = o.get("text") if isinstance(o, dict) else None
        if not isinstance(t, str) or not t:
            continue
        n_docs += 1
        n_chars += len(t)
        freq.update(t)
print("    文档 %s 行 | 字符 %s | 唯一字符 %d | JSON失败 %d | 用时 %.1fs" % (
    "{:,}".format(n_docs), "{:,}".format(n_chars), len(freq), bad, time.time() - T0), flush=True)

print("[2/3] 构建词表（自适应阈值，目标 4k-8k）...", flush=True)
# 找最小阈值使词表（不含特殊）<= 8000
threshold = 1
for T in (1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 500, 800, 1200, 2000, 3500, 5000, 8000, 12000, 20000, 35000):
    vsize = sum(1 for c, n in freq.items() if n >= T)
    if vsize <= 8000:
        threshold = T
        break
kept = {c: n for c, n in freq.items() if n >= threshold}
covered_chars = sum(kept.values())
coverage = covered_chars / max(n_chars, 1)
# 词表按频率降序分配 ID（特殊 token 占 0-3）
vocab = dict(SPECIAL)
nid = len(SPECIAL)
for c, n in sorted(kept.items(), key=lambda x: (-x[1], x[0])):
    vocab[c] = nid
    nid += 1
print("    阈值=%d | 词表大小=%d（含 4 特殊）| 覆盖率=%.4f%% | 未覆盖字符=%s" % (
    threshold, len(vocab), coverage * 100,
    "{:,}".format(n_chars - covered_chars)), flush=True)

print("[3/3] 落盘 tokenizer.json ...", flush=True)
tok = {
    "model_type": "char-level",
    "version": "chesi-tok-v1",
    "trained_on": "chesi_10b_corpus.jsonl",
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "corpus_stats": {"docs": n_docs, "chars": n_chars, "unique_chars": len(freq)},
    "special_tokens": SPECIAL,
    "freq_threshold": threshold,
    "coverage": coverage,
    "vocab_size": len(vocab),
    "vocab": vocab,
}
with open(os.path.join(OUTDIR, "tokenizer.json"), "w", encoding="utf-8") as f:
    json.dump(tok, f, ensure_ascii=False, separators=(",", ":"))

top50 = sorted(kept.items(), key=lambda x: -x[1])[:50]
rep = []
rep.append("# 澈思自训 tokenizer v1 训练报告")
rep.append("")
rep.append("> 生成时间：%s｜训练器：`tokenizer/train_tokenizer.py`（纯 Python 标准库，零第三方依赖）" % tok["trained_at"])
rep.append("")
rep.append("## 规格")
rep.append("")
rep.append("| 项 | 值 |")
rep.append("|---|---|")
rep.append("| 类型 | 字符级（char-level） |")
rep.append("| 训练语料 | `H:\\ChesiAI-Data\\chesi_pretrain_corpus\\chesi_10b_corpus.jsonl` |")
rep.append("| 语料规模 | %s 文档 / %s 字符 |" % ("{:,}".format(n_docs), "{:,}".format(n_chars)))
rep.append("| 唯一字符 | %s |" % "{:,}".format(len(freq)))
rep.append("| 频率阈值 | %d |" % threshold)
rep.append("| **词表大小** | **%d**（含 4 特殊 token） |" % len(vocab))
rep.append("| 特殊 token | <pad>=0 / <unk>=1 / <bos>=2 / <eos>=3 |")
rep.append("| **字符覆盖率** | **%.4f%%**（未覆盖 %s 字符） |" % (coverage * 100, "{:,}".format(n_chars - covered_chars)))
rep.append("| 压缩比 | ~1.0 token/字符（字符级固有） |")
rep.append("")
rep.append("## Top 50 高频字符")
rep.append("")
rep.append("```")
rep.append("".join(c for c, _ in top50))
rep.append("```")
rep.append("")
rep.append("## 判定")
rep.append("")
if 4000 <= len(vocab) <= 8000 and coverage > 0.995:
    rep.append("✅ **词表落在计划规格 4k-8k 区间，覆盖率 >99.5%，可直接用于 Stage 1（10M 字符级模型）。**")
elif len(vocab) < 4000:
    rep.append("⚠️ 词表 <4k（语料字符多样性低于预期）——仍可用，但建议 Stage 2 前升级 BPE。")
else:
    rep.append("⚠️ 词表 >8k——需提高阈值重训。")
rep.append("")
rep.append("## 使用")
rep.append("")
rep.append("```python")
rep.append("from tokenizer.chesi_tokenizer import ChesiTokenizer")
rep.append("tk = ChesiTokenizer.load(r'G:\\LianAIdamoxinkaifwj\\tokenizer\\tokenizer.json')")
rep.append("ids = tk.encode('澈思很高兴认识你')      # -> [id, id, ...]")
rep.append("txt = tk.decode(ids)                    # 往返一致（覆盖范围内）")
rep.append("```")
rep.append("")
rep.append("> 升级路径：Stage 2（110M）前在此字符基座上训练 BPE 合并（目标 32k-64k），本 tokenizer 的 encode/decode 接口不变。")
with open(os.path.join(OUTDIR, "TOKENIZER_REPORT.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(rep) + "\n")

print()
print("完成：词表 %d | 覆盖率 %.4f%% | 总用时 %.1fs" % (len(vocab), coverage * 100, time.time() - T0))
print("输出：%s" % os.path.join(OUTDIR, "tokenizer.json"))
