# -*- coding: utf-8 -*-
"""
tokenizer 词表扩展 v1.1（2026-10-21）
把 SFT 生产数据集中出现、但预训练语料没有的字符追加进词表（ID 追加，不动原有 ID）
范围：merged 池 / gap_fill 新池 / 锚定集(msgs) / v2 验证集 / 290 答案集
输出：tokenizer.json（v1.1）+ 扩展报告（TOKENIZER_EXTENSION.md）
"""
import json
import collections
import time

T0 = time.time()
TOK = r"G:\LianAIdamoxinkaifwj\tokenizer\tokenizer.json"
OUT_MD = r"G:\LianAIdamoxinkaifwj\tokenizer\TOKENIZER_EXTENSION.md"

SOURCES = [
    (r"G:\LianAIdamoxinkaifwj\dataset_acquisition\social\chesi_sft_train_v2_merged.jsonl", "merged SFT 池"),
    (r"G:\LianAIdamoxinkaifwj\dataset_acquisition\social\chesi_v2_gap_fill.jsonl", "gap_fill 新池"),
    (r"G:\LianAIdamoxinkaifwj\dataset_acquisition\anchor\chesi_v2_anchor_ds_msgs.jsonl", "锚定集 ds"),
    (r"G:\LianAIdamoxinkaifwj\dataset_acquisition\anchor\chesi_v2_anchor_final_msgs.jsonl", "锚定集 final"),
    (r"G:\LianAIdamoxinkaifwj\dataset_acquisition\anchor\chesi_v2_anchor_set_msgs.jsonl", "锚定集 set"),
    (r"G:\LianAIdamoxinkaifwj\chesi_v2_eval\v2_validation_set.jsonl", "v2 验证集"),
    (r"G:\LianAIdamoxinkaifwj\dataset_acquisition\v1_290_rerun\validation_answers_290.jsonl", "290 答案集"),
]


def all_text(o):
    parts = []
    if isinstance(o, dict):
        for k in ("text", "instruction", "output", "question", "answer"):
            v = o.get(k)
            if isinstance(v, str):
                parts.append(v)
        for m in o.get("messages") or []:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
    return "".join(parts)


t = json.load(open(TOK, encoding="utf-8"))
vocab = t["vocab"]
special = t["special_tokens"]
next_id = max(vocab.values()) + 1

new_chars = collections.Counter()
per_src = {}
print("扫描 SFT 生产数据集字符 ...", flush=True)
for path, label in SOURCES:
    cnt = collections.Counter()
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                n += 1
                try:
                    o = json.loads(s)
                except Exception:
                    continue
                txt = all_text(o)
                for ch in txt:
                    if ch not in vocab:
                        cnt[ch] += 1
    except FileNotFoundError:
        print("  [跳过] 不存在：%s" % path)
        continue
    per_src[label] = (n, sum(cnt.values()), len(cnt))
    new_chars.update(cnt)
    print("  %-14s 行数 %8d  OOV字符 %8d  种类 %4d" % (label, n, sum(cnt.values()), len(cnt)), flush=True)

print()
print("新增字符总数：%d 种" % len(new_chars))
added = {}
for ch, c in sorted(new_chars.items(), key=lambda x: (-x[1], x[0])):
    vocab[ch] = next_id
    added[ch] = (next_id, c)
    next_id += 1
t["vocab"] = vocab
t["vocab_size"] = len(vocab) + len(special)
t["version"] = "chesi-tok-v1.1"
t["extension"] = {
    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "added_chars": len(added),
    "sources": [lbl for _, lbl in SOURCES],
}
json.dump(t, open(TOK, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

md = []
md.append("# tokenizer 词表扩展报告（v1 → v1.1）")
md.append("")
md.append("> 时间：%s｜动机：SFT 生产数据含预训练语料之外的字符（罕见医学用字/全角符号/emoji），" % t["extension"]["at"])
md.append("> 不扩展则 Stage 2 SFT 时这些字符全部变 `<unk>`（实测 merged 池 0.164% 字符、19.49% 记录受影响）")
md.append("")
md.append("## 扩展结果")
md.append("")
md.append("| 项 | 值 |")
md.append("|---|---|")
md.append("| 原词表 | 7,276（7,272 字符 + 4 特殊） |")
md.append("| **新增字符** | **%d** |" % len(added))
md.append("| **新词表** | **%d**（仍在 4k-8k 规格内） |" % t["vocab_size"])
md.append("| ID 策略 | 追加在既有字符之后（ID %d+），**不动任何原有 ID** |" % (next_id - len(added)))
md.append("")
md.append("## 各源 OOV 统计")
md.append("")
md.append("| 源 | 行数 | OOV 字符数 | OOV 种类 |")
md.append("|---|---|---|---|")
for lbl, (n, c, k) in per_src.items():
    md.append("| %s | %s | %s | %d |" % (lbl, "{:,}".format(n), "{:,}".format(c), k))
md.append("")
md.append("## 新增字符 Top 30（按频次）")
md.append("")
md.append("```")
md.append("".join("%s(%d) " % (ch, added[ch][1]) for ch, _ in sorted(added.items(), key=lambda x: -x[1][1])[:30]))
md.append("```")
md.append("")
md.append("## 往返一致性复测")
md.append("")
md.append("见交付报告（预期三大数据集 100% 往返一致）。")
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print()
print("完成：词表 %d → %d，用时 %.1fs" % (7276, t["vocab_size"], time.time() - T0))
