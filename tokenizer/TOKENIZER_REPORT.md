# 澈思自训 tokenizer v1 训练报告

> 生成时间：2026-10-21 09:56:38｜训练器：`tokenizer/train_tokenizer.py`（纯 Python 标准库，零第三方依赖）

## 规格

| 项 | 值 |
|---|---|
| 类型 | 字符级（char-level） |
| 训练语料 | `H:\ChesiAI-Data\chesi_pretrain_corpus\chesi_10b_corpus.jsonl` |
| 语料规模 | 1,174,481 文档 / 2,529,315,824 字符 |
| 唯一字符 | 7,272 |
| 频率阈值 | 1 |
| **词表大小** | **7276**（含 4 特殊 token） |
| 特殊 token | <pad>=0 / <unk>=1 / <bos>=2 / <eos>=3 |
| **字符覆盖率** | **100.0000%**（未覆盖 0 字符） |
| 压缩比 | ~1.0 token/字符（字符级固有） |

## Top 50 高频字符

```
的
。，和 一学这理是实论了识发在*方展有解与关中知#不践要能习者化域领研究深基进从可性持新法系现对
```

## 判定

✅ **词表落在计划规格 4k-8k 区间，覆盖率 >99.5%，可直接用于 Stage 1（10M 字符级模型）。**

## 使用

```python
from tokenizer.chesi_tokenizer import ChesiTokenizer
tk = ChesiTokenizer.load(r'G:\LianAIdamoxinkaifwj\tokenizer\tokenizer.json')
ids = tk.encode('澈思很高兴认识你')      # -> [id, id, ...]
txt = tk.decode(ids)                    # 往返一致（覆盖范围内）
```

> 升级路径：Stage 2（110M）前在此字符基座上训练 BPE 合并（目标 32k-64k），本 tokenizer 的 encode/decode 接口不变。
