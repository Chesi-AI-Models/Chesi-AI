# -*- coding: utf-8 -*-
"""
澈思 tokenizer v1 推理器（纯 Python 标准库，零第三方依赖）
配套 tokenizer.json（字符级，词表 7,276）
"""
import json


class ChesiTokenizer:
    def __init__(self, vocab, special):
        self.vocab = vocab            # char -> id
        self.special = special        # name -> id
        self.id2char = {i: c for c, i in vocab.items()}

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            t = json.load(f)
        return cls(t["vocab"], t["special_tokens"])

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.special["<bos>"])
        unk = self.special["<unk>"]
        v = self.vocab
        for ch in text:
            ids.append(v.get(ch, unk))
        if add_eos:
            ids.append(self.special["<eos>"])
        return ids

    def decode(self, ids, skip_special=True):
        out = []
        for i in ids:
            if skip_special and i < len(self.special):
                continue
            c = self.id2char.get(i)
            if c is not None:
                out.append(c)
        return "".join(out)

    def roundtrip_ok(self, text):
        return self.decode(self.encode(text)) == text


if __name__ == "__main__":
    import sys
    tk = ChesiTokenizer.load(r"G:\LianAIdamoxinkaifwj\tokenizer\tokenizer.json")
    print("vocab_size =", tk.vocab_size)
    for s in ["澈思很高兴认识你。", "利润率以进价为基数计算，售价 = 进价 × (1 + 利润率)。", "print('hello 世界')"]:
        ids = tk.encode(s)
        back = tk.decode(ids)
        print("OK " if back == s else "FAIL", "| chars=%d tokens=%d ratio=%.3f | %s..." % (
            len(s), len(ids), len(ids) / max(len(s), 1), s[:20]))
