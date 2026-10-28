#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""docs/*.md → 同风格独立 HTML（零依赖，纯标准库）"""
import html, os, re, sys

CSS = r'''
:root{--bg:#fff;--bg-soft:#f8fafc;--bg-card:#fff;--fg:#0f172a;--fg-soft:#475569;--fg-dim:#64748b;
--line:#e2e8f0;--line-soft:#f1f5f9;--accent:#1d4ed8;--accent-soft:#eff6ff;--accent-fg:#1e40af;
--code-bg:#0f172a;--code-fg:#e2e8f0;--shadow:0 1px 2px rgba(15,23,42,.04),0 4px 16px rgba(15,23,42,.06)}
@media (prefers-color-scheme:dark){:root{--bg:#0a0e17;--bg-soft:#0f1520;--bg-card:#121a27;
--fg:#e8eef7;--fg-soft:#a8b6c9;--fg-dim:#7b8a9e;--line:#1e2a3a;--line-soft:#16202e;
--accent:#60a5fa;--accent-soft:#132033;--accent-fg:#93c5fd;--code-bg:#060a12;--code-fg:#c9d6e5;
--shadow:0 1px 2px rgba(0,0,0,.3),0 4px 16px rgba(0,0,0,.25)}}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
body{background:var(--bg);color:var(--fg);font:16px/1.8 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--bg) 90%,transparent);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
.nav{max-width:900px;margin:0 auto;padding:0 22px;display:flex;align-items:center;gap:14px;height:56px}
.brand{display:flex;align-items:center;gap:8px;font-weight:800;font-size:16px;color:var(--fg)}
.logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(145deg,#1e3a8a,#1d4ed8);color:#fff;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:800}
.nav .sp{flex:1}.nav a.back{font-size:14px;color:var(--fg-soft)}
main{max-width:900px;margin:0 auto;padding:44px 22px 80px}
h1{font-size:clamp(25px,4.4vw,34px);font-weight:800;letter-spacing:-.02em;line-height:1.3;margin:0 0 10px}
h2{font-size:clamp(19px,3vw,24px);font-weight:800;letter-spacing:-.015em;margin:44px 0 14px;padding-top:22px;border-top:1px solid var(--line)}
h2:first-of-type{border-top:0;padding-top:0;margin-top:32px}
h3{font-size:17px;font-weight:700;margin:28px 0 10px}
h4{font-size:15.5px;font-weight:700;margin:22px 0 8px}
p{margin:0 0 14px;color:var(--fg-soft);line-height:1.85}
strong,b{color:var(--fg);font-weight:700}
ul,ol{margin:0 0 16px 24px;color:var(--fg-soft);line-height:1.85}
li{margin-bottom:5px}
blockquote{border-left:3px solid var(--accent);background:var(--accent-soft);padding:13px 17px;border-radius:0 10px 10px 0;margin:16px 0;font-size:14.5px;color:var(--fg-soft)}
blockquote p:last-child{margin-bottom:0}
hr{border:0;border-top:1px solid var(--line);margin:32px 0}
code{background:var(--bg-soft);border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:.88em;color:var(--accent-fg);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--code-bg);color:var(--code-fg);border-radius:12px;padding:16px 18px;overflow-x:auto;font-size:13.5px;line-height:1.7;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin:0 0 18px;box-shadow:var(--shadow)}
pre code{background:none;border:0;padding:0;color:inherit;font-size:inherit}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);border-radius:12px;margin:0 0 20px;box-shadow:var(--shadow)}
table{width:100%;border-collapse:collapse;font-size:14px;min-width:480px}
th{text-align:left;padding:11px 14px;font-weight:700;font-size:12.5px;color:var(--fg-dim);letter-spacing:.04em;text-transform:uppercase;background:var(--bg-soft);border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:11px 14px;border-bottom:1px solid var(--line-soft);color:var(--fg-soft);vertical-align:top}
tr:last-child td{border-bottom:0}
footer{max-width:900px;margin:0 auto;padding:26px 22px 50px;border-top:1px solid var(--line);font-size:13px;color:var(--fg-dim);line-height:1.9}
@media(max-width:640px){main{padding:28px 18px 60px}}
'''

def inline(t):
    codes=[]
    def stash(m):
        codes.append(m.group(1)); return f"\x00CODE{len(codes)-1}\x00"
    t=re.sub(r"`([^`]+)`", stash, t)
    t=html.escape(t)
    t=re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t=re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", t)
    t=re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    for i,c in enumerate(codes):
        t=t.replace(f"\x00CODE{i}\x00", f"<code>{html.escape(c)}</code>")
    return t

def md_to_html(md, title):
    L=md.splitlines(); out=[]; i=0; n=len(L); stack=[]; para=[]
    def flush():
        if para: out.append("<p>"+inline(" ".join(para))+"</p>"); para.clear()
    def close():
        while stack: out.append(f"</{stack.pop()}>")
    while i<n:
        s=L[i].rstrip()
        if s.startswith("```"):
            flush(); close(); i+=1; buf=[]
            while i<n and not L[i].rstrip().startswith("```"): buf.append(L[i]); i+=1
            i+=1; out.append("<pre><code>"+html.escape("\n".join(buf))+"</code></pre>"); continue
        if not s.strip(): flush(); close(); i+=1; continue
        if s.lstrip().startswith("|") and i+1<n and re.match(r"^\s*\|[\s:|-]+\|\s*$", L[i+1]):
            flush(); close()
            def cells(r):
                r=r.strip(); r=r[1:] if r.startswith("|") else r; r=r[:-1] if r.endswith("|") else r
                return [c.strip() for c in r.split("|")]
            head=cells(s); i+=2; body=[]
            while i<n and L[i].lstrip().startswith("|"): body.append(cells(L[i])); i+=1
            t=['<div class="tw"><table><thead><tr>']+[f"<th>{inline(c)}</th>" for c in head]+["</tr></thead><tbody>"]
            for r in body: t.append("<tr>"+"".join(f"<td>{inline(c)}</td>" for c in r)+"</tr>")
            t.append("</tbody></table></div>"); out.append("".join(t)); continue
        m=re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush(); close(); lv=len(m.group(1))
            out.append(f"<h{lv}>{inline(m.group(2).strip())}</h{lv}>"); i+=1; continue
        if re.match(r"^\s*([-*_])\1{2,}\s*$", s):
            flush(); close(); out.append("<hr>"); i+=1; continue
        if s.lstrip().startswith(">"):
            flush(); close(); buf=[]
            while i<n and L[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", L[i])); i+=1
            out.append("<blockquote><p>"+inline(" ".join(x for x in buf if x.strip()))+"</p></blockquote>"); continue
        m=re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", s)
        if m:
            flush(); want="ul" if m.group(2) in "-*+" else "ol"
            if not stack or stack[-1]!=want:
                close(); out.append(f"<{want}>"); stack.append(want)
            out.append(f"<li>{inline(m.group(3).strip())}</li>"); i+=1; continue
        para.append(s.strip()); i+=1
    flush(); close()
    body="\n".join(out)
    fav="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='20' fill='%230f172a'/%3E%3Ctext x='50' y='68' font-size='54' text-anchor='middle' fill='%2360a5fa' font-family='sans-serif' font-weight='700'%3E澈%3C/text%3E%3C/svg%3E"
    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0f172a">
<title>{html.escape(title)} — 澈思 Chesi AI</title>
<link rel="icon" href="{fav}">
<style>{CSS}</style>
</head>
<body>
<header><div class="nav">
  <a href="index.html" class="brand"><span class="logo">澈</span>澈思 Chesi</a>
  <span class="sp"></span>
  <a class="back" href="index.html">← 返回首页</a>
</div></header>
<main>
{body}
</main>
<footer>
  代码与权重：<b>Apache-2.0</b>。本页所有数字均来自实测，带口径；未实测项已明确标注。<br>
  <a href="index.html">澈思 Chesi AI</a> ·
  <a href="https://github.com/Chesi-AI-Models/Chesi-AI">GitHub</a> ·
  <a href="https://huggingface.co/Chesi-AI-Models/Chesi-AI-stage1">HuggingFace</a>
</footer>
</body>
</html>
'''

def main():
    d = sys.argv[1] if len(sys.argv)>1 else "docs"
    for src,dst,title in [("REPRODUCE.md","REPRODUCE.html","复现指南"),
                          ("CORPUS_LEDGER.md","CORPUS_LEDGER.html","语料台账")]:
        sp=os.path.join(d,src)
        if not os.path.exists(sp): print("  skip:",sp); continue
        out=md_to_html(open(sp,encoding="utf-8").read(), title)
        # 保险：任何漏网的 HF 裸链一律改镜像（大陆可直连）
        out=out.replace("https://huggingface.co/datasets/","https://hf-mirror.com/datasets/")
        out=out.replace("https://huggingface.co/","https://hf-mirror.com/")
        dp=os.path.join(d,dst)
        open(dp,"w",encoding="utf-8",newline="\n").write(out)
        print(f"  OK {sp} -> {dp}  {len(out):,} chars")

if __name__=="__main__":
    main()
