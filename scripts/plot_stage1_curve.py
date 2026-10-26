#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_stage1_curve.py — 从训练历史生成 Stage-1 loss 曲线（纯标准库，零依赖）

用法：
    python plot_stage1_curve.py --history <history.json> --out docs

输入：verify_stage1.py / train.py 写出的 history.json，结构：
    {"cfg": {...}, "backend": {...}, "history": [{"step":20,"loss":...,"lr":...,"eval":null|float}, ...]}

输出：
    docs/stage1_loss_curve.svg   矢量图（GitHub README 原生支持）
    docs/stage1_loss_curve.png   位图（若本机有 matplotlib，额外产出）

口径声明（务必与 README 一致）：
    · 图中 eval 曲线 = 训练期内记录的 eval（末点 1.3694 @ step 20000）
    · 验收判据① 的 eval = verify_stage1.py 的独立复评（32 blocks held-out）= 1.5618
    · 两者为不同口径，均 < 阈值 1.8
"""
import argparse
import json
import os

# ---- 绘图参数 ----
W, H = 960, 520
ML, MR, MT, MB = 78, 210, 58, 88      # 左边距/右边距/上边距/下边距（底部留白给脚注）
BG = "#ffffff"
AXIS = "#94a3b8"
GRID = "#eef2f7"
TRAIN = "#93c5fd"
EVAL = "#1d4ed8"
THRESH = "#dc2626"
TEXT = "#0f172a"
SUB = "#475569"


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_svg(hist):
    steps = [r["step"] for r in hist]
    loss = [r["loss"] for r in hist]
    evs = [(r["step"], r["eval"]) for r in hist if r.get("eval") is not None]

    xmin, xmax = 0, max(steps)
    allv = loss + [v for _, v in evs]
    ymin, ymax = 0.9, max(allv) * 1.06

    pw, ph = W - ML - MR, H - MT - MB
    X = lambda s: ML + (s - xmin) / (xmax - xmin) * pw
    Y = lambda v: MT + (ymax - v) / (ymax - ymin) * ph

    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">')
    p.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

    # 标题
    p.append(f'<text x="{ML}" y="26" font-size="17" font-weight="600" fill="{TEXT}">'
             f'Chesi AI &#183; Stage-1 from-scratch pretraining</text>')
    p.append(f'<text x="{ML}" y="45" font-size="12" fill="{SUB}">'
             f'13,639,680 params &#183; 20,000 steps &#183; batch 4 &#183; seq 512 &#183; lr 6e-4 &#183; RTX 4080 (torch 2.5.1)</text>')

    # Y 轴刻度与网格
    y0 = 1.0
    while y0 <= ymax:
        if y0 >= ymin:
            y = Y(y0)
            p.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            p.append(f'<text x="{ML-10}" y="{y+4:.1f}" font-size="11" fill="{SUB}" text-anchor="end">{y0:.1f}</text>')
        y0 += 0.5

    # X 轴刻度
    for xv in range(0, xmax + 1, 5000):
        x = X(xv)
        p.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+ph}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{x:.1f}" y="{MT+ph+20}" font-size="11" fill="{SUB}" text-anchor="middle">{xv:,}</text>')

    # 坐标轴
    p.append(f'<line x1="{ML}" y1="{MT+ph}" x2="{ML+pw}" y2="{MT+ph}" stroke="{AXIS}" stroke-width="1.2"/>')
    p.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+ph}" stroke="{AXIS}" stroke-width="1.2"/>')
    p.append(f'<text x="{ML+pw/2}" y="{MT+ph+42}" font-size="12" fill="{TEXT}" text-anchor="middle">training step</text>')
    p.append(f'<text x="22" y="{MT+ph/2}" font-size="12" fill="{TEXT}" text-anchor="middle" transform="rotate(-90 22 {MT+ph/2})">loss</text>')

    # 阈值线 1.8（标签移到绘图区右上空白处 + 白色底衬，避免被曲线压住）
    yty = Y(1.8)
    p.append(f'<line x1="{ML}" y1="{yty:.1f}" x2="{ML+pw}" y2="{yty:.1f}" stroke="{THRESH}" stroke-width="1.4" stroke-dasharray="6,4"/>')
    tlx = ML + pw - 8
    p.append(f'<rect x="{tlx-158:.1f}" y="{yty-21:.1f}" width="162" height="17" rx="3" fill="#ffffff" opacity="0.92"/>')
    p.append(f'<text x="{tlx}" y="{yty-8:.1f}" font-size="11" fill="{THRESH}" font-weight="600" text-anchor="end">acceptance criterion &#8804; 1.8</text>')

    # train loss 折线
    tp = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(steps, loss))
    p.append(f'<polyline points="{tp}" fill="none" stroke="{TRAIN}" stroke-width="1.1" opacity="0.85"/>')

    # eval loss 折线
    ep = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in evs)
    p.append(f'<polyline points="{ep}" fill="none" stroke="{EVAL}" stroke-width="1.9"/>')

    # 末点标注（置于点的左下 + 白色底衬，避免压住曲线）
    last_s, last_v = evs[-1]
    lx, ly = X(last_s), Y(last_v)
    p.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" fill="{EVAL}"/>')
    tx, ty = lx - 14, ly + 22
    p.append(f'<rect x="{tx-160:.1f}" y="{ty-13:.1f}" width="164" height="17" rx="3" fill="#ffffff" opacity="0.92"/>')
    p.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="12" font-weight="600" fill="{EVAL}" text-anchor="end">'
             f'eval {last_v:.4f} @ step {last_s:,}</text>')

    # 右侧图例与说明
    lx0 = ML + pw + 22
    p.append(f'<text x="{lx0}" y="{MT+4}" font-size="12" font-weight="600" fill="{TEXT}">Legend</text>')
    p.append(f'<line x1="{lx0}" y1="{MT+22}" x2="{lx0+22}" y2="{MT+22}" stroke="{EVAL}" stroke-width="1.9"/>')
    p.append(f'<text x="{lx0+28}" y="{MT+26}" font-size="11.5" fill="{TEXT}">eval (training-time)</text>')
    p.append(f'<line x1="{lx0}" y1="{MT+40}" x2="{lx0+22}" y2="{MT+40}" stroke="{TRAIN}" stroke-width="1.1"/>')
    p.append(f'<text x="{lx0+28}" y="{MT+44}" font-size="11.5" fill="{TEXT}">train loss</text>')
    p.append(f'<line x1="{lx0}" y1="{MT+58}" x2="{lx0+22}" y2="{MT+58}" stroke="{THRESH}" stroke-width="1.4" stroke-dasharray="6,4"/>')
    p.append(f'<text x="{lx0+28}" y="{MT+62}" font-size="11.5" fill="{TEXT}">criterion 1.8</text>')

    p.append(f'<text x="{lx0}" y="{MT+92}" font-size="11.5" font-weight="600" fill="{TEXT}">Acceptance (verify_stage1.py)</text>')
    p.append(f'<text x="{lx0}" y="{MT+110}" font-size="11.5" fill="{TEXT}">eval loss 1.5618</text>')
    p.append(f'<text x="{lx0}" y="{MT+126}" font-size="10.5" fill="{SUB}">32-block independent re-eval</text>')
    p.append(f'<text x="{lx0}" y="{MT+146}" font-size="11.5" fill="{TEXT}">rel. diff 1.85e-07</text>')
    p.append(f'<text x="{lx0}" y="{MT+162}" font-size="11.5" fill="{TEXT}">bit-identical (2 runs)</text>')
    p.append(f'<text x="{lx0}" y="{MT+182}" font-size="10.5" fill="{SUB}">wall-clock 26.1 min</text>')
    p.append(f'<text x="{lx0}" y="{MT+198}" font-size="10.5" fill="{SUB}">on a single RTX 4080</text>')

    # 底部口径脚注
    p.append(f'<text x="{ML}" y="{H-16}" font-size="10.5" fill="{SUB}">'
             f'Curve: training-time eval from history.json (last point {last_v:.4f}). '
             f'Acceptance eval 1.5618 is verify_stage1.py&#39;s independent 32-block re-evaluation &#8212; different basis, same &#8804;1.8 result.</text>')

    p.append('</svg>')
    return "\n".join(p)


def try_png(hist, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[skip PNG] matplotlib unavailable: {e}")
        return
    steps = [r["step"] for r in hist]
    loss = [r["loss"] for r in hist]
    evs = [(r["step"], r["eval"]) for r in hist if r.get("eval") is not None]
    fig, ax = plt.subplots(figsize=(9.6, 4.7), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(steps, loss, color="#93c5fd", lw=1.1, label="train loss", zorder=2)
    ax.plot([s for s, _ in evs], [v for _, v in evs], color="#1d4ed8", lw=1.9,
            label="eval (training-time)", zorder=3)
    ax.axhline(1.8, color="#dc2626", lw=1.4, ls="--", label="acceptance criterion 1.8")
    ax.plot([evs[-1][0]], [evs[-1][1]], "o", color="#1d4ed8", ms=5, zorder=4)
    ax.annotate(f"eval {evs[-1][1]:.4f} @ step {evs[-1][0]:,}",
                xy=(evs[-1][0], evs[-1][1]), xytext=(-12, 14),
                textcoords="offset points", ha="right", fontsize=11,
                fontweight="bold", color="#1d4ed8")
    ax.set_title("Chesi AI · Stage-1 from-scratch pretraining", fontsize=15,
                 fontweight="600", loc="left", pad=22)
    ax.text(0, 1.015, "13,639,680 params · 20,000 steps · batch 4 · seq 512 · lr 6e-4 · RTX 4080 (torch 2.5.1)",
            transform=ax.transAxes, fontsize=10, color="#475569")
    ax.set_xlabel("training step", fontsize=11)
    ax.set_ylabel("loss", fontsize=11)
    ax.grid(axis="y", color="#eef2f7")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#94a3b8")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.text(0.012, 0.015,
             "Acceptance eval 1.5618 = verify_stage1.py independent 32-block re-evaluation (different basis, same <=1.8 result).",
             fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = os.path.join(outdir, "stage1_loss_curve.png")
    fig.savefig(path)
    print(f"[PNG] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.history, encoding="utf-8") as f:
        data = json.load(f)
    hist = data["history"]
    svg = build_svg(hist)
    spath = os.path.join(args.out, "stage1_loss_curve.svg")
    with open(spath, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"[SVG] {spath}  ({len(svg):,} bytes)")
    try_png(hist, args.out)


if __name__ == "__main__":
    main()
