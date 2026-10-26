#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
push_all.py — 把本地工作树同步到 GitHub（走 Contents API，不走 git 协议）

为什么需要它：本机 Git 走 Windows 系统代理（WinINET），当代理进程未运行时
`git push` 全部失败，但 `api.github.com` 直连正常。本脚本用 GitHub Contents API
（PUT /repos/{owner}/{repo}/contents/{path}）直接写文件，绕开 git 协议。

用法：
    python push_all.py --repo Chesi-AI-Models/Chesi-AI --branch main
    python push_all.py --repo Lianyilin-rgb/Chesi-AI --branch main      # 用默认 gh 身份
    python push_all.py --repo owner/name --branch main --dry-run        # 只看会改什么

说明：
  · 逐文件比较远端 sha 与本地内容，内容相同则跳过（幂等）
  · 需要对目标仓库有写权限的凭据（gh auth 登录的身份，或 GH_TOKEN 环境变量）
  · 不会删除远端文件（删除需人工确认）
"""
import argparse
import base64
import fnmatch
import json
import os
import subprocess
import sys

# 与 .gitignore 对齐的排除规则
SKIP_PATTERNS = [
    "*.pt", "*.pth", "*.ckpt", "*.safetensors", "*.jsonl", "*.parquet",
    "*.log", "*.pyc", "__pycache__/*", ".git/*", "out/*", "runs/*",
    "data/*", "corpus/*", "datasets/*", ".DS_Store",
]
SKIP_DIRS = {".git", "__pycache__", "out", "runs", "data", "corpus", "datasets", ".idea"}


def api(method, path, data=None, repo_ok=True):
    cmd = ["gh", "api", path, "-X", method]
    if data is not None:
        tmp = os.path.join(os.getcwd(), ".push_all_payload.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        cmd += ["--input", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        if repo_ok:
            return None
        print(f"ERR {method} {path}: {r.stderr.strip()[:200]}")
        sys.exit(1)
    return r.stdout


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fn, pat) for pat in SKIP_PATTERNS):
                continue
            yield full, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ap.add_argument("--message", default="chore: sync working tree via Contents API")
    ap.add_argument("--mirror-of", default=None,
                    help="若本仓库是镜像，填主仓库地址（owner/name）；同步时会自动在 README 顶部插入镜像提示")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mirror_note = {}
    if args.mirror_of:
        canonical = f"https://github.com/{args.mirror_of}"
        mirror_note = {
            "README.md": f"> **本仓库为镜像**：主仓库在 <{canonical}>，后续更新以主仓库为准。\n",
            "README.en.md": f"> **This repo is a mirror**: the canonical repo is <{canonical}>.\n",
        }

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"根目录不存在: {root}")
        return 2
    print(f"目标仓库: {args.repo}@{args.branch}\n源目录: {root}\n")

    created = updated = skipped = failed = 0
    for full, rel in walk(root):
        with open(full, "rb") as f:
            content = f.read()
        # 镜像仓库：README 顶部自动插入/保留镜像提示
        if rel in mirror_note:
            try:
                txt = content.decode("utf-8")
            except UnicodeDecodeError:
                txt = None
            if txt is not None:
                marker = "本仓库为镜像" if rel == "README.md" else "This repo is a mirror"
                if marker not in txt:
                    lines = txt.split("\n")
                    for i, ln in enumerate(lines):
                        if ln.startswith("## "):
                            lines.insert(i, mirror_note[rel].rstrip())
                            break
                    txt = "\n".join(lines)
                    content = txt.encode("utf-8")
        b64 = base64.b64encode(content).decode()
        cur = api("GET", f"repos/{args.repo}/contents/{rel}?ref={args.branch}")
        if cur:
            meta = json.loads(cur)
            remote_sha = meta.get("sha")
            if remote_sha:
                # 用 blob sha 粗比：内容相同则跳过（避免重复下载 base64）
                import hashlib
                git_blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if meta.get("sha") == git_blob:
                    print(f"  [skip] {rel}")
                    skipped += 1
                    continue
                payload = {"message": args.message, "content": b64, "sha": remote_sha, "branch": args.branch}
                action = "update"
            else:
                payload = {"message": args.message, "content": b64, "branch": args.branch}
                action = "create"
        else:
            payload = {"message": args.message, "content": b64, "branch": args.branch}
            action = "create"

        if args.dry_run:
            print(f"  [{action}] {rel}")
            created += action == "create"
            updated += action == "update"
            continue

        out = api("PUT", f"repos/{args.repo}/contents/{rel}", payload)
        if out:
            print(f"  [ok:{action}] {rel}")
            created += action == "create"
            updated += action == "update"
        else:
            print(f"  [FAIL] {rel}")
            failed += 1

    print(f"\n结果: 新建 {created} / 更新 {updated} / 跳过 {skipped} / 失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
