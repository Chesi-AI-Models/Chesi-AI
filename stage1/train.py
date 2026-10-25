# -*- coding: utf-8 -*-
"""
澈思 Stage-1 训练器（设备无关 + 多卡 + 断点续训）

设计目标（docs/OPENI_MULTIMODAL_PLAN.md 第 6 节）：
  - 一条命令适配 CPU / NVIDIA CUDA / 昇腾 NPU / CUDA 兼容国产栈（沐曦 MACA、天数等以其 torch 构建为准）
  - 多卡：torchrun 启动自动进 DDP；单卡直接 python 启动，行为完全一致
  - FP32 项目锁定（DECISION_LOG 2026-10-06）；--dtype bf16 需用户单独授权，会打 WARNING
  - 断点续训：checkpoint 含 model/optimizer/step/config；MinIO 自动上传在平台侧配置

用法：
  本地 CPU 冒烟：  python train.py --steps 120 --batch-size 4 --out checkpoints/smoke
  启智单卡：       python train.py --corpus /tmp/dataset/xxx.jsonl --steps 20000 --device auto
  启智多卡：       torchrun --nproc_per_node=4 train.py --corpus ... --device auto
  续训：           python train.py --resume checkpoints/run_x/ckpt_latest.pt --steps 20000
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from model import ChesiGPT
from data import PackedCorpus, load_tokenizer, make_batch

ROOT = Path(__file__).resolve().parent


# ---------- 设备解析（启智三卡适配核心） ----------

def resolve_device(spec: str = "auto") -> torch.device:
    """auto 探测链：cuda（NVIDIA + CUDA 兼容国产栈）→ npu（昇腾）→ xpu（Intel）→ musa（摩尔线程）→ cpu"""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    for name in ("npu", "xpu", "musa"):
        mod = getattr(torch, name, None)
        if mod is not None:
            try:
                if mod.is_available():
                    return torch.device(name)
            except Exception:
                pass
    return torch.device("cpu")


def backend_info(device: torch.device) -> dict:
    info = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["capability"] = list(torch.cuda.get_device_capability(device))
    else:
        info["gpu"] = "n/a (CPU 或国产后端，名称以各栈 nvidia-smi/npu-smi 等价工具为准)"
    return info


# ---------- 训练核心（verify_stage1.py 复用） ----------

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(p, 1.0))))


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith("bias") else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def train_run(cfg, log_fn=print):
    """执行一次训练，返回 dict(history=[{step, loss, lr, eval}], model, device, backend)"""
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))

    tk = load_tokenizer(cfg.tokenizer_json, stage1_vocab_size=cfg.stage1_vocab_size, full_vocab=cfg.full_vocab)
    model_vocab = tk.vocab_size if cfg.full_vocab else cfg.stage1_vocab_size + 4
    model = ChesiGPT(
        vocab_size=model_vocab, seq_len=cfg.seq_len, n_layer=cfg.n_layer,
        d_model=cfg.d_model, n_head=cfg.n_head, dropout=cfg.dropout,
    ).to(device)
    log_fn(f"[stage1] 模型参数（唯一计，tied head 只算一次）: {model.num_params_unique():,}")

    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None)

    opt = build_optimizer(model, cfg.lr, cfg.weight_decay)
    start_step = 0
    if cfg.resume and Path(cfg.resume).exists():
        ck = torch.load(cfg.resume, map_location=device)
        state = ck["model"] if "model" in ck else ck
        (model.module if is_ddp else model).load_state_dict(state)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        log_fn(f"[stage1] 续训自 step {start_step}")

    dtype = torch.float32
    if cfg.dtype == "bf16":
        log_fn("[stage1] ⚠️ WARNING：bf16 违反项目 FP32 锁定纪律（DECISION_LOG 2026-10-06），需用户单独授权！")
        dtype = torch.bfloat16
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype, enabled=(cfg.dtype == "bf16"))

    corpus = PackedCorpus(cfg.corpus, tk, seq_len=cfg.seq_len, text_field=cfg.text_field,
                          eval_stride=cfg.eval_stride, max_blocks=None)
    it = iter(corpus)
    buf_train, buf_eval = [], []
    history, t0 = [], time.time()
    running, nrun = 0.0, 0
    step_t0 = time.perf_counter()   # 段计时起点（step_ms 用）
    last_log_step = start_step

    def fill(n):
        nonlocal buf_train, buf_eval
        while len(buf_train) < n:
            try:
                block, is_eval = next(it)
            except StopIteration:
                return False
            (buf_eval if is_eval else buf_train).append(block)
        return True

    step = start_step
    while step < cfg.steps:
        if not fill(cfg.batch_size * cfg.grad_accum):
            log_fn(f"[stage1] 语料耗尽于 step {step}（共 {step} 步）")
            break
        lr = cosine_lr(step, cfg.steps, cfg.lr, cfg.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(cfg.grad_accum):
            batch = buf_train[: cfg.batch_size]
            del buf_train[: cfg.batch_size]
            x, y = make_batch(batch, device)
            with autocast_ctx:
                _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            running += float(loss.item()); nrun += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0 or step == cfg.steps:
            avg = running / max(nrun, 1); running, nrun = 0.0, 0
            el = time.time() - t0
            _now = time.perf_counter()
            _seg = step - last_log_step
            step_ms = (_now - step_t0) * 1000.0 / max(_seg, 1)
            step_t0 = _now
            last_log_step = step
            log_fn(f"[stage1] step {step}/{cfg.steps} loss {avg:.4f} lr {lr:.2e} "
                   f"elapsed {el/60:.1f}m step_ms {step_ms:.1f} dev={device}")
            history.append({"step": step, "loss": avg, "lr": lr, "eval": None})

        if step % cfg.eval_every == 0 and buf_eval:
            model.eval()
            with torch.no_grad():
                tot, cnt = 0.0, 0
                for b in buf_eval[: cfg.eval_batches]:
                    x, y = make_batch([b], device)
                    with autocast_ctx:
                        _, l = model(x, y)
                    tot += float(l.item()); cnt += 1
            model.train()
            log_fn(f"[stage1]   [eval @step {step}] loss {tot/max(cnt,1):.4f} ({cnt} blocks)")
            if history:
                history[-1]["eval"] = tot / max(cnt, 1)
            buf_eval = buf_eval[cfg.eval_batches:]

        if rank == 0 and step % cfg.ckpt_every == 0:
            save_ckpt(cfg, model, opt, step, is_ddp, tag="latest")

    if rank == 0:
        save_ckpt(cfg, model, opt, step, is_ddp, tag="final")
        out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump({"cfg": vars(cfg), "backend": backend_info(device), "history": history}, f,
                      ensure_ascii=False, indent=1)
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
    return {"history": history, "model": model, "device": device, "backend": backend_info(device)}


def save_ckpt(cfg, model, opt, step, is_ddp, tag="latest"):
    out_dir = Path(cfg.out); out_dir.mkdir(parents=True, exist_ok=True)
    state = (model.module if is_ddp else model).state_dict()
    torch.save({"model": state, "opt": opt.state_dict(), "step": step,
                "cfg": vars(cfg)}, out_dir / f"ckpt_{tag}.pt")


# ---------- CLI ----------

def build_parser():
    p = argparse.ArgumentParser(description="澈思 Stage-1 训练器（设备无关）")
    p.add_argument("--corpus", default=str(Path(r"H:\ChesiAI-Data\chesi_pretrain_corpus\chesi_10b_corpus.jsonl")),
                   help="jsonl 语料路径（支持逗号分隔/glob）")
    p.add_argument("--tokenizer-json", default=str(ROOT.parent / "tokenizer" / "tokenizer.json"))
    p.add_argument("--stage1-vocab-size", type=int, default=7276)
    p.add_argument("--full-vocab", action="store_true", help="用全量 11,387 词表（Stage 2+）")
    p.add_argument("--text-field", default="text")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=40)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-stride", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "smoke"))
    p.add_argument("--resume", default="")
    return p


if __name__ == "__main__":
    cfg = build_parser().parse_args()
    print(f"[stage1] backend: {json.dumps(backend_info(resolve_device(cfg.device)), ensure_ascii=False)}")
    print(f"[stage1] cfg: steps={cfg.steps} bs={cfg.batch_size}x{cfg.grad_accum} seq={cfg.seq_len} "
          f"lr={cfg.lr} dtype={cfg.dtype} device={cfg.device}")
    print(f"[stage1] 注意：冒烟可默认用 10b 语料；正式基座训练必须换开源人类语料（C+ 决策，DECISION_LOG 2026-10-21）")
    train_run(cfg)
