"""路线 A GER：Qwen3 QLoRA 纠错微调脚本（本地 6GB / 云端 24GB 通用）。

设计约定（与 infer_ger.py 严格同源）：
  - 训练样本的 prompt **完全复用** ``prompts.render_chat_prompt``（plain 模板、
    ``enable_thinking=False``），保证训练/推理输入分布一致；
  - 训练目标 = ``ref.lower() + "<|im_end|>"``：LibriSpeech ref 全大写无标点，
    N-best 候选在 prompt 中已统一小写，推理输出也是小写裸句——因此目标只做
    ``lower()``，**绝不**用 ``normalize_for_wer``（它会把 ``quilter's`` 改写成
    ``quilter is``、``don't→do not``、``0→zero``，与推理输出格式不一致）。
    WER 规范化只在评测阶段做；
  - Assistant-only loss：精确按 tokenizer 切分计算 prompt 长度，prompt 段
    labels 全部置 -100，仅「目标转写 + EOS」参与 loss；
  - 等效 batch = ``batch_size * grad_accum``；不 drop_last，epoch 末尾不足
    一个等效 batch 的累积会**强制 flush 成一步**（loss 按实际微批数归一），
    因此每 epoch 的 ``optimizer.step()`` 次数 = ceil(ceil(n/bs)/grad_accum)。

显存（Qwen3-1.7B NF4 + r=16 + gradient checkpointing + batch=1）：
    ~1.2GB 量化权重 + 优化器/梯度 + 512 token 激活，本地 4.5GB 安全线内。

用法：
    # 本地 NF4 冒烟（10-15 步，输出到 /tmp 不污染仓库）
    python route_a_ger/train_qlora.py --nbest outputs/nbest/dev-clean.jsonl \
        --max_steps 15 --grad_accum 4 --max_seq_len 512 \
        --output_dir /tmp/qlora_smoke

    # 云端 A10 24G 正式训练（bf16 + 全量 N-best）
    python route_a_ger/train_qlora.py --nbest outputs/nbest/train-clean-100.jsonl \
        --model Qwen/Qwen3-1.7B --epochs 3 --bf16 \
        --output_dir outputs/qlora/ger-qwen3-1.7b-train-clean-100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.profiler import cuda_cleanup, query_gpu_used_mb
from route_a_ger.prompts import SYSTEM_PROMPT, render_chat_prompt

logger = logging.getLogger("train_qlora")

IGNORE_INDEX = -100
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# 样本构造（纯函数，便于 CPU 单测）
# ---------------------------------------------------------------------------
def build_target(ref: str) -> str:
    """训练目标文本：ref 小写 + <|im_end|>（与推理输出格式一致）。

    只做 lower()/strip()，不做 EnglishTextNormalizer——理由见模块 docstring。
    """
    return ref.strip().lower() + "<|im_end|>"


def encode_example(
    tokenizer,
    prompt: str,
    target: str,
    max_seq_len: int,
) -> Optional[dict]:
    """把 (prompt, target) 编码成带 assistant-only labels 的训练样本。

    Args:
        tokenizer: 需支持 ``tokenizer(text, add_special_tokens=False)["input_ids"]``
        prompt: 已渲染的 chat prompt（末尾是 assistant 起始头）
        target: ``build_target`` 产物（目标转写 + <|im_end|>）
        max_seq_len: 总序列上限（prompt + target）

    Returns:
        ``{"input_ids": [...], "labels": [...], "attention_mask": [...],
           "truncated": bool}``；样本不可用时返回 None。

    截断策略：
        - target 本身超长 -> None（无法在不破坏标签的前提下塞入）；
        - prompt+target 超长 -> 从 **左侧** 截断 prompt，保留末尾 assistant
          起始头与用户指令，target 始终完整。
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + target, add_special_tokens=False)["input_ids"]

    # 边界一致性：BPE 在 prompt 末尾 "\n\n" 与 target 首词之间不应发生跨边界
    # 合并。若 tokenizer 行为异常则跳过该样本，绝不产生错位标签。
    if full_ids != prompt_ids + target_ids:
        return None

    if len(target_ids) >= max_seq_len:
        return None

    truncated = False
    if len(full_ids) > max_seq_len:
        keep = max_seq_len - len(target_ids)
        prompt_ids = prompt_ids[-keep:] if keep > 0 else []
        truncated = True

    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "truncated": truncated,
    }


def collate_examples(batch: List[dict], pad_token_id: int) -> dict:
    """右侧 padding 的 collator（batch=1 时无 padding；训练态因果 LM 用右侧）。"""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad)
        attention.append([1] * len(x["input_ids"]) + [0] * pad)
        labels.append(x["labels"] + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def load_nbest(path: Path, max_samples: Optional[int] = None) -> List[dict]:
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    return records[:max_samples] if max_samples else records


def build_dataset(
    records: List[dict],
    tokenizer,
    template: str,
    max_seq_len: int,
    use_scores: bool = False,
) -> tuple[List[dict], dict]:
    """把 N-best 记录编码为训练样本，并统计跳过原因。"""
    examples: List[dict] = []
    stats = {"total": len(records), "overlong": 0, "mismatch": 0, "truncated": 0}
    for rec in records:
        prompt = render_chat_prompt(
            tokenizer, rec["nbest"], SYSTEM_PROMPT, use_scores, template
        )
        ex = encode_example(
            tokenizer, prompt, build_target(rec["ref"]), max_seq_len
        )
        if ex is None:
            # 区分「target 超长」与「tokenizer 边界错位」：两者都跳过
            target_len = len(
                tokenizer(build_target(rec["ref"]), add_special_tokens=False)[
                    "input_ids"
                ]
            )
            if target_len >= max_seq_len:
                stats["overlong"] += 1
            else:
                stats["mismatch"] += 1
            logger.warning("跳过样本 %s（target_len=%d）", rec.get("id"), target_len)
            continue
        if ex.pop("truncated"):
            stats["truncated"] += 1
        examples.append(ex)
    return examples, stats


# ---------------------------------------------------------------------------
# 步数规划（纯函数，便于 CPU 单测）
# ---------------------------------------------------------------------------
def micro_batches_per_epoch(n_examples: int, batch_size: int) -> int:
    """每个 epoch 的 micro-batch 数（不 drop_last，尾部不足 batch_size 也算一批）。"""
    if n_examples <= 0:
        raise ValueError("n_examples 必须 > 0")
    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")
    return math.ceil(n_examples / batch_size)


def steps_per_epoch(n_examples: int, batch_size: int, grad_accum: int) -> int:
    """每个 epoch 的 ``optimizer.step()`` 次数（与训练循环严格一致）。

    epoch 末尾不足 ``grad_accum`` 的累积会**强制 flush 成一步**（不 drop_last、
    不丢梯度），故为 ``ceil(micro_batches / grad_accum)``；小样本场景下
    ``len(examples) < batch_size * grad_accum`` 时结果恒 >= 1，不会除零。
    """
    if grad_accum < 1:
        raise ValueError("grad_accum 必须 >= 1")
    m = micro_batches_per_epoch(n_examples, batch_size)
    return max(1, math.ceil(m / grad_accum))


def micro_counts_per_epoch(
    n_examples: int, batch_size: int, grad_accum: int
) -> List[int]:
    """每步实际累积的 micro-batch 数：前面的步为 ``grad_accum``，末步为余量。

    该序列之和恒等于 ``micro_batches_per_epoch``，训练循环按此除 loss，
    保证尾部 flush 步的梯度尺度与整步一致（不会被 grad_accum 除得过小）。
    """
    m = micro_batches_per_epoch(n_examples, batch_size)
    n_steps = steps_per_epoch(n_examples, batch_size, grad_accum)
    counts = [grad_accum] * (n_steps - 1)
    counts.append(m - grad_accum * (n_steps - 1))
    return counts


def iter_step_targets(
    n_examples: int,
    batch_size: int,
    grad_accum: int,
    n_epochs: int,
    total_steps: int,
) -> List[int]:
    """按 epoch 顺序给出每一步应累积的 micro-batch 数，共 ``total_steps`` 项。

    训练循环直接消费该序列，因此 ``len(list(iter_step_targets(...)))`` 就是
    实际 ``optimizer.step()`` 次数，与 scheduler 的 ``total_steps`` 严格一致。
    """
    counts = micro_counts_per_epoch(n_examples, batch_size, grad_accum)
    targets: List[int] = []
    for _ in range(max(0, n_epochs)):
        for c in counts:
            if len(targets) >= total_steps:
                return targets
            targets.append(c)
    return targets


# ---------------------------------------------------------------------------
# 显存采样（nvidia-smi 进程级口径，后台线程）
# ---------------------------------------------------------------------------
class VramSampler(threading.Thread):
    """后台按固定间隔采样 nvidia-smi 已用显存，记录峰值（MB）。"""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self.peak_used_mb: Optional[float] = None

    def run(self) -> None:
        while not self._stop_evt.is_set():
            used = query_gpu_used_mb()
            if used is not None:
                self.peak_used_mb = max(self.peak_used_mb or 0.0, used)
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=5)


# ---------------------------------------------------------------------------
# 模型与训练
# ---------------------------------------------------------------------------
def _load_model(args):
    """加载 4-bit NF4 底座 + 注入 LoRA，返回 (model, tokenizer)。"""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    logger.info(
        "加载底座 %s | NF4 双量化 | compute_dtype=%s",
        args.model, "bf16" if args.bf16 else "fp16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        dtype=compute_dtype,  # transformers 4.57 新参数名（torch_dtype 已弃用）
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"  # 训练态右侧 padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 冻结底座 + 梯度检查点（use_reentrant=False 兼容 bnb 4bit 与 PEFT）
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def train(args) -> int:
    from transformers import get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 显存基线：模型加载前采样（WSL2 下含 Windows 侧常驻占用，差分口径抵消）
    baseline_used = query_gpu_used_mb()
    sampler = VramSampler()
    sampler.start()
    t_load0 = time.perf_counter()
    model, tokenizer = _load_model(args)
    load_sec = time.perf_counter() - t_load0
    logger.info("模型加载完成，耗时 %.1fs", load_sec)

    records = load_nbest(args.nbest, args.max_samples)
    examples, data_stats = build_dataset(
        records, tokenizer, args.template, args.max_seq_len, args.use_scores
    )
    if not examples:
        raise RuntimeError("没有可用训练样本（检查 --max_seq_len 与数据路径）")
    logger.info(
        "训练样本 %d/%d | 跳过 overlong=%d mismatch=%d | 左截断=%d",
        len(examples), data_stats["total"], data_stats["overlong"],
        data_stats["mismatch"], data_stats["truncated"],
    )

    pad_id = tokenizer.pad_token_id
    trainable = [p for p in model.parameters() if p.requires_grad]
    # 注意：bnb Params4bit 以 packed uint8 存储，sum(p.numel()) 会少算一半；
    # 用 peft 的 num_parameters()（= print_trainable_parameters 口径）取真实量。
    n_trainable = sum(p.numel() for p in trainable)
    n_total = model.num_parameters() if hasattr(model, "num_parameters") else (
        sum(p.numel() for p in model.parameters())
    )

    from bitsandbytes.optim import PagedAdamW8bit

    optimizer = PagedAdamW8bit(trainable, lr=args.lr)
    # 每 epoch 的 optimizer.step() 次数：ceil(ceil(n/bs) / grad_accum)，尾部强制 flush
    spe = steps_per_epoch(len(examples), args.batch_size, args.grad_accum)
    total_steps = args.max_steps or int(math.ceil(spe * args.epochs))
    total_steps = max(1, total_steps)
    # 轮数：max_steps 优先（可跨多轮取满，如 50 样本 / grad_accum=4 仅 12 步，
    # 要跑 15 步需进入第 2 轮）；否则按 epochs（允许小数，向上取整）。
    if args.max_steps:
        n_epochs = max(1, math.ceil(total_steps / spe))
    else:
        n_epochs = max(1, math.ceil(args.epochs))
    # 训练循环按该序列消费：其长度即真实 optimizer.step() 次数，
    # 与 scheduler 的 total_steps 严格一致（不再依赖 micro 计数跨轮结转）。
    step_targets = iter_step_targets(
        len(examples), args.batch_size, args.grad_accum, n_epochs, total_steps
    )
    total_steps = len(step_targets)
    warmup = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=not args.bf16)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()

    def _epoch_batches():
        order = list(range(len(examples)))
        random.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            yield collate_examples([examples[i] for i in idx], pad_id)

    step = 0
    micro = 0          # 当前步已累积的 micro-batch 数
    micro_target = 0   # 当前步应累积的 micro-batch 数（末步可能 < grad_accum）
    step_losses: List[float] = []
    accum_loss = 0.0
    t_train0 = time.perf_counter()
    for _epoch in range(n_epochs):
        for batch in _epoch_batches():
            if step >= total_steps:
                break
            if micro == 0:
                micro_target = step_targets[step]
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"非有限 loss: {loss.item()}")
            scaler.scale(loss / micro_target).backward()
            accum_loss += loss.item()
            micro += 1
            if micro < micro_target:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            step += 1
            avg_loss = accum_loss / micro
            step_losses.append(avg_loss)
            accum_loss = 0.0
            micro = 0
            # 3 步滑动平均：单步受样本难度影响波动大，趋势看平滑值
            ma3 = sum(step_losses[-3:]) / len(step_losses[-3:])
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                logger.info(
                    "step %d/%d | loss=%.4f | ma3=%.4f | lr=%.2e",
                    step, total_steps, avg_loss, ma3, scheduler.get_last_lr()[0],
                )
        if step >= total_steps:
            break

    train_sec = time.perf_counter() - t_train0
    model.eval()
    torch_peak = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else None
    )
    sampler.stop()
    smi_peak = None
    if sampler.peak_used_mb is not None and baseline_used is not None:
        smi_peak = max(sampler.peak_used_mb - baseline_used, 0.0)

    # 仅保存轻量 LoRA adapter + 摘要（tokenizer 复用底座缓存，不重复落盘）
    model.save_pretrained(str(out_dir))

    summary = {
        "model": args.model,
        "nbest": str(args.nbest),
        "output_dir": str(out_dir),
        "lora": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "trainable_params": n_trainable,
            "total_params": n_total,
            "trainable_pct": round(100.0 * n_trainable / max(n_total, 1), 4),
        },
        "train": {
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "grad_accum": args.grad_accum,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "dtype": "bf16" if args.bf16 else "fp16",
            "template": args.template,  # 推理必须用同模板才能对齐分布
            "num_examples": len(examples),
            **data_stats,
        },
        "result": {
            "initial_loss": round(step_losses[0], 6) if step_losses else None,
            "final_loss": round(step_losses[-1], 6) if step_losses else None,
            # 末 3 步均值：单步受样本难度影响波动大，趋势以此为准
            "final_loss_ma3": (
                round(sum(step_losses[-3:]) / len(step_losses[-3:]), 6)
                if step_losses else None
            ),
            "steps": step,
            "load_sec": round(load_sec, 2),
            "train_sec": round(train_sec, 2),
            "peak_vram_mb_torch": round(torch_peak, 1) if torch_peak else None,
            "peak_vram_mb_smi": round(smi_peak, 1) if smi_peak is not None else None,
            "loss_curve": [round(x, 6) for x in step_losses],
        },
    }
    with open(out_dir / "qlora_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(
        "训练完成：%d 步 | loss %.4f -> %.4f | Peak VRAM(torch)=%.0fMB "
        "| 耗时 %.1fs | adapter 已保存到 %s",
        step,
        summary["result"]["initial_loss"],
        summary["result"]["final_loss"],
        torch_peak or 0.0,
        train_sec,
        out_dir,
    )

    del model
    cuda_cleanup()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen3 QLoRA GER 纠错微调")
    p.add_argument("--nbest", type=Path, required=True, help="N-best 缓存 JSONL")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--template", default="plain", choices=["plain", "numbered"])
    p.add_argument("--use_scores", action="store_true")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--output_dir", default="outputs/qlora/ger-qwen3-1.7b-lora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES,
        help="LoRA 注入模块（默认 q/k/v/o/gate/up/down_proj）",
    )
    dtype = p.add_mutually_exclusive_group()
    dtype.add_argument(
        "--fp16", action="store_true", help="fp16 计算精度（本地默认）"
    )
    dtype.add_argument(
        "--bf16", action="store_true", help="bf16 计算精度（云端 A10 推荐）"
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.grad_accum < 1 or args.batch_size < 1:
        raise SystemExit("--grad_accum / --batch_size 必须 >= 1")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
