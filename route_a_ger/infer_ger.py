"""路线 A：读取 N-best 缓存，经 LLM 后端批量生成式纠错并评测。

用法：
    # NF4 冒烟（dev-clean 50 条）
    python route_a_ger/infer_ger.py --nbest outputs/nbest/dev-clean.jsonl \
        --model Qwen/Qwen3-1.7B --quant_mode nf4 --max_samples 50

    # 三精度全量（test-clean）
    python route_a_ger/infer_ger.py --nbest outputs/nbest/test-clean.jsonl \
        --model Qwen/Qwen3-1.7B --quant_mode fp16

    # WSL2 下 Windows 侧常驻 ~1.4GB 致空闲显存贴着阈值边缘（fp16 峰值实际
    # ~3.6GB），默认预检放宽到 4.4GB；仍被拒时可用 --max_vram_gb 进一步放宽：
    python route_a_ger/infer_ger.py ... --quant_mode fp16 --max_vram_gb 4.0

    # 管线冒烟（无 GPU/模型/外部进程；Mock 回显 Top-1，WER 必须等于 beam-1 基线）
    python route_a_ger/infer_ger.py --nbest outputs/nbest/dev-clean.jsonl \
        --backend mock --template plain --exp_name ger-mock-dev-clean-50

    # 定制 llama.cpp（server 模式，模型常驻；TTFT/显存差分统计走此模式）
    python route_a_ger/infer_ger.py --nbest outputs/nbest/dev-clean.jsonl \
        --backend llamacpp --quant_mode q4_k_m --llamacpp_url http://127.0.0.1:8080

    # 挂载 QLoRA adapter（Task 1.6 产物；仅 hf_bnb 后端）。
    # 注意 --template 必须与训练时一致（train_qlora.py 默认 plain）
    python route_a_ger/infer_ger.py --nbest outputs/nbest/dev-clean.jsonl \
        --quant_mode nf4 --lora_path /tmp/qlora_smoke --template plain --max_samples 5

断点续跑（宿舍断电/分批跑）：
    首跑生成逐条 checkpoint（默认开启，outputs/<exp_name>/checkpoint.jsonl），
    Ctrl+C / 断电后重跑同一命令加 --resume 即跳过已完成样本。
    统计口径可加：总推理时长 = Σ逐条 gen_sec（模型加载/中断间隔不计入），
    RTF / Tokens/s 跨段不因分段统计失真；peak VRAM 跨段取 max。
    python route_a_ger/infer_ger.py ... --exp_name ger-1.7b-nf4-test-clean --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from common.llm_backends.hf_bnb import HFBNBackend
from common.llm_backends.llamacpp_custom import LlamaCppBackend
from common.llm_backends.mock import MockLLMBackend
from common.metrics import (
    append_summary_row,
    compute_wer,
    export_error_cases,
    export_predictions,
)
from common.profiler import (
    InferenceProfiler,
    current_peak_vram_mb,
    query_gpu_mem_mb,
    query_gpu_used_mb,
    query_process_vram_mb,
)
from common.text_norm import normalize_for_wer
from route_a_ger.prompts import (
    SYSTEM_PROMPT,
    dynamic_max_new_tokens,
    extract_answer,
    render_chat_prompt,
)

logger = logging.getLogger("infer_ger")


def load_nbest(path: Path, max_samples: int | None = None) -> List[dict]:
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    return records[:max_samples] if max_samples else records


# ---------------------------------------------------------------------------
# LoRA：训练/推理模板一致性 + 结果标签隔离
# ---------------------------------------------------------------------------
def resolve_lora_template(lora_path: str | Path) -> str | None:
    """读取 adapter 目录下 qlora_summary.json 的训练 template。

    文件不存在、JSON 损坏或无 ``train.template`` 字段（旧 adapter）返回 None。
    """
    summary = Path(lora_path) / "qlora_summary.json"
    if not summary.exists():
        return None
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("qlora_summary.json 读取失败，跳过模板校验: %s", summary)
        return None
    return (data.get("train") or {}).get("template")


def check_lora_template(lora_path: str | Path, template: str) -> str | None:
    """返回与当前 --template 冲突的训练模板；一致/无元数据返回 None。"""
    train_template = resolve_lora_template(lora_path)
    if train_template and train_template != template:
        return train_template
    return None


def quant_mode_label(quant_mode: str, lora_path: str | None) -> str:
    """评估标签：LoRA 推理与零样本严格隔离（如 nf4 -> nf4_qlora）。

    该标签同时进入默认 exp_name 与 summary.csv 的 quant_mode 字段，
    保证 LoRA 结果不会覆盖/混淆已有零样本档。
    """
    return f"{quant_mode}_qlora" if lora_path else quant_mode


def default_exp_name(
    model: str, quant_mode: str, split: str, lora_path: str | None
) -> str:
    """默认实验名；LoRA 场景带 ``_qlora`` 后缀，避免覆盖零样本目录。"""
    return f"ger-{model.split('/')[-1]}-{quant_mode_label(quant_mode, lora_path)}-{split}"


def _query_llamaserver_vram() -> float | None:
    """pgrep llama-server -> nvidia-smi compute-apps 抓进程显存（MB）。

    llamacpp server 模式的进程级口径：模型权重 + KV cache 全量。
    pgrep 找不到进程 / nvidia-smi 列不出（WSL2 限制）返回 None。
    """
    import subprocess

    try:
        pids = subprocess.run(
            ["pgrep", "-f", "llama-server"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()
    except Exception:  # noqa: BLE001 - pgrep 失败视为不可用
        return None
    for pid in pids:
        v = query_process_vram_mb(int(pid))
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------- checkpoint
def checkpoint_path(base: Path) -> Path:
    """逐条落盘的 checkpoint 文件路径（与产物同目录）。"""
    return base / "checkpoint.jsonl"


def load_checkpoint(path: Path) -> dict:
    """读回 checkpoint：{id -> 逐条记录}。最后半行损坏（断电截断）自动丢弃。"""
    done: dict = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("checkpoint 末行损坏，已丢弃（断电截断）: %s", path)
                break
            done[rec["id"]] = rec
    return done


def append_checkpoint(f, rec: dict) -> None:
    """单条追加 + flush + fsync：宿主断电后已写入行不丢（page cache 落盘）。"""
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def merge_stats(done: dict, stats: dict) -> dict:
    """把 checkpoint 历史段与本段的统计合并为全量口径（函数见模块 docstring）。

    - total_infer_sec 统一为 Σ逐条 gen_sec：历史段（checkpoint）+ 本段，
      模型加载、段间中断间隔、checkpoint fsync 开销均不计入——单段与
      续跑口径一致，RTF / Tokens/s 由其派生不失真；
    - total_audio_sec / total_new_tokens：逐条累加，跨段精确可加；
    - peak_vram_mb：跨段取 max（任一段为 None 时取另一段）；
    - ttft：取历史段首条记录（无则保留本段值）。
    """
    hist_gen_sec = sum(r["gen_sec"] for r in done.values())
    hist_audio = sum(r["duration"] for r in done.values())
    hist_tokens = sum(r["completion_tokens"] for r in done.values())
    total_gen = hist_gen_sec + stats.get("_cur_gen_sec", 0.0)
    total_audio = hist_audio + stats.get("_cur_audio_sec", 0.0)
    total_tokens = hist_tokens + stats.get("_cur_new_tokens", 0)
    merged = dict(stats)
    merged["total_infer_sec"] = round(total_gen, 3)
    merged["total_audio_sec"] = round(total_audio, 3)
    merged["total_new_tokens"] = total_tokens
    merged["rtf"] = round(total_gen / max(total_audio, 1e-6), 4)
    merged["tokens_per_sec"] = (
        round(total_tokens / total_gen, 2) if total_gen > 0 else 0.0
    )
    hist_vram = done_first_vram(done)
    cur_vram = stats.get("peak_vram_mb")
    vram_vals = [v for v in (hist_vram, cur_vram) if v is not None]
    merged["peak_vram_mb"] = max(vram_vals) if vram_vals else None
    hist_ttft = _hist_first_ttft(done)
    if hist_ttft is not None:
        merged["ttft"] = hist_ttft
    return merged


def done_first_vram(done: dict) -> float | None:
    """历史段逐条记录中的最大显存峰值（checkpoint 逐条存了 vram_mb）。"""
    vals = [r["vram_mb"] for r in done.values() if r.get("vram_mb") is not None]
    return max(vals) if vals else None


def _hist_first_ttft(done: dict) -> float | None:
    """历史段首个样本的 TTFT 秒（有则沿用，None 不覆盖本段值）。"""
    for r in done.values():
        if r.get("ttft_sec") is not None:
            return r["ttft_sec"]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="路线 A GER 批量纠错评测")
    parser.add_argument(
        "--nbest", type=Path, required=True, help="N-best 缓存 JSONL 路径"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-1.7B", help="HF model id 或本地路径"
    )
    parser.add_argument(
        "--quant_mode", default=None,
        choices=["fp16", "int8", "nf4", "q8_0", "q4_k_m", "gguf"],
        help="精度档：hf_bnb 用 fp16/int8/nf4；llamacpp 用 q8_0/q4_k_m/gguf 标注",
    )
    parser.add_argument(
        "--backend", default="hf_bnb",
        choices=["hf_bnb", "llamacpp", "mock"],
        help="推理后端：hf_bnb（默认）| llamacpp（定制版）| mock（管线冒烟）",
    )
    parser.add_argument(
        "--lora_path", default=None,
        help="LoRA adapter 目录（仅 hf_bnb 后端；QLoRA 微调产物）",
    )
    parser.add_argument(
        "--llamacpp_bin", type=Path, default=None,
        help="定制 llama-cli 可执行路径（llamacpp CLI 模式，红线 7.5：CLI 注入不硬编码）",
    )
    parser.add_argument(
        "--llamacpp_url", default=None,
        help="llama-server 根地址（llamacpp server 模式，如 http://127.0.0.1:8080）",
    )
    parser.add_argument(
        "--gguf_model", type=Path, default=None,
        help="GGUF 模型路径（llamacpp CLI 模式必须）",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--template", default="numbered", choices=["numbered", "plain"]
    )
    parser.add_argument("--use_scores", action="store_true")
    parser.add_argument(
        "--exp_name", default=None, help="实验名（默认自动生成）"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("outputs"),
        help="实验输出根目录",
    )
    parser.add_argument(
        "--write_summary", action="store_true",
        help="写入 outputs/metrics/summary.csv",
    )
    parser.add_argument(
        "--max_vram_gb", type=float, default=None,
        help="加载前空闲显存预检阈值（GB）；默认 fp16=4.4、量化档不预检",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续跑：从 outputs/<exp_name>/checkpoint.jsonl 跳过已完成样本",
    )
    parser.add_argument(
        "--vram_baseline_mb", type=float, default=None,
        help="启动 llama-server 前记录的 nvidia-smi 空闲显存（MB）；"
             "llamacpp 后端显存峰值 = 当前已用 - 基线（含权重全量，最准口径）",
    )
    parser.add_argument(
        "--no_checkpoint", action="store_true",
        help="禁用逐条 checkpoint（默认开启）",
    )
    args = parser.parse_args()

    # quant_mode 语义随后端变化：hf_bnb=bnb 档；llamacpp=GGUF 档标注；mock=仅记录
    if args.quant_mode is None:
        args.quant_mode = {"hf_bnb": "fp16", "llamacpp": "gguf", "mock": "fp16"}[
            args.backend
        ]
    if args.backend == "hf_bnb" and args.quant_mode not in ("fp16", "int8", "nf4"):
        parser.error(f"hf_bnb 后端不支持 quant_mode={args.quant_mode}")
    if args.lora_path and args.backend != "hf_bnb":
        parser.error(
            f"--lora_path 仅支持 hf_bnb 后端（当前 backend={args.backend}）；"
            "llamacpp 需先将 adapter 合并导出为 GGUF"
        )
    if args.lora_path:
        # 硬保护：训练/推理模板必须一致，否则静默分布错配会污染 WER
        train_template = check_lora_template(args.lora_path, args.template)
        if train_template:
            parser.error(
                f"--template={args.template} 与训练模板 {train_template} 不一致；"
                f"请改用 --template {train_template}"
            )
    if args.backend == "llamacpp":
        if not (args.llamacpp_bin or args.llamacpp_url):
            parser.error(
                "llamacpp 后端需 --llamacpp_bin（CLI 模式）或 --llamacpp_url（server 模式）二选一"
            )
        if args.llamacpp_bin and not args.gguf_model:
            parser.error("llamacpp CLI 模式必须提供 --gguf_model")
        if args.quant_mode not in ("q8_0", "q4_k_m", "gguf"):
            parser.error(
                f"llamacpp 后端 quant_mode 请标注 GGUF 档（q8_0/q4_k_m/gguf），当前 {args.quant_mode}"
            )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    split = args.nbest.stem  # e.g. "dev-clean" / "test-clean"
    records = load_nbest(args.nbest, args.max_samples)
    logger.info(
        "GER 评测开始：%d 条 | backend=%s | model=%s | quant=%s | template=%s | lora=%s",
        len(records), args.backend, args.model, args.quant_mode, args.template,
        args.lora_path or "none",
    )

    # ---------------- 后端构造（按 --backend 分支） ----------------
    if args.backend == "mock":
        backend = MockLLMBackend()
    elif args.backend == "llamacpp":
        backend = LlamaCppBackend(
            model_path=args.gguf_model,
            bin_path=args.llamacpp_bin,
            server_url=args.llamacpp_url,
            tokenizer_ref=args.model,  # 与渲染 prompt 同源的 HF tokenizer
        )
    else:  # hf_bnb
        # fp16 权重 ~3.4GB + KV cache，预检阈值默认 4.4GB；量化档权重小，不预检。
        # WSL2 下 Windows 侧常驻占用使空闲显存可能贴阈值边缘，可用 --max_vram_gb 放宽。
        vram_budget = args.max_vram_gb
        if vram_budget is None:
            vram_budget = 4.4 if args.quant_mode == "fp16" else None
        backend = HFBNBackend(
            model_id=args.model,
            quant_mode=args.quant_mode,
            max_vram_gb=vram_budget,
            lora_path=args.lora_path,
        )

    # 预渲染全部 prompt（模板渲染是 CPU 操作，与 GPU 生成解耦）
    backend._load()  # 确保 tokenizer 可用（llamacpp: 校验二进制/server 就绪）
    if args.backend == "hf_bnb":
        render_tokenizer = backend.tokenizer
    else:
        # llamacpp/mock 无进程内 tokenizer，用离线 AutoTokenizer（模型已缓存）
        render_tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        render_chat_prompt(
            render_tokenizer,
            r["nbest"],
            SYSTEM_PROMPT,
            args.use_scores,
            args.template,
        )
        for r in records
    ]
    max_new = [dynamic_max_new_tokens(r["nbest"]) for r in records]

    # 进程外后端（llamacpp server）显存口径（三级，取最先可用者）：
    # ① --vram_baseline_mb：用户在启动 llama-server 前记录的空闲显存，
    #    峰值 = 当前已用 - 基线（含权重 + KV cache 全量，最准）；
    # ② server 进程 compute-apps：nvidia-smi 抓 llama-server 进程占用
    #    （WSL2 下 Linux 侧进程可见；Windows 侧进程不可见）；
    # ③ 脚本内前后差分：仅反映推理增量，漏计权重（最后兜底）。
    # 陷阱：llama-server 在本脚本启动前就已加载权重，单纯脚本内差分
    #       会漏计权重显存 —— 必须用 ① 或 ②。
    server_vram_mb: float | None = None
    vram_source = ""
    if args.backend == "llamacpp" and args.llamacpp_url:
        if args.vram_baseline_mb is not None:
            server_vram_mb = None  # 用基线口径，进入生成后按需计算
            vram_source = "baseline"
        else:
            server_vram_mb = _query_llamaserver_vram()
            vram_source = "process" if server_vram_mb is not None else "diff"
    ext_gpu_base_mb = (
        query_gpu_used_mb()
        if args.backend == "llamacpp" and args.llamacpp_url
        else None
    )

    profiler = InferenceProfiler()
    hyps_raw: List[str] = []
    refs: List[str] = []
    total_audio = 0.0
    total_completion_tokens = 0
    cur_gen_sec = 0.0  # 本段逐条生成耗时累加（口径可加，不含模型加载）

    # ---------------- 断点续跑：checkpoint 载入 ----------------
    # LoRA 结果隔离：默认实验名与 summary.quant_mode 使用 nf4_qlora 标签
    quant_label = quant_mode_label(args.quant_mode, args.lora_path)
    exp_name = args.exp_name or default_exp_name(
        args.model, args.quant_mode, split, args.lora_path
    )
    base = args.out_dir / exp_name
    base.mkdir(parents=True, exist_ok=True)
    ckpt_file = checkpoint_path(base)
    if args.resume:
        record_ids = {r["id"] for r in records}
        done = {
            rid: rec
            for rid, rec in load_checkpoint(ckpt_file).items()
            if rid in record_ids  # 防 --max_samples 变化等导致的脏统计
        }
        logger.info(
            "--resume：checkpoint %d/%d 条已完成，跳过", len(done), len(records)
        )
    else:
        done = {}
        # 首跑截断旧 checkpoint（exp_name 复用时防串档）
        if ckpt_file.exists() and not args.no_checkpoint:
            ckpt_file.unlink()
            logger.warning("checkpoint 已存在，首跑将覆盖: %s", ckpt_file)

    ckpt_fp = None
    if not args.no_checkpoint:
        ckpt_fp = open(ckpt_file, "a", encoding="utf-8")

    try:
        with profiler.track():
            for i, (rec, prompt, mnt) in enumerate(zip(records, prompts, max_new)):
                if rec["id"] in done:
                    continue  # 已完成样本：跳过生成，统计走 checkpoint 合并
                t0 = time.perf_counter()
                result = backend.generate(prompt, max_new_tokens=mnt)
                gen_sec = time.perf_counter() - t0
                # 进程外后端自报精确 TTFT（llama.cpp server timings 口径）
                if result.ttft_ms is not None:
                    profiler.report_ttft(result.ttft_ms / 1000.0)
                hyp = extract_answer(result.text)
                hyps_raw.append(hyp)
                refs.append(rec["ref"])
                total_audio += rec["duration"]
                total_completion_tokens += result.completion_tokens
                cur_gen_sec += gen_sec
                # 逐条 checkpoint：flush+fsync，断电后以最后完整行为准
                if ckpt_fp is not None:
                    append_checkpoint(ckpt_fp, {
                        "id": rec["id"],
                        "hyp_raw": hyp,
                        "ref": rec["ref"],
                        "duration": rec["duration"],
                        "completion_tokens": result.completion_tokens,
                        "gen_sec": round(gen_sec, 6),
                        "ttft_sec": None,  # HF 后端非流式无逐条 TTFT；server 模式经 profiler 注入
                        "vram_mb": current_peak_vram_mb(),
                    })
                if (i + 1) % 10 == 0:
                    logger.info(
                        "进度 %d/%d | tokens=%d", i + 1, len(records),
                        total_completion_tokens,
                    )
    finally:
        if ckpt_fp is not None:
            ckpt_fp.close()
        # 进程外后端显存统计（三级口径见上方注释，取最先可用者）。
        # 必须在 track() 退出之后注入：track 的 finally 会用 torch allocator
        # 读数覆盖 _peak_vram_mb，llamacpp 进程内无 torch 分配，覆盖会丢值。
        if vram_source == "baseline":
            fields = query_gpu_mem_mb()
            if fields is not None:
                total, used = fields
                # 基线条目是"启动 server 前的空闲"→ 换算成启动前已用
                baseline_used = total - args.vram_baseline_mb
                peak = max(used - baseline_used, 0.0)
                profiler.report_peak_vram(peak)
                logger.info(
                    "llamacpp 显存（基线口径）: 峰值 %.0f MB "
                    "(当前已用 %.0f - 启动前已用 %.0f)",
                    peak, used, baseline_used,
                )
        elif vram_source == "process":
            v = _query_llamaserver_vram()
            if v is not None:
                # 进程口径已含权重；推理期间 KV cache 增长取两次较大值
                peak = max(v, server_vram_mb or v)
                profiler.report_peak_vram(peak)
                logger.info(
                    "llamacpp 显存（进程口径）: 峰值 %.0f MB", peak,
                )
        elif ext_gpu_base_mb is not None:  # diff 兜底
            cur = query_gpu_used_mb()
            if cur is not None:
                profiler.report_peak_vram(cur - ext_gpu_base_mb)
                logger.info(
                    "llamacpp 进程级显存差分: %.0f MB (base=%.0f)"
                    "（注意：漏计 server 预载权重）",
                    cur - ext_gpu_base_mb, ext_gpu_base_mb,
                )
        stats = profiler.finalize(
            total_audio_sec=max(total_audio, 1e-6),
            total_new_tokens=total_completion_tokens,
        )
        # 分段统计口径合并：Σgen_sec 替代墙钟（模型加载/中断间隔不计入），
        # RTF/Tokens/s 可加不失真；peak VRAM 跨段取 max；TTFT 取历史首条
        stats["_cur_gen_sec"] = cur_gen_sec
        stats["_cur_audio_sec"] = total_audio
        stats["_cur_new_tokens"] = total_completion_tokens
        stats = merge_stats(done, stats)
        backend.unload()

    # 规范化 + WER：续跑时全量 = checkpoint 历史段 + 本段（保持文件顺序；
    # 本段 hyps_raw 只含新生成样本，按游标对位，skipped 从 checkpoint 取回）
    if done:
        merged_hyps: List[str] = []
        merged_refs: List[str] = []
        cur_idx = 0
        for r in records:
            if r["id"] in done:
                merged_hyps.append(done[r["id"]]["hyp_raw"])
                merged_refs.append(done[r["id"]]["ref"])
            else:
                merged_hyps.append(hyps_raw[cur_idx])
                merged_refs.append(refs[cur_idx])
                cur_idx += 1
        hyps_raw, refs = merged_hyps, merged_refs
    hyps_norm = [normalize_for_wer(h) for h in hyps_raw]
    refs_norm = [normalize_for_wer(r) for r in refs]
    wer = compute_wer(refs_norm, hyps_norm, normalize=False)

    # 声学基线对照（同一子集）
    baseline = compute_wer(refs_norm, [normalize_for_wer(r["nbest"][0]["text"]) for r in records], normalize=False)
    logger.info(
        "GER 结果：WER=%.4f (S=%d D=%d I=%d) | beam-1 基线=%.4f | 提升=%.2f pts",
        wer["wer"], wer["sub"], wer["del"], wer["ins"],
        baseline["wer"], (baseline["wer"] - wer["wer"]) * 100,
    )

    # 产物归档（exp_name/base 已在 checkpoint 段确定）
    export_predictions(
        base / "predictions.jsonl",
        (
            {"id": r["id"], "ref": r["ref"], "hyp_raw": h_raw,
             "hyp": normalize_for_wer(h_raw), "beam1": r["nbest"][0]["text"]}
            for r, h_raw in zip(records, hyps_raw)
        ),
    )
    export_error_cases(
        base / "cases.jsonl",
        (
            {"id": r["id"], "ref": r["ref"], "hyp": normalize_for_wer(h_raw),
             "beam1": r["nbest"][0]["text"]}
            for r, h_raw in zip(records, hyps_raw)
        ),
    )

    if args.write_summary:
        append_summary_row(
            args.out_dir / "metrics" / "summary.csv",
            exp_name=exp_name, route="A", split=split,
            model=args.model,
            backend={"hf_bnb": "hf_bnb", "llamacpp": "llamacpp",
                      "mock": "mock"}[args.backend],
            quant_mode=quant_label,  # LoRA: nf4_qlora，与零样本 nf4 严格隔离
            wer=wer["wer"], sub=wer["sub"], **{"del": wer["del"]},
            ins=wer["ins"],
            peak_vram_mb=stats["peak_vram_mb"],
            total_audio_sec=stats["total_audio_sec"],
            total_infer_sec=stats["total_infer_sec"],
            rtf=stats["rtf"], tokens_per_sec=stats["tokens_per_sec"],
        )
        logger.info("已写入 summary.csv")

    logger.info("产物目录：%s", base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
