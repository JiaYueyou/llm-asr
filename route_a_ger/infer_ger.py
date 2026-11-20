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
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
from common.profiler import InferenceProfiler, query_gpu_used_mb
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
    args = parser.parse_args()

    # quant_mode 语义随后端变化：hf_bnb=bnb 档；llamacpp=GGUF 档标注；mock=仅记录
    if args.quant_mode is None:
        args.quant_mode = {"hf_bnb": "fp16", "llamacpp": "gguf", "mock": "fp16"}[
            args.backend
        ]
    if args.backend == "hf_bnb" and args.quant_mode not in ("fp16", "int8", "nf4"):
        parser.error(f"hf_bnb 后端不支持 quant_mode={args.quant_mode}")
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
        "GER 评测开始：%d 条 | backend=%s | model=%s | quant=%s | template=%s",
        len(records), args.backend, args.model, args.quant_mode, args.template,
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

    # 进程外后端（llamacpp server）显存差分基线：进入生成前测一次
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

    try:
        with profiler.track():
            for i, (rec, prompt, mnt) in enumerate(zip(records, prompts, max_new)):
                result = backend.generate(prompt, max_new_tokens=mnt)
                # 进程外后端自报精确 TTFT（llama.cpp server timings 口径）
                if result.ttft_ms is not None:
                    profiler.report_ttft(result.ttft_ms / 1000.0)
                hyps_raw.append(extract_answer(result.text))
                refs.append(rec["ref"])
                total_audio += rec["duration"]
                total_completion_tokens += result.completion_tokens
                if (i + 1) % 10 == 0:
                    logger.info(
                        "进度 %d/%d | tokens=%d", i + 1, len(records),
                        total_completion_tokens,
                    )
            # 进程外后端显存差分（nvidia-smi 口径，常驻占用在差分中抵消）
            if ext_gpu_base_mb is not None:
                cur = query_gpu_used_mb()
                if cur is not None:
                    profiler.report_peak_vram(cur - ext_gpu_base_mb)
                    logger.info(
                        "llamacpp 进程级显存差分: %.0f MB (base=%.0f)",
                        cur - ext_gpu_base_mb, ext_gpu_base_mb,
                    )
    finally:
        stats = profiler.finalize(
            total_audio_sec=max(total_audio, 1e-6),
            total_new_tokens=total_completion_tokens,
        )
        backend.unload()

    # 规范化 + WER（text_norm 内含思考标签/前缀防御性剥离）
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

    # 产物归档
    exp_name = args.exp_name or (
        f"ger-{args.model.split('/')[-1]}-{args.quant_mode}-{split}"
    )
    base = args.out_dir / exp_name
    base.mkdir(parents=True, exist_ok=True)
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
            Path("outputs/metrics/summary.csv"),
            exp_name=exp_name, route="A", split=split,
            model=args.model,
            backend={"hf_bnb": "hf_bnb", "llamacpp": "llamacpp",
                      "mock": "mock"}[args.backend],
            quant_mode=args.quant_mode,
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
