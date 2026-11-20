"""路线 A：读取 N-best 缓存，经 LLM 后端批量生成式纠错并评测。

用法：
    # NF4 冒烟（dev-clean 50 条）
    python route_a_ger/infer_ger.py --nbest outputs/nbest/dev-clean.jsonl \
        --model Qwen/Qwen3-1.7B --quant_mode nf4 --max_samples 50

    # 三精度全量（test-clean）
    python route_a_ger/infer_ger.py --nbest outputs/nbest/test-clean.jsonl \
        --model Qwen/Qwen3-1.7B --quant_mode fp16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.llm_backends.hf_bnb import HFBNBackend
from common.metrics import (
    append_summary_row,
    compute_wer,
    export_error_cases,
    export_predictions,
)
from common.profiler import InferenceProfiler
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
        "--quant_mode", default="fp16", choices=["fp16", "int8", "nf4"]
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    split = args.nbest.stem  # e.g. "dev-clean" / "test-clean"
    records = load_nbest(args.nbest, args.max_samples)
    logger.info(
        "GER 评测开始：%d 条 | model=%s | quant=%s | template=%s",
        len(records), args.model, args.quant_mode, args.template,
    )

    backend = HFBNBackend(
        model_id=args.model,
        quant_mode=args.quant_mode,
        max_vram_gb=4.5 if args.quant_mode == "fp16" else None,
    )

    # 预渲染全部 prompt（模板渲染是 CPU 操作，与 GPU 生成解耦）
    backend._load()  # 确保 tokenizer 可用
    prompts = [
        render_chat_prompt(
            backend.tokenizer,
            r["nbest"],
            SYSTEM_PROMPT,
            args.use_scores,
            args.template,
        )
        for r in records
    ]
    max_new = [dynamic_max_new_tokens(r["nbest"]) for r in records]

    profiler = InferenceProfiler()
    hyps_raw: List[str] = []
    refs: List[str] = []
    total_audio = 0.0
    total_completion_tokens = 0

    try:
        with profiler.track():
            for i, (rec, prompt, mnt) in enumerate(zip(records, prompts, max_new)):
                result = backend.generate(prompt, max_new_tokens=mnt)
                hyps_raw.append(extract_answer(result.text))
                refs.append(rec["ref"])
                total_audio += rec["duration"]
                total_completion_tokens += result.completion_tokens
                if (i + 1) % 10 == 0:
                    logger.info(
                        "进度 %d/%d | tokens=%d", i + 1, len(records),
                        total_completion_tokens,
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
            model=args.model, backend="hf_bnb", quant_mode=args.quant_mode,
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
