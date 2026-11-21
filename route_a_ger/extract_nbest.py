"""路线 A：wav2vec2-base-100h + pyctcdecode 导出 Top-5 N-best 并静态缓存。

产出契约（PROJECT_SPEC.md 第 5.2 节）：
    outputs/nbest/<split>.jsonl，每行：
    {"id": ..., "ref": ..., "duration": ...,
     "ctc_greedy": ...,
     "nbest": [{"text": ..., "score": ...}, ...]}

score 为 pyctcdecode 的 lm_score + logltoken（CTC 对数似然），
值越大越好；nbest 按 score 降序排列。

用法：
    # 冒烟（本地 3060，8 条）
    python route_a_ger/extract_nbest.py --split dev-clean --max_samples 8

    # 全量
    python route_a_ger/extract_nbest.py --split test-clean
    python route_a_ger/extract_nbest.py --split test-other

    # 云端离线：使用本地下载的模型目录
    python route_a_ger/extract_nbest.py --split train-clean-100 \
        --model /mnt/workspace/models/wav2vec2-base-100h
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.metrics import append_summary_row, compute_wer  # noqa: E402
from common.profiler import InferenceProfiler, cuda_cleanup  # noqa: E402

logger = logging.getLogger("extract_nbest")

MODEL_ID = "facebook/wav2vec2-base-100h"
N_BEST = 5
BEAM_WIDTH = 32  # pyctcdecode 束宽；Top-5 需要 >5 的束宽才稳定


def load_model(device: str, model_id: str = MODEL_ID) -> tuple:
    """加载 wav2vec2 CTC 模型（FP16 可选）与处理器。

    ``model_id`` 支持 HF repo id 或本地目录（云端离线场景传下载好的目录）。
    """
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()
    if device.startswith("cuda"):
        model = model.half()  # 95M 参数，FP16 足够且省显存
    return processor, model


def build_decoder(processor) -> "build_ctcdecoder":
    """从处理器词表构建 pyctcdecode 解码器（无外部 LM 的纯声学解码）。

    wav2vec2-base-100h 词表布局（32 列）：
        0: <pad>  1: <s>  2: </s>  3: <unk>  4: '|'(词分隔)  5-31: A-Z
    pyctcdecode 约定 index 0 为 blank，且 labels 不允许重复。
    处理：<pad> 作 blank；<s>/</s>/<unk> 为无效列，须映射为不冲突的
    不可见占位符（Unicode 私有区），既不参与文本输出也不与 blank 重复。
    """
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]

    _INVISIBLE = [chr(0xE000 + i) for i in range(3)]  # 私有区占位符
    labels = []
    for i, t in enumerate(sorted_vocab):
        if i == 0:  # <pad> -> blank
            labels.append("")
        elif t == "<s>":
            labels.append(_INVISIBLE[0])
        elif t == "</s>":
            labels.append(_INVISIBLE[1])
        elif t == "<unk>":
            labels.append(_INVISIBLE[2])
        elif t == "|":
            labels.append(" ")  # 词分隔符 -> 空格
        else:
            labels.append(t)
    assert len(labels) == len(set(labels)), "labels 存在重复"
    return build_ctcdecoder(
        labels=labels,
        unigrams=None,  # 无 KenLM 时传空，仅用 CTC 分数束搜索
    )


@torch.no_grad()
def extract_one(
    audio_path: str,
    processor,
    model,
    decoder,
    device: str,
    n_best: int = N_BEST,
    beam_width: int = BEAM_WIDTH,
) -> tuple:
    """单条音频 -> (ctc_greedy, nbest_list)。"""
    speech, sr = sf.read(audio_path, dtype="float32")
    assert sr == 16000, f"{audio_path} 采样率 {sr} != 16000"
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")

    input_values = inputs.input_values.to(device)
    if device.startswith("cuda"):
        input_values = input_values.half()

    logits = model(input_values).logits[0]
    # pyctcdecode 接收 float32 CPU 张量的 log_softmax
    log_probs = torch.log_softmax(logits.float().cpu(), dim=-1).numpy()

    # CTC greedy：标准折叠 = 合并连续重复（blank 打断合并）→ 删 blank。
    # 注意 decode() 默认 group_tokens=True 会在已折叠序列上二次折叠
    # （L L -> L），必须显式关闭，否则 MIDDLE -> MIDLE。
    greedy_ids = np.argmax(log_probs, axis=-1)
    prev = -1
    kept = []
    for i in greedy_ids.tolist():
        if i != 0 and i != prev:  # 0 = blank
            kept.append(i)
        prev = i
    ctc_greedy = processor.tokenizer.decode(kept, group_tokens=False)

    # beam search Top-N（score = lm_score + logltoken，越大越好）
    beams = decoder.decode_beams(log_probs, beam_width=beam_width)[
        :n_best
    ]
    nbest = []
    for beam in beams:
        nbest.append(
            {
                "text": beam[0].lower(),
                # beam: (text, last_lm_state, output_tokens, lm_score, logltoken)
                "score": round(float(beam[-2] + beam[-1]), 4),
            }
        )
    # decode_beams 返回已按 score 降序；防御性再排一次
    nbest.sort(key=lambda x: x["score"], reverse=True)
    return ctc_greedy, nbest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="wav2vec2 + pyctcdecode 导出 Top-N N-best 缓存"
    )
    parser.add_argument(
        "--split",
        required=True,
        help="LibriSpeech split 名（如 test-clean / dev-clean）",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"wav2vec2 模型：HF repo id 或本地目录（默认 {MODEL_ID}）",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="清单路径；默认 data/manifests/<split>.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/nbest"),
        help="N-best 缓存输出目录（默认 outputs/nbest）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理样本数（冒烟测试用）",
    )
    parser.add_argument(
        "--n_best",
        type=int,
        default=N_BEST,
        help=f"N-best 条数（默认 {N_BEST}）",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=BEAM_WIDTH,
        help=f"pyctcdecode 束宽（默认 {BEAM_WIDTH}）",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备（默认自动）",
    )
    parser.add_argument(
        "--write_summary",
        action="store_true",
        help="将声学基线 WER 写入 outputs/metrics/summary.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    manifest = args.manifest or Path(f"data/manifests/{args.split}.jsonl")
    records = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    if args.max_samples:
        records = records[: args.max_samples]
    logger.info("split=%s 读入 %d 条样本", args.split, len(records))

    processor, model = load_model(args.device, args.model)
    decoder = build_decoder(processor)

    output_path = args.output_dir / f"{args.split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profiler = InferenceProfiler()
    hyps_greedy: List[str] = []
    refs: List[str] = []
    total_audio = 0.0

    with profiler.track(), open(output_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            try:
                greedy, nbest = extract_one(
                    rec["audio_path"],
                    processor,
                    model,
                    decoder,
                    args.device,
                    n_best=args.n_best,
                    beam_width=args.beam_width,
                )
            except Exception as exc:  # noqa: BLE001 - 单条失败不中断批量任务
                logger.error("样本 %s 失败：%s", rec["id"], exc)
                continue
            out = {
                "id": rec["id"],
                "ref": rec["text"],
                "duration": rec["duration"],
                "ctc_greedy": greedy,
                "nbest": nbest,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                f.flush()
                logger.info("进度 %d/%d", i + 1, len(records))
            hyps_greedy.append(greedy)
            refs.append(rec["text"])
            total_audio += rec["duration"]

    # 显存释放（红线 7.4）
    del model
    cuda_cleanup()

    stats = profiler.finalize(total_audio_sec=max(total_audio, 1e-6))
    wer = compute_wer(refs, hyps_greedy)
    logger.info(
        "纯声学贪心基线 split=%s: WER=%.4f (S=%d D=%d I=%d), RTF=%.4f",
        args.split,
        wer["wer"],
        wer["sub"],
        wer["del"],
        wer["ins"],
        stats["rtf"],
    )

    if args.write_summary:
        append_summary_row(
            Path("outputs/metrics/summary.csv"),
            exp_name=f"acoustic-greedy-{args.split}",
            route="A",
            split=args.split,
            model=args.model,
            backend="wav2vec2",
            quant_mode="fp16",
            wer=wer["wer"],
            sub=wer["sub"],
            **{"del": wer["del"]},
            ins=wer["ins"],
            peak_vram_mb=stats["peak_vram_mb"],
            total_audio_sec=stats["total_audio_sec"],
            total_infer_sec=stats["total_infer_sec"],
            rtf=stats["rtf"],
            tokens_per_sec=0.0,  # 声学前端无 LLM token 概念，占位
        )
        logger.info("已写入 outputs/metrics/summary.csv")

    logger.info("N-best 缓存：%s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
