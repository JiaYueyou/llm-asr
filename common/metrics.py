"""评测指标模块（路线 A / B 共享）。

核心职责（PROJECT_SPEC.md 第 5.3 节契约）：
  1. 基于 jiwer 计算整体 WER 及 S/D/I（替换/删除/插入）分解；
  2. 以固定字段顺序追加写入 outputs/metrics/summary.csv：
     exp_name, route, split, model, backend, quant_mode,
     wer, sub, del, ins,
     peak_vram_mb, total_audio_sec, total_infer_sec, rtf, tokens_per_sec
  3. 导出单条预测对比与错例（WER 贡献 > 0）JSONL。

用法：
    from common.metrics import compute_wer, append_summary_row

    wer, sub, dele, ins = compute_wer(refs, hyps)
    append_summary_row(csv_path, exp_name=..., route=..., ...)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Sequence

from jiwer import process_words

from common.text_norm import normalize_for_wer

logger = logging.getLogger("metrics")

# summary.csv 固定字段（契约顺序，勿改动）
SUMMARY_FIELDS = [
    "exp_name",
    "route",
    "split",
    "model",
    "backend",
    "quant_mode",
    "wer",
    "sub",
    "del",
    "ins",
    "peak_vram_mb",
    "total_audio_sec",
    "total_infer_sec",
    "rtf",
    "tokens_per_sec",
]


def compute_wer(
    refs: Sequence[str],
    hyps: Sequence[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """计算整体 WER 与 S/D/I 分解。

    Args:
        refs: 参考文本列表（LibriSpeech 全大写无标点）。
        hyps: 模型假设文本列表，与 refs 一一对应。
        normalize: 是否先过 normalize_for_wer（评测路径必须 True；
            测试构造已知答案时可置 False 跳过规范化）。

    Returns:
        {"wer": 0.1234, "sub": 2, "del": 1, "ins": 0}
        错误数为累计词数（int），wer 为比率（float）。

    空字符串约定：
        - ref 非空 + hyp 空：全部计为删除；
        - ref 空 + hyp 非空：全部计为插入（jiwer 支持，WER 按无穷大处理，
          此处钳制为 1.0 防止 inf 污染汇总）。
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs({len(refs)}) 与 hyps({len(hyps)}) 数量不一致")
    if len(refs) == 0:
        return {"wer": 0.0, "sub": 0, "del": 0, "ins": 0}

    if normalize:
        refs = [normalize_for_wer(r) for r in refs]
        hyps = [normalize_for_wer(h) for h in hyps]

    # jiwer 不接受空串（会 raise）。空串样本直接按语义累计错误数，
    # 非空部分才送 jiwer 对齐——避免占位符把"整句删除"扭曲成
    # "1 替换 + N-1 删除"。
    sub = dele = ins = 0
    n_ref_words = 0

    pairs = []
    for r, h in zip(refs, hyps):
        r_words, h_words = r.split(), h.split()
        if not r_words and not h_words:
            continue
        if not h_words:  # hyp 空：ref 全部计删除
            dele += len(r_words)
            n_ref_words += len(r_words)
            continue
        if not r_words:  # ref 空：hyp 全部计插入
            ins += len(h_words)
            continue
        pairs.append((r, h))

    if pairs:
        result = process_words([p[0] for p in pairs], [p[1] for p in pairs])
        sub += result.substitutions
        dele += result.deletions
        ins += result.insertions
        n_ref_words += result.hits + result.substitutions + result.deletions

    if n_ref_words == 0:
        # 全部 ref 为空：WER 无定义；仅当 hyp 也全空才计 0
        wer = 0.0 if (sub + dele + ins) == 0 else 1.0
    else:
        # 钳制由插入主导的极端值（WER 可 >1），保证汇总表可读
        wer = min((sub + dele + ins) / n_ref_words, 1.0)
    return {"wer": wer, "sub": sub, "del": dele, "ins": ins}


def per_utterance_wer(ref: str, hyp: str, normalize: bool = True) -> Dict[str, float]:
    """单条样本的 WER 与 S/D/I（错例导出用）。"""
    return compute_wer([ref], [hyp], normalize=normalize)


def append_summary_row(csv_path: Path, **row: object) -> None:
    """按契约字段追加一行到 summary.csv；文件不存在时先写表头。

    必填字段（契约顺序）缺一时报错；额外字段忽略。
    浮点字段统一保留 4 位小数，整数计数保持整数。
    """
    csv_path = Path(csv_path)
    missing = [f for f in SUMMARY_FIELDS if f not in row]
    if missing:
        raise KeyError(f"summary 行缺少契约字段: {missing}")

    float_fields = {"wer", "sub", "del", "ins", "peak_vram_mb",
                    "total_audio_sec", "total_infer_sec", "rtf", "tokens_per_sec"}

    formatted = {}
    for f in SUMMARY_FIELDS:
        v = row[f]
        if f in float_fields and v is not None:
            # sub/del/ins 虽为计数，但契约示例按数值写入，统一数值格式
            formatted[f] = round(float(v), 4) if isinstance(v, float) else v
        else:
            formatted[f] = v
        if formatted[f] is None:
            formatted[f] = ""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(formatted)


def export_predictions(
    out_path: Path,
    records: Iterable[Dict],
) -> int:
    """导出单条预测对比 JSONL。

    每条记录应包含：id, ref, hyp, 以及可选的 wer/sub/del/ins、
    nbest_top（路线 A 的声学前端 Top-1）。字段原样写入。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info("导出预测对比 %d 条 -> %s", n, out_path)
    return n


def export_error_cases(
    out_path: Path,
    records: Iterable[Dict],
) -> int:
    """导出错例 JSONL：仅保留 per-utterance WER > 0 的样本。

    records 中每条需含 ref/hyp；函数内部补充逐条 wer/sub/del/ins。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            m = per_utterance_wer(rec.get("ref", ""), rec.get("hyp", ""))
            if m["wer"] <= 0:
                continue
            enriched = {**rec, **m}
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            n += 1
    logger.info("导出错例 %d 条 -> %s", n, out_path)
    return n
