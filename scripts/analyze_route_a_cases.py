#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路线 A 定性分析：句子改写率 / 改好改坏统计 + 代表性样本挑选。

只读 outputs/ 下已有的 predictions.jsonl，不写入 outputs/，结果输出到
docs/route_a_analysis.json（纯文本 JSON，可入库复现）。

用法：
    python scripts/analyze_route_a_cases.py [--out docs/route_a_analysis.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.metrics import compute_wer  # noqa: E402
from common.text_norm import normalize_for_wer  # noqa: E402

# 参与统计的实验目录（与 summary.csv 一一对应）
EXPS = {
    "test-clean": [
        ("FP16", "ger-1.7b-fp16-test-clean"),
        ("INT8", "ger-1.7b-int8-test-clean"),
        ("Q8_0", "ger-1.7b-q8_0-llamacpp-test-clean"),
        ("Q4_K_M", "ger-1.7b-q4km-llamacpp-test-clean"),
        ("NF4", "ger-1.7b-nf4-test-clean"),
        ("NF4 + QLoRA", "ger-Qwen3-1.7B-nf4_qlora-test-clean"),
    ],
    "test-other": [
        ("FP16", "ger-Qwen3-1.7B-fp16-test-other"),
        ("Q8_0", "ger-1.7b-q8_0-llamacpp-test-other"),
        ("Q4_K_M", "ger-1.7b-q4km-llamacpp-test-other"),
        ("NF4", "ger-1.7b-nf4-test-other"),
        ("NF4 + QLoRA", "ger-Qwen3-1.7B-nf4_qlora-test-other"),
    ],
}


def load_preds(exp: str) -> list[dict]:
    path = REPO / "outputs" / exp / "predictions.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sent_wer(ref: str, hyp: str) -> float:
    """句级 WER；两边都空视为 0，参考非空而输出空视为 1。"""
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return compute_wer([ref], [hyp])["wer"]


def analyse(rows: list[dict]) -> dict:
    n = len(rows)
    rewritten = improved = worsened = same = 0
    delta_sum = 0.0
    per_sample: list[dict] = []
    for r in rows:
        ref = normalize_for_wer(r["ref"])
        hyp = normalize_for_wer(r.get("hyp_raw") or r.get("hyp", ""))
        base = normalize_for_wer(r.get("beam1", ""))
        w_h, w_b = sent_wer(ref, hyp), sent_wer(ref, base)
        changed = hyp != base
        rewritten += changed
        if changed and w_h < w_b:
            improved += 1
        elif changed and w_h > w_b:
            worsened += 1
        elif changed:
            same += 1  # 文本变了但句级 WER 不变
        delta_sum += w_b - w_h
        per_sample.append(
            {
                "id": r["id"],
                "ref": ref,
                "beam1": base,
                "hyp": hyp,
                "w_beam1": round(w_b, 4),
                "w_hyp": round(w_h, 4),
                "changed": changed,
            }
        )
    return {
        "n": n,
        "rewrite_rate": round(rewritten / n, 4),
        "rewritten": rewritten,
        "improved": improved,
        "worsened": worsened,
        "changed_neutral": same,
        "improve_share_of_rewrites": round(improved / rewritten, 4) if rewritten else 0.0,
        "worsen_share_of_rewrites": round(worsened / rewritten, 4) if rewritten else 0.0,
        "mean_delta_wer_pts": round(delta_sum / n * 100, 3),
        "_samples": per_sample,
    }


def load_nbest(split: str) -> dict[str, list[str]]:
    """id -> 归一化后的 Top-5 候选文本，用于判定 QLoRA 是否跳出候选集。"""
    path = REPO / "outputs" / "nbest" / f"{split}.jsonl"
    out: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["id"]] = [normalize_for_wer(c["text"]) for c in r["nbest"]]
    return out


def pick_cases(res: dict[str, dict[str, dict]]) -> dict:
    """挑选 4-6 条高代表性样本。res[split][label] -> analyse() 结果。"""
    cases: dict[str, list[dict]] = {}
    clean_nf4 = res["test-clean"]["NF4"]
    clean_q4 = res["test-clean"]["Q4_K_M"]
    other_q4 = res["test-other"]["Q4_K_M"]
    clean_qlora = res["test-clean"]["NF4 + QLoRA"]
    other_nf4 = res["test-other"]["NF4"]
    other_qlora = res["test-other"]["NF4 + QLoRA"]
    nb_clean = load_nbest("test-clean")
    nb_other = load_nbest("test-other")

    def find(pool: dict, pred, k=1, min_words=6, max_words=40):
        """在 pool['_samples'] 中挑 pred 成立、参考长度适中的样本。"""
        cand = [
            s
            for s in pool["_samples"]
            if pred(s) and min_words <= len(s["ref"].split()) <= max_words
        ]
        cand.sort(key=lambda s: (len(s["ref"].split()), s["id"]))
        return cand[:k]

    def is_insert_hallucination(s: dict) -> bool:
        """NF4 插词/复读：输出显著长于 beam-1，且句级 WER 变差。"""
        return (
            s["w_hyp"] > s["w_beam1"]
            and len(s["hyp"].split()) >= 1.3 * len(s["beam1"].split())
            and s["w_beam1"] <= 0.35
        )

    def is_moderate_insert(s: dict) -> bool:
        """NF4 轻量插词：句级 WER 抬升 8 pts 以上、输出多 1–4 词。"""
        nw, bw, hw = len(s["ref"].split()), len(s["beam1"].split()), len(s["hyp"].split())
        return (
            10 <= nw <= 30
            and bw == nw
            and 1 <= hw - bw <= 4
            and s["w_beam1"] < 0.12
            and s["w_hyp"] - s["w_beam1"] >= 0.08
        )

    def is_truncation(s: dict) -> bool:
        """Q4_K_M 截断：输出不足参考 60% 且非粘连（字符/词比正常）。"""
        hw = len(s["hyp"].split())
        cpw = len(s["hyp"]) / max(hw, 1)
        return (
            len(s["ref"].split()) >= 10
            and hw <= 0.6 * len(s["ref"].split())
            and cpw < 15
        )

    def outside_nbest(s: dict, nb: dict[str, list[str]]) -> bool:
        """QLoRA 输出不在 Top-5 候选中——证明模型重组出候选集外的正确文本。"""
        return s["hyp"] not in nb.get(s["id"], [])

    # 1) NF4 轻量插词（test-clean）
    cases["NF4 插词 (test-clean)"] = find(clean_nf4, is_moderate_insert)
    # 2) Q4_K_M 截断漏词（test-clean）
    cases["Q4_K_M 截断 (test-clean)"] = find(clean_q4, is_truncation)
    # 3) QLoRA 跳出 Top-5 候选（test-clean）
    cases["QLoRA 跳出候选 (test-clean)"] = find(
        clean_qlora,
        lambda s: s["w_beam1"] >= 0.25 and s["w_hyp"] == 0.0 and outside_nbest(s, nb_clean),
    )
    # 4) QLoRA 跳出 Top-5 候选（test-other）
    cases["QLoRA 跳出候选 (test-other)"] = find(
        other_qlora,
        lambda s: s["w_beam1"] >= 0.3 and s["w_hyp"] == 0.0 and outside_nbest(s, nb_other),
    )
    # 5) NF4 整句复读（test-other，噪声集放大）
    cases["NF4 复读 (test-other)"] = find(other_nf4, is_insert_hallucination)
    # 6) Q4_K_M 截断（test-other）
    cases["Q4_K_M 截断 (test-other)"] = find(other_q4, is_truncation)
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs" / "route_a_analysis.json"))
    args = ap.parse_args()

    results: dict[str, dict] = {}
    for split, exps in EXPS.items():
        results[split] = {}
        for label, exp in exps:
            rows = load_preds(exp)
            results[split][label] = analyse(rows)

    cases = pick_cases(results)

    # 控制台摘要
    print(f"{'split':<11}{'config':<14}{'rewrite':>9}{'improved':>10}{'worsened':>10}{'neutral':>9}{'dWER(pts)':>11}")
    for split, per in results.items():
        for label, st in per.items():
            print(
                f"{split:<11}{label:<14}{st['rewrite_rate']*100:>8.1f}%"
                f"{st['improved']:>10}{st['worsened']:>10}{st['changed_neutral']:>9}"
                f"{st['mean_delta_wer_pts']:>11.2f}"
            )

    print("\n=== 代表性样本 ===")
    for key, lst in cases.items():
        for s in lst:
            print(f"[{key}] {s['id']} w_beam1={s['w_beam1']} w_hyp={s['w_hyp']}")
            print("  ref  :", s["ref"])
            print("  beam1:", s["beam1"])
            print("  hyp  :", s["hyp"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": {
            split: {
                label: {kk: vv for kk, vv in st.items() if kk != "_samples"}
                for label, st in per.items()
            }
            for split, per in results.items()
        },
        "cases": cases,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
