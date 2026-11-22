#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路线 A 论文级 SVG 图表生成（纯本地数据，零编造）。

数据源（唯一事实来源）：
  outputs/metrics/summary.csv           # whisper-normalizer 1.0.1 统一口径
  outputs/logs/route-a-*-qlora.log      # Task 1.6b 云端 1782 步真实训练日志
  outputs/nbest/*.jsonl                 # Top-5 Oracle 参考线（按 1.0.1 重算）

用法：
    python scripts/plot_route_a.py [--outdir docs/figures]

配色：Nature / IEEE / ICML 常用低饱和多色科研色板，四图统一映射。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PREVIEW = Path("/tmp/route_a_fig_preview")
PREVIEW.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

from common.metrics import compute_wer  # noqa: E402
from common.text_norm import normalize_for_wer  # noqa: E402

# ---------------------------------------------------------------------------
# 全局视觉常量（四图严格共用）
# ---------------------------------------------------------------------------
BG = "#FFFFFF"
GRID = "#E5E7EB"
TEXT = "#1F2937"
MUTED = "#7F8C8D"

PALETTE = {
    "Beam-1": "#7F8C8D",        # 中性石板灰（传统声学基线）
    "FP16": "#4E79A7",          # 深青蓝（全精度参考）
    "INT8": "#93B2DF",          # 浅钢蓝
    "Q8_0": "#2A9D8F",          # 青绿（端侧效率甜点）
    "Q4_K_M": "#E9B44C",        # 柔和琥珀（K-quant 激进档）
    "NF4": "#E67C50",           # 暖珊瑚（bnb 4-bit）
    "NF4 + QLoRA": "#C04F15",   # 赤陶红（本工作最优）
}
ORACLE = "#7E6B9B"              # 灰紫（Top-5 候选上界）
D_COLOR = "#4E79A7"             # 删除错误
I_COLOR = "#E67C50"             # 插入错误

CONFIG_ORDER = ["Beam-1", "FP16", "INT8", "Q8_0", "Q4_K_M", "NF4", "NF4 + QLoRA"]
# exp_name 前缀 -> 展示名（test-clean / test-other 两套目录名并存）
EXP_KEYS = [
    ("acoustic-beam1", "Beam-1"),
    ("ger-1.7b-fp16", "FP16"),
    ("ger-Qwen3-1.7B-fp16", "FP16"),
    ("ger-1.7b-int8", "INT8"),
    ("ger-1.7b-q8_0", "Q8_0"),
    ("ger-1.7b-q4km", "Q4_K_M"),
    ("ger-1.7b-nf4-test", "NF4"),
    ("ger-Qwen3-1.7B-nf4_qlora", "NF4 + QLoRA"),
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.8,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.major.size": 0,
            "ytick.major.size": 3.5,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "path",  # 文字转路径，GitHub 渲染零字体依赖
            "figure.dpi": 150,
        }
    )


def style_axes(ax, ygrid: bool = True, xgrid: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def load_summary() -> dict[tuple[str, str], dict]:
    """(split, config) -> summary 行。"""
    path = REPO / "outputs" / "metrics" / "summary.csv"
    out: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in ("test-clean", "test-other"):
                continue
            for prefix, label in EXP_KEYS:
                if row["exp_name"].startswith(prefix):
                    # 同 split 同 config 只保留首条（history 无重复脏行）
                    out.setdefault((split, label), row)
                    break
    return out


def load_oracle() -> dict[str, float]:
    """按 1.0.1 口径重算 Top-5 Oracle（逐句取句级 WER 最小候选）。"""
    res: dict[str, float] = {}
    for split in ("test-clean", "test-other"):
        rows = [json.loads(l) for l in (REPO / "outputs" / "nbest" / f"{split}.jsonl").open(encoding="utf-8") if l.strip()]
        refs, sel = [], []
        for r in rows:
            ref = normalize_for_wer(r["ref"])
            cands = [normalize_for_wer(c["text"]) for c in r["nbest"]]
            scored = [(compute_wer([ref], [c])["wer"], i) for i, c in enumerate(cands)]
            refs.append(ref)
            sel.append(cands[min(scored)[1]])
        res[split] = compute_wer(refs, sel)["wer"] * 100
    return res


def load_loss_curve() -> tuple[list[int], list[float], int]:
    """从 Task 1.6b 训练日志提取 (step, loss) 与总步数。"""
    logs = sorted((REPO / "outputs" / "logs").glob("*qlora*.log"))
    if not logs:
        raise FileNotFoundError("未找到 QLoRA 训练日志 outputs/logs/*qlora*.log")
    pat = re.compile(r"step (\d+)/(\d+) \| loss=([0-9.]+)")
    best: tuple[list[int], list[float], int] | None = None
    for log in logs:
        steps, losses, total = [], [], 0
        for line in log.open(encoding="utf-8", errors="ignore"):
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
                total = int(m.group(2))
        if steps and (best is None or len(steps) > len(best[0])):
            best = (steps, losses, total)
    if not best:
        raise RuntimeError("训练日志中未解析到 step/loss 行")
    return best


# ---------------------------------------------------------------------------
# 图 1：WER 双集对比
# ---------------------------------------------------------------------------
def fig1_wer_comparison(data, oracle, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.04, wspace=0.12, hspace=0.04)

    for ax, split in zip(axes, ("test-clean", "test-other")):
        labels = [c for c in CONFIG_ORDER if (split, c) in data]
        vals = [float(data[(split, c)]["wer"]) * 100 for c in labels]
        xs = np.arange(len(labels))
        bars = ax.bar(
            xs,
            vals,
            width=0.66,
            color=[PALETTE[c] for c in labels],
            edgecolor=BG,
            linewidth=0.6,
            zorder=3,
        )
        # Beam-1 为参考基线：叠加斜线纹理，作为颜色之外的第二重身份编码
        for b, c in zip(bars, labels):
            if c == "Beam-1":
                b.set_hatch("////")
                b.set_edgecolor("#FFFFFF")
                b.set_linewidth(1.0)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.05,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=TEXT,
            )
        # 参考线：Beam-1 与 Top-5 Oracle
        beam1 = vals[labels.index("Beam-1")] if "Beam-1" in labels else None
        xr = len(labels) - 0.5
        if beam1 is not None:
            ax.axhline(beam1, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
            ax.text(
                xr + 0.08,
                beam1,
                f"Beam-1\n{beam1:.2f}%",
                ha="left",
                va="center",
                fontsize=8.0,
                color=MUTED,
                clip_on=False,
            )
        orc = oracle[split]
        ax.axhline(orc, color=ORACLE, linestyle=(0, (1.5, 2)), linewidth=1.1, zorder=2)
        ax.text(
            xr + 0.08,
            orc,
            f"Oracle\n{orc:.2f}%",
            ha="left",
            va="center",
            fontsize=8.0,
            color=ORACLE,
            clip_on=False,
        )
        ax.set_xlim(-0.62, xr + 0.02)

        ax.set_xticks(xs)
        ax.set_xticklabels(
            [c.replace(" + ", "\n+ ") if c == "NF4 + QLoRA" else c for c in labels],
            fontsize=9.0,
        )
        ax.set_ylabel("WER (%)")
        ax.set_title(split, fontsize=11.5, pad=16, fontweight="semibold")
        if split == "test-other":
            ax.text(
                0.5,
                1.015,
                "INT8 not evaluated on test-other",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#9CA3AF",
                style="italic",
            )
        lo = min(vals + [orc]) - 0.9
        hi = max(vals + [orc]) + 0.55
        ax.set_ylim(lo, hi)
        style_axes(ax)

    handles = []
    for c in CONFIG_ORDER:
        kw = {"hatch": "////", "edgecolor": "#FFFFFF", "linewidth": 1.0} if c == "Beam-1" else {"edgecolor": "none"}
        handles.append(Patch(facecolor=PALETTE[c], label=c, **kw))
    handles += [
        Line2D([0], [0], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, label="Beam-1 baseline"),
        Line2D([0], [0], color=ORACLE, linestyle=(0, (1.5, 2)), linewidth=1.1, label="Top-5 oracle"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5,
        fontsize=8.8,
        handlelength=1.5,
        columnspacing=1.4,
    )
    path = outdir / "fig1_wer_comparison.svg"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(PREVIEW / "_preview_fig1.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 2：S/D/I 中 D 与 I 的对称分化
# ---------------------------------------------------------------------------
def fig2_sdi_divergence(data, outdir: Path) -> Path:
    cfgs = ["FP16", "Q8_0", "NF4", "Q4_K_M", "NF4 + QLoRA"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.04, wspace=0.10)

    for ax, split in zip(axes, ("test-clean", "test-other")):
        labels = [c for c in cfgs if (split, c) in data]
        d = [int(data[(split, c)]["del"]) for c in labels]
        i = [int(data[(split, c)]["ins"]) for c in labels]
        xs = np.arange(len(labels))
        w = 0.36
        bd = ax.bar(xs - w / 2, d, width=w, color=D_COLOR, edgecolor=BG, linewidth=0.6, zorder=3, label="Deletion (D)")
        bi = ax.bar(xs + w / 2, i, width=w, color=I_COLOR, edgecolor=BG, linewidth=0.6, zorder=3, label="Insertion (I)")
        for bars in (bd, bi):
            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 8,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    color=TEXT,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels([c.replace(" + ", "\n+ ") if c == "NF4 + QLoRA" else c for c in labels], fontsize=9.2)
        ax.set_title(split, fontsize=11.5, pad=9, fontweight="semibold")
        ax.set_ylim(0, max(max(d), max(i)) * 1.16)
        style_axes(ax)

    axes[0].set_ylabel("Error count (words)")
    handles = [
        Patch(facecolor=D_COLOR, edgecolor="none", label="Deletion (D)"),
        Patch(facecolor=I_COLOR, edgecolor="none", label="Insertion (I)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=9.4, handlelength=1.5)
    path = outdir / "fig2_sdi_divergence.svg"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(PREVIEW / "_preview_fig2.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 3：3D 权衡空间（显存 × 延迟 × 精度）
# ---------------------------------------------------------------------------
def fig3_3d_tradeoff(data, outdir: Path) -> Path:
    cfgs = ["FP16", "INT8", "Q8_0", "Q4_K_M", "NF4", "NF4 + QLoRA"]
    # 标签偏移量（屏幕点坐标 dx, dy）与对齐方式；2D 偏移保证任何视角下零遮挡
    offs = {
        "test-clean": {
            "FP16": ((9, 0), "left"),
            "INT8": ((9, 1), "left"),
            "Q8_0": ((-9, -1), "right"),
            "Q4_K_M": ((-9, 3), "right"),
            "NF4": ((-9, 6), "right"),
            "NF4 + QLoRA": ((10, 1), "left"),
        },
        "test-other": {
            "FP16": ((9, 0), "left"),
            "Q8_0": ((-9, -3), "right"),
            "Q4_K_M": ((-9, 3), "right"),
            "NF4": ((-9, 5), "right"),
            "NF4 + QLoRA": ((10, -3), "left"),
        },
    }
    xr = (1.05, 3.85)
    yr = (-1.72, -0.16)
    zr = (0.0, 14.6)

    fig = plt.figure(figsize=(12.6, 5.6))
    pending: list[tuple] = []  # 投影完成后再打标签，保证屏幕坐标零遮挡
    for idx, split in enumerate(("test-clean", "test-other"), start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.set_facecolor(BG)
        labels = [c for c in cfgs if (split, c) in data]
        for c in labels:
            row = data[(split, c)]
            x = float(row["peak_vram_mb"]) / 1024.0
            y = np.log10(max(float(row["rtf"]), 1e-4))
            z = float(row["wer"]) * 100
            col = PALETTE[c]
            # 垂直落线 + 底面投影
            ax.plot([x, x], [y, y], [zr[0], z], color=col, linewidth=1.1, alpha=0.55, zorder=2)
            ax.scatter([x], [y], [zr[0]], color=col, s=18, alpha=0.4, edgecolors="none", depthshade=False)
            ax.scatter([x], [y], [z], color=col, s=66, edgecolors=BG, linewidths=0.9, depthshade=False, zorder=5)
            (dx, dy), ha = offs[split][c]
            pending.append((ax, c, x, y, z, dx, dy, ha))

        ax.set_xlabel("Peak VRAM (GB)", labelpad=8, fontsize=9.4)
        ax.set_ylabel("log$_{10}$(RTF)", labelpad=8, fontsize=9.4)
        ax.set_zlabel("WER (%)", labelpad=6, fontsize=9.4)
        ax.set_title(split, fontsize=11.5, pad=0, fontweight="semibold")
        ax.view_init(elev=23, azim=-56)
        ax.set_box_aspect((1.3, 1.0, 0.9))
        # 坐标轴与网格风格
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(BG)
            axis.pane.set_edgecolor(GRID)
            axis._axinfo["grid"]["color"] = GRID
            axis._axinfo["grid"]["linewidth"] = 0.8
            axis.line.set_color("#9CA3AF")
            axis.line.set_linewidth(0.8)
        ax.tick_params(colors=TEXT, labelsize=8.2, pad=1.5)
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_zlim(*zr)
        ax.zaxis.set_major_locator(matplotlib.ticker.MultipleLocator(3))

    fig.tight_layout(w_pad=2.0)
    # 视图确定后投影到 2D，再按屏幕点偏移打标签
    from mpl_toolkits.mplot3d import proj3d

    fig.canvas.draw()
    for ax, c, x, y, z, dx, dy, ha in pending:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.annotate(
            c,
            xy=(x2, y2),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.3,
            color=TEXT,
            ha=ha,
            va="center",
            annotation_clip=False,
            zorder=7,
        )

    path = outdir / "fig3_3d_tradeoff.svg"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(PREVIEW / "_preview_fig3.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 4：QLoRA 收敛曲线
# ---------------------------------------------------------------------------
def fig4_qlora_training_loss(outdir: Path) -> Path:
    steps, losses, total = load_loss_curve()
    steps_a = np.asarray(steps)
    loss_a = np.asarray(losses)
    # 滑动平均（窗口 25 步 ≈ 1/70 epoch，足以平滑 batch 噪声）
    win = 25
    kernel = np.ones(win) / win
    smooth = np.convolve(loss_a, kernel, mode="valid")
    smooth_x = steps_a[win - 1:]

    fig, ax = plt.subplots(figsize=(9.4, 4.2), layout="constrained")
    ax.plot(steps_a, loss_a, color=PALETTE["INT8"], linewidth=0.55, alpha=0.45, zorder=2, label="Step loss")
    ax.plot(smooth_x, smooth, color=PALETTE["FP16"], linewidth=1.9, zorder=4, label="Moving average (w=25)")

    half = total / 2.0
    ax.axvline(half, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.text(half - total * 0.012, ax.get_ylim()[1] * 0.94, "Epoch 1", ha="right", va="top", fontsize=8.8, color=MUTED)
    ax.text(half + total * 0.012, ax.get_ylim()[1] * 0.94, "Epoch 2", ha="left", va="top", fontsize=8.8, color=MUTED)

    ax.annotate(
        f"final {loss_a[-1]:.3f}",
        xy=(steps_a[-1], loss_a[-1]),
        xytext=(-62, 26),
        textcoords="offset points",
        fontsize=8.6,
        color=TEXT,
        arrowprops=dict(arrowstyle="-", color="#9CA3AF", linewidth=0.8),
    )
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss")
    ax.set_xlim(0, total)
    ax.set_ylim(0, min(loss_a.max(), 1.0) * 1.05)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=9.0, handlelength=1.8)
    path = outdir / "fig4_qlora_training_loss.svg"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(PREVIEW / "_preview_fig4.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 图 5（可选加餐）：S/D/I 3D 柱状矩阵
# ---------------------------------------------------------------------------
def fig5_3d_sdi_landscape(data, outdir: Path) -> Path | None:
    cfgs = ["FP16", "Q8_0", "NF4", "Q4_K_M", "NF4 + QLoRA"]
    splits = ["test-clean", "test-other"]
    kinds = [("del", "D", D_COLOR), ("ins", "I", I_COLOR)]

    fig = plt.figure(figsize=(11.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    bw = 0.30  # 每组内 D / I 并排，互不遮挡
    for xi, c in enumerate(cfgs):
        for yi, split in enumerate(splits):
            if (split, c) not in data:
                continue
            row = data[(split, c)]
            for zi, (key, _, col) in enumerate(kinds):
                v = int(row[key])
                x0 = xi + (zi - 0.5) * bw
                ax.bar3d(x0 - bw / 2, yi - 0.34, 0, bw, 0.68, v, color=col, alpha=0.92, edgecolor=BG, linewidth=0.4, shade=True)

    ax.set_xticks(np.arange(len(cfgs)))
    ax.set_xticklabels(cfgs, fontsize=8.8)
    ax.set_yticks(np.arange(len(splits)))
    ax.set_yticklabels(splits, fontsize=9.0)
    ax.set_zlabel("Error count (words)", labelpad=8, fontsize=9.4)
    ax.view_init(elev=26, azim=-62)
    ax.set_box_aspect((1.55, 1.0, 0.78))
    ax.set_zlim(0, 1250)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(BG)
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.8
        axis.line.set_color("#9CA3AF")
        axis.line.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=8.2, pad=1.5)
    handles = [Patch(facecolor=D_COLOR, label="Deletion (D)"), Patch(facecolor=I_COLOR, label="Insertion (I)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=9.0, handlelength=1.4)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    path = outdir / "fig5_3d_sdi_landscape.svg"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(PREVIEW / "_preview_fig5.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "docs" / "figures"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    setup_style()
    data = load_summary()
    oracle = load_oracle()
    print("Oracle:", {k: round(v, 4) for k, v in oracle.items()})

    paths = [
        fig1_wer_comparison(data, oracle, outdir),
        fig2_sdi_divergence(data, outdir),
        fig3_3d_tradeoff(data, outdir),
        fig4_qlora_training_loss(outdir),
    ]
    try:
        p5 = fig5_3d_sdi_landscape(data, outdir)
        if p5:
            paths.append(p5)
    except Exception as exc:  # 加餐图失败不影响主图
        print(f"[warn] fig5 跳过：{exc}")

    for p in paths:
        print(f"生成 {p.relative_to(REPO)}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
