#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""路线 A 论文级 SVG 图表生成（数据 100% 来自本地实验产物）。

数据源：
  outputs/metrics/summary.csv           # whisper-normalizer 1.0.1 统一口径
  outputs/nbest/*.jsonl                 # Top-5 Oracle（按 1.0.1 重算）
  outputs/logs/route-a-*-qlora.log      # Task 1.6b 云端 1782 步训练日志
  docs/route_a_analysis.json            # 改写率统计（analyze_route_a_cases.py）

用法：
    python scripts/analyze_route_a_cases.py   # 先生成改写率统计
    python scripts/plot_route_a.py            # → docs/figures/fig{1..5}_*.svg
    python scripts/plot_route_a.py --png-check-dir /tmp/fig_check   # 附带 PNG 自检

图号严格按 docs/ROUTE_A_REPORT.md 正文出现顺序：
  fig1 WER 主对比 / fig2 QLoRA 收敛 / fig3 S/D/I 分化 /
  fig4 三维权衡 / fig5 改写结构

绘图规范：低饱和顶刊色板、内向刻度、细坐标轴、深色柱体描边、
(a)/(b) 子图编号；图内不写解释性文字。
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
sys.path.insert(0, str(REPO))

from common.metrics import compute_wer  # noqa: E402
from common.text_norm import normalize_for_wer  # noqa: E402

# PNG 自检输出目录（默认不落盘；--png-check-dir 指定后启用）
PNG_CHECK_DIR: Path | None = None

# ---------------------------------------------------------------------------
# 全局视觉规范
# ---------------------------------------------------------------------------
BG = "#FFFFFF"
GRID = "#E5E7EB"
AXIS = "#2B2D42"
TEXT = "#1F2937"
MUTED = "#6C757D"
INSET_FRAME = "#CBD5E1"

PALETTE = {
    "Beam-1": "#6C757D",        # 中性石板灰
    "FP16": "#3B6291",          # 深钢蓝
    "INT8": "#7DA6D2",          # 柔和天蓝
    "Q8_0": "#2A9D8F",          # 青绿
    "Q4_K_M": "#E9B44C",        # 暖琥珀
    "NF4": "#D96B43",           # 赤陶红
    "NF4 + QLoRA": "#B83A1D",   # 深砖红
}
ORACLE = "#7B6D9A"              # 灰紫
D_COLOR = "#3B6291"             # 删除错误
I_COLOR = "#D96B43"             # 插入错误
RATIO_COLOR = "#7B6D9A"         # I/D 比值
NEUTRAL = "#CDD2D8"             # 改写中性
GOOD = "#2A9D8F"
BAD = "#D96B43"

CONFIG_ORDER = ["Beam-1", "FP16", "INT8", "Q8_0", "Q4_K_M", "NF4", "NF4 + QLoRA"]
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
            "font.size": 9.2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "path",
            "figure.dpi": 150,
        }
    )


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)


def panel_label(ax, text: str, x: float = -0.02, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=TEXT,
    )


def save_fig(fig, path: Path, stem: str) -> Path:
    """保存 SVG；若启用自检则同步渲染 200 DPI PNG 到独立目录。"""
    fig.savefig(path, bbox_inches="tight")
    if PNG_CHECK_DIR is not None:
        PNG_CHECK_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG_CHECK_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def load_summary() -> dict[tuple[str, str], dict]:
    path = REPO / "outputs" / "metrics" / "summary.csv"
    out: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            if split not in ("test-clean", "test-other"):
                continue
            for prefix, label in EXP_KEYS:
                if row["exp_name"].startswith(prefix):
                    out.setdefault((split, label), row)
                    break
    return out


def load_oracle() -> dict[str, float]:
    res: dict[str, float] = {}
    for split in ("test-clean", "test-other"):
        rows = [
            json.loads(line)
            for line in (REPO / "outputs" / "nbest" / f"{split}.jsonl").open(encoding="utf-8")
            if line.strip()
        ]
        refs, sel = [], []
        for r in rows:
            ref = normalize_for_wer(r["ref"])
            cands = [normalize_for_wer(c["text"]) for c in r["nbest"]]
            scored = [(compute_wer([ref], [c])["wer"], i) for i, c in enumerate(cands)]
            refs.append(ref)
            sel.append(cands[min(scored)[1]])
        res[split] = compute_wer(refs, sel)["wer"] * 100
    return res


def load_loss_curve() -> tuple[np.ndarray, np.ndarray, int]:
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
    return np.asarray(best[0]), np.asarray(best[1]), best[2]


def load_rewrite_stats() -> dict:
    path = REPO / "docs" / "route_a_analysis.json"
    if not path.exists():
        raise FileNotFoundError(
            "缺少 docs/route_a_analysis.json，请先运行 python scripts/analyze_route_a_cases.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))["stats"]


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------
def rolling_stats(x: np.ndarray, y: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """居中滑动均值与 ±1 标准差（valid 段）。"""
    mean = np.convolve(y, np.ones(win) / win, mode="valid")
    sq = np.convolve(y * y, np.ones(win) / win, mode="valid")
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return x[win - 1:], mean, std


# ---------------------------------------------------------------------------
# 图 1：WER 主对比
# ---------------------------------------------------------------------------
def fig1_wer_comparison(data, oracle, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.03, wspace=0.10)

    for idx, (ax, split) in enumerate(zip(axes, ("test-clean", "test-other"))):
        labels = [c for c in CONFIG_ORDER if (split, c) in data]
        vals = [float(data[(split, c)]["wer"]) * 100 for c in labels]
        xs = np.arange(len(labels))
        orc = oracle[split]
        lo = min(vals + [orc]) - 0.55

        # Top-5 Oracle 以下的“超越区”
        ax.axhspan(lo, orc, color=ORACLE, alpha=0.07, lw=0, zorder=0)
        ax.axhline(orc, color=ORACLE, linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)

        bars = ax.bar(
            xs,
            vals,
            width=0.62,
            color=[PALETTE[c] for c in labels],
            edgecolor=AXIS,
            linewidth=0.6,
            zorder=3,
        )
        # 全部数值标签统一置于柱顶外侧正上方
        beam1 = vals[labels.index("Beam-1")]
        for b, c, v in zip(bars, labels, vals):
            kw = dict(ha="center", va="bottom", fontsize=8.2, color=TEXT, zorder=6)
            y = v + 0.035
            # 参考线若落在标签字身区间，则将标签抬至线上方，杜绝压线
            for ref in (beam1, orc):
                if v + 0.015 < ref < v + 0.16:
                    y = max(y, ref + 0.045)
            # 参考线仅擦过柱顶（如 4.22 与 Oracle 4.25）时，加极窄纯白底避让
            if any(v < ref <= v + 0.05 for ref in (beam1, orc)):
                kw["bbox"] = dict(facecolor=BG, edgecolor="none", pad=0.6, alpha=0.9)
            ax.text(b.get_x() + b.get_width() / 2, y, f"{v:.2f}", **kw)

        ax.axhline(beam1, color=MUTED, linestyle=(0, (1, 1.6)), linewidth=0.9, zorder=2)

        # 参考线标签置于右侧留白
        xr = len(labels) - 0.42
        ax.text(xr + 0.10, beam1, f"Beam-1\n{beam1:.2f}%", ha="left", va="center", fontsize=7.6, color=MUTED, linespacing=1.15)
        ax.text(xr + 0.10, orc, f"Oracle\n{orc:.2f}%", ha="left", va="center", fontsize=7.6, color=ORACLE, linespacing=1.15)

        ax.set_xlim(-0.60, xr + 0.05)
        ax.set_ylim(lo, max(vals) + 0.42)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [c.replace(" + ", "\n+ ") if c == "NF4 + QLoRA" else c for c in labels],
            fontsize=8.4,
        )
        ax.set_ylabel("WER (%)")
        panel_label(ax, f"({'a' if idx == 0 else 'b'}) {split}", x=0.0, y=1.02)
        style_axes(ax)

    # 柱体颜色已由 X 轴标签完整表达，底部仅保留参考线图例
    handles = [
        Line2D([0], [0], color=MUTED, linestyle=(0, (1, 1.6)), linewidth=0.9, label="Beam-1 baseline"),
        Line2D([0], [0], color=ORACLE, linestyle=(0, (4, 2)), linewidth=0.9, label="Top-5 oracle"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        fontsize=8.0,
        handlelength=1.8,
        columnspacing=2.0,
    )
    path = outdir / "fig1_wer_comparison.svg"
    return save_fig(fig, path, "fig1_wer_comparison")


# ---------------------------------------------------------------------------
# 图 3：S/D/I 分化（上：I/D 比值，下：D / I 词数）
# ---------------------------------------------------------------------------
def fig3_sdi_divergence(data, outdir: Path) -> Path:
    cfgs = ["FP16", "Q8_0", "NF4", "Q4_K_M", "NF4 + QLoRA"]
    fig = plt.figure(figsize=(11.2, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2], hspace=0.08, wspace=0.12)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.03)

    for idx, split in enumerate(("test-clean", "test-other")):
        ax_r = fig.add_subplot(gs[0, idx])
        ax_b = fig.add_subplot(gs[1, idx], sharex=ax_r)

        labels = [c for c in cfgs if (split, c) in data]
        d = np.array([int(data[(split, c)]["del"]) for c in labels])
        i = np.array([int(data[(split, c)]["ins"]) for c in labels])
        xs = np.arange(len(labels))
        ratio = i / np.maximum(d, 1)
        w = 0.34

        # --- 上薄面板：I/D 比值 ---
        ax_r.axhline(1.0, color=MUTED, linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
        ax_r.plot(
            xs,
            ratio,
            color=RATIO_COLOR,
            marker="D",
            markersize=5.2,
            markerfacecolor=BG,
            markeredgewidth=1.0,
            linewidth=1.15,
            zorder=4,
        )
        # 逐点标注：标签朝远离 I/D=1.0 参考线的一侧放置，避免压线
        for x, r in zip(xs, ratio):
            above = r >= 1.0
            ax_r.text(
                x,
                r + 0.15 if above else r - 0.15,
                f"{r:.2f}",
                ha="center",
                va="bottom" if above else "top",
                fontsize=7.4,
                color=RATIO_COLOR,
                zorder=6,
                bbox=dict(facecolor=BG, edgecolor="none", pad=0.4, alpha=0.9),
            )
        rlo, rhi = ratio.min(), ratio.max()
        pad = (rhi - rlo) * 0.34 + 0.18
        ax_r.set_ylim(max(0.0, rlo - pad), rhi + pad)
        ax_r.set_ylabel("Insertion / Deletion", fontsize=8.6)
        ax_r.set_yticks([0.5, 1.0, 1.5, 2.0, 2.5] if rhi > 1.8 else [0.5, 1.0, 1.5])
        style_axes(ax_r)
        ax_r.spines["bottom"].set_visible(False)
        ax_r.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        panel_label(ax_r, f"({'a' if idx == 0 else 'b'}) {split}", x=0.0, y=1.06)

        # --- 下主面板：D / I 分组柱 ---
        bd = ax_b.bar(xs - w / 2, d, width=w, color=D_COLOR, edgecolor=AXIS, linewidth=0.6, zorder=3)
        bi = ax_b.bar(xs + w / 2, i, width=w, color=I_COLOR, edgecolor=AXIS, linewidth=0.6, zorder=3)
        top = max(d.max(), i.max())
        for bars in (bd, bi):
            for b in bars:
                h = b.get_height()
                ax_b.text(
                    b.get_x() + b.get_width() / 2,
                    h + top * 0.015,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    color=TEXT,
                )
        ax_b.set_ylim(0, top * 1.12)
        ax_b.set_xticks(xs)
        ax_b.set_xticklabels(
            [c.replace(" + ", "\n+ ") if c == "NF4 + QLoRA" else c for c in labels],
            fontsize=8.4,
        )
        ax_b.set_ylabel("Error count (words)", fontsize=8.6)
        style_axes(ax_b)

    handles = [
        Patch(facecolor=D_COLOR, edgecolor=AXIS, linewidth=0.6, label="Deletion (D)"),
        Patch(facecolor=I_COLOR, edgecolor=AXIS, linewidth=0.6, label="Insertion (I)"),
        Line2D([0], [0], color=RATIO_COLOR, marker="D", markerfacecolor=BG, markeredgewidth=1.0,
               linewidth=1.15, markersize=5.2, label="I/D ratio (upper)"),
        Line2D([0], [0], color=MUTED, linestyle=(0, (4, 2)), linewidth=0.8, label="I/D = 1.0"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8.0, handlelength=1.6, columnspacing=1.4)
    path = outdir / "fig3_sdi_divergence.svg"
    return save_fig(fig, path, "fig3_sdi_divergence")


# ---------------------------------------------------------------------------
# 图 4：3D 三维权衡空间
# ---------------------------------------------------------------------------
def fig4_3d_tradeoff(data, oracle, outdir: Path) -> Path:
    cfgs = ["FP16", "INT8", "Q8_0", "Q4_K_M", "NF4", "NF4 + QLoRA"]
    offs = {
        "test-clean": {
            "FP16": ((10, 2), "left"),
            "INT8": ((10, 2), "left"),
            "Q8_0": ((-11, -4), "right"),
            "Q4_K_M": ((-11, 5), "right"),
            "NF4": ((-11, 8), "right"),
            "NF4 + QLoRA": ((12, 1), "left"),
        },
        "test-other": {
            "FP16": ((10, 2), "left"),
            "Q8_0": ((-12, 2), "right"),
            "Q4_K_M": ((-11, 6), "right"),
            "NF4": ((-11, 9), "right"),
            "NF4 + QLoRA": ((12, -2), "left"),
        },
    }
    xr = (1.12, 3.62)
    yr = (-1.72, -0.16)
    zr = {"test-clean": (3.8, 6.6), "test-other": (10.2, 13.6)}
    rtf_ticks = [0.03, 0.05, 0.10, 0.20, 0.50]

    fig = plt.figure(figsize=(12.6, 5.6))
    pending: list[tuple] = []
    for idx, split in enumerate(("test-clean", "test-other")):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        ax.set_facecolor(BG)
        labels = [c for c in cfgs if (split, c) in data]
        zlo, zhi = zr[split]
        orc = oracle[split]

        # Oracle 半透明参考面
        gx, gy = np.meshgrid(np.linspace(xr[0], xr[1], 2), np.linspace(yr[0], yr[1], 2))
        ax.plot_surface(gx, gy, np.full_like(gx, orc), color=ORACLE, alpha=0.08, shade=False, linewidth=0, zorder=0)

        pts = {}
        for c in labels:
            row = data[(split, c)]
            x = float(row["peak_vram_mb"]) / 1024.0
            y = np.log10(max(float(row["rtf"]), 1e-4))
            z = float(row["wer"]) * 100
            pts[c] = (x, y, z)
            col = PALETTE[c]
            # 仅保留垂直落线 + 底面投影点（不向侧墙/后墙投影）
            ax.plot([x, x], [y, y], [zlo, z], color=col, linewidth=1.0, alpha=0.55, zorder=2)
            ax.scatter([x], [y], [zlo], color=col, s=15, alpha=0.40, edgecolors="none", depthshade=False)
            ax.scatter([x], [y], [z], color=col, s=58, edgecolors=AXIS, linewidths=0.7, depthshade=False, zorder=5)
            pending.append((ax, c, x, y, z, *offs[split][c]))

        # 3D 帕累托前沿：Q4_K_M → Q8_0 → NF4 + QLoRA
        front = ["Q4_K_M", "Q8_0", "NF4 + QLoRA"]
        if all(c in pts for c in front):
            fx = [pts[c][0] for c in front]
            fy = [pts[c][1] for c in front]
            fz = [pts[c][2] for c in front]
            ax.plot(fx, fy, fz, color=MUTED, linestyle=(0, (5, 3)), linewidth=1.0, alpha=0.85, zorder=4)

        ax.set_xlabel("Peak VRAM (GB)", labelpad=7, fontsize=9)
        ax.set_ylabel("RTF", labelpad=7, fontsize=9)
        ax.set_zlabel("WER (%)", labelpad=6, fontsize=9)
        ax.set_xticks([1.5, 2.0, 2.5, 3.0, 3.5])
        ax.set_yticks(np.log10(rtf_ticks))
        ax.set_yticklabels([f"{t:.2f}" for t in rtf_ticks], fontsize=7.8)
        ax.view_init(elev=22, azim=-48)
        ax.set_box_aspect((1.28, 1.0, 0.92))
        ax.set_xlim(*xr)
        ax.set_ylim(*yr)
        ax.set_zlim(zlo, zhi)
        ax.zaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.6 if split == "test-clean" else 1.0))
        ax.tick_params(colors=AXIS, labelsize=7.8, pad=1, direction="in")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(BG)
            axis.pane.set_edgecolor("#BFC4CC")
            axis.pane.set_alpha(1.0)
            axis._axinfo["grid"]["color"] = "#D8DCE2"
            axis._axinfo["grid"]["linewidth"] = 0.7
            axis._axinfo["grid"]["alpha"] = 0.25
            axis.line.set_color(AXIS)
            axis.line.set_linewidth(0.8)
        ax.text2D(
            0.0,
            1.0,
            f"({'a' if idx == 0 else 'b'}) {split}",
            transform=ax.transAxes,
            fontsize=10.2,
            fontweight="bold",
            color=TEXT,
        )

    fig.tight_layout(w_pad=1.6)
    fig.canvas.draw()
    from mpl_toolkits.mplot3d import proj3d

    for ax, c, x, y, z, xytext, ha in pending:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.annotate(
            c,
            xy=(x2, y2),
            xytext=xytext,
            textcoords="offset points",
            fontsize=7.9,
            color=TEXT,
            ha=ha,
            va="center",
            annotation_clip=False,
            zorder=7,
            bbox=dict(facecolor=BG, alpha=0.8, edgecolor="none", pad=0.8),
            arrowprops=dict(arrowstyle="-", color="#9CA3AF", linewidth=0.55, shrinkA=1, shrinkB=2),
        )

    # 底部单行极简图例
    handles = [
        Line2D([0], [0], color=MUTED, linestyle=(0, (5, 3)), linewidth=1.1, label="Pareto frontier"),
        Patch(facecolor=ORACLE, alpha=0.18, edgecolor="none", label="Top-5 Oracle plane"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8.2, handlelength=1.9, columnspacing=2.2)

    path = outdir / "fig4_3d_tradeoff.svg"
    return save_fig(fig, path, "fig4_3d_tradeoff")


# ---------------------------------------------------------------------------
# 图 2：QLoRA 收敛曲线 + inset zoom
# ---------------------------------------------------------------------------
def fig2_qlora_training_loss(outdir: Path) -> Path:
    steps, losses, total = load_loss_curve()
    win = 25
    sx, mean, std = rolling_stats(steps, losses, win)

    fig, ax = plt.subplots(figsize=(9.0, 3.8), layout="constrained")
    ax.plot(steps, losses, color=PALETTE["INT8"], linewidth=0.45, alpha=0.22, zorder=1, label="Step loss")
    ax.fill_between(sx, mean - std, mean + std, color=PALETTE["FP16"], alpha=0.16, linewidth=0, zorder=2, label="±1 std")
    ax.plot(sx, mean, color=PALETTE["FP16"], linewidth=1.8, zorder=3, label="Moving average (w=25)")

    half = total / 2.0
    ax.axvline(half, color=MUTED, linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    ax.set_xlim(0, total)
    ax.set_ylim(0.0, 0.42)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss")
    style_axes(ax)

    # Epoch 分界标签：虚线顶侧偏左
    ax.text(
        half - total * 0.010,
        0.975,
        "Epoch 1 | Epoch 2",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7.8,
        color=MUTED,
        zorder=6,
    )

    # inset：Epoch 2 收敛尾部，严格置于分界虚线右侧，纯白不透明
    axins = ax.inset_axes([0.54, 0.54, 0.42, 0.40], zorder=10)
    tail = sx >= 900
    axins.plot(steps[steps >= 900], losses[steps >= 900], color=PALETTE["INT8"], linewidth=0.4, alpha=0.25, zorder=1)
    axins.fill_between(sx[tail], (mean - std)[tail], (mean + std)[tail], color=PALETTE["FP16"], alpha=0.16, linewidth=0, zorder=2)
    axins.plot(sx[tail], mean[tail], color=PALETTE["FP16"], linewidth=1.4, zorder=3)
    axins.set_xlim(900, total)
    axins.set_ylim(0.0, 0.13)
    axins.set_xticks([1000, 1200, 1400, 1600])
    axins.set_yticks([0.0, 0.04, 0.08, 0.12])
    axins.tick_params(labelsize=6.6, direction="in", colors=AXIS, pad=1.5)
    for side in ("top", "right", "left", "bottom"):
        axins.spines[side].set_visible(True)
        axins.spines[side].set_color(INSET_FRAME)
        axins.spines[side].set_linewidth(0.8)
    axins.set_facecolor(BG)
    axins.patch.set_alpha(1.0)
    # 末步标注：置于 inset 右下空白区，短引线不穿越滑动平均线
    axins.annotate(
        f"Final step: {losses[-1]:.3f}",
        xy=(steps[-1], losses[-1]),
        xytext=(-6, -8),
        textcoords="offset points",
        fontsize=6.6,
        color=TEXT,
        ha="right",
        va="top",
        zorder=11,
        bbox=dict(facecolor=BG, edgecolor="none", pad=0.4, alpha=0.9),
        arrowprops=dict(arrowstyle="-", color="#9CA3AF", linewidth=0.6, shrinkA=1, shrinkB=2),
    )

    # 图例：Step loss 项加深可见度（仅图例内提升 alpha）
    handles, labels = ax.get_legend_handles_labels()
    handles[0].set_alpha(0.55)
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=7.8, handlelength=1.6)

    path = outdir / "fig2_qlora_training_loss.svg"
    return save_fig(fig, path, "fig2_qlora_training_loss")


# ---------------------------------------------------------------------------
# 图 5：改写结构 + 句均 ΔWER
# ---------------------------------------------------------------------------
def fig5_rewrite_analysis(stats, outdir: Path) -> Path:
    cfgs = ["FP16", "INT8", "Q8_0", "Q4_K_M", "NF4", "NF4 + QLoRA"]
    splits = [("test-clean", "clean"), ("test-other", "other")]

    rows = []
    for split, tag in splits:
        for c in cfgs:
            st = stats.get(split, {}).get(c)
            if st is None:
                continue
            rew = st["rewritten"]
            rows.append(
                {
                    "label": f"{c} ({tag})",
                    "cfg": c,
                    "split": split,
                    "improved": st["improved"] / rew,
                    "neutral": st["changed_neutral"] / rew,
                    "worsened": st["worsened"] / rew,
                    "delta": st["mean_delta_wer_pts"],
                }
            )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 4.9), gridspec_kw={"width_ratios": [1.35, 1.0]}, layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.04, hspace=0.03)

    # (a) 水平堆叠条：clean / other 两组之间留 0.5 行距并加细分隔虚线
    n_clean = sum(1 for r in rows if r["split"] == "test-clean")
    ys = []
    y_cur = (len(rows) + 0.5) - 1.0
    for k, r in enumerate(rows):
        if k == n_clean:
            y_cur -= 0.5  # 组间额外间隔
        ys.append(y_cur)
        y_cur -= 1.0
    ys = np.asarray(ys)
    left = np.zeros(len(rows))
    for key, col, name in (
        ("improved", GOOD, "Improved"),
        ("neutral", NEUTRAL, "Unchanged WER"),
        ("worsened", BAD, "Worsened"),
    ):
        vals = np.array([r[key] for r in rows])
        ax1.barh(ys, vals, left=left, height=0.62, color=col, edgecolor=AXIS, linewidth=0.55, label=name, zorder=3)
        for y, v, l in zip(ys, vals, left):
            if v >= 0.07:
                ax1.text(l + v / 2, y, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.8, color=BG if key != "neutral" else TEXT)
        left += vals
    # 组间极细浅灰水平虚线
    y_sep = (ys[n_clean - 1] + ys[n_clean]) / 2.0
    ax1.axhline(y_sep, color="#CBD5E1", linestyle=(0, (3, 3)), linewidth=0.7, zorder=2)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([r["label"] for r in rows], fontsize=7.6)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel("Share of rewritten utterances")
    ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7.8, handlelength=1.2, columnspacing=1.1)
    panel_label(ax1, "(a) Rewrite composition", x=0.0, y=1.03)
    style_axes(ax1, grid_axis="x")

    # (b) 双向棒棒糖图：颜色编码增益方向，形状编码数据集
    ybase = np.arange(len(cfgs))[::-1]
    for split, tag, marker, size, dy in (
        ("test-clean", "clean", "o", 44, 0.15),
        ("test-other", "other", "D", 32, -0.15),
    ):
        xs = []
        for c in cfgs:
            row = next((r for r in rows if r["cfg"] == c and r["split"] == split), None)
            xs.append(np.nan if row is None else row["delta"])
        xs = np.array(xs)
        yv = ybase + dy
        colors = [GOOD if v >= 0 else BAD for v in xs]
        ax2.hlines(yv, 0, np.nan_to_num(xs), color=colors, linewidth=1.3, zorder=3)
        ax2.scatter(xs, yv, s=size, c=colors, marker=marker, edgecolors=AXIS, linewidths=0.7, zorder=4)
        for y, v in zip(yv, xs):
            if np.isnan(v):
                continue
            ax2.text(
                v + (0.08 if v >= 0 else -0.08),
                y,
                f"{v:+.2f}",
                ha="left" if v >= 0 else "right",
                va="center",
                fontsize=6.7,
                color=TEXT,
            )
    ax2.axvline(0, color=AXIS, linewidth=0.9, zorder=2)
    ax2.set_yticks(ybase)
    ax2.set_yticklabels(cfgs, fontsize=7.6)
    ax2.set_ylim(-0.6, len(cfgs) - 0.4)
    ax2.set_xlabel("Mean per-utterance ΔWER (pts)")
    ax2.set_xlim(-1.25, 2.75)
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    panel_label(ax2, "(b) Utterance-level gain", x=0.0, y=1.03)
    style_axes(ax2, grid_axis="x")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BG, markeredgecolor=AXIS, markeredgewidth=0.8, markersize=5.2, label="test-clean"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=BG, markeredgecolor=AXIS, markeredgewidth=0.8, markersize=4.8, label="test-other"),
        Patch(facecolor=GOOD, label="Gain (ΔWER > 0)"),
        Patch(facecolor=BAD, label="Loss (ΔWER < 0)"),
    ]
    ax2.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=7.4, handlelength=1.1, columnspacing=1.1)

    path = outdir / "fig5_rewrite_analysis.svg"
    return save_fig(fig, path, "fig5_rewrite_analysis")


def main() -> None:
    global PNG_CHECK_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(REPO / "docs" / "figures"))
    ap.add_argument(
        "--png-check-dir",
        default=None,
        help="额外渲染 200 DPI PNG 到该目录用于自检（不纳入版本控制）",
    )
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.png_check_dir:
        PNG_CHECK_DIR = Path(args.png_check_dir)

    setup_style()
    data = load_summary()
    oracle = load_oracle()
    stats = load_rewrite_stats()
    print("Oracle:", {k: round(v, 4) for k, v in oracle.items()})

    # 图号严格对齐 docs/ROUTE_A_REPORT.md 正文出现顺序
    paths = [
        fig1_wer_comparison(data, oracle, outdir),
        fig2_qlora_training_loss(outdir),
        fig3_sdi_divergence(data, outdir),
        fig4_3d_tradeoff(data, oracle, outdir),
        fig5_rewrite_analysis(stats, outdir),
    ]
    for p in paths:
        print(f"生成 {p.relative_to(REPO)}  ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
