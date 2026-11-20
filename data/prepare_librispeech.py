#!/usr/bin/env python
"""扫描 LibriSpeech (OpenSLR 12) 目录树，生成统一 JSONL 音频清单。

遵循 PROJECT_SPEC.md 第 5 节数据契约：
    {"id": "1089-134686-0000",
     "audio_path": "/abs/path/to/1089-134686-0000.flac",
     "text": "HE HOPED THERE WOULD BE STEW FOR DINNER",
     "duration": 10.435}

目录结构（LibriSpeech 官方解压后）：
    <root>/LibriSpeech/<split>/<speaker_id>/<chapter_id>/<utt_id>.flac
    <root>/LibriSpeech/<split>/<speaker_id>/<chapter_id>/<speaker>-<chapter>.trans.txt

用法示例：
    # 仅扫描已有数据（--librispeech_root 指向包含 LibriSpeech/ 的目录）
    python data/prepare_librispeech.py --librispeech_root data \
        --splits test-clean test-other dev-clean

    # 下载并解压后生成清单（OpenSLR 12）
    python data/prepare_librispeech.py --download \
        --librispeech_root data --splits test-clean test-other dev-clean

    # 本地小样本冒烟测试
    python data/prepare_librispeech.py --librispeech_root data \
        --splits dev-clean --max_samples 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import soundfile as sf
from tqdm.auto import tqdm

logger = logging.getLogger("prepare_librispeech")

# OpenSLR 12 官方下载地址
OPENSLR_BASE_URL = "https://www.openslr.org/resources/12"
# 各 split 的 tar.gz 体积（MB），用于下载前提示
SPLIT_SIZE_MB = {
    "dev-clean": 337,
    "test-clean": 346,
    "test-other": 328,
    "train-clean-100": 6387,
}


def download_split(split: str, librispeech_root: Path) -> Path:
    """从 OpenSLR 12 下载并解压指定 split 到 librispeech_root。

    返回解压后的 split 目录（<root>/LibriSpeech/<split>）。
    幂等：split 目录已存在且非空时跳过下载。
    """
    split_dir = librispeech_root / "LibriSpeech" / split
    if split_dir.is_dir() and any(split_dir.iterdir()):
        logger.info("split=%s 已存在于 %s，跳过下载", split, split_dir)
        return split_dir

    url = f"{OPENSLR_BASE_URL}/{split}.tar.gz"
    tar_path = librispeech_root / f"{split}.tar.gz"
    librispeech_root.mkdir(parents=True, exist_ok=True)

    size_mb = SPLIT_SIZE_MB.get(split, "?")
    logger.info("下载 %s (~%s MB) -> %s", url, size_mb, tar_path)
    try:
        urllib.request.urlretrieve(url, tar_path)
    except Exception as exc:  # noqa: BLE001 - 下载失败需给出明确指引
        raise RuntimeError(
            f"下载失败：{url}\n"
            f"可手动下载后放置于 {tar_path} 再重跑；"
            f"国内网络可尝试镜像 https://us.openslr.org/resources/12/{split}.tar.gz"
        ) from exc

    logger.info("解压 %s ...", tar_path)
    # 官方 tar 内部顶层目录为 LibriSpeech/，解压到 root 后得到 root/LibriSpeech/<split>
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(librispeech_root)

    # 释放磁盘：解压完成即删除压缩包（train-clean-100 达 6GB+）
    tar_path.unlink(missing_ok=True)
    logger.info("split=%s 就绪：%s", split, split_dir)
    return split_dir


def parse_transcriptions(trans_file: Path) -> dict:
    """解析 <speaker>-<chapter>.trans.txt，返回 {utt_id: text}。

    文件格式（每行）：
        1089-134686-0000 HE HOPED THERE WOULD BE STEW FOR DINNER
    """
    utt2text: dict = {}
    # LibriSpeech 官方为纯 ASCII；按二进制读取以防个别文件含非 UTF-8 字节
    with open(trans_file, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            utt2text[utt_id] = text.upper()
    return utt2text


def build_manifest_for_split(
    librispeech_root: Path,
    split: str,
    output_path: Path,
    max_samples: Optional[int] = None,
    verify_audio: bool = False,
) -> int:
    """扫描单个 split，写出 JSONL 清单，返回写入条数。"""
    split_dir = librispeech_root / "LibriSpeech" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"未找到 split 目录：{split_dir}\n"
            f"请确认 --librispeech_root 指向包含 LibriSpeech/ 的目录，"
            f"或使用 --download 自动下载。"
        )

    # 收集所有 trans 文件：split/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt
    trans_files = sorted(split_dir.glob("*/*/*.trans.txt"))
    if not trans_files:
        raise FileNotFoundError(f"{split_dir} 下未发现任何 .trans.txt，目录结构异常")

    records = []
    n_skipped = 0
    for trans_file in tqdm(trans_files, desc=f"scan {split}", unit="chapter"):
        chapter_dir = trans_file.parent
        utt2text = parse_transcriptions(trans_file)
        for utt_id, text in sorted(utt2text.items()):
            flac_path = chapter_dir / f"{utt_id}.flac"
            if not flac_path.is_file():
                logger.warning("缺少音频文件，跳过：%s", flac_path)
                n_skipped += 1
                continue
            if verify_audio:
                _ = sf.read(str(flac_path), dtype="float32")
            duration = get_duration(flac_path)
            records.append(
                {
                    "id": utt_id,
                    "audio_path": str(flac_path.resolve()),
                    "text": text,
                    "duration": round(duration, 3),
                }
            )

    if n_skipped:
        logger.warning("split=%s 共跳过 %d 条缺失音频", split, n_skipped)

    # max_samples：按 LibriSpeech 原始顺序（说话人分组）截断，保证可复现
    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]
        logger.info("split=%s 应用 --max_samples=%d 截断", split, max_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_hours = sum(r["duration"] for r in records) / 3600.0
    logger.info(
        "split=%s 写入 %d 条 -> %s (总时长 %.2f 小时)",
        split,
        len(records),
        output_path,
        total_hours,
    )
    return len(records)


def get_duration(flac_path: Path) -> float:
    """读取 FLAC 时长（秒）。仅读头部元信息，不解码全部数据。"""
    info = sf.info(str(flac_path))
    return info.frames / info.samplerate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成 LibriSpeech JSONL 清单（PROJECT_SPEC.md 第 5 节契约）"
    )
    parser.add_argument(
        "--librispeech_root",
        type=Path,
        default=Path("data"),
        help="LibriSpeech 根目录（其下应包含 LibriSpeech/<split>）；"
        "下载的 tar.gz 也解压到这里（默认：data）",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test-clean", "test-other"],
        help="要处理的 split 列表（默认：test-clean test-other）",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/manifests"),
        help="JSONL 清单输出目录（默认：data/manifests）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="每个 split 最多写入的样本数（本地冒烟测试用）",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="先从 OpenSLR 12 下载缺失的 split（已存在的跳过）",
    )
    parser.add_argument(
        "--verify_audio",
        action="store_true",
        help="额外用 soundfile 完整校验每个 FLAC 可解码（较慢，仅排查数据问题时用）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.download:
        for split in args.splits:
            download_split(split, args.librispeech_root)

    total = 0
    for split in args.splits:
        out_path = args.output_dir / f"{split}.jsonl"
        n = build_manifest_for_split(
            args.librispeech_root,
            split,
            out_path,
            max_samples=args.max_samples,
            verify_audio=args.verify_audio,
        )
        total += n

    logger.info("全部完成：%d 个 split，共 %d 条样本", len(args.splits), total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
