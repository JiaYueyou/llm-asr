"""文本规范化模块（路线 A / B 共享）。

核心职责（PROJECT_SPEC.md 第 7.3 条）：
  参考文本（LibriSpeech 全大写无标点）与模型输出在送入 jiwer 之前，
  必须统一经过 EnglishTextNormalizer，并防御性剥离 LLM 可能残留的
  思考标签、对话前缀、包裹引号等污染。

用法：
    from common.text_norm import normalize_for_wer
    hyp = normalize_for_wer("Corrected text: HE HOPED THERE WOULD BE STEW")
"""

from __future__ import annotations

import re

# whisper-normalizer 新旧版本的导入路径不同：
#   - 顶层导出（部分版本）
#   - whisper_normalizer.english 子模块（0.1.x 实测路径）
try:
    from whisper_normalizer import EnglishTextNormalizer
except ImportError:  # pragma: no cover - 取决于安装版本
    from whisper_normalizer.english import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()

# Qwen3 思考模式标签：闭合（ 是 enable_thinking=False 时 Qwen3 模板注入的空标签。
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)

# 常见对话前缀（LLM 复述指令时产生），大小写不敏感，仅当出现在行首时剥离
_CHAT_PREFIXES = [
    "corrected text",
    "corrected transcription",
    "corrected sentence",
    "correction",
    "transcription",
    "transcript",
    "asr correction",
    "output",
    "answer",
    "result",
    "final text",
    "hypothesis",
]
_PREFIX_ALTS = "|".join(re.escape(p) for p in _CHAT_PREFIXES)
# 前缀后可跟 ":"、"："、"-" 或空白
_LEADING_PREFIX_RE = re.compile(
    rf"^\s*(?:\d+[.)]\s*)?(?:{_PREFIX_ALTS})\s*[:：\-–—]\s*",
    re.IGNORECASE,
)

# 包裹引号（英文/中文、单/双/波浪）
_WRAP_QUOTE_RE = re.compile(r'^[\s"“”‘’\'「『]+|[\s"“”‘’\'」』]+$', re.DOTALL)


def strip_think_tags(text: str) -> str:
    """剥离 Qwen3 思考标签：<think>...</think>（含空体）及未闭合残留。"""
    if "<think>" not in text:
        return text
    text = _THINK_TAG_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text


def strip_chat_prefix(text: str) -> str:
    """剥离行首的对话式前缀，如 "Corrected text:" / "1) Transcription:"。"""
    # 前缀可能被模型重复输出（如 "Corrected text: Corrected text: ..."）
    for _ in range(3):
        new_text = _LEADING_PREFIX_RE.sub("", text, count=1)
        if new_text == text:
            break
        text = new_text
    return text


def strip_wrapping_quotes(text: str) -> str:
    """剥离首尾包裹引号（EnglishTextNormalizer 不处理中文引号等）。"""
    return _WRAP_QUOTE_RE.sub("", text).strip()


def pre_normalize(text: str) -> str:
    """送入 EnglishTextNormalizer 之前的防御性清洗。

    顺序：剥离思考标签 -> 统一换行 -> 剥离对话前缀 -> 压缩空白。
    每步都防御 None/空串。
    """
    if not text or not isinstance(text, str):
        return ""
    text = strip_think_tags(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # LLM 输出多行时（如逐句编号），保留行结构交给 normalizer 前先合并：
    # ASR 候选本身不含换行，出现换行几乎必为 LLM 格式产物
    text = " ".join(line.strip() for line in text.split("\n"))
    text = strip_chat_prefix(text)
    text = strip_wrapping_quotes(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_wer(text: str) -> str:
    """WER 计算前的统一规范化入口（ref 与 hyp 都必须过此函数）。

    流程：pre_normalize 防御性清洗 -> EnglishTextNormalizer。
    """
    cleaned = pre_normalize(text)
    if not cleaned:
        return ""
    return _normalizer(cleaned)
