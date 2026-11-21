"""路线 A GER（生成式纠错）Prompt 模板（严格受控，禁用思维链）。

设计约定：
  - wav2vec2 输出为全大写；BPE 对全大写文本切分过碎，既浪费 token 又
    干扰 LLM 语言先验 -> N-best 统一 .lower() 后进 prompt；
  - 声学分数默认不进 prompt（CTC log 分与 LLM 词先验不同纲），
    留作云端消融项；
  - chat template 渲染（含 enable_thinking=False）在调用方 infer_ger.py。
"""

from __future__ import annotations
from typing import List, Optional

SYSTEM_PROMPT = (
    "You are an ASR error correction assistant. "
    "You correct speech recognition transcriptions. "
    "Output ONLY the corrected transcription on a single line. "
    "Do not list or repeat the candidates. "
    "Do not number your answer. "
    "No explanations, no punctuation."
)


def _truncate_repetition(
    words: List[str],
    short_unit: int = 4,
    long_unit: int = 6,
    short_threshold: float = 0.95,
    long_threshold: float = 0.85,
) -> List[str]:
    """复读截断：检测「首段 ≈ 其后副本」的整句/分句复读并截到首段边界。

    低比特量化（尤其 NF4）偶发 EOS 前循环复读，且常是**近似重复**而非逐字
    精确重复（如 ``emerald → emerold → emeralds``、``seized → seizes``），
    旧的精确 20-gram 匹配抓不到；本函数改用词级模糊匹配。

    算法：令输出词序列为 words，枚举首段长度 s（``short_unit ≤ s ≤ n//2``），
    将 ``words[:s]`` 与紧随其后的等长窗口 ``words[s:2s]`` 做
    ``SequenceMatcher`` 词级相似度；一旦命中即认定复读，截断到 ``words[:s]``
    （保留首遍完整内容）。取**最小**命中 s = 首遍真实边界。

    两级阈值（严格防误截文学文本正常排比/短语重复）：
      - **短单元**（``short_unit ≤ s < long_unit``，即 4–5 词）：要求近乎
        逐字精确 ``short_threshold=0.95``——短叠词/短语复读（``squeak``×N、
        ``anders face grew red``×N）能捕获，而正常短排比偶尔重复 4–5 词时
        相似度远达不到 0.95，不会误截；
      - **长单元**（``s ≥ long_unit``，即 ≥6 词）：``long_threshold=0.85``——
        覆盖 NF4 微变长句复读；真实朗读转写中 6+ 词近逐字复现基本不可能，
        正常文本不会巧合达到。实测 ``we shall fight on the beaches ...`` 等
        经典排比最大相似度仅 0.83，安全。
      - 仅当 ``n ≥ 2*short_unit`` 才检测，短输出直接放过。
    """
    from difflib import SequenceMatcher

    n = len(words)
    if n < 2 * short_unit:
        return words
    # 取相似度最高的候选边界（并列取最小 s = 首遍最短单元）。
    # 不能用「首个达阈值即返回」——错位一个词的首段（如 12 词）相似度
    # 0.92 会先于正确的 13 词边界（1.0）命中，导致丢掉边界处一词；取最高
    # 相似度可精确落在首遍真实边界。
    best_s: int | None = None
    best_ratio = 0.0
    for s in range(short_unit, n // 2 + 1):
        threshold = long_threshold if s >= long_unit else short_threshold
        ratio = SequenceMatcher(None, words[:s], words[s:s + s]).ratio()
        if ratio >= threshold and ratio > best_ratio + 1e-9:
            best_ratio = ratio
            best_s = s
    return words[:best_s] if best_s is not None else words


def extract_answer(raw_text: str) -> str:
    """从 LLM 原始输出中提取最终答案（防御性解析）。

    三层防御：
      1. 复读截断：低比特量化偶发 EOS 前循环复读（红线 7.2 的兜底），
         支持精确与近似（微变）重复，按首个重复单元截断（短单元 4–5 词需
         0.95、长单元 ≥6 词需 0.85，见 ``_truncate_repetition``）；
      2. 多行输出 -> 取最后一个非空行（答案通常在最后）；
      3. 该行若以 "N." / "N)" 开头则剥离编号。
    """
    import re

    text = raw_text.strip()
    if not text:
        return ""

    # 1) 复读循环截断：词级近似重复检测（含短句复读，无 40 词硬门槛）。
    #    仅在实际截断时重建文本——否则保留原换行，供第 2 步「多行取末行」用。
    orig_words = text.split()
    trunc_words = _truncate_repetition(orig_words)
    if trunc_words != orig_words:
        text = " ".join(trunc_words)

    # 2) 多行取末行
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    last = lines[-1]

    # 3) 剥离行首编号
    m = re.match(r"^\d+[.)]\s*(.+)$", last)
    return m.group(1).strip() if m else last


def build_ger_prompt(
    nbest: List[dict],
    use_scores: bool = False,
    template: str = "numbered",
) -> str:
    """Top-N 候选 -> 用户消息文本。

    Args:
        nbest: [{"text": ..., "score": ...}, ...]，来自 outputs/nbest 缓存
        use_scores: 是否在候选项后附声学分数（默认 False）
        template: "numbered"（编号列表）| "plain"（候选直接拼接）

    Returns:
        已 .lower() 的候选构造的用户消息字符串。
    """
    if not nbest:
        raise ValueError("nbest 不能为空")

    cands = []
    for c in nbest:
        t = c["text"].strip().lower()
        if use_scores:
            t = f"{t} (score: {c['score']:.2f})"
        cands.append(t)

    if template == "plain":
        body = "  ".join(cands)
        return (
            "A speech recognizer produced these candidate transcriptions "
            f"for the same utterance: {body}\n"
            "Output the single best corrected sentence in plain words only."
        )

    lines = ["Candidate transcriptions from a speech recognizer:"]
    for i, c in enumerate(cands, 1):
        lines.append(f"{i}. {c}")
    lines.append(
        "Correct the transcription errors and output the single best "
        "corrected sentence in plain words only."
    )
    return "\n".join(lines)


def dynamic_max_new_tokens(nbest: List[dict]) -> int:
    """按 Top-1 候选词数动态设生成上限（红线 7.2 的放宽版）。

    上限 = int(top1_len_words * 2.0) + 24：
      2.0 系数覆盖 LLM 复读早期阶段 + 小写改写的 token 膨胀；
      +24 为指令遵循与格式余量。纠错输出不该长于候选的 2 倍。
    """
    if not nbest:
        return 24
    top1_words = len(nbest[0]["text"].split())
    return int(top1_words * 2.0) + 24


def render_chat_prompt(
    tokenizer,
    nbest: List[dict],
    system_prompt: Optional[str] = None,
    use_scores: bool = False,
    template: str = "numbered",
) -> str:
    """渲染完整 chat prompt（Qwen3 必须显式关思考模式，红线 7.1）。"""
    from transformers import AutoTokenizer  # 局部导入避免循环依赖

    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": build_ger_prompt(nbest, use_scores, template)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
