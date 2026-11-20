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


def extract_answer(raw_text: str) -> str:
    """从 LLM 原始输出中提取最终答案（防御性解析）。

    三层防御：
      1. 复读截断：低比特量化偶发 EOS 前循环复读（红线 7.2 的兜底），
         以首个 20+ 词 n-gram 的二次出现为循环点，截掉其后内容；
      2. 多行输出 -> 取最后一个非空行（答案通常在最后）；
      3. 该行若以 "N." / "N)" 开头则剥离编号。
    """
    import re

    text = raw_text.strip()
    if not text:
        return ""

    # 1) 复读循环截断：滑窗找重复 20 词片段。
    #    循环单位可能是任意长度的整句；首个重复 n-gram 的第二次出现位置 i2
    #    即首遍完整内容的结束边界，保留 words[:i2]（而非 seen[key]+N，
    #    否则首遍长句会被截短）。
    words = text.split()
    if len(words) > 40:
        N = 20
        seen = {}
        for i in range(len(words) - N):
            key = " ".join(words[i : i + N])
            if key in seen:
                words = words[:i]  # i = 第二次出现起点 = 首遍完整边界
                text = " ".join(words)
                break
            seen[key] = i

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
