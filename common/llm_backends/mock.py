"""Mock 推理后端：无 GPU / 无模型权重 / 无外部进程跑通全评测管线。

用途（Task 1.5 骨架期）：
  - 定制 llama.cpp 与 GGUF 就绪前，验证 infer_ger.py 的 --backend 分支、
    prompt 渲染、extract_answer 解析、WER 汇总与产物导出全链路；
  - 行为契约：从渲染好的 prompt 中提取 Top-1 候选并原样回显
    （"零纠错 GER"）。因此对任意 N-best 子集，冒烟 WER 必须与
    beam-1 基线完全一致——这是一个强端到端断言，任何不一致都说明
    管线某处（渲染/解析/规范化）被改坏。

实现说明：
  - 回显需要从 prompt 里反解候选。这依赖两个已知格式：Qwen chat 模板
    的 user 段标记，与 plain 模板的引导句（prompts.build_ger_prompt）。
    作为 Mock 可接受此耦合；解析失败时回退固定句（仍保持确定性）。
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

from common.llm_backends.base import BaseLLMBackend, GenerationResult

# Qwen chat 模板 user 段：<|im_start|>user\n...<|im_end|>
_USER_SEG_RE = re.compile(r"user\n(.*?)<\|im_end\|>", re.S)
# plain 模板引导句（见 prompts.build_ger_prompt）：候选以双空格拼接
_PLAIN_BODY_RE = re.compile(r"utterance:\s*(.*?)\n", re.S)

_FALLBACK_ANSWER = "this is a mock backend response"


def _extract_first_candidate(prompt: str) -> Optional[str]:
    """从渲染好的 chat prompt 反解 Top-1 候选；失败返回 None。"""
    m = _USER_SEG_RE.search(prompt)
    if not m:
        return None
    m2 = _PLAIN_BODY_RE.search(m.group(1))
    if not m2:
        return None
    return m2.group(1).split("  ")[0].strip() or None


class MockLLMBackend(BaseLLMBackend):
    """确定性 Mock：回显 Top-1 候选（零纠错 GER 语义）。"""

    backend_name = "mock"

    def __init__(self, sleep_s: float = 0.0):
        """sleep_s 模拟单条生成延迟（RTF 冒烟用），默认 0。"""
        self.sleep_s = sleep_s
        self.calls = 0

    @staticmethod
    def _heuristic_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        import time

        def _one(p: str) -> GenerationResult:
            self.calls += 1
            if self.sleep_s:
                time.sleep(self.sleep_s)
            # max_new_tokens 截断模拟：回显词数不超过上限词数的 1.5 倍
            text = _extract_first_candidate(p) or _FALLBACK_ANSWER
            cap = max(1, int(max_new_tokens))
            words = text.split()
            if len(words) > cap:
                text = " ".join(words[:cap])
            return GenerationResult(
                text=text,
                prompt_tokens=self._heuristic_tokens(p),
                completion_tokens=self._heuristic_tokens(text),
                ttft_ms=None,
            )

        if isinstance(prompt, str):
            return _one(prompt)
        return [_one(p) for p in prompt]

    def count_tokens(self, text: str) -> int:
        return self._heuristic_tokens(text)

    def unload(self) -> None:
        """Mock 无外部资源，显式置空计数便于断言生命周期。"""
        self.calls = 0
