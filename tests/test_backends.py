"""common/llm_backends 与 route_a_ger/prompts 单元测试（无需 GPU/模型下载）。

运行：pytest tests/test_backends.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.llm_backends.base import BaseLLMBackend, GenerationResult  # noqa: E402
from route_a_ger.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_ger_prompt,
    dynamic_max_new_tokens,
)


# ---------------------------------------------------------------------------
# GenerationResult 数据结构
# ---------------------------------------------------------------------------
class TestGenerationResult:
    def test_defaults(self):
        r = GenerationResult(text="hello", prompt_tokens=5, completion_tokens=2)
        assert r.ttft_ms is None
        assert r.text == "hello"


# ---------------------------------------------------------------------------
# BaseLLMBackend 抽象
# ---------------------------------------------------------------------------
class TestBaseBackend:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseLLMBackend()  # type: ignore[abstract]

    def test_minimal_impl(self):
        class Dummy(BaseLLMBackend):
            backend_name = "dummy"

            def generate(self, prompt, max_new_tokens, temperature=0.0,
                         repetition_penalty=1.05):
                return GenerationResult(prompt, 1, 1)

            def count_tokens(self, text):
                return len(text.split())

            def unload(self):
                pass

        d = Dummy()
        out = d.generate("test", 10)
        assert out.text == "test"
        assert d.count_tokens("a b c") == 3
        d.warmup()  # 默认空操作不报错


# ---------------------------------------------------------------------------
# build_ger_prompt
# ---------------------------------------------------------------------------
class TestBuildGerPrompt:
    NBEST = [
        {"text": "HE HOPED THERE WOULD BE STU", "score": -12.41},
        {"text": "HE HOPED THERE WOULD BE STEW", "score": -12.89},
    ]

    def test_lowercase_enforced(self):
        p = build_ger_prompt(self.NBEST)
        assert "he hoped there would be stu" in p
        assert "STU" not in p  # 全大写输入必须被 lower

    def test_numbered_template(self):
        p = build_ger_prompt(self.NBEST, template="numbered")
        assert "1. he hoped" in p and "2. he hoped" in p

    def test_plain_template(self):
        p = build_ger_prompt(self.NBEST, template="plain")
        assert "1." not in p
        assert "he hoped there would be stu" in p

    def test_scores_off_by_default(self):
        p = build_ger_prompt(self.NBEST)
        assert "score" not in p

    def test_scores_on(self):
        p = build_ger_prompt(self.NBEST, use_scores=True)
        assert "score: -12.41" in p

    def test_empty_nbest_raises(self):
        with pytest.raises(ValueError):
            build_ger_prompt([])

    def test_system_prompt_has_no_case_rule(self):
        # 微调要求：删除 "no capitalization changes"
        assert "capitalization" not in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# dynamic_max_new_tokens
# ---------------------------------------------------------------------------
class TestDynamicMaxNewTokens:
    def test_formula(self):
        # top1 = 8 词 -> int(8*2.0)+24 = 40
        nb = [{"text": " ".join(["w"] * 8), "score": 0}]
        assert dynamic_max_new_tokens(nb) == 40

    def test_short_candidate(self):
        nb = [{"text": "hi", "score": 0}]
        assert dynamic_max_new_tokens(nb) == 26  # int(1*2)+24

    def test_empty_guard(self):
        assert dynamic_max_new_tokens([]) == 24
