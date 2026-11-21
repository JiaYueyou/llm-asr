"""QLoRA 训练样本构造单元测试（无需 GPU/模型）。

覆盖核心防雷点：
  - Assistant-only loss masking 精确性：prompt 段 labels 全为 -100，
    仅 target + EOS 参与 loss，且切分点与 tokenizer 精确对齐；
  - 超长样本左侧截断后标签仍严格对齐（target 不被截）；
  - tokenizer 跨边界合并时跳过样本（绝不产生错位标签）；
  - collate 的 padding 不污染 loss（pad labels = -100，attention = 0）。

运行：pytest tests/test_qlora.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route_a_ger.train_qlora import (  # noqa: E402
    IGNORE_INDEX,
    build_target,
    collate_examples,
    encode_example,
)


class CharTokenizer:
    """字符级假 tokenizer：天然满足 concat 可加性。"""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class BoundaryMergeTokenizer(CharTokenizer):
    """把跨边界的 "XY" 合并成单 token，模拟 BPE 边界合并异常。"""

    def __call__(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            if text[i:i + 2] == "XY":
                ids.append(999)  # 合并 token
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}


TOK = CharTokenizer()


class TestBuildTarget:
    def test_lowercase_and_eos(self):
        assert build_target("HE HOPED") == "he hoped<|im_end|>"

    def test_strip_whitespace(self):
        assert build_target("  He Hoped \n") == "he hoped<|im_end|>"

    def test_no_text_normalization(self):
        # 训练目标不做 EnglishTextNormalizer：撇号/数字必须原样保留
        assert build_target("QUILTER'S") == "quilter's<|im_end|>"


class TestAssistantOnlyMasking:
    PROMPT = "system<|im_start|>user hi<|im_end|>\n<|im_start|>assistant\n"
    TARGET = "the answer<|im_end|>"

    def test_prompt_labels_all_ignored(self):
        ex = encode_example(TOK, self.PROMPT, self.TARGET, max_seq_len=1000)
        assert ex is not None
        n_prompt = len(self.PROMPT)
        assert len(ex["input_ids"]) == len(self.PROMPT) + len(self.TARGET)
        assert ex["labels"][:n_prompt] == [IGNORE_INDEX] * n_prompt

    def test_target_labels_exact_and_eos_counted(self):
        ex = encode_example(TOK, self.PROMPT, self.TARGET, max_seq_len=1000)
        n_prompt = len(self.PROMPT)
        target_ids = [ord(c) for c in self.TARGET]
        assert ex["labels"][n_prompt:] == target_ids
        # EOS 必须参与 loss（最后一个 label 是 <|im_end|> 的末字符）
        assert ex["labels"][-1] == ord(">")
        assert ex["labels"][-1] != IGNORE_INDEX

    def test_attention_mask_all_ones(self):
        ex = encode_example(TOK, self.PROMPT, self.TARGET, max_seq_len=1000)
        assert ex["attention_mask"] == [1] * len(ex["input_ids"])

    def test_exact_boundary_has_no_off_by_one(self):
        # 切分点：最后一个 -100 紧邻第一个 target token
        ex = encode_example(TOK, self.PROMPT, self.TARGET, max_seq_len=1000)
        n_prompt = len(self.PROMPT)
        assert ex["labels"][n_prompt - 1] == IGNORE_INDEX
        assert ex["labels"][n_prompt] == ord(self.TARGET[0])


class TestTruncation:
    def test_left_truncation_keeps_target_intact(self):
        prompt = "P" * 30
        target = "TT<|im_end|>"  # 12 字符
        ex = encode_example(TOK, prompt, target, max_seq_len=20)
        assert ex is not None
        assert len(ex["input_ids"]) == 20
        n_keep_prompt = 20 - len(target)
        assert ex["labels"][:n_keep_prompt] == [IGNORE_INDEX] * n_keep_prompt
        assert ex["labels"][n_keep_prompt:] == [ord(c) for c in target]

    def test_target_too_long_returns_none(self):
        assert encode_example(
            TOK, "short", "T" * 20, max_seq_len=10
        ) is None

    def test_boundary_merge_returns_none(self):
        # prompt 末尾 X 与 target 开头 Y 被合并 -> 拒绝，避免标签错位
        assert encode_example(
            BoundaryMergeTokenizer(), "aX", "Yb", max_seq_len=100
        ) is None

    def test_normal_boundary_not_rejected(self):
        ex = encode_example(
            BoundaryMergeTokenizer(), "aZ", "Yb", max_seq_len=100
        )
        assert ex is not None


class TestCollate:
    def test_padding_does_not_pollute_loss(self):
        ex1 = encode_example(TOK, "ab", "cd", max_seq_len=100)
        ex2 = encode_example(TOK, "abcdef", "gh", max_seq_len=100)
        batch = collate_examples([ex1, ex2], pad_token_id=0)
        # 短样本右侧补 4 个位置：labels=-100、attention=0
        assert batch["labels"][0][-4:].tolist() == [IGNORE_INDEX] * 4
        assert batch["attention_mask"][0][-4:].tolist() == [0] * 4
        assert batch["input_ids"][0][-4:].tolist() == [0] * 4
        # 长样本无 padding，target 全部保留
        assert batch["labels"][1][-2:].tolist() == [ord("g"), ord("h")]
