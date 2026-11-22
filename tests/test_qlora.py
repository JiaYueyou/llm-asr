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

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from route_a_ger.train_qlora import (  # noqa: E402
    IGNORE_INDEX,
    build_target,
    collate_examples,
    encode_example,
    iter_step_targets,
    micro_batches_per_epoch,
    micro_counts_per_epoch,
    steps_per_epoch,
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


def simulate_optimizer_steps(
    n_examples: int,
    batch_size: int,
    grad_accum: int,
    n_epochs: int,
    total_steps: int,
) -> tuple[int, int]:
    """按 train() 的累积逻辑跑一遍（不涉及模型），返回 (step 次数, 消耗微批数)。

    与训练循环同构：micro 达到当前步目标即 step；epoch 末尾不足目标也 flush。
    """
    targets = iter_step_targets(
        n_examples, batch_size, grad_accum, n_epochs, total_steps
    )
    micro_per_epoch = micro_batches_per_epoch(n_examples, batch_size)
    step = 0
    micro = 0
    micro_target = 0
    consumed = 0
    for _ in range(n_epochs):
        for _b in range(micro_per_epoch):
            if step >= total_steps:
                return step, consumed
            if micro == 0:
                micro_target = targets[step]
            consumed += 1
            micro += 1
            if micro < micro_target:
                continue
            step += 1
            micro = 0
    return step, consumed


class TestStepsPerEpoch:
    """步数规划必须与真实 optimizer.step() 次数严格一致。"""

    @pytest.mark.parametrize(
        "n, bs, ga",
        [
            (128, 4, 8),   # 整除：32 micro -> 4 步
            (100, 4, 8),   # 非整除：25 micro -> 4 步（末步 1 micro）
            (64, 8, 4),    # 整除（旧配置）
            (60, 8, 4),    # 非整除：8 micro -> 2 步（末步 flush）
            (7, 2, 3),     # 非整除 + 末步 flush
        ],
    )
    def test_total_steps_match_optimizer_steps(self, n, bs, ga):
        spe = steps_per_epoch(n, bs, ga)
        m = micro_batches_per_epoch(n, bs)
        assert m == math.ceil(n / bs)
        assert spe == math.ceil(m / ga)

        epochs = 3
        total_steps = spe * epochs
        steps, consumed = simulate_optimizer_steps(n, bs, ga, epochs, total_steps)
        assert steps == total_steps
        # 不 drop_last：每个 epoch 的全部微批都被消费
        assert consumed == micro_batches_per_epoch(n, bs) * epochs

    def test_micro_counts_sum_equals_micro_batches(self):
        for n, bs, ga in [(128, 4, 8), (100, 4, 8), (7, 2, 3), (5, 1, 4)]:
            counts = micro_counts_per_epoch(n, bs, ga)
            assert len(counts) == steps_per_epoch(n, bs, ga)
            assert sum(counts) == micro_batches_per_epoch(n, bs)
            # 仅末步可能小于 grad_accum，其余均为 grad_accum
            assert counts[:-1] == [ga] * (len(counts) - 1)
            assert 1 <= counts[-1] <= ga

    @pytest.mark.parametrize(
        "n, bs, ga",
        [
            (3, 4, 8),   # 1 micro < 一个等效 batch
            (1, 8, 4),
            (7, 4, 8),
            (1, 1, 1),
        ],
    )
    def test_small_sample_no_zero_division(self, n, bs, ga):
        spe = steps_per_epoch(n, bs, ga)
        assert spe >= 1  # 不会出现 steps_per_epoch == 0

        steps, consumed = simulate_optimizer_steps(n, bs, ga, 1, spe)
        assert steps == spe == 1
        assert consumed == micro_batches_per_epoch(n, bs)

        # max_steps 模式下 n_epochs 由 ceil(total/spe) 推出，同样不能除零
        n_epochs = max(1, math.ceil(spe / spe))
        assert len(iter_step_targets(n, bs, ga, n_epochs, spe)) == spe

    def test_max_steps_capped_exactly(self):
        # 样本不足一整轮时，max_steps 模式可跨轮取满且不多跑
        targets = iter_step_targets(5, 1, 4, n_epochs=3, total_steps=5)
        assert len(targets) == 5
        steps, _ = simulate_optimizer_steps(5, 1, 4, 3, 5)
        assert steps == 5

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            steps_per_epoch(0, 1, 1)
        with pytest.raises(ValueError):
            steps_per_epoch(10, 0, 1)
        with pytest.raises(ValueError):
            steps_per_epoch(10, 1, 0)
