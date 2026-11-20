"""common/ 模块单元测试。

运行：pytest tests/test_common.py -v
（不依赖 GPU；profiler 在 CPU 模式下降级路径也被覆盖）
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.metrics import (  # noqa: E402
    SUMMARY_FIELDS,
    append_summary_row,
    compute_wer,
    export_error_cases,
    export_predictions,
)
from common.profiler import InferenceProfiler  # noqa: E402
from common.text_norm import (  # noqa: E402
    normalize_for_wer,
    pre_normalize,
    strip_chat_prefix,
    strip_think_tags,
)

REF = "HE HOPED THERE WOULD BE STEW FOR DINNER"


# ---------------------------------------------------------------------------
# text_norm
# ---------------------------------------------------------------------------
class TestStripThinkTags:
    def test_complete_tag(self):
        text = "<think>" + "reasoning" + "</think>" + " he is going to school"
        assert strip_think_tags(text) == "he is going to school"

    def test_empty_tag_qwen3_style(self):
        # enable_thinking=False 时 Qwen3 模板注入空 think 体
        text = "<think></think>HE IS"
        assert strip_think_tags(text) == "HE IS"

    def test_multiline_think(self):
        text = (
            "<think>" + "First\nSecond line" + "</think>"
            "The corrected sentence."
        )
        assert strip_think_tags(text) == "The corrected sentence."

    def test_unclosed_tag(self):
        # 流式中断/截断时的残留
        text = "He is going" + "<think>" + "half though"
        assert strip_think_tags(text) == "He is going"

    def test_no_tag_untouched(self):
        assert strip_think_tags(REF) == REF


class TestStripChatPrefix:
    def test_corrected_text(self):
        assert strip_chat_prefix("Corrected text: HE IS") == "HE IS"
        assert strip_chat_prefix("corrected text:HE IS") == "HE IS"

    def test_transcription_with_dash(self):
        assert strip_chat_prefix("Transcription - HE IS") == "HE IS"

    def test_numbered_prefix(self):
        assert strip_chat_prefix("1) Output: HE IS") == "HE IS"

    def test_repeated_prefix(self):
        assert strip_chat_prefix("Corrected text: Corrected text: HE IS") == "HE IS"

    def test_plain_text_untouched(self):
        # 正文里恰好出现 "corrected" 一词不应被误删（无冒号引导）
        assert strip_chat_prefix("THE TEXT WAS CORRECTED BY HIM") == (
            "THE TEXT WAS CORRECTED BY HIM"
        )

    def test_chinese_colon(self):
        assert strip_chat_prefix("答案：HE IS") == "答案：HE IS"
        # 中文前缀不在剥离白名单内（本项目仅英文评测），但冒号形式不误伤其他


class TestPreNormalize:
    def test_newline_merge(self):
        assert pre_normalize("HE IS\nGOING") == "HE IS GOING"

    def test_wrapping_quotes(self):
        assert pre_normalize('"HE IS"') == "HE IS"
        assert pre_normalize("“HE IS”") == "HE IS"

    def test_empty_and_none(self):
        assert pre_normalize("") == ""
        assert pre_normalize(None) == ""  # type: ignore[arg-type]

    def test_full_pipeline_think_plus_prefix(self):
        text = (
            "<think>" + "Internal reasoning here" + "</think>"
            "Corrected text: HE HOPED THERE WOULD BE STEW FOR DINNER"
        )
        assert pre_normalize(text) == "HE HOPED THERE WOULD BE STEW FOR DINNER"


class TestNormalizeForWer:
    def test_case_and_punct(self):
        assert normalize_for_wer("He hoped there would be stew for dinner!") == (
            "he hoped there would be stew for dinner"
        )

    def test_number_expansion(self):
        # 数字需展开为词（LibriSpeech ref 里全为词形）
        assert normalize_for_wer("IT COST 12 DOLLARS") == "it cost $12"

    def test_ref_passthrough(self):
        assert normalize_for_wer(REF) == (
            "he hoped there would be stew for dinner"
        )

    def test_empty(self):
        assert normalize_for_wer("") == ""


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class TestComputeWer:
    def test_perfect_match(self):
        m = compute_wer([REF], [REF], normalize=False)
        assert (m["wer"], m["sub"], m["del"], m["ins"]) == (0.0, 0, 0, 0)

    def test_deletion(self):
        # ref 8 词 vs hyp 6 词：删 2
        m = compute_wer(["A B C D E F G H"], ["A B C D E F"], normalize=False)
        assert (m["del"], m["wer"]) == (2, 2 / 8)

    def test_insertion(self):
        m = compute_wer(["A B"], ["A B C D"], normalize=False)
        assert (m["ins"], m["wer"]) == (2, 1.0)

    def test_substitution(self):
        m = compute_wer(["A B C"], ["A X C"], normalize=False)
        assert (m["sub"], m["wer"]) == (1, 1 / 3)

    def test_mixed_multi_utterance(self):
        m = compute_wer(
            ["A B C", "D E", "F"],
            ["A X C", "D E", "F G"],
            normalize=False,
        )
        # 语料级累计：sub=1, ins=1
        assert (m["sub"], m["del"], m["ins"]) == (1, 0, 1)
        assert m["wer"] == pytest.approx(2 / 6)

    def test_empty_hyp_counts_as_deletion(self):
        m = compute_wer(["A B"], [""], normalize=False)
        assert (m["del"], m["wer"]) == (2, 1.0)

    def test_empty_ref_counts_as_insertion_clamped(self):
        m = compute_wer([""], ["A B"], normalize=False)
        assert m["ins"] == 2
        assert m["wer"] == 1.0  # 钳制而非 inf

    def test_both_empty(self):
        m = compute_wer([""], [""] * 1, normalize=False)
        assert m["wer"] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_wer(["A"], ["A", "B"], normalize=False)

    def test_empty_input(self):
        m = compute_wer([], [])
        assert m["wer"] == 0.0

    def test_with_normalization_pipeline(self):
        # 端到端：前缀+大小写+标点差异不影响 WER
        m = compute_wer([REF], ["Corrected text: He hoped there would be stew for dinner."])
        assert m["wer"] == 0.0


# ---------------------------------------------------------------------------
# summary.csv
# ---------------------------------------------------------------------------
class TestAppendSummaryRow:
    BASE = dict(
        exp_name="smoke",
        route="A",
        split="test-clean",
        model="Qwen3-1.7B",
        backend="hf",
        quant_mode="fp16",
        wer=0.0523,
        sub=3,
        **{"del": 1},
        ins=0,
        peak_vram_mb=None,
        total_audio_sec=60.0,
        total_infer_sec=30.0,
        rtf=0.5,
        tokens_per_sec=42.0,
    )

    def _write(self, tmp_path, **override):
        path = tmp_path / "summary.csv"
        append_summary_row(path, **{**self.BASE, **override})
        return path

    def test_header_written_once(self, tmp_path):
        path = self._write(tmp_path)
        self._write(tmp_path, exp_name="second")  # 同文件追加
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 2
        assert list(rows[0].keys()) == SUMMARY_FIELDS

    def test_value_roundtrip(self, tmp_path):
        path = self._write(tmp_path, wer=0.12345, sub=3, **{"del": 1}, ins=0)
        row = next(csv.DictReader(open(path)))
        assert float(row["wer"]) == pytest.approx(0.1235, abs=1e-4)
        assert int(row["sub"]) == 3

    def test_missing_field_raises(self, tmp_path):
        bad = {k: v for k, v in self.BASE.items() if k != "route"}
        with pytest.raises(KeyError):
            append_summary_row(tmp_path / "s.csv", **bad)

    def test_none_rendered_empty(self, tmp_path):
        path = self._write(tmp_path)
        row = next(csv.DictReader(open(path)))
        assert row["peak_vram_mb"] == ""


# ---------------------------------------------------------------------------
# 预测/错例导出
# ---------------------------------------------------------------------------
class TestExport:
    RECS = [
        {"id": "u1", "ref": "A B", "hyp": "A B"},
        {"id": "u2", "ref": "C D", "hyp": "C X"},
        {"id": "u3", "ref": "E", "hyp": "E F"},
    ]

    def test_export_predictions(self, tmp_path):
        p = tmp_path / "predictions.jsonl"
        n = export_predictions(p, self.RECS)
        assert n == 3
        assert sum(1 for _ in open(p)) == 3

    def test_export_error_cases_filters(self, tmp_path):
        p = tmp_path / "cases.jsonl"
        n = export_error_cases(p, self.RECS)
        assert n == 2  # u2 (sub) 与 u3 (ins)
        import json

        lines = [json.loads(x) for x in open(p)]
        assert lines[0]["sub"] == 1
        assert lines[1]["ins"] == 1


# ---------------------------------------------------------------------------
# profiler（CPU 降级路径）
# ---------------------------------------------------------------------------
class TestProfiler:
    def test_track_and_finalize(self):
        prof = InferenceProfiler()
        with prof.track():
            prof.mark_ttft()
            prof.add_tokens(10)
            prof.add_tokens(5)
        stats = prof.finalize(total_audio_sec=100.0)
        # 取整后的 total_infer_sec 可能为 0.0（亚毫秒上下文），
        # 但 tokens_per_sec 基于未取整原值计算，必然为正
        assert stats["rtf"] == pytest.approx(
            stats["total_infer_sec"] / 100.0, abs=1e-6
        )
        assert stats["total_new_tokens"] == 15
        assert stats["tokens_per_sec"] > 0

    def test_finalize_override_tokens(self):
        prof = InferenceProfiler()
        with prof.track():
            pass
        stats = prof.finalize(total_audio_sec=10.0, total_new_tokens=100)
        assert stats["total_new_tokens"] == 100

    def test_report_ttft_external(self):
        prof = InferenceProfiler()
        with prof.track():
            prof.report_ttft(0.123)
            prof.report_ttft(9.9)  # 首次为准
        assert prof.finalize(1.0)["ttft"] == 0.123

    def test_mark_ttft_outside_track_raises(self):
        with pytest.raises(RuntimeError):
            InferenceProfiler().mark_ttft()

    def test_invalid_audio_sec(self):
        prof = InferenceProfiler()
        with prof.track():
            pass
        with pytest.raises(ValueError):
            prof.finalize(total_audio_sec=0.0)

    def test_peak_vram_none_on_cpu(self):
        prof = InferenceProfiler()
        with prof.track():
            pass
        # CPU 环境下降级为 None（GPU 环境为 float）
        if prof.peak_vram_mb is not None:
            assert prof.peak_vram_mb >= 0

    def test_reuse_across_tracks(self):
        prof = InferenceProfiler()
        with prof.track():
            prof.add_tokens(1)
        with prof.track():
            pass  # reset 后上一轮 token 不残留
        assert prof.finalize(1.0)["total_new_tokens"] == 0
