"""common/llm_backends 与 route_a_ger/prompts 单元测试（无需 GPU/模型下载）。

运行：pytest tests/test_backends.py -v
"""

from __future__ import annotations

import json  # noqa: F401 - payload/timings 断言用
import subprocess  # noqa: F401 - monkeypatch CompletedProcess 用
import sys
import urllib.error  # noqa: F401 - server 异常测试用
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.llm_backends.base import BaseLLMBackend, GenerationResult  # noqa: E402
from common.llm_backends.llamacpp_custom import LlamaCppBackend  # noqa: E402
from common.llm_backends.mock import MockLLMBackend  # noqa: E402
from route_a_ger.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    _truncate_repetition,
    build_ger_prompt,
    dynamic_max_new_tokens,
    extract_answer,
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


# ---------------------------------------------------------------------------
# extract_answer 复读截断（近似重复 / 短句复读 / 正常排比不误截）
# ---------------------------------------------------------------------------
class TestExtractAnswerRepetition:
    def test_nf4_near_duplicate_emerald(self):
        """NF4 微变复读：emerald→emerold / seized→seizes 近似重复应截到首遍。"""
        unit = ("at the emerald city where our princess osma lives green is "
                "the popular color")
        text = (unit + " at the emerold city where our princess osma lives "
                "green is the popular color")
        assert extract_answer(text) == unit

    def test_nf4_near_duplicate_seized(self):
        unit = "then he tossed it down and seized the next"
        text = (unit + " then he tossed it down and seizes the next "
                "then he tossed it down and seizes the next")
        assert extract_answer(text) == unit

    def test_exact_long_duplicate(self):
        """精确长句复读（>40 词旧守卫本已覆盖，新逻辑须保持一致）。"""
        unit = "the bear shook his shaggy sides and then a well known voice replied"
        text = unit + " " + unit + " " + unit
        assert extract_answer(text) == unit

    def test_short_sentence_duplicate(self):
        """短句复读（8 词，单元 4 词）——旧 len>40 硬门槛会放过，新版须捕获。"""
        text = "you are positive then you are positive then"
        assert extract_answer(text) == "you are positive then"

    def test_short_token_repeat(self):
        """短叠词复读（squeak×8）应截断到单个单元。"""
        text = "squeak " * 8
        out = extract_answer(text)
        assert out == "squeak squeak squeak squeak"  # 短单元 s=4 命中，保留前 4

    def test_no_false_truncation_parallelism(self):
        """经典文学排比（正常文本）不得被误截。"""
        cases = [
            "we shall fight on the beaches we shall fight on the landing grounds "
            "we shall fight in the fields",
            "it was the best of times it was the worst of times it was the age of wisdom",
            "he said that he would come and he said that he would stay",
            "she walked through the garden and she walked through the orchard",
            "he opened the door and he opened the window and he opened the gate",
        ]
        for s in cases:
            assert extract_answer(s) == s, f"正常句被误截: {s!r}"

    def test_no_false_truncation_short_repeats(self):
        """合法短叠词/短语（< 4 词单元）不触发截断。"""
        for s in ["no no no that is not what i meant at all",
                  "a a b b c c is a pattern not a repetition",
                  "very very good indeed and very very kind"]:
            assert extract_answer(s) == s

    def test_short_output_untouched(self):
        """短输出（< 2*short_unit=8 词）直接放过。"""
        assert extract_answer("he said hello there") == "he said hello there"

    def test_truncate_repetition_direct(self):
        """_truncate_repetition 直接接口 + 边界（s 取首遍真实边界）。"""
        words = ("the cat sat on the mat " * 3).split()
        assert _truncate_repetition(words) == "the cat sat on the mat".split()

    def test_empty_and_multiline_not_broken(self):
        """原有行为不被破坏：空串返回空、多行取末行、行首编号剥离。"""
        assert extract_answer("") == ""
        assert extract_answer("first line\nsecond line") == "second line"
        assert extract_answer("1. the answer here") == "the answer here"


# ---------------------------------------------------------------------------
# LlamaCppBackend（无需二进制 / server —— 纯函数 + 离线构造校验）
# ---------------------------------------------------------------------------
class TestLlamaCppBackend:
    def _cli_backend(self, tmp_path):
        bin_path = tmp_path / "llama-cli"
        bin_path.write_text("#!/bin/sh\nexit 0\n")  # 存在性校验用占位
        model = tmp_path / "model.q4_k_m.gguf"
        model.write_text("gguf")
        return LlamaCppBackend(
            model_path=model, bin_path=bin_path, n_gpu_layers=99,
            ctx_size=2048, repeat_penalty=1.1, extra_args=["--flash-attn"],
        )

    def test_mode_exclusive(self):
        with pytest.raises(ValueError):
            LlamaCppBackend(bin_path="x", server_url="http://h:8080")
        with pytest.raises(ValueError):
            LlamaCppBackend()  # 两者全缺

    def test_cli_requires_model(self):
        with pytest.raises(ValueError):
            LlamaCppBackend(bin_path="x")  # CLI 模式缺 model_path

    def test_cli_args_determinism(self, tmp_path):
        b = self._cli_backend(tmp_path)
        args = b._build_cli_args("hello", 32)
        assert args[0] == str(b.bin_path)
        # 确定性红线 7.2：temp=0 + repeat-penalty 显式传入
        assert args[args.index("--temp") + 1] == "0.0"
        assert (
            "--repeat-penalty" in args
            and args[args.index("--repeat-penalty") + 1] == "1.10"
        )
        # 续写语义：禁对话包装 + 禁 prompt 回显
        assert "-no-cnv" in args and "--no-display-prompt" in args
        # 红线 7.5：定制参数经 extra_args 注入，无环境路径硬编码
        assert "-n" in args and args[args.index("-n") + 1] == "32"
        assert "-ngl" in args and args[args.index("-ngl") + 1] == "99"
        assert "-c" in args and args[args.index("-c") + 1] == "2048"
        assert "--flash-attn" in args

    def test_cli_args_missing_binary(self, tmp_path):
        b = LlamaCppBackend(
            model_path=tmp_path / "m.gguf",
            bin_path=tmp_path / "nonexistent-cli",
        )
        with pytest.raises(FileNotFoundError):
            b._load()

    def test_server_payload(self, tmp_path):
        b = LlamaCppBackend(
            server_url="http://127.0.0.1:8080", repeat_penalty=1.1
        )
        payload = b._build_request_payload("hi", 16, temperature=0.0, repeat_penalty=1.1)
        assert payload["temperature"] == 0.0
        assert payload["n_predict"] == 16
        assert payload["repeat_penalty"] == 1.1
        assert payload["stream"] is False
        # 显式控制项（cache_prompt 默认关 / seed / stop）
        assert payload["cache_prompt"] is False
        assert payload["seed"] == 42
        assert payload["stop"] == ["<|im_end|>", "<|endoftext|>"]

    def test_server_response_parse(self):
        """新版 timings（predicted_* + cache_n）解析。"""
        data = {
            "content": "corrected text",
            "timings": {
                "prompt_n": 100, "prompt_ms": 500.0, "cache_n": 20,
                "predicted_n": 8, "predicted_ms": 400.0,
            },
        }
        r = LlamaCppBackend._parse_server_response(data, "p" * 400)
        assert r.text == "corrected text"
        assert r.prompt_tokens == 120  # prompt_n + cache_n
        assert r.completion_tokens == 8
        # TTFT ≈ prompt_ms + predicted_ms/predicted_n = 500 + 400/8
        assert r.ttft_ms == pytest.approx(550.0)

    def test_server_response_degraded(self):
        # timings 缺失：降级为启发式 token 数，ttft=None
        data = {"content": "abcd"}
        r = LlamaCppBackend._parse_server_response(data, "p" * 40)
        assert r.completion_tokens == 1  # len//4
        assert r.ttft_ms is None

    def test_endpoint_normalization(self):
        b = LlamaCppBackend(server_url="http://h:8080/")
        assert b._endpoint("completion") == "http://h:8080/completion"
        # 用户传完整端点时应归一化，而非拼出 /completion/health
        b2 = LlamaCppBackend(server_url="http://h:8080/completion")
        assert b2._endpoint("health") == "http://h:8080/health"
        assert b2._endpoint("completion") == "http://h:8080/completion"

    def test_generate_dispatch_batch(self, tmp_path, monkeypatch):
        """generate() 按 CLI 模式派发且支持批量；monkeypatch 进程调用。

        注意：不能 setattr 模块级 subprocess 的属性（会污染真模块），
        直接 patch 该模块绑定的 subprocess.run 本身。
        """
        b = self._cli_backend(tmp_path)
        captured = {}

        def fake_run(args, **kw):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0, stdout="out line\n", stderr="")

        import common.llm_backends.llamacpp_custom as lc

        monkeypatch.setattr(lc.subprocess, "run", fake_run, raising=True)
        out = b.generate("prompt text", max_new_tokens=8)
        assert out.text == "out line"
        assert captured["args"][captured["args"].index("-n") + 1] == "8"
        outs = b.generate(["p1", "p2"], max_new_tokens=8)
        assert [o.text for o in outs] == ["out line", "out line"]

    def test_generate_cli_failure_raises(self, tmp_path, monkeypatch):
        b = self._cli_backend(tmp_path)

        def fake_run(args, **kw):
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="boom")

        import common.llm_backends.llamacpp_custom as lc

        monkeypatch.setattr(lc.subprocess, "run", fake_run, raising=True)
        with pytest.raises(RuntimeError, match="退出码 1"):
            b.generate("p", max_new_tokens=4)

    def test_count_tokens_heuristic(self, monkeypatch):
        """server 模式 /tokenize 失败（网络异常）时回落 len//4 启发式。"""
        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:8080")

        def fake_urlopen(req, *a, **kw):
            raise urllib.error.URLError("conn refused")

        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        assert b.count_tokens("a" * 40) == 10  # len//4 回落

    def test_count_tokens_server_tokenize(self):
        """server 模式无 tokenizer_ref 时走 POST /tokenize 精确计数。"""
        import urllib.request as urlreq

        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:8080")
        captured = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"tokens": [1, 2, 3, 4, 5]}'

        def fake_urlopen(req, *a, **kw):
            captured["url"] = req.full_url
            captured["body"] = json.loads(req.data.decode())
            return _Resp()

        orig = urlreq.urlopen
        lc.urllib.request.urlopen = fake_urlopen
        try:
            assert b.count_tokens("hello") == 5
            assert captured["url"] == "http://h:8080/tokenize"
            assert captured["body"] == {"content": "hello"}
        finally:
            lc.urllib.request.urlopen = orig

    # ------------------------------------------------ server 模式 generate
    def _patch_urlopen(self, monkeypatch, resp_factory):
        import common.llm_backends.llamacpp_custom as lc

        monkeypatch.setattr(lc.urllib.request, "urlopen", resp_factory)

    def test_generate_server_full_flow(self, monkeypatch):
        """server 模式全链路：payload 透传 + 真实 timings（predicted_*）解析。"""
        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:8080", cache_prompt=True)
        b._loaded = True  # 跳过 /health 探测（本测试聚焦 /completion 链路）
        captured = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({
                    "content": "corrected text",
                    "timings": {
                        "prompt_n": 100, "prompt_ms": 500.0, "cache_n": 20,
                        "predicted_n": 8, "predicted_ms": 400.0,
                    },
                }).encode()

        def fake_urlopen(req, *a, **kw):
            captured["url"] = req.full_url
            captured["payload"] = json.loads(req.data.decode())
            return _Resp()

        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        r = b.generate("prompt", max_new_tokens=16, temperature=0.0,
                       repetition_penalty=1.2)
        assert captured["url"] == "http://h:8080/completion"
        p = captured["payload"]
        # 参数透传 + 显式控制项
        assert p["temperature"] == 0.0
        assert p["repeat_penalty"] == 1.2
        assert p["cache_prompt"] is True
        assert p["seed"] == 42
        assert p["stop"] == ["<|im_end|>", "<|endoftext|>"]
        assert p["n_predict"] == 16
        # timings：prompt_tokens = prompt_n + cache_n = 120；TTFT = 500+400/8
        assert r.text == "corrected text"
        assert r.prompt_tokens == 120
        assert r.completion_tokens == 8
        assert r.ttft_ms == pytest.approx(550.0)

    def test_generate_server_rep_fallback(self, monkeypatch):
        """repetition_penalty 未传（None）回落 self.repeat_penalty（1.10）。"""
        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:8080", repeat_penalty=1.05)
        b._loaded = True  # 跳过 /health 探测
        captured = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"content": "x"}'

        def fake_urlopen(req, *a, **kw):
            captured["payload"] = json.loads(req.data.decode())
            return _Resp()

        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        b.generate("p", max_new_tokens=4)  # 不传 repetition_penalty
        assert captured["payload"]["repeat_penalty"] == 1.05

    def test_generate_server_urlerror(self, monkeypatch):
        """网络断连（URLError）报带端点与超时信息的 RuntimeError。"""
        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:8080")
        b._loaded = True  # 跳过 /health 探测

        def fake_urlopen(req, *a, **kw):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="completion.*connection refused"):
            b.generate("p", max_new_tokens=4)

    def test_generate_server_timeout(self, monkeypatch):
        """超时（TimeoutError）同样转为带端点信息的 RuntimeError。"""
        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:8080", timeout_s=1.5)
        b._loaded = True  # 跳过 /health 探测

        def fake_urlopen(req, *a, **kw):
            raise TimeoutError("timed out")

        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="超时 1.5s"):
            b.generate("p", max_new_tokens=4)

    def test_server_response_legacy_eval_fallback(self):
        """旧版 llama-server timings（eval_n/eval_ms）兼容 fallback。"""
        data = {
            "content": "old text",
            "timings": {
                "prompt_n": 120, "prompt_ms": 600.0,
                "eval_n": 8, "eval_ms": 400.0,
            },
        }
        r = LlamaCppBackend._parse_server_response(data, "p" * 400)
        assert r.text == "old text"
        assert r.prompt_tokens == 120
        assert r.completion_tokens == 8
        assert r.ttft_ms == pytest.approx(650.0)  # 600 + 400/8

    def test_server_response_degraded(self):
        # timings 缺失：降级为启发式 token 数，ttft=None
        data = {"content": "abcd"}
        r = LlamaCppBackend._parse_server_response(data, "p" * 40)
        assert r.completion_tokens == 1  # len//4
        assert r.ttft_ms is None
        # prompt_n+cache_n 均为 0 时 prompt_tokens 也降级
        data2 = {"content": "x", "timings": {"prompt_ms": 5.0}}
        r2 = LlamaCppBackend._parse_server_response(data2, "p" * 40)
        assert r2.prompt_tokens == 10  # len//4

    def test_health_503_model_loading(self, monkeypatch):
        """_load() 探测 /health 503 时明确提示模型仍在加载。"""
        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:8080")

        class _HTTPError(urllib.error.HTTPError):
            def __init__(self):
                super().__init__("http://h:8080/health", 503, "loading", None, None)

        def fake_urlopen(req, *a, **kw):
            raise _HTTPError()

        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="503.*仍在加载"):
            b._load()

    def test_health_conn_refused(self, monkeypatch):
        """server 未启动（连接拒绝）报未就绪并提示确认端口。"""
        import common.llm_backends.llamacpp_custom as lc

        b = LlamaCppBackend(server_url="http://h:9999")

        def fake_urlopen(req, *a, **kw):
            raise urllib.error.URLError("refused")

        monkeypatch.setattr(lc.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError, match="无法连接.*9999"):
            b._load()

    # ------------------------------------------------ CLI 模式补充
    def test_cli_args_passthrough_and_controls(self, tmp_path):
        """generate() 传入的 temperature/rep 透传 CLI；seed/stop 显式携带。"""
        b = self._cli_backend(tmp_path)  # seed=42 默认
        args = b._build_cli_args("hi", 16, temperature=0.3, repetition_penalty=1.25)
        assert args[args.index("--temp") + 1] == "0.3"
        assert args[args.index("--repeat-penalty") + 1] == "1.25"
        assert args[args.index("-s") + 1] == "42"
        assert args[args.index("-r") + 1] == "<|im_end|>"  # 首个停用词
        # 自定义 stop/seed
        b2 = LlamaCppBackend(
            model_path=tmp_path / "m.gguf", bin_path=tmp_path / "cli",
            stop=["END"], seed=7,
        )
        (tmp_path / "m.gguf").write_text("g")
        (tmp_path / "cli").write_text("#!/bin/sh\n")
        args2 = b2._build_cli_args("hi", 16)
        assert args2[args2.index("-s") + 1] == "7"
        assert args2[args2.index("-r") + 1] == "END"

    def test_generate_cli_timeout(self, tmp_path, monkeypatch):
        """llama-cli 超时转 RuntimeError（含 timeout_s 与二进制路径）。"""
        import common.llm_backends.llamacpp_custom as lc

        b = self._cli_backend(tmp_path)

        def fake_run(args, **kw):
            raise subprocess.TimeoutExpired(cmd=args, timeout=b.timeout_s)

        monkeypatch.setattr(lc.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="超时"):
            b.generate("p", max_new_tokens=4)

    def test_generate_cli_passthrough(self, tmp_path, monkeypatch):
        """generate() 的 temperature/rep 透传到 CLI argv（默认 0.0/1.10）。"""
        import common.llm_backends.llamacpp_custom as lc

        b = self._cli_backend(tmp_path)
        captured = {}

        def fake_run(args, **kw):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0, stdout="out\n", stderr="")

        monkeypatch.setattr(lc.subprocess, "run", fake_run)
        b.generate("p", max_new_tokens=4, temperature=0.0, repetition_penalty=1.3)
        a = captured["args"]
        assert a[a.index("--temp") + 1] == "0.0"
        assert a[a.index("--repeat-penalty") + 1] == "1.30"


# ---------------------------------------------------------------------------
# MockLLMBackend（回显 Top-1 候选 —— 零纠错 GER 语义）
# ---------------------------------------------------------------------------
class TestMockLLMBackend:
    def _rendered_prompt(self):
        """最小可解析的 chat prompt（模拟 Qwen 模板渲染产物）。"""
        user = build_ger_prompt(
            [
                {"text": "THE CAT SAT", "score": -1.0},
                {"text": "THE CAT SEAT", "score": -1.2},
            ],
            template="plain",
        )
        return f"user\n{user}<|im_end|>"

    def test_echo_top1_single(self):
        b = MockLLMBackend()
        r = b.generate(self._rendered_prompt(), max_new_tokens=64)
        assert r.text == "the cat sat"
        assert b.calls == 1

    def test_echo_top1_batch(self):
        b = MockLLMBackend()
        rs = b.generate([self._rendered_prompt()] * 3, max_new_tokens=64)
        assert len(rs) == 3
        assert all(r.text == "the cat sat" for r in rs)
        assert b.calls == 3

    def test_unparseable_fallback_deterministic(self):
        b = MockLLMBackend()
        r = b.generate("garbage prompt", max_new_tokens=64)
        assert r.text == "this is a mock backend response"
        r2 = b.generate("garbage prompt", max_new_tokens=64)
        assert r2.text == r.text  # 确定性

    def test_max_new_tokens_truncation(self):
        b = MockLLMBackend()
        r = b.generate(self._rendered_prompt(), max_new_tokens=1)
        # 词数上限 1*1.5 截断（heuristic cap），至少保留 1 词
        assert len(r.text.split()) <= 2

    def test_unload_resets_calls(self):
        b = MockLLMBackend()
        b.generate(self._rendered_prompt(), max_new_tokens=64)
        b.unload()
        assert b.calls == 0

    def test_backend_name(self):
        assert MockLLMBackend().backend_name == "mock"
        assert LlamaCppBackend(server_url="http://h:1").backend_name == "llamacpp"


# ---------------------------------------------------------------------------
# profiler 进程级显存查询（llamacpp 外部进程口径）
# ---------------------------------------------------------------------------
class TestProcessVramQuery:
    def test_query_process_vram_found(self, monkeypatch):
        """compute-apps 列出目标 pid 时返回其显存占用。

        profiler.query_process_vram_mb 内部局部 import subprocess，
        patch 顶层 subprocess.run 即可生效。
        """
        import subprocess as sp

        import common.profiler as prof

        fake_output = "12345, 2357\n67890, 100\n"

        def fake_run(cmd, **kw):
            class _P:
                stdout = fake_output
            return _P()

        monkeypatch.setattr(sp, "run", fake_run)
        assert prof.query_process_vram_mb(12345) == 2357.0
        assert prof.query_process_vram_mb(67890) == 100.0

    def test_query_process_vram_missing_pid(self, monkeypatch):
        """pid 不在列表（进程退出/未用 GPU）返回 None 而非报错。"""
        import subprocess as sp

        import common.profiler as prof

        def fake_run(cmd, **kw):
            class _P:
                stdout = "99999, 500\n"
            return _P()

        monkeypatch.setattr(sp, "run", fake_run)
        assert prof.query_process_vram_mb(12345) is None

    def test_query_process_vram_smerror(self, monkeypatch):
        """nvidia-smi 失败（异常）返回 None。"""
        import subprocess as sp

        import common.profiler as prof

        def fake_run(cmd, **kw):
            raise sp.CalledProcessError(1, "nvidia-smi")

        monkeypatch.setattr(sp, "run", fake_run)
        assert prof.query_process_vram_mb(123) is None


# ---------------------------------------------------------------------------
# infer_ger checkpoint：逐条落盘 / --resume / 统计口径可加
# ---------------------------------------------------------------------------
class TestCheckpointResume:
    def _run_cli(self, args: list, capsys=None):
        """以子进程跑 infer_ger.py（隔离 sys.argv/全局态，冒烟即真实路径）。

        强制 HF_HUB_OFFLINE=1：mock 后端虽不下载模型，但 infer_ger 仍需
        AutoTokenizer 渲染 prompt；transformers 4.57 的 _patch_mistral_regex
        在未设 OFFLINE 时会发 HEAD 请求，无网环境下重试后直接失败。
        """
        import os
        import subprocess as sp

        return sp.run(
            [sys.executable, "route_a_ger/infer_ger.py", *args],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )

    def test_load_checkpoint_corrupt_tail(self, tmp_path):
        """断电截断的末行（无换行的半行 JSON）必须被丢弃，完整行保留。"""
        from route_a_ger.infer_ger import load_checkpoint

        f = tmp_path / "checkpoint.jsonl"
        f.write_text(
            '{"id": "a", "hyp_raw": "x", "ref": "r", "duration": 1.0, '
            '"completion_tokens": 2, "gen_sec": 0.5, "ttft_sec": null, '
            '"vram_mb": null}\n'
            '{"id": "b", "hyp_raw": "trunca',  # 断电截断行
            encoding="utf-8",
        )
        done = load_checkpoint(f)
        assert set(done) == {"a"}

    def test_load_checkpoint_missing(self, tmp_path):
        from route_a_ger.infer_ger import load_checkpoint

        assert load_checkpoint(tmp_path / "nope.jsonl") == {}

    def test_append_checkpoint_fsync(self, tmp_path):
        """append_checkpoint 必须逐条 fsync（防宿主断电丢 page cache）。"""
        import io
        import json as jsonlib

        from route_a_ger.infer_ger import append_checkpoint

        f = io.open(tmp_path / "ck.jsonl", "w", encoding="utf-8")
        rec = {"id": "a", "hyp_raw": "x", "gen_sec": 0.1}
        append_checkpoint(f, rec)
        f.close()
        line = (tmp_path / "ck.jsonl").read_text(encoding="utf-8")
        assert jsonlib.loads(line)["id"] == "a"

    def test_merge_stats_additive(self):
        """两段统计合并 = Σ逐条：RTF/Tokens/s 不受段间间隔影响（口径核心）。"""
        from route_a_ger.infer_ger import merge_stats

        # 历史段：3 条，共 6s 生成 / 30s 音频 / 60 tokens / 峰值 3000MB
        done = {
            f"utt-{i}": {
                "id": f"utt-{i}", "hyp_raw": "h", "ref": "r",
                "duration": 10.0, "completion_tokens": 20,
                "gen_sec": 2.0, "ttft_sec": 0.5, "vram_mb": 3000.0,
            }
            for i in range(3)
        }
        # 本段：1 条，2s / 10s / 20 tokens / 峰值 2800MB（低于历史）
        cur_stats = {
            "peak_vram_mb": 2800.0, "ttft": 0.9,
            "_cur_gen_sec": 2.0, "_cur_audio_sec": 10.0, "_cur_new_tokens": 20,
        }
        m = merge_stats(done, cur_stats)
        assert m["total_infer_sec"] == 8.0          # 6 + 2，Σgen_sec
        assert m["total_audio_sec"] == 40.0         # 30 + 10
        assert m["total_new_tokens"] == 80
        assert m["rtf"] == 0.2                      # 8/40
        assert m["tokens_per_sec"] == 10.0          # 80/8
        assert m["peak_vram_mb"] == 3000.0          # 跨段 max
        assert m["ttft"] == 0.5                     # 历史段首条 TTFT 优先

    def test_merge_stats_empty_history_uses_gen_sec(self):
        """无历史段（首跑被 Ctrl+C 后 checkpoint 为空）也统一 Σgen_sec 口径。"""
        from route_a_ger.infer_ger import merge_stats

        cur = {"peak_vram_mb": 100.0, "ttft": 0.2,
               "_cur_gen_sec": 3.0, "_cur_audio_sec": 6.0, "_cur_new_tokens": 9}
        m = merge_stats({}, cur)
        assert m["total_infer_sec"] == 3.0
        assert m["rtf"] == 0.5
        assert m["tokens_per_sec"] == 3.0

    def test_mock_two_segment_resume_e2e(self, tmp_path):
        """端到端：段1 跑 20 条 -> 段2 --resume 跑全 50 条。

        强断言：① WER 与 beam-1 基线完全一致（Mock 零纠错 + 合并顺序正确）；
        ② checkpoint 恰 50 行；③ summary 行的 total_infer_sec ≈ Σgen_sec
        （而非段2 墙钟——段2 含 tokenizer 加载等开销，若口径错误必然偏大）；
        ④ predictions.jsonl 恰 50 条且顺序与源文件一致。
        """
        import json as jsonlib

        out = tmp_path / "out"
        common = [
            "--nbest", "outputs/nbest/dev-clean.jsonl",
            "--backend", "mock", "--template", "plain",
            "--out_dir", str(out), "--write_summary",
        ]
        # 段1：20 条（无 --resume，新 checkpoint）
        r1 = self._run_cli(common + ["--max_samples", "20", "--exp_name", "e2e"])
        assert r1.returncode == 0, r1.stderr
        ck1 = (out / "e2e" / "checkpoint.jsonl").read_text().strip().splitlines()
        assert len(ck1) == 20

        # 段2：全 50 条 --resume
        r2 = self._run_cli(common + ["--exp_name", "e2e", "--resume"])
        assert r2.returncode == 0, r2.stderr
        assert "--resume：checkpoint 20/50 条已完成" in r2.stderr + r2.stdout

        ck2 = (out / "e2e" / "checkpoint.jsonl").read_text().strip().splitlines()
        assert len(ck2) == 50

        # WER 与 beam-1 基线一致（断言在 stdout 的 GER 结果行）
        line = [l for l in (r2.stdout + r2.stderr).splitlines() if "GER 结果" in l]
        assert line, "缺少 GER 结果日志行"
        assert "WER=0.0904" in line[0]
        assert "基线=0.0904" in line[0]

        # 统计口径：summary 的 total_infer_sec 应等于 Σgen_sec（毫秒级）
        rows = (out / "metrics" / "summary.csv").read_text().strip().splitlines()
        assert len(rows) == 3  # header + 段1 + 段2
        header = rows[0].split(",")
        row2 = dict(zip(header, rows[2].split(",")))
        sum_gen = sum(jsonlib.loads(l)["gen_sec"] for l in ck2)
        assert abs(float(row2["total_infer_sec"]) - sum_gen) < 0.05
        # RTF = Σgen / Σaudio（音频时长固定，可精确核对）
        import json as j
        durs = [j.loads(l)["duration"] for l in open("outputs/nbest/dev-clean.jsonl")]
        assert abs(float(row2["rtf"]) - sum_gen / sum(durs)) < 0.001

        # predictions 顺序与完整性
        preds = [j.loads(l) for l in open(out / "e2e" / "predictions.jsonl")]
        src = [j.loads(l) for l in open("outputs/nbest/dev-clean.jsonl")]
        assert [p["id"] for p in preds] == [s["id"] for s in src]


# ---------------------------------------------------------------------------
# LoRA adapter 挂载参数（Task 1.6）
# ---------------------------------------------------------------------------
class TestLoraPath:
    def test_backend_stores_lora_path(self):
        from common.llm_backends.hf_bnb import HFBNBackend

        b = HFBNBackend(model_id="Qwen/Qwen3-1.7B", quant_mode="nf4",
                        lora_path="/tmp/adapter")
        assert b.lora_path == "/tmp/adapter"
        assert b.model is None and b.tokenizer is None  # 构造不加载权重

    def test_backend_default_no_lora(self):
        from common.llm_backends.hf_bnb import HFBNBackend

        assert HFBNBackend(model_id="x").lora_path is None

    @pytest.mark.parametrize("backend", ["mock", "llamacpp"])
    def test_infer_ger_rejects_lora_on_non_hf_backend(self, tmp_path, backend):
        """--lora_path 与非 hf_bnb 后端组合必须在参数校验阶段拦截（不加载模型）。"""
        import os
        import subprocess as sp

        nbest = tmp_path / "dummy.jsonl"
        nbest.write_text("", encoding="utf-8")
        extra = (
            ["--llamacpp_url", "http://127.0.0.1:1"]
            if backend == "llamacpp" else []
        )
        r = sp.run(
            [
                sys.executable, "route_a_ger/infer_ger.py",
                "--nbest", str(nbest), "--backend", backend,
                "--lora_path", "/tmp/adapter", *extra,
            ],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        assert r.returncode == 2
        assert "--lora_path" in r.stderr
        assert "hf_bnb" in r.stderr


# ---------------------------------------------------------------------------
# LoRA：模板硬保护 + 结果隔离（Task 1.6 收口）
# ---------------------------------------------------------------------------
class TestLoraIsolation:
    def test_quant_mode_label(self):
        from route_a_ger.infer_ger import quant_mode_label

        assert quant_mode_label("nf4", None) == "nf4"
        assert quant_mode_label("nf4", "/tmp/adapter") == "nf4_qlora"

    def test_default_exp_name_isolated_from_zero_shot(self):
        from route_a_ger.infer_ger import default_exp_name

        zero = default_exp_name("Qwen/Qwen3-1.7B", "nf4", "test-clean", None)
        lora = default_exp_name(
            "Qwen/Qwen3-1.7B", "nf4", "test-clean", "/tmp/adapter"
        )
        assert zero == "ger-Qwen3-1.7B-nf4-test-clean"
        assert lora == "ger-Qwen3-1.7B-nf4_qlora-test-clean"
        assert zero != lora  # 目录/产物不互相覆盖

    def test_resolve_and_check_template(self, tmp_path):
        from route_a_ger.infer_ger import (
            check_lora_template,
            resolve_lora_template,
        )

        assert resolve_lora_template(tmp_path) is None  # 无 summary
        (tmp_path / "qlora_summary.json").write_text(
            json.dumps({"train": {"template": "plain"}}), encoding="utf-8"
        )
        assert resolve_lora_template(tmp_path) == "plain"
        assert check_lora_template(tmp_path, "plain") is None
        assert check_lora_template(tmp_path, "numbered") == "plain"

    def test_malformed_summary_is_not_fatal(self, tmp_path):
        from route_a_ger.infer_ger import resolve_lora_template

        (tmp_path / "qlora_summary.json").write_text("{bad", encoding="utf-8")
        assert resolve_lora_template(tmp_path) is None

    def test_cli_rejects_template_mismatch(self, tmp_path):
        """--template 与训练模板冲突必须在加载模型前 parser.error（退出码 2）。"""
        import os
        import subprocess as sp

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "qlora_summary.json").write_text(
            json.dumps({"train": {"template": "plain"}}), encoding="utf-8"
        )
        r = sp.run(
            [
                sys.executable, "route_a_ger/infer_ger.py",
                "--nbest", str(tmp_path / "dummy.jsonl"),
                "--backend", "hf_bnb", "--quant_mode", "nf4",
                "--lora_path", str(adapter), "--template", "numbered",
            ],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
        )
        assert r.returncode == 2
        assert "训练模板 plain" in r.stderr
        assert "--template plain" in r.stderr
