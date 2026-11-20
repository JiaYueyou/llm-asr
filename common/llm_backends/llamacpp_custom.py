"""定制 llama.cpp 推理后端（CLI 进程 / HTTP server 双模式，路线 A/B 共享）。

设计约定（PROJECT_SPEC.md 红线 7.5 + BaseLLMBackend 接口）：
  - 路径与端口全部由调用方配置注入（--llamacpp_bin / --llamacpp_url），
    本层严禁硬编码任何环境相关路径；
  - chat template 由调用方（infer_ger.py）用 HF tokenizer 渲染
    （enable_thinking=False），本层以"纯续写"语义发送——CLI 模式
    -no-cnv、server 模式 /completion 端点，与 HF 后端共享同一评测管线；
  - 确定性解码：temperature=0 + 显式 seed；repeat_penalty 默认 1.10
    （generate() 显式传入时以传入值为准，不再静默忽略）；
  - 统计口径：
      * server 模式：响应 timings 提供精确 prompt/解码 token 数与耗时，
        TTFT ≈ prompt_ms + predicted_ms/predicted_n（prompt 评估 + 单步解码）；
        新版字段为 predicted_n / predicted_ms（旧版为 eval_n / eval_ms，
        做兼容 fallback）；
      * CLI 模式：每次 generate 重起进程重载模型，RTF 不可比（仅用于
        定制版功能调试）；token 数为 len//4 启发式，显存无差分可测；
  - 显存：模型在进程外运行，torch allocator 口径无效；由调用方用
    common.profiler.query_gpu_used_mb() 做评测前后 nvidia-smi 差分注入
    profiler.report_peak_vram()（差分口径可抵消 Windows 侧常驻占用）。

定制版 llama.cpp 就绪后的联调项（当前为骨架，管线冒烟用 MockLLMBackend）：
  - flag 拼写以定制版 `llama-cli --help` 为准，特殊 flag 走 extra_args；
  - CLI 模式 stderr 的 eval 统计解析（替换 len//4 启发式 token 数）；
  - 若定制版可自报显存（server /props 或日志），替换 nvidia-smi 差分。
"""

from __future__ import annotations

import json
import logging
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

from common.llm_backends.base import BaseLLMBackend, GenerationResult

logger = logging.getLogger("llamacpp")

# 默认停用词：Qwen3 chat 模板的结束标记（模板由调用方渲染，续写至 EOS 标记即停）
DEFAULT_STOP: List[str] = ["<|im_end|>", "<|endoftext|>"]


def _heuristic_tokens(text: str) -> int:
    """无精确 token 统计时的启发式估计（英文 ~4 字符/token）。"""
    return max(1, len(text) // 4)


class LlamaCppBackend(BaseLLMBackend):
    """定制 llama.cpp 后端。bin_path / server_url 二选一：

    - bin_path（CLI 模式）：每条样本起一个 llama-cli 进程。参数拼装为
      纯函数 _build_cli_args，可离线单测；
    - server_url（server 模式）：对接 llama-server 的 /completion
      端点（模型常驻），批量评测与 RTF/TTFT 统计必须走此模式。
    """

    backend_name = "llamacpp"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        bin_path: Optional[Union[str, Path]] = None,
        server_url: Optional[str] = None,
        n_gpu_layers: Optional[int] = None,
        ctx_size: Optional[int] = None,
        repeat_penalty: float = 1.10,
        extra_args: Optional[List[str]] = None,
        timeout_s: float = 300.0,
        tokenizer_ref: Optional[str] = None,
        cache_prompt: bool = False,
        stop: Optional[List[str]] = None,
        seed: int = 42,
    ):
        if bool(bin_path) == bool(server_url):
            raise ValueError("bin_path 与 server_url 必须二选一")
        if bin_path and not model_path:
            raise ValueError("CLI 模式必须提供 GGUF model_path（-m）")
        self.model_path = Path(model_path) if model_path else None
        self.bin_path = Path(bin_path) if bin_path else None
        self.server_url = server_url
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.repeat_penalty = repeat_penalty
        self.extra_args = list(extra_args or [])
        self.timeout_s = timeout_s
        # count_tokens 用 HF 同源 tokenizer（与渲染 prompt 的 tokenizer 一致）
        self.tokenizer_ref = tokenizer_ref
        # 显式控制项：prompt cache（评测逐条独立，默认关防串档）、
        # 停用词（默认 Qwen3 结束标记）、随机种子（确定性对齐）
        self.cache_prompt = cache_prompt
        self.stop = list(stop) if stop is not None else list(DEFAULT_STOP)
        self.seed = seed
        self._tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        """模式相关的启动校验（幂等）。

        CLI 模式校验二进制与 GGUF 存在性；server 模式探测 /health
        确保 llama-server 已就绪（模型加载由服务端负责）。
        """
        if self._loaded:
            return
        if self.bin_path is not None:
            if not self.bin_path.exists():
                raise FileNotFoundError(f"llama-cli 不存在: {self.bin_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"GGUF 模型不存在: {self.model_path}")
            logger.info(
                "CLI 模式: %s | model=%s | ngl=%s",
                self.bin_path, self.model_path,
                "默认" if self.n_gpu_layers is None else self.n_gpu_layers,
            )
        else:
            self._check_server_health()
        self._loaded = True

    def _check_server_health(self) -> None:
        """探测 /health；503 明确提示模型加载中，其余异常统一报未就绪。"""
        health = self._endpoint("health")
        try:
            with urllib.request.urlopen(health, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"/health 状态码 {resp.status}")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                raise RuntimeError(
                    f"llama-server 返回 503：模型仍在加载中（{health}），"
                    f"请等待加载完成后再启动评测"
                ) from e
            raise RuntimeError(
                f"llama-server 未就绪（{health} 状态码 {e.code}）"
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(
                f"无法连接 llama-server（{health}）: {e}；"
                f"请确认 server 已启动且端口正确"
            ) from e
        logger.info("server 模式: %s（模型由服务端加载）", self.server_url)

    def _endpoint(self, name: str) -> str:
        """URL 归一化：允许传根地址或任一已知端点（先剥掉已知后缀再拼）。

        如传 "http://h:8080/completion" 求 health 时，须归一到
        "http://h:8080/health" 而非错误的 "http://h:8080/completion/health"。
        """
        base = self.server_url.rstrip("/")
        for suffix in ("completion", "health", "props", "tokenize"):
            if base.endswith(f"/{suffix}"):
                base = base[: -(len(suffix) + 1)]
                break
        return f"{base}/{name}"

    # ------------------------------------------------------------- CLI 模式
    def _build_cli_args(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.10,
    ) -> List[str]:
        """拼装 llama-cli 参数（纯函数，供单测校验确定性 flag）。

        -no-cnv: -p 即完整上文（模板已由调用方渲染），不做对话包装；
        --no-display-prompt: stdout 只含生成文本，解析无需截断 prompt 回显；
        -s: 显式随机种子（temp=0 下进一步对齐确定性）。
        """
        args = [
            str(self.bin_path),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(int(max_new_tokens)),
            "--temp", str(temperature),
            "--repeat-penalty", f"{repetition_penalty:.2f}",
            "-s", str(self.seed),
            "-no-cnv",
            "--no-display-prompt",
        ]
        if self.stop:
            args += ["-r", self.stop[0]]  # llama-cli 反斜杠转义多停用词繁琐，取首个
        if self.n_gpu_layers is not None:
            args += ["-ngl", str(self.n_gpu_layers)]
        if self.ctx_size is not None:
            args += ["-c", str(self.ctx_size)]
        args += self.extra_args
        return args

    @staticmethod
    def _parse_cli_stdout(stdout: str) -> str:
        """--no-display-prompt 下 stdout 仅含生成文本；strip 容错尾部换行。"""
        return stdout.strip()

    def _generate_cli(
        self, prompt: str, max_new_tokens: int,
        temperature: float, repetition_penalty: float,
    ) -> GenerationResult:
        try:
            proc = subprocess.run(
                self._build_cli_args(
                    prompt, max_new_tokens, temperature, repetition_penalty
                ),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"llama-cli 超时（> {self.timeout_s}s），进程已终止: "
                f"{self.bin_path}"
            ) from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-cli 退出码 {proc.returncode}: "
                f"{proc.stderr.strip()[-500:]}"
            )
        text = self._parse_cli_stdout(proc.stdout)
        return GenerationResult(
            text=text,
            prompt_tokens=_heuristic_tokens(prompt),
            completion_tokens=_heuristic_tokens(text),
            ttft_ms=None,  # 非流式 CLI 无法分离首 token 时刻
        )

    # ---------------------------------------------------------- server 模式
    def _build_request_payload(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        repeat_penalty: float = 1.10,
    ) -> dict:
        """llama-server /completion 请求体（纯函数，供单测）。

        cache_prompt 默认 False：GER 评测每条 prompt 独立，避免服务端
        前缀缓存把 TTFT/prompt_ms 统计口径搅乱（云端批跑可显式开启）。
        """
        payload = {
            "prompt": prompt,
            "n_predict": int(max_new_tokens),
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "stream": False,
            "cache_prompt": self.cache_prompt,
            "seed": self.seed,
        }
        if self.stop:
            payload["stop"] = list(self.stop)
        return payload

    @staticmethod
    def _parse_server_response(data: dict, prompt: str) -> GenerationResult:
        """解析 /completion 响应（纯函数，供单测）。

        timings 字段（llama-server 新版；旧版为 eval_n/eval_ms）：
            prompt_n / prompt_ms : prompt 评估 token 数与耗时
            cache_n              : KV cache 命中的前缀 token 数（计入已处理
                                   prompt，故 prompt_tokens = prompt_n+cache_n）
            predicted_n/predicted_ms（旧 eval_n/eval_ms）：解码 token 数与耗时
        TTFT ≈ prompt_ms + predicted_ms/predicted_n（prompt 评估 + 单步解码），
        为近似口径；timings 缺失时降级为启发式 token 数、ttft=None。
        """
        text = str(data.get("content", ""))
        timings = data.get("timings") or {}
        # 新版字段优先，旧版 eval_* fallback
        pred_n = timings.get("predicted_n") or timings.get("eval_n")
        pred_ms = timings.get("predicted_ms")
        if pred_ms is None:
            pred_ms = timings.get("eval_ms")
        prompt_n = int(timings.get("prompt_n") or 0)
        cache_n = int(timings.get("cache_n") or 0)
        prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        ttft_ms = None
        if pred_n:
            ttft_ms = prompt_ms + float(pred_ms or 0.0) / float(pred_n)
        prompt_tokens = prompt_n + cache_n
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens or _heuristic_tokens(prompt),
            completion_tokens=int(pred_n or 0) or _heuristic_tokens(text),
            ttft_ms=ttft_ms,
        )

    def _generate_server(
        self, prompt: str, max_new_tokens: int,
        temperature: float, repetition_penalty: float,
    ) -> GenerationResult:
        payload = json.dumps(
            self._build_request_payload(
                prompt, max_new_tokens, temperature, repetition_penalty
            )
        ).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint("completion"),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(
                f"llama-server /completion HTTP {e.code}: {body}"
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError(
                f"llama-server /completion 请求失败（超时 {self.timeout_s}s 或"
                f"连接断开）: {self._endpoint('completion')} -> {e}"
            ) from e
        return self._parse_server_response(data, prompt)

    # -------------------------------------------------------------- generate
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = None,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """生成；repetition_penalty 未传（None）时回落 self.repeat_penalty。

        temperature / repetition_penalty 显式传入时必须生效（透传到
        CLI 参数与 server payload），不再静默忽略。
        """
        self._load()
        rep = repetition_penalty if repetition_penalty is not None else self.repeat_penalty
        if self.bin_path:
            gen = lambda p: self._generate_cli(  # noqa: E731
                p, max_new_tokens, temperature, rep)
        else:
            gen = lambda p: self._generate_server(  # noqa: E731
                p, max_new_tokens, temperature, rep)
        if isinstance(prompt, str):
            return gen(prompt)
        return [gen(p) for p in prompt]

    # ------------------------------------------------------------- utilities
    def count_tokens(self, text: str) -> int:
        """token 精确计数优先级：HF 同源 tokenizer > server /tokenize > 启发式。"""
        if self.tokenizer_ref is not None:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_ref)
            return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
        if self.server_url is not None:
            n = self._server_tokenize(text)
            if n is not None:
                return n
        return _heuristic_tokens(text)

    def _server_tokenize(self, text: str) -> Optional[int]:
        """POST /tokenize 取精确 token 数；失败（网络/字段缺失）返回 None。"""
        req = urllib.request.Request(
            self._endpoint("tokenize"),
            data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            tokens = data.get("tokens")
            return len(tokens) if isinstance(tokens, list) else None
        except Exception as e:  # noqa: BLE001 - 任何失败都回落启发式
            logger.debug("/tokenize 失败，回落启发式: %s", e)
            return None

    def unload(self) -> None:
        """llama.cpp 运行在进程外，无进程内资源可释放（显存差分由调用方管理）。"""
        logger.info(
            "[%s] 进程外后端，无需卸载（显存由 nvidia-smi 差分统计）",
            self.backend_name,
        )
