"""定制 llama.cpp 推理后端（CLI 进程 / HTTP server 双模式，路线 A/B 共享）。

设计约定（PROJECT_SPEC.md 红线 7.5 + BaseLLMBackend 接口）：
  - 路径与端口全部由调用方配置注入（--llamacpp_bin / --llamacpp_url），
    本层严禁硬编码任何环境相关路径；
  - chat template 由调用方（infer_ger.py）用 HF tokenizer 渲染
    （enable_thinking=False），本层以"纯续写"语义发送——CLI 模式
    -no-cnv、server 模式 /completion 端点，与 HF 后端共享同一评测管线；
  - 确定性解码：temp=0 + repeat_penalty 1.10（红线 7.2）；
  - 统计口径：
      * server 模式：响应 timings 提供精确 prompt/eval token 数与耗时，
        TTFT ≈ prompt_ms + eval_ms/eval_n（prompt 评估 + 单步解码）；
      * CLI 模式：每次 generate 重起进程重载模型，RTF 不可比（仅用于
        定制版功能调试）；token 数为 len//4 启发式，显存无差分可测；
  - 显存：模型在进程外运行，torch allocator 口径无效；server 模式由
    调用方用 query_gpu_used_mb() 做评测前后 nvidia-smi 差分注入
    profiler.report_peak_vram()。

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
from common.profiler import query_gpu_used_mb

logger = logging.getLogger("llamacpp")


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
            health = self._endpoint("health")
            try:
                with urllib.request.urlopen(health, timeout=self.timeout_s) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"/health 状态码 {resp.status}")
            except Exception as e:  # noqa: BLE001 - 统一转为启动失败
                raise RuntimeError(
                    f"llama-server 未就绪（{health}）: {e}；"
                    f"请先启动 server 并确认端口"
                ) from e
            logger.info("server 模式: %s（模型由服务端加载）", self.server_url)
        self._loaded = True

    def _endpoint(self, name: str) -> str:
        """URL 归一化：允许传根地址或任一已知端点（先剥掉已知后缀再拼）。

        如传 "http://h:8080/completion" 求 health 时，须归一到
        "http://h:8080/health" 而非错误的 "http://h:8080/completion/health"。
        """
        base = self.server_url.rstrip("/")
        for suffix in ("completion", "health", "props"):
            if base.endswith(f"/{suffix}"):
                base = base[: -(len(suffix) + 1)]
                break
        return f"{base}/{name}"

    # ------------------------------------------------------------- CLI 模式
    def _build_cli_args(self, prompt: str, max_new_tokens: int) -> List[str]:
        """拼装 llama-cli 参数（纯函数，供单测校验确定性 flag）。

        -no-cnv: -p 即完整上文（模板已由调用方渲染），不做对话包装；
        --no-display-prompt: stdout 只含生成文本，解析无需截断 prompt 回显。
        """
        args = [
            str(self.bin_path),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(int(max_new_tokens)),
            "--temp", "0",
            "--repeat-penalty", f"{self.repeat_penalty:.2f}",
            "-no-cnv",
            "--no-display-prompt",
        ]
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

    def _generate_cli(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        proc = subprocess.run(
            self._build_cli_args(prompt, max_new_tokens),
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
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
    @staticmethod
    def _build_request_payload(
        prompt: str, max_new_tokens: int, repeat_penalty: float,
    ) -> dict:
        """llama-server /completion 请求体（纯函数，供单测）。"""
        return {
            "prompt": prompt,
            "n_predict": int(max_new_tokens),
            "temperature": 0.0,
            "repeat_penalty": repeat_penalty,
            "stream": False,
        }

    @staticmethod
    def _parse_server_response(data: dict, prompt: str) -> GenerationResult:
        """解析 /completion 响应（纯函数，供单测）。

        timings 字段（llama-server 标准输出）:
            prompt_n / prompt_ms: prompt 评估 token 数与耗时
            eval_n / eval_ms    : 生成 token 数与总解码耗时
        TTFT ≈ prompt_ms + eval_ms/eval_n（prompt 评估 + 单步解码），
        为近似口径；timings 缺失时降级为启发式 token 数、ttft=None。
        """
        text = str(data.get("content", ""))
        timings = data.get("timings") or {}
        eval_n = timings.get("eval_n")
        prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        eval_ms = timings.get("eval_ms")
        ttft_ms = None
        if eval_n:
            ttft_ms = prompt_ms + float(eval_ms or 0.0) / float(eval_n)
        return GenerationResult(
            text=text,
            prompt_tokens=int(timings.get("prompt_n") or 0)
            or _heuristic_tokens(prompt),
            completion_tokens=int(eval_n or 0) or _heuristic_tokens(text),
            ttft_ms=ttft_ms,
        )

    def _generate_server(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        payload = json.dumps(
            self._build_request_payload(prompt, max_new_tokens, self.repeat_penalty)
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
        return self._parse_server_response(data, prompt)

    # -------------------------------------------------------------- generate
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        self._load()
        gen = (
            (lambda p: self._generate_cli(p, max_new_tokens))
            if self.bin_path
            else (lambda p: self._generate_server(p, max_new_tokens))
        )
        if isinstance(prompt, str):
            return gen(prompt)
        return [gen(p) for p in prompt]

    # ------------------------------------------------------------- utilities
    def count_tokens(self, text: str) -> int:
        """用 HF 同源 tokenizer 精确计数；未提供 tokenizer_ref 时启发式。"""
        if self.tokenizer_ref is None:
            return _heuristic_tokens(text)
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_ref)
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    def unload(self) -> None:
        """llama.cpp 运行在进程外，无进程内资源可释放（显存差分由调用方管理）。"""
        logger.info(
            "[%s] 进程外后端，无需卸载（显存由 nvidia-smi 差分统计）",
            self.backend_name,
        )
