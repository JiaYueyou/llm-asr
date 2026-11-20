"""性能与显存监控模块（路线 A / B 共享）。

核心职责（PROJECT_SPEC.md 第 7.4 条）：
  1. GPU Peak VRAM：每轮评测前自动 reset_peak_memory_stats；
  2. 总推理耗时、RTF = 总推理耗时 / 总音频时长；
  3. TTFT（首 token 延迟）与 Tokens/s（解码吞吐）。

设计约束：
  - 无 CUDA 环境（CI / 纯 CPU 单测）优雅降级：peak_vram_mb = None；
  - 只依赖 torch 的基础 API，不引入后端特定逻辑（llama.cpp 后端通过
    手动 report() 注入自己的耗时/token 数即可复用同一套统计）。

用法：
    profiler = InferenceProfiler()
    with profiler.track():           # 评测全程
        for utt in utts:
            profiler.mark_ttft()     # 收到首 token 时调用
            ...
    stats = profiler.finalize(total_audio_sec=123.4, total_new_tokens=4567)
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from typing import Dict, Optional

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch 为硬依赖，仅防御性处理
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _cuda_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def cuda_cleanup() -> None:
    """显式释放显存与 Python 引用（切换模型/后端前调用，红线 7.4）。"""
    gc.collect()
    if _cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class InferenceProfiler:
    """单轮评测的计时 / 显存 / 吞吐统计器（非线程安全）。

    生命周期：track() 开始 -> 逐样本 mark_ttft()/累加 token -> finalize() 汇总。
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.reset()

    def reset(self) -> None:
        self._t_start: Optional[float] = None
        self._t_end: Optional[float] = None
        self._ttft: Optional[float] = None
        self._total_new_tokens: int = 0
        self._peak_vram_mb: Optional[float] = None

    @contextmanager
    def track(self):
        """评测全程上下文：重置峰值统计并计时。

        进入时调用 torch.cuda.reset_peak_memory_stats()（红线 7.4），
        保证 peak 反映本轮而非历史最大。

        注意：CUDA 上下文未初始化时 reset_peak_memory_stats 会抛
        Invalid device argument，故先做一次微小分配触发懒初始化。
        """
        if _cuda_available():
            if not torch.cuda.is_initialized():
                torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(self.device_id)
        self.reset()
        self._t_start = time.perf_counter()
        try:
            yield self
        finally:
            self._t_end = time.perf_counter()
            if _cuda_available():
                self._peak_vram_mb = torch.cuda.max_memory_allocated(
                    self.device_id
                ) / (1024**2)
                # max_memory_allocated 是 torch allocator 口径；
                # nvidia-smi 的进程级占用可通过 report_peak_vram() 覆盖

    def mark_ttft(self) -> float:
        """记录首 token 时刻（相对 track() 开始的秒数）。

        对每个样本，LLM 后端在拿到首段生成文本时调用一次；
        多次调用以首次为准。
        """
        if self._t_start is None:
            raise RuntimeError("mark_ttft() 必须在 track() 上下文内调用")
        if self._ttft is None and self._t_start is not None:
            self._ttft = time.perf_counter() - self._t_start
        return self._ttft

    def report_ttft(self, ttft_sec: float) -> None:
        """外部后端（如 llama.cpp server）直接注入 TTFT，单位秒。"""
        if self._ttft is None:
            self._ttft = float(ttft_sec)

    def add_tokens(self, n: int) -> None:
        """累加本轮生成的 token 数（用于 Tokens/s）。"""
        if n < 0:
            raise ValueError("token 数不能为负")
        self._total_new_tokens += int(n)

    def report_peak_vram(self, peak_mb: float) -> None:
        """外部注入进程级显存峰值（如 nvidia-smi / llama.cpp 统计）。"""
        self._peak_vram_mb = float(peak_mb)

    @property
    def total_infer_sec(self) -> float:
        if self._t_start is None or self._t_end is None:
            raise RuntimeError("total_infer_sec 需在 track() 上下文结束后读取")
        return self._t_end - self._t_start

    @property
    def total_new_tokens(self) -> int:
        return self._total_new_tokens

    @property
    def peak_vram_mb(self) -> Optional[float]:
        return self._peak_vram_mb

    @property
    def ttft(self) -> Optional[float]:
        return self._ttft

    def finalize(
        self,
        total_audio_sec: float,
        total_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        """汇总统计，返回可直接填入 summary.csv 的字段值。

        Returns:
            {"peak_vram_mb": 4123.5 or None, "total_infer_sec": 98.7,
             "rtf": 0.51, "ttft": 0.87, "tokens_per_sec": 34.2,
             "total_audio_sec": ..., "total_new_tokens": ...}
        """
        if total_audio_sec <= 0:
            raise ValueError("total_audio_sec 必须为正")
        infer_sec_raw = self.total_infer_sec  # 未取整原值，避免 round(0.0004)->0 导致下游失真
        if total_new_tokens is not None:
            self._total_new_tokens = total_new_tokens
        tokens_per_sec = (
            self._total_new_tokens / infer_sec_raw if infer_sec_raw > 0 else 0.0
        )
        return {
            "peak_vram_mb": self._peak_vram_mb,
            "total_audio_sec": round(total_audio_sec, 3),
            "total_infer_sec": round(infer_sec_raw, 3),
            "rtf": round(infer_sec_raw / total_audio_sec, 4),
            "ttft": round(self._ttft, 4) if self._ttft is not None else None,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "total_new_tokens": self._total_new_tokens,
        }


def query_gpu_free_mb(device_id: int = 0) -> Optional[float]:
    """通过 nvidia-smi 查询指定 GPU 的空闲显存（MB）；失败返回 None。

    用于评测前的显存余量检查（本地 4.5GB 安全上限）与
    外部后端的进程级占用统计。
    """
    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        total, used = (float(x) for x in out.split(","))
        return total - used
    except Exception:  # noqa: BLE001 - 任何失败都按"不可用"处理
        return None
