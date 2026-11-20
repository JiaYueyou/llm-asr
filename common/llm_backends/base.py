"""可插拔 LLM 推理后端：统一抽象接口（路线 A/B 共享）。

设计约定：
  - 调用方（infer_ger.py / infer_e2e.py）负责渲染 chat template
    （含 enable_thinking=False），本层只接收已套模板的字符串；
  - 所有后端强制确定性解码；
  - 返回结构统一 GenerationResult，耗时/显存聚合由 common.profiler 完成。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class GenerationResult:
    """单次生成的统一返回结构（所有后端一致）。"""

    text: str                      # 原始生成文本（未清洗）
    prompt_tokens: int            # prompt 侧 token 数
    completion_tokens: int         # 生成侧 token 数
    ttft_ms: Optional[float] = None  # 首 token 延迟（ms）；非流式后端为 None


class BaseLLMBackend(ABC):
    """文本 LLM 后端抽象基类。

    实现方（hf_bnb / llamacpp_custom）必须：
      1. generate() 内强制 do_sample=False / temperature=0 类确定性解码；
      2. unload() 彻底释放显存（红线 7.4）；
      3. 不感知 Qwen3 特有逻辑（模板渲染在调用方）。
    """

    backend_name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """生成。prompt 为单个或批量（已套模板）字符串。

        temperature 仅为接口兼容保留，实现方必须走确定性路径。
        返回：单 prompt -> 单结果；批量 -> 结果列表（与输入同序）。
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """估算文本 token 数（各后端用自家 tokenizer）。"""

    @abstractmethod
    def unload(self) -> None:
        """卸载模型并释放显存：del + gc + empty_cache + ipc_collect。"""

    def warmup(self) -> None:
        """可选预热钩子（默认空操作）。"""
        return None

    def _load(self) -> None:
        """懒加载/就绪校验钩子（默认空操作）。

        有加载需求的后端覆写：hf_bnb 加载权重与 tokenizer，
        llamacpp 校验二进制/探测 server；调用方统一在首次生成前
        调用一次，Mock 等无状态后端走默认空操作即可。
        """
        return None
