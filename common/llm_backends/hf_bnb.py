"""HF transformers + bitsandbytes 推理后端（FP16 / INT8 / NF4）。

显存预算（Qwen3-1.7B，本地 3060 4.5GB 安全线）：
    fp16: ~3.4GB 权重 + KV cache（动态 max_new_tokens 控制在 <100MB）
    int8: ~1.9GB
    nf4 : ~1.2GB（double quant）

可选 ``lora_path``：加载底座后通过 peft 挂载 QLoRA adapter（Task 1.6 产物），
底座保持原量化精度，adapter 以低秩增量参与前向。
"""

from __future__ import annotations

import gc
import logging
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.llm_backends.base import BaseLLMBackend, GenerationResult
from common.profiler import cuda_cleanup, query_gpu_free_mb

logger = logging.getLogger("hf_bnb")


class HFBNBackend(BaseLLMBackend):
    """Transformers + bitsandbytes 后端。"""

    backend_name = "hf_bnb"

    def __init__(
        self,
        model_id: str,
        quant_mode: str = "fp16",
        device_map: str = "cuda:0",
        max_vram_gb: Optional[float] = None,
        repetition_penalty: float = 1.10,
        lora_path: Optional[str] = None,
    ):
        if quant_mode not in ("fp16", "int8", "nf4"):
            raise ValueError(f"未知 quant_mode: {quant_mode}（可选 fp16/int8/nf4）")
        self.model_id = model_id
        self.quant_mode = quant_mode
        self.device_map = device_map
        self.max_vram_gb = max_vram_gb
        self.repetition_penalty = repetition_penalty
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if self.model is not None:
            return
        if self.device_map.startswith("cuda"):
            free_mb = query_gpu_free_mb()
            if free_mb is not None:
                logger.info("加载前 GPU 空闲显存: %.0f MB", free_mb)
                if self.max_vram_gb and free_mb < self.max_vram_gb * 1024:
                    raise RuntimeError(
                        f"GPU 空闲 {free_mb:.0f}MB 低于预算 {self.max_vram_gb}GB，"
                        f"先清理残留进程或换更低的量化档"
                    )

        common = dict(
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
        )
        if self.quant_mode == "fp16":
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **common)
        elif self.quant_mode == "int8":
            # transformers 4.57：统一走 quantization_config
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                **common,
            )
        else:  # nf4
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                **common,
            )
        # 可选挂载 LoRA adapter（QLoRA 微调产物）：底座保持量化，仅附加低秩权重
        if self.lora_path:
            from peft import PeftModel

            logger.info("挂载 LoRA adapter: %s", self.lora_path)
            model = PeftModel.from_pretrained(
                model, self.lora_path, is_trainable=False
            )
        model.eval()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # batch 输入时右侧 pad 会污染左侧对齐，统一 left
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        repetition_penalty: float = 1.10,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        self._load()
        single = isinstance(prompt, str)
        prompts = [prompt] if single else list(prompt)

        # 外层已渲染 chat template，这里绝不能再加 special tokens
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded.input_ids.to(self.model.device)
        attention_mask = encoded.attention_mask.to(self.model.device)
        prompt_tokens = int(attention_mask.sum(dim=1).tolist()[0]) if single else None

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,          # 红线 7.2：确定性解码
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # 去掉 prompt 部分，仅保留新生成
        new_tokens = outputs[:, input_ids.shape[1]:]

        results: List[GenerationResult] = []
        for row_in, row_out in zip(input_ids, new_tokens):
            text = self.tokenizer.decode(row_out, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=int(row_in.sum().item()),
                    completion_tokens=int(row_out.numel()),
                    ttft_ms=None,  # 非流式生成无法测 TTFT，由外层 profiler 打点
                )
            )
        return results[0] if single else results

    # ------------------------------------------------------------- utilities
    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            self._load()
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def unload(self) -> None:
        """卸载模型并彻底释放显存（切换 quant_mode 前必须调用）。"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        cuda_cleanup()
        logger.info("[%s/%s] 已卸载并清理显存", self.model_id, self.quant_mode)

    def warmup(self) -> None:
        """单条 1-token 生成预热（触发 CUDA kernel 编译与显存池分配）。"""
        self._load()
        _ = self.generate("x", max_new_tokens=1)
