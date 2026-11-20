# Project Specification: Small-Scale Quantized LLM for Edge-Oriented ASR (Qwen3)

## 1. 项目目标与分阶段定位
本项目研究 8B 参数以内开源小参数 LLM 及其量化模型在英文语音识别（LibriSpeech / OpenSLR 12）中的准确率与端侧效率权衡，采用“共享底座 + 分阶段演进”的工程架构：

- **阶段一（路线 A：级联生成式纠错 GER）**
  - **交付目标**：完成课程/阶段性实验报告。
  - **核心内容**：使用 `wav2vec2-base-100h + pyctcdecode` 导出 Top-5 N-best 候选文本并静态缓存；基于 Qwen3 系列执行零样本/LoRA 纠错与多精度量化评测；接入自研/定制版 `llama.cpp` 后端进行推理加速与端侧实时率（RTF）统计。
- **阶段二（路线 B：端到端 Speech-LLM）**
  - **交付目标**：扩展完整对比实验，支撑普刊论文撰写（含端侧部署代理指标与可行性论证，暂不做真机部署）。
  - **核心内容**：复用阶段一数据与评测底座，构建 `Frozen Whisper-small Encoder + Trainable Cross-Modal Projector + Frozen Qwen3` 端到端语音大模型；重点对比路线 A 与路线 B，并开展“全链路量化 vs. 跨模态混合精度量化（Encoder/Projector 保持 FP16，仅 LLM 量化）”消融实验。

---

## 2. 硬件与运行环境约束
所有代码必须严格适配“本地 6GB 显存调试 + 云端 24GB 显存全量批跑”的双环境切换，禁止写死路径或显存配置：

- **本地开发环境**：WSL2 (Linux) + NVIDIA RTX 3060 **6GB VRAM**（实际安全可用显存约 **5.2 GB**）。
  - 本地承担任务：数据清洗与清单构建、`wav2vec2` 推理及 N-best 缓存导出、`Qwen3-1.7B` 全精度/量化推理与 QLoRA 微调调试、定制 `llama.cpp` 后端联调、路线 B 小批量（50 条样本）前向/反向逻辑冒烟测试。
- **云端算力环境**：Linux + 单卡 RTX 4090 (24GB VRAM)。
  - 云端承担任务：`Qwen3-4B` / `Qwen3-8B` 全精度及量化批量评测、完整 LoRA 纠错微调、路线 B 在 `train-clean-100` 上的 Projector 全量训练。

---

## 3. 核心技术选型

| 模块 | 选型方案 | 备注 |
| :--- | :--- | :--- |
| **数据集** | LibriSpeech (`test-clean`, `test-other`, `dev-clean`, `train-clean-100`) | 阶段一先在本地下载 `dev/test`；训练子集按需抽取或云端全量拉取 |
| **LLM 骨干家族** | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` | **强制关闭思考模式**（`enable_thinking=False`） |
| **路线 A 声学前端** | `facebook/wav2vec2-base-100h` + `pyctcdecode` | 导出 Top-5 N-best 及对数概率；可选挂 4-gram KenLM 作为传统基线 |
| **路线 B 声学编码器** | `openai/whisper-small` (仅 Encoder，冻结) | 50Hz 声学帧经 5 帧拼接下采样至 10Hz，接 2 层 Linear+ReLU/SwiGLU MLP 投影层 |
| **科研量化后端** | HuggingFace `transformers` + `bitsandbytes` + `peft` | 支持 FP16/BF16、INT8 (`load_in_8bit`)、INT4 (`nf4` + `double_quant`) |
| **加速与端侧评测后端** | 个人定制 `llama.cpp`（GGUF，如 `Q8_0` / `Q4_K_M`） | 封装为独立 Backend 适配器，与 HF 后端共用评测和 RTF 统计接口 |
| **评测与规范化** | `jiwer` + `whisper-normalizer` (`EnglishTextNormalizer`) | 统一输出 WER、S/D/I 错误分解、Peak VRAM、总耗时、RTF、TTFT、Tokens/s |

---

## 4. 统一项目目录结构（必须严格遵守解耦设计）

```text
llm-asr-quant/
├── PROJECT_SPEC.md              # 本任务规格文档
├── configs/                     # YAML 配置文件（区分 local_3060 / cloud_4090）
│   ├── route_a_local.yaml
│   ├── route_a_cloud.yaml
│   └── route_b_e2e.yaml
├── data/
│   ├── prepare_librispeech.py   # 扫描 LibriSpeech 生成统一 JSONL 清单
│   └── manifests/               # test-clean.jsonl, test-other.jsonl, train-clean-100.jsonl
├── common/                      # 【阶段一、二 100% 共享核心】
│   ├── text_norm.py             # EnglishTextNormalizer + 剥离 LLM 多余前缀/标点
│   ├── metrics.py               # jiwer 计算 WER 及 S/D/I，导出汇总 CSV 与错例 JSONL
│   ├── profiler.py              # 显存峰值 (PyTorch/nvidia-smi)、RTF、TTFT、吞吐量统计
│   └── llm_backends/            # 可插拔推理后端接口
│       ├── base.py              # 统一抽象接口 BaseLLMBackend.generate(prompt)
│       ├── hf_bnb.py            # Transformers + bitsandbytes (FP16/INT8/NF4)
│       └── llamacpp_custom.py   # 个人定制 llama.cpp 接口封装（CLI/Server/Binding）
├── route_a_ger/                 # 【阶段一：级联纠错实验】
│   ├── extract_nbest.py         # wav2vec2-base-100h + pyctcdecode 导出并缓存 Top-5
│   ├── prompts.py               # 严格受控的 GER Prompt 模板（禁用思维链）
│   ├── infer_ger.py             # 读取缓存 JSONL，切换不同 Backend 批量纠错并评测
│   └── finetune_qlora.py        # QLoRA 微调脚本（适配 6GB 本地调试与 24GB 正式训练）
├── route_b_e2e/                 # 【阶段二：端到端 Speech-LLM】
│   ├── projector.py             # 下采样 + 2层 MLP 跨模态投影层定义
│   ├── model_speech_llm.py      # Whisper-small Encoder + Projector + Qwen3 联合前向封装
│   ├── train_projector.py       # 冻结 Encoder & LLM，仅训练 Projector
│   └── infer_e2e.py             # 端到端评测（支持全量化 vs 混合精度量化开关）
├── scripts/                     # 一键批跑 Shell 脚本（便于云端无人值守跑完全部组合）
└── outputs/                     # 按实验名自动归档：predictions/、metrics/、cases/
```

---

## 5. 数据接口契约（JSONL Schema）

1. **音频清单 (`data/manifests/*.jsonl`)**
```json
{"id": "1089-134686-0000", "audio_path": "/abs/path/to/1089-134686-0000.flac", "text": "HE HOPED THERE WOULD BE STEW FOR DINNER", "duration": 10.435}
```

2. **路线 A N-best 静态缓存 (`outputs/nbest/*.jsonl`)**
```json
{
  "id": "1089-134686-0000",
  "ref": "HE HOPED THERE WOULD BE STEW FOR DINNER",
  "duration": 10.435,
  "ctc_greedy": "HE HOPED THERE WOULD BE STU FOR DINNER",
  "nbest": [
    {"text": "HE HOPED THERE WOULD BE STU FOR DINNER", "score": -12.41},
    {"text": "HE HOPED THERE WOULD BE STEW FOR DINNER", "score": -12.89}
  ]
}
```

3. **统一评测指标输出 (`outputs/metrics/summary.csv`)**
必须包含字段：`exp_name, route, split, model, backend, quant_mode, wer, sub, del, ins, peak_vram_mb, total_audio_sec, total_infer_sec, rtf, tokens_per_sec`

---

## 6. 分阶段开发任务清单

### Stage 1: 共享底座 + 路线 A（当前优先）
- [ ] **Task 1.1**：初始化 WSL2 Python 虚拟环境与 `requirements.txt`，编写 `data/prepare_librispeech.py` 并生成 `test-clean` / `test-other` 清单。
- [ ] **Task 1.2**：实现 `common/text_norm.py`、`common/metrics.py`、`common/profiler.py` 并附带单元测试。
- [ ] **Task 1.3**：实现 `route_a_ger/extract_nbest.py`，在本地 3060 上跑完 `test-clean` 与 `test-other` 并缓存 Top-5 JSONL，产出纯声学基线 WER。
- [ ] **Task 1.4**：实现 `common/llm_backends/hf_bnb.py` 与 `route_a_ger/infer_ger.py`，在本地跑通 `Qwen3-1.7B` 的 FP16 / INT8 / NF4 零样本评测。
- [ ] **Task 1.5**：实现 `common/llm_backends/llamacpp_custom.py`，对接本地自研/定制 `llama.cpp`，跑通 GGUF 模型推理与真实端侧 RTF/显存统计。
- [ ] **Task 1.6**：实现 `route_a_ger/finetune_qlora.py`，用 `train-clean-100` 子集抽取 N-best，在本地 3060 调试 QLoRA 纠错微调并评估效果提升。
- [ ] **Task 1.7**：编写 `scripts/run_route_a_cloud.sh`，上云端 4090 一键跑完 `4B` / `8B` 全组合并回收实验报告图表数据。

### Stage 2: 路线 B 端到端扩展（Stage 1 完成后启动）
- [ ] **Task 2.1**：实现 `route_b_e2e/projector.py`（5 帧拼接下采样 + 2 层 MLP）与 `model_speech_llm.py`（接入 `openai/whisper-small` Encoder 与 Qwen3）。
- [ ] **Task 2.2**：实现 `route_b_e2e/train_projector.py`，在本地 3060 上用 50 条样本跑通梯度流和 Checkpoint 保存，确认无显存泄漏。
- [ ] **Task 2.3**：在云端 4090 上使用 `train-clean-100` 训练 `1.7B / 4B / 8B` 对应的 Projector。
- [ ] **Task 2.4**：实现 `route_b_e2e/infer_e2e.py`，完成“全量化 vs. 跨模态混合精度量化（仅量化 LLM）”消融实验，与路线 A 汇总写入同一 `summary.csv`。

---

## 7. 关键工程红线（开发时必须遵守）
1. **Qwen3 思考模式禁用**：所有调用 Qwen3 的地方（HF `apply_chat_template(..., enable_thinking=False)` 以及 `llama.cpp` 传参/模板）必须显式关闭 thinking mode，且后处理代码需做防御性剥离（防止残留思考标签污染 WER）。
2. **确定性解码与防复读**：ASR 评测阶段一律设置 `do_sample=False`（或 `temperature=0.0`），限制 `max_new_tokens`（建议按输入词数动态设上限，如 `int(ref_len * 1.5) + 16`），并配置合理 `repetition_penalty`（如 `1.05`–`1.1`），防止低比特量化诱发死循环复读。
3. **文本规范化一致性**：参考文本（LibriSpeech 全大写无标点）与模型输出在送入 `jiwer` 前，必须同时经过 `EnglishTextNormalizer`，并剥离类似 `"Corrected text:"` 的对话前缀。
4. **6GB 显存防护**：本地配置文件默认 `batch_size=1`、开启 `gradient_checkpointing=True`、每次切换模型显式调用 `del model; gc.collect(); torch.cuda.empty_cache()`；`profiler.py` 需在每轮评测前调用 `torch.cuda.reset_peak_memory_stats()`。
5. **`llama.cpp` 后端解耦**：由于使用个人定制版 `llama.cpp`，在 `llamacpp_custom.py` 中通过配置文件读取自定义可执行文件路径或服务端口，严禁硬编码路径；需同时采集精确的 Prompt 评估耗时、生成耗时、TTFT、Tokens/s 和进程级显存占用。