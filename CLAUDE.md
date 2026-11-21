# CLAUDE.md — llm-asr-quant 项目协作规范

## 项目定位与规范索引
研究 8B 参数以内开源小参数 LLM（Qwen3 系列）及其量化模型在英文 ASR
（LibriSpeech）中的准确率与端侧效率权衡。两阶段演进：
- 阶段一（Route A）：wav2vec2 + pyctcdecode 导出 Top-5 N-best，Qwen3 零样本/QLoRA GER 纠错，多精度量化评测，接定制 llama.cpp 测 RTF。
- 阶段二（Route B）：Whisper-small Encoder + Projector + Qwen3 端到端 Speech-LLM（暂未启动）。

**`docs/PROJECT_SPEC.md` 是架构、数据契约、任务清单的唯一事实来源（SSOT）**。
修改核心逻辑 / 数据 schema / 任务顺序前，必须先查阅该文件。

## 目录架构速览
- `common/` — 阶段一二 100% 共享：`text_norm.py`（EnglishTextNormalizer + 防御性清洗）、`metrics.py`（jiwer WER + S/D/I + summary.csv 契约）、`profiler.py`（Peak VRAM/RTF/TTFT/Tokens/s）、`llm_backends/`（`base.py` 抽象 / `hf_bnb.py` FP16·INT8·NF4 / `llamacpp_custom.py` 预留）。
- `route_a_ger/` — Route A 专用：`extract_nbest.py`（声学前端 N-best 缓存）、`prompts.py`（受控 GER 模板 + `extract_answer` 防复读解析）、`infer_ger.py`（批量纠错评测）。
- `data/` — `prepare_librispeech.py` 清单脚本；`manifests/`（不入库，各环境自生成）。
- `outputs/` — `nbest/` 静态缓存、`metrics/summary.csv`、按实验名归档的 predictions/cases（不入库）。
- `tests/` — pytest 单元测试（`test_common.py` / `test_backends.py`）。
- `route_b_e2e/`、`scripts/` — 阶段二与云端批跑（暂未实现）。

## 开发与运行约束
- **环境**：WSL2 + RTX 3060 Laptop 6GB（**安全上限 4.5GB**，勿超）。Conda 环境名 `llm-asr`（Python 3.10）：
  `source /home/yzy/software/miniconda3/etc/profile.d/conda.sh && conda activate llm-asr`
- **PyTorch**：必须 >= 2.6（transformers 4.57 CVE 检查拒绝旧版 torch 加载 .bin 权重）。当前 torch 2.6.0+cu124。
- **网络**：huggingface.co 直连不稳定；模型下载用 `HF_ENDPOINT=https://hf-mirror.com`（镜像下载数据可能文件错配，优先官方源+断点续传）；模型已在本地缓存后一律 `export HF_HUB_OFFLINE=1` 避免重试阻塞。
- **显存红线**：切换模型/量化档前必须 `backend.unload()` 或 `cuda_cleanup()`；每次评测前 `reset_peak_memory_stats`（profiler.track() 已内置）；本地配置默认 batch_size=1。
- **Qwen3 红线**：所有调用强制 `enable_thinking=False` + `do_sample=False` + `repetition_penalty≥1.05`；`max_new_tokens` 动态上限 `int(top1_words*2.0)+24`；输出经 `extract_answer()`（防复读/列表退化）+ `normalize_for_wer()`（防思考标签泄漏）。
- **测试**：改 `common/` 或 `prompts.py` 后必须 `python -m pytest tests/ -q` 全绿再提交。
- **代码风格**：中文注释、与现有模块一致的 logging 格式、CLI 参数带 `--max_samples` 冒烟通道。

## 会话交接约定
- **每次开启新会话时：若根目录存在 `TODO_HANDOFF.md`，优先读取该文件恢复当前进度**，再按 `docs/PROJECT_SPEC.md` 的任务清单推进。
- `TODO_HANDOFF.md` 每完成一个 Task 由当次会话更新；`CLAUDE.md` 只记录长期不变的规范。
- 提交节奏：每个 Task 完成后建议 `git add` 相关源码与测试（outputs/、manifests/ 不入库）。
