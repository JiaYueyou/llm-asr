# llm-asr-quant

> 小参数 LLM（Qwen3 系列，≤8B）在英文语音识别（LibriSpeech）中的**生成式纠错（GER）**与**多精度量化**研究：
> 准确率—显存—延迟的三维权衡，以及端侧部署的可行性边界。

本仓库为两阶段工程。**阶段一（路线 A）已完成闭环**；阶段二（路线 B）为规划中的端到端 Speech-LLM 扩展。

| 阶段 | 路线 | 架构 | 状态 |
|---|---|---|---|
| 一 | Route A：级联 GER | `wav2vec2-base-100h` + `pyctcdecode` Top-5 N-best → Qwen3-1.7B 纠错 → 多精度量化评测 | ✅ 完成（1.7B 双测试集全档） |
| 二 | Route B：端到端 Speech-LLM | 冻结 `whisper-small` Encoder + 10 Hz 跨模态投影层 + 冻结 Qwen3，对比全链路量化 vs. 跨模态混合精度量化 | 🕒 规划中 |

阶段一的完整实验报告见 **[`docs/ROUTE_A_REPORT.md`](docs/ROUTE_A_REPORT.md)**；架构与数据契约的唯一事实来源是 [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md)。

---

## 1. 核心结果速览（阶段一）

全部数字来自 `outputs/metrics/summary.csv`，统一使用 **whisper-normalizer 1.0.1** 评测口径。

### 1.1 WER（%）

| 配置 | test-clean | test-other | 说明 |
|---|---:|---:|---|
| Beam-1（声学基线） | 5.84 | 13.04 | pyctcdecode Top-1，无 LLM |
| Top-5 Oracle（候选上界） | 4.25 | 11.16 | 逐句取 Top-5 最优候选 |
| FP16（hf_bnb） | 5.22 | 11.70 | 全精度参考 |
| INT8（hf_bnb） | 5.21 | — | test-other 因断电窗口跳过 |
| **Q8_0（llama.cpp）** | **5.21** | **11.76** | 端侧效率甜点 |
| Q4_K_M（llama.cpp） | 5.99 | 12.96 | K-quant 激进档，质量受损 |
| NF4（hf_bnb） | 6.10 | 13.08 | 4-bit 精度悬崖最明显 |
| **NF4 + QLoRA（本工作）** | **4.22** | **10.72** | 双集击穿 Top-5 Oracle |

### 1.2 端侧效率（test-clean）

| 配置 | Peak VRAM | RTF | Tokens/s |
|---|---:|---:|---:|
| FP16 | 3399 MB | 0.1415 | 22.06 |
| INT8 | 2155 MB | 0.5786 | 5.38 |
| **Q8_0** | **1931 MB** | **0.0400** | **77.97** |
| Q4_K_M | 1330 MB | 0.0404 | 76.82 |
| NF4 | 1423 MB | 0.1918 | 18.10 |
| NF4 + QLoRA | 1502 MB | 0.2523 | 12.33 |

**三条结论**：

1. **4-bit 存在真实精度悬崖**：NF4 在 clean 集比 FP16 差 0.88 pts、在噪声集差 1.38 pts，且错误方向是**插入**（I=679/1017）而非替换。
2. **两个 4-bit 后端错误方向相反**：NF4 偏插词（复读/加词），`Q4_K_M` 偏漏词（截断/粘连），在双集上对称复现（见 `fig2`）。
3. **Q8_0 是零样本帕累托最优**：WER 与 INT8 持平，但 RTF 快 14.5×、显存少 224 MB；相对 FP16 提速 3.5×（clean）–4.4×（other），显存少约 1.5 GB。
4. **QLoRA 是唯一击穿 Oracle 的配置**：仅用 1.5 GB 推理显存，在 clean 集比 Oracle 低 0.03 pts、other 集低 0.44 pts。

图表见 [`docs/figures/`](docs/figures/)（全部 SVG 矢量）：

- `fig1_wer_comparison.svg` — 双集 WER 对比 + Beam-1 / Oracle 参考线
- `fig2_sdi_divergence.svg` — 删除/插入错误的双集对称分化
- `fig3_3d_tradeoff.svg` — 显存 × 延迟 × 精度三维权衡空间
- `fig4_qlora_training_loss.svg` — QLoRA 1782 步真实收敛曲线
- `fig5_3d_sdi_landscape.svg` — S/D/I × 方法 × 数据集三维柱状矩阵

---

## 2. 项目目录

```text
llm-asr-quant/
├── common/                      # 阶段一、二 100% 共享
│   ├── text_norm.py             # EnglishTextNormalizer + 思考标签/前缀/引号防御性剥离
│   ├── metrics.py               # jiwer WER + S/D/I + summary.csv 契约
│   ├── profiler.py              # Peak VRAM / RTF / TTFT / Tokens-per-sec
│   └── llm_backends/
│       ├── base.py              # BaseLLMBackend 抽象接口
│       ├── hf_bnb.py            # transformers + bitsandbytes（FP16 / INT8 / NF4 / LoRA）
│       ├── llamacpp_custom.py   # 定制 llama.cpp（CLI / server 双模式）
│       └── mock.py              # 管线冒烟用零纠错后端
├── route_a_ger/                 # 阶段一：级联纠错
│   ├── extract_nbest.py         # wav2vec2 + pyctcdecode → Top-5 静态缓存
│   ├── prompts.py               # 受控 GER 模板 + extract_answer 防复读
│   ├── infer_ger.py             # 批量纠错评测（--resume / --template plain）
│   └── train_qlora.py           # QLoRA 低秩适配训练
├── route_b_e2e/                 # 阶段二：端到端 Speech-LLM（未启动）
├── data/prepare_librispeech.py  # LibriSpeech 清单构建
├── scripts/
│   ├── run_route_a_cloud.sh     # 魔搭 A10 一键流水线
│   ├── plot_route_a.py          # 论文级 SVG 图表复现脚本
│   └── analyze_route_a_cases.py # 改写率统计 + 定性样本挑选
├── docs/
│   ├── PROJECT_SPEC.md          # 架构/契约/任务清单 SSOT
│   ├── ROUTE_A_REPORT.md        # 阶段一实验报告
│   ├── CLOUD_RUNBOOK.md         # 云端运行手册
│   └── figures/                 # 出版级 SVG 图表
├── tests/                       # pytest（136 passed）
└── outputs/                     # 实验产物（gitignore，各环境自生成）
```

---

## 3. 环境配置

### 3.1 硬件与显存红线

| 环境 | 硬件 | 可用显存 | 承担任务 |
|---|---|---|---|
| 本地 | RTX 3060 Laptop | 6 GB（**安全上限 4.5 GB**） | 1.7B 全档推理、QLoRA 冒烟、llama.cpp 联调 |
| 云端 | 魔搭 A10 | 22.2 GiB（23 GB 空闲） | QLoRA 全量训练、1.7B 双集批跑 |

### 3.2 安装

```bash
source /home/yzy/software/miniconda3/etc/profile.d/conda.sh
conda activate llm-asr          # Python 3.10

# PyTorch 必须 >= 2.6：transformers 4.57 因 CVE-2025-32434 拒绝旧版加载 .bin 权重
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# whisper-normalizer 1.0.1 声明 Requires-Python>=3.11，py3.10 需跳过版本检查
pip install --ignore-requires-python whisper-normalizer==1.0.1
```

> **口径警告**：`whisper-normalizer` 0.1.x 会把所有格 `shelley's` 展开为 `shelley is`，
> 1.0.x 输出 `shelleys`。两者 WER 相差约 0.10 pts，**混用会导致跨环境结果不可比**。
> 本仓库所有数字统一为 **1.0.1**。

### 3.3 离线与镜像

模型已缓存后一律离线运行，避免 transformers 的 HEAD 探测导致脚本假死：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

首次下载可用 `export HF_ENDPOINT=https://hf-mirror.com`（镜像偶有文件错配，模型优先官方源 + `wget -c` 断点续传）。

---

## 4. 核心推理规范（红线）

1. **思考模式强制关闭**：`enable_thinking=False`；后处理再做防御性剥离（`normalize_for_wer`），防止残留思考标签污染 WER。
2. **确定性解码**：`do_sample=False`、`temperature=0`、`repetition_penalty=1.10`；`max_new_tokens = int(top1_words * 2.0) + 24`。
3. **模板锁定 `--template plain`**：`numbered` 编号列表会诱导模型逐条复述候选（曾导致 NF4 WER 24.4%）。
4. **防复读三层防护**：`extract_answer()` 词级近似重复检测（短单元 ≥0.95、长单元 ≥0.85 相似度）→ 多行取末行 → 剥行首编号。
5. **断点续跑**：所有批跑一律带 `--resume`，逐条 fsync checkpoint，统计口径跨段不失真。
6. **llama.cpp 批跑必须 `--cache-ram 0`**：默认 8 GB 主机 RAM prompt 缓存会在 WSL2（总内存 7.5 GB）触发 OOM。
7. **显存卫生**：切换精度档前 `backend.unload()`；每次评测前 `reset_peak_memory_stats()`。

---

## 5. 一键复现

### 5.1 声学基线 + N-best 缓存

```bash
python data/prepare_librispeech.py --root /path/to/LibriSpeech
python route_a_ger/extract_nbest.py --manifest data/manifests/test-clean.jsonl \
  --out outputs/nbest/test-clean.jsonl
```

### 5.2 hf_bnb 量化评测（FP16 / INT8 / NF4）

```bash
python route_a_ger/infer_ger.py --nbest outputs/nbest/test-clean.jsonl \
  --model Qwen/Qwen3-1.7B --backend hf_bnb --quant_mode nf4 \
  --template plain --resume --write_summary \
  --exp_name ger-1.7b-nf4-test-clean
```

### 5.3 llama.cpp GGUF 评测

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits   # 记基线 MB
nohup /path/to/llama.cpp/build/bin/llama-server \
  -m /path/to/Qwen3-1.7B-Q8_0.gguf -np 1 -ngl 99 -c 2048 --cache-ram 0 \
  --host 127.0.0.1 --port 18080 > /tmp/llama_server.log 2>&1 &

python route_a_ger/infer_ger.py --nbest outputs/nbest/test-clean.jsonl \
  --backend llamacpp --quant_mode q8_0 --template plain \
  --llamacpp_url http://127.0.0.1:18080 --vram_baseline_mb <基线> \
  --resume --write_summary --exp_name ger-1.7b-q8_0-llamacpp-test-clean

pkill -f "build/bin/llama-server -m"
```

### 5.4 QLoRA 训练与 LoRA 推理

```bash
# 训练（云端 A10；本地 6 GB 只能冒烟）
python route_a_ger/train_qlora.py --nbest outputs/nbest/train-clean-100.jsonl \
  --model Qwen/Qwen3-1.7B --output_dir outputs/checkpoints/qwen3_1.7b_qlora_100h \
  --epochs 2 --batch_size 4 --grad_accum 8 --lr 2e-4 --dtype bf16

# 推理（必须 --template plain，脚本会与 adapter 的 qlora_summary.json 硬校验）
python route_a_ger/infer_ger.py --nbest outputs/nbest/test-clean.jsonl \
  --model Qwen/Qwen3-1.7B --backend hf_bnb --quant_mode nf4 \
  --lora_path outputs/checkpoints/qwen3_1.7b_qlora_100h \
  --template plain --resume --write_summary \
  --exp_name ger-Qwen3-1.7B-nf4_qlora-test-clean
```

### 5.5 云端一键流水线

```bash
bash scripts/run_route_a_cloud.sh --stage download    # CPU 实例
bash scripts/run_route_a_cloud.sh --stage nbest       # GPU 实例
bash scripts/run_route_a_cloud.sh --stage qlora
bash scripts/run_route_a_cloud.sh --stage sweep
bash scripts/run_route_a_cloud.sh --stage pack
```

详见 [`docs/CLOUD_RUNBOOK.md`](docs/CLOUD_RUNBOOK.md)。

### 5.6 图表与案例分析复现

```bash
python scripts/plot_route_a.py            # → docs/figures/*.svg
python scripts/analyze_route_a_cases.py   # → docs/route_a_analysis.json
```

---

## 6. 测试

```bash
HF_HUB_OFFLINE=1 python -m pytest tests/ -q
# 136 passed
```

覆盖：文本规范化、WER/S/D/I 契约、显存与 RTF 统计、hf_bnb 与 llama.cpp 后端、
`extract_answer` 防复读、断点续跑端到端、QLoRA 训练/推理模板一致性校验。

---

## 7. 引用与致谢

- 声学前端：`facebook/wav2vec2-base-100h` + [`pyctcdecode`](https://github.com/kensho-technologies/pyctcdecode)
- 语言模型：[Qwen3](https://github.com/QwenLM/Qwen3)（1.7B）
- 量化：HuggingFace [`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes)、[`llama.cpp`](https://github.com/ggml-org/llama.cpp)（定制版）
- 评测：[`jiwer`](https://github.com/jitsi/jiwer) + [`whisper-normalizer`](https://github.com/kurianbenoy/whisper_normalizer)
