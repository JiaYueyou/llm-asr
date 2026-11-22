# Route A 云端运行手册（魔搭 ModelScope 免费实例）

> 目标：用免费 **CPU 实例**下数据/模型，用免费 **24G GPU 实例**（剩余额度约 33h）
> 跑 N-best、QLoRA 与全组合评测，最后打包回本地。
> 所有持久化内容统一放在 `/mnt/workspace`（可用 `WORKSPACE_ROOT` 覆盖）。

## 0. 目录规划（持久化）

```text
${WORKSPACE_ROOT:-/mnt/workspace}/
├── llm-asr-quant/            # 代码仓库（上传或 git clone）
├── hf_home/                  # HF_HOME
├── modelscope_cache/         # MODELSCOPE_CACHE
├── models/                   # MODELS_ROOT
│   ├── Qwen3-1.7B/  Qwen3-4B/  Qwen3-8B/
│   └── wav2vec2-base-100h/
├── data/                     # DATA_ROOT
│   ├── LibriSpeech/train-clean-100/
│   └── manifests/train-clean-100.jsonl
└── outputs/                  # OUTPUT_ROOT
    ├── nbest/                # 上传本地 dev/test 缓存 + 云端生成的 train-clean-100
    ├── checkpoints/qwen3_1.7b_qlora_100h/
    ├── metrics/summary.csv
    ├── logs/
    └── route_a_cloud_results.tar.gz
```

脚本会自动导出上述 5 个环境变量；`WORKSPACE_ROOT=/your/path` 可整体改根目录。

## 1. 准备工作（本地 → 云端）

1. 把代码仓库放到 `${WORKSPACE_ROOT}/llm-asr-quant`（Notebook 上传 / `git clone`）。
2. **上传本地 N-best 缓存**（云端不重新抽 dev/test，省 GPU 时间）：
   - 目标路径：`${WORKSPACE_ROOT}/outputs/nbest/`
   - 文件：`dev-clean.jsonl`、`test-clean.jsonl`、`test-other.jsonl`
   - 来源：本地 `outputs/nbest/`（每条约 2.5MB）
3. 可选预检（不执行任何命令，只打印）：

```bash
cd ${WORKSPACE_ROOT}/llm-asr-quant
bash scripts/run_route_a_cloud.sh --stage all --dry-run | less
```

## 2. CPU 实例：下载模型与数据（不耗 GPU 时长）

推荐镜像：`ubuntu22.04-py312-torch2.3.1-1.40.1`。

```bash
cd ${WORKSPACE_ROOT}/llm-asr-quant

# 只补轻依赖 + 检查环境（不会重装 torch）
bash scripts/run_route_a_cloud.sh --stage setup

# 下载 Qwen3-1.7B/4B/8B、wav2vec2-base-100h、LibriSpeech train-clean-100
# 已存在自动跳过；可断点续跑
bash scripts/run_route_a_cloud.sh --stage download
```

- ModelScope 无镜像时脚本自动回退 `HF_ENDPOINT=https://hf-mirror.com`。
- LibriSpeech 优先 `openslr.org`，失败回退 `us.openslr.org`；6.4GB tar 解压后删除。
- 下载完成后生成 `data/manifests/train-clean-100.jsonl`。

> ⚠️ 不要在此实例跑 `nbest`：CPU 镜像 torch 2.3.1，transformers 4.57 会因
> CVE-2025-32434 拒绝加载 `.bin` 权重（wav2vec2 仓库无 safetensors）。
> N-best 抽取必须切到 GPU 镜像（torch 更新）。

## 3. GPU 实例：抽取、训练、评测

推荐镜像：`ubuntu22.04-cuda13.0.3-py312-torch2.13.0-1.40.1`，24G 显存。
`/mnt/workspace` 在同一账号下持久可见。

```bash
cd ${WORKSPACE_ROOT}/llm-asr-quant

# 3.1 环境检查：打印 GPU 型号/显存/CUDA + bnb 4bit 前向 + bf16 状态
bash scripts/run_route_a_cloud.sh --stage setup
```

### 3.1 train-clean-100 Top-5 N-best

```bash
nohup bash scripts/run_route_a_cloud.sh --stage nbest \
  > ${WORKSPACE_ROOT}/outputs/nohup-nbest.log 2>&1 &
tail -f ${WORKSPACE_ROOT}/outputs/nohup-nbest.log
```

- 使用本地 `wav2vec2-base-100h` + `pyctcdecode`，`beam_width=32`、无 KenLM。
- 输出：`outputs/nbest/train-clean-100.jsonl`。
- 若 dev/test 缓存未上传，脚本会打印缺失提示。

### 3.2 QLoRA 微调 + 挂载评测

```bash
nohup bash scripts/run_route_a_cloud.sh --stage qlora \
  > ${WORKSPACE_ROOT}/outputs/nohup-qlora.log 2>&1 &
tail -f ${WORKSPACE_ROOT}/outputs/nohup-qlora.log
```

- `Qwen3-1.7B` NF4 QLoRA，`template=plain`，等效 batch=32（4×8，微批更小以
  降低 A10 上的激活峰值），2 epochs。每 epoch 步数 = `ceil(ceil(n/4)/8)`，
  不足一个等效 batch 的尾部会强制 flush 成一步。
- 自动检测 bf16：支持则 `--bf16`，否则回退 fp16。
- 产物：`outputs/checkpoints/qwen3_1.7b_qlora_100h/`（adapter + summary JSON）。
- 训练完成后自动在 `test-clean` / `test-other` 挂载评测：
  `--quant_mode nf4 --template plain --resume --write_summary --max_vram_gb 22.0`。
  `summary.csv` 中标记为 **`nf4_qlora`**，与零样本 `nf4` 严格隔离。

### 3.3 全组合评测（按优先级）

```bash
nohup bash scripts/run_route_a_cloud.sh --stage sweep \
  > ${WORKSPACE_ROOT}/outputs/nohup-sweep.log 2>&1 &
tail -f ${WORKSPACE_ROOT}/outputs/nohup-sweep.log
```

顺序（任一组合失败不阻断后续）：
1. `Qwen3-1.7B` × `test-other`：fp16 / int8 / nf4
2. `Qwen3-4B` × `test-clean`、`test-other`：fp16 / int8 / nf4
3. `Qwen3-8B` × `test-clean`、`test-other`：fp16 / int8 / nf4

全部使用本地模型目录 + `hf_bnb` + `--template plain --resume --write_summary
--max_vram_gb 22.0`。8B FP16 权重约 16.4GB，24G 实例需显式放开预检阈值（22.0）。

### 3.4 tmux 方式（更推荐，断网不中断）

```bash
tmux new -s routea
# 在会话内执行上面的 bash 命令（不需要 nohup）
# Ctrl-b d 脱离；tmux attach -t routea 恢复
```

脚本自身会把输出 tee 到 `outputs/logs/route-a-<时间>-<stage>.log`。

## 4. 打包下载回本地

```bash
bash scripts/run_route_a_cloud.sh --stage pack
```

产物：`${WORKSPACE_ROOT}/outputs/route_a_cloud_results.tar.gz`，包含：

- `outputs/metrics/summary.csv`
- `outputs/checkpoints/qwen3_1.7b_qlora_100h/`（LoRA adapter + qlora_summary.json）
- 各实验目录下 `predictions.jsonl` / `cases.jsonl`
- `outputs/logs/`

从 Notebook 文件浏览器下载该 tar.gz 即可。

## 5. 断点续跑与实例重启

- **`/mnt/workspace` 持久**：代码、模型、数据、N-best、输出与日志都在里面，
  实例重启/切换 CPU↔GPU 不会丢。
- **进程会死**：重启后重新 `tmux`/`nohup` 执行同一 stage。
- **评测自动续跑**：`infer_ger.py --resume` 读取每个实验目录的
  `checkpoint.jsonl`，只跑未完成样本；统计口径跨段可加。
- **QLoRA 训练不续跑**：脚本当前不做训练 checkpoint；重跑会覆盖 adapter。
  建议训练前确认 GPU 稳定，或用 tmux。
- **系统环境可能重置**：实例重启后重跑 `--stage setup`（只补依赖，不动 torch）。
- **下载中断**：模型目录非空即视为完成；若怀疑损坏，删除对应
  `models/<name>/` 重下。LibriSpeech 下载用 `curl --continue-at -`，可续传。
- **GPU 额度**：CPU 阶段不耗 GPU；GPU 实例用完及时停止/关机。按上述顺序，
  N-best 约 1–3h、QLoRA 约 1–2h、sweep 约 6–10h，总计可控制在 33h 内。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| `--max_vram_gb` 预检失败 | 确认无残留进程；8B FP16 必须传 `22.0`（脚本已内置） |
| bnb 导入报 CUDA 版本不匹配 | 重跑 `--stage setup`；不要手动装固定版本 torch |
| 模板校验报错 | LoRA 训练模板是 `plain`，推理必须 `--template plain`（脚本已传） |
| 缺少 `test-*.jsonl` | 从本地上传至 `${WORKSPACE_ROOT}/outputs/nbest/` |
| `modelscope download` 失败 | 脚本自动回退 HF 镜像；仍失败可手动下载到对应目录 |
| OOM | 先释放显存（重开进程）；8B 只用 fp16/int8/nf4，勿加 batch |
