#!/usr/bin/env bash
# =============================================================================
# Route A 云端一键流水线（目标：魔搭 ModelScope 免费 24G GPU / CPU 实例）
#
# 设计原则：CPU 阶段与 GPU 阶段严格拆分，节省 GPU 时长：
#   download  → 推荐在【免费 CPU 实例】跑（下模型/数据，不耗 GPU）
#   nbest / qlora / sweep → 【GPU 实例】
#   setup     → 两者通用（只补轻依赖，绝不重装 torch）
#   pack      → 两者通用（打包含评测结果与 LoRA 权重）
#
# 所有持久化产物统一在 ${WORKSPACE_ROOT}（默认 /mnt/workspace）下：
#   ${HF_HOME} / ${MODELSCOPE_CACHE} / ${DATA_ROOT} / ${MODELS_ROOT} / ${OUTPUT_ROOT}
#
# 用法：
#   bash scripts/run_route_a_cloud.sh --stage download          # CPU 实例
#   bash scripts/run_route_a_cloud.sh --stage nbest             # GPU 实例
#   bash scripts/run_route_a_cloud.sh --stage qlora
#   bash scripts/run_route_a_cloud.sh --stage sweep
#   bash scripts/run_route_a_cloud.sh --stage pack
#   bash scripts/run_route_a_cloud.sh --stage all --dry-run     # 只打印不执行
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
STAGE="all"
DRY_RUN=false
VALID_STAGES="setup download nbest qlora sweep pack all"

usage() {
    cat <<'EOF'
用法: bash scripts/run_route_a_cloud.sh [--stage <stage>] [--dry-run]

  --stage   setup | download | nbest | qlora | sweep | pack | all（默认 all）
  --dry-run 仅打印完整命令，不执行（只读 import 检测仍会运行）

环境变量:
  WORKSPACE_ROOT  持久化根目录（默认 /mnt/workspace）
  REPO_ROOT       代码仓库目录（默认脚本上一级）

推荐流程: CPU 实例跑 --stage setup/download → 切 GPU 实例跑 nbest/qlora/sweep
          → --stage pack 后下载 tar.gz。详见 docs/CLOUD_RUNBOOK.md
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            [[ $# -ge 2 ]] || { echo "错误: --stage 需要参数" >&2; exit 2; }
            STAGE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "错误: 未知参数 $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! grep -qw "$STAGE" <<<"$VALID_STAGES"; then
    echo "错误: 未知 stage '$STAGE'（可选: $VALID_STAGES）" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 路径与持久化环境（全部位于 WORKSPACE_ROOT 下）
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace}"

export HF_HOME="${WORKSPACE_ROOT}/hf_home"
export MODELSCOPE_CACHE="${WORKSPACE_ROOT}/modelscope_cache"
export DATA_ROOT="${WORKSPACE_ROOT}/data"
export MODELS_ROOT="${WORKSPACE_ROOT}/models"
export OUTPUT_ROOT="${WORKSPACE_ROOT}/outputs"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

QWEN_17="${MODELS_ROOT}/Qwen3-1.7B"
QWEN_4B="${MODELS_ROOT}/Qwen3-4B"
QWEN_8B="${MODELS_ROOT}/Qwen3-8B"
ASR_MODEL="${MODELS_ROOT}/wav2vec2-base-100h"

NBEST_DIR="${OUTPUT_ROOT}/nbest"
MANIFEST_TRAIN="${DATA_ROOT}/manifests/train-clean-100.jsonl"
CKPT_DIR="${OUTPUT_ROOT}/checkpoints/qwen3_1.7b_qlora_100h"
LOG_DIR="${OUTPUT_ROOT}/logs"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
run() {
    if $DRY_RUN; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# 依赖文件检查：dry-run 下只告警（本地尚无云端产物），真实运行缺文件则中止
require_file() {
    local path="$1"
    [[ -f "$path" ]] && return 0
    if $DRY_RUN; then
        echo "[dry-run] 注意：云端应存在但本地缺失 -> $path"
    else
        echo "错误: 缺少文件 $path（请先完成前置 stage 或从本地上传）" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    [[ -d "$path" ]] && return 0
    if $DRY_RUN; then
        echo "[dry-run] 注意：云端应存在但本地缺失 -> $path"
    else
        echo "错误: 缺少目录 $path（请先完成 download）" >&2
        exit 1
    fi
}

# 日志：真实运行时全部 stdout/stderr tee 到 OUTPUT_ROOT/logs（dry-run 不落盘）
if ! $DRY_RUN; then
    mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$DATA_ROOT" "$MODELS_ROOT" "$NBEST_DIR" \
             "$HF_HOME" "$MODELSCOPE_CACHE"
    LOG_FILE="${LOG_DIR}/route-a-$(date +%Y%m%d-%H%M%S)-${STAGE}.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "日志: $LOG_FILE"
fi

BF16_ARG=()
detect_bf16() {
    if python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)' 2>/dev/null; then
        BF16_ARG=(--bf16)
        echo "bf16: 可用（训练使用 --bf16）"
    else
        BF16_ARG=()
        echo "bf16: 不可用（训练回退 fp16）"
    fi
}

# ---------------------------------------------------------------------------
# stage: setup —— 只补轻依赖 + GPU/bnb/bf16 兼容性检查（不重装 torch）
# ---------------------------------------------------------------------------
stage_setup() {
    echo "== [setup] 环境检查与缺失依赖安装 =="
    run python -c "import sys,torch; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"

    # 版本敏感项：transformers 需 >=4.51（Qwen3），peft/bitsandbytes 需满足 QLoRA
    if ! python -c "import transformers,packaging.version as v; assert v.parse(transformers.__version__) >= v.parse('4.51')" >/dev/null 2>&1; then
        run python -m pip install -q "transformers>=4.51,<5"
    fi
    if ! python -c "import peft,packaging.version as v; assert v.parse(peft.__version__) >= v.parse('0.11')" >/dev/null 2>&1; then
        run python -m pip install -q "peft>=0.11"
    fi
    if ! python -c "import bitsandbytes,packaging.version as v; assert v.parse(bitsandbytes.__version__) >= v.parse('0.43')" >/dev/null 2>&1; then
        run python -m pip install -q "bitsandbytes>=0.43"
    fi

    # 纯 import 检查的轻依赖
    local -A PKGS=(
        [pyctcdecode]=pyctcdecode
        [jiwer]=jiwer
        [whisper_normalizer]=whisper-normalizer
        [soundfile]=soundfile
        [datasets]=datasets
        [tqdm]=tqdm
        [yaml]=PyYAML
        [modelscope]=modelscope
        [huggingface_hub]=huggingface_hub
    )
    local mod pkg
    for mod in "${!PKGS[@]}"; do
        pkg="${PKGS[$mod]}"
        if ! python -c "import $mod" >/dev/null 2>&1; then
            run python -m pip install -q "$pkg"
        fi
    done

    # GPU 兼容性：打印型号/显存 + 真正跑一次 16x16 NF4 前向验证 bnb kernel
    read -r -d '' GPU_CHECK <<'PY' || true
import torch
if not torch.cuda.is_available():
    print("CPU 环境：跳过 GPU / bitsandbytes 检查（推荐在此阶段跑 download）")
    raise SystemExit(0)
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | VRAM {p.total_memory / 1024**3:.1f} GiB | CUDA {torch.version.cuda}")
print("bf16 supported:", torch.cuda.is_bf16_supported())
import bitsandbytes as bnb
compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
layer = bnb.nn.Linear4bit(16, 16, bias=False, quant_type="nf4", compute_dtype=compute).cuda()
x = torch.randn(1, 16, device="cuda", dtype=compute)
y = layer(x)
print("bitsandbytes 4bit forward OK:", tuple(y.shape), y.dtype)
PY
    run python -c "$GPU_CHECK"
}

# ---------------------------------------------------------------------------
# stage: download —— 模型 + LibriSpeech train-clean-100（推荐 CPU 实例）
# ---------------------------------------------------------------------------
download_model() {
    local ms_id="$1" dest="$2"
    if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null || true)" ]]; then
        echo "已存在，跳过: $dest"
        return 0
    fi
    run mkdir -p "$dest"
    echo "下载模型 $ms_id -> $dest"
    if ! run modelscope download --model "$ms_id" --local_dir "$dest"; then
        echo "ModelScope 无镜像/下载失败，回退 HF 镜像: $ms_id"
        run env HF_ENDPOINT=https://hf-mirror.com python -c \
            "from huggingface_hub import snapshot_download; snapshot_download('$ms_id', local_dir='$dest')"
    elif $DRY_RUN; then
        echo "[dry-run] 若 ModelScope 无镜像，将回退: HF_ENDPOINT=https://hf-mirror.com snapshot_download('$ms_id', local_dir='$dest')"
    fi
}

stage_download() {
    echo "== [download] 模型与数据下载（推荐在免费 CPU 实例运行，不耗 GPU 时长）=="
    # download 是唯一联网阶段：清除可能从 shell 继承的离线开关
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    download_model "Qwen/Qwen3-1.7B" "$QWEN_17"
    download_model "Qwen/Qwen3-4B" "$QWEN_4B"
    download_model "Qwen/Qwen3-8B" "$QWEN_8B"
    download_model "facebook/wav2vec2-base-100h" "$ASR_MODEL"

    # LibriSpeech train-clean-100（~6.4GB），已解压则跳过
    local split="train-clean-100"
    local split_dir="${DATA_ROOT}/LibriSpeech/${split}"
    if [[ -d "$split_dir" ]] && [[ -n "$(ls -A "$split_dir" 2>/dev/null || true)" ]]; then
        echo "已存在，跳过下载: $split_dir"
    elif $DRY_RUN; then
        echo "[dry-run] curl -L --fail --retry 3 -o ${DATA_ROOT}/${split}.tar.gz https://www.openslr.org/resources/12/${split}.tar.gz"
        echo "[dry-run] （失败回退 https://us.openslr.org/resources/12/${split}.tar.gz）"
        echo "[dry-run] tar -xzf ${DATA_ROOT}/${split}.tar.gz -C ${DATA_ROOT} && rm -f ${DATA_ROOT}/${split}.tar.gz"
    else
        mkdir -p "$DATA_ROOT"
        local tar="${DATA_ROOT}/${split}.tar.gz"
        local ok=0 base
        for base in https://www.openslr.org/resources/12 https://us.openslr.org/resources/12; do
            echo "下载 $base/$split.tar.gz ..."
            if curl -L --fail --retry 3 --continue-at - -o "$tar" "$base/$split.tar.gz"; then
                ok=1; break
            fi
        done
        [[ "$ok" -eq 1 ]] || { echo "错误: LibriSpeech 下载失败，请手动放置 $tar 后重跑" >&2; exit 1; }
        tar -xzf "$tar" -C "$DATA_ROOT"
        rm -f "$tar"
    fi

    # 生成 train-clean-100 清单（幂等，覆盖写）
    run python data/prepare_librispeech.py \
        --librispeech_root "$DATA_ROOT" \
        --splits "$split" \
        --output_dir "${DATA_ROOT}/manifests"
}

# ---------------------------------------------------------------------------
# stage: nbest —— train-clean-100 Top-5（GPU）
# ---------------------------------------------------------------------------
stage_nbest() {
    echo "== [nbest] train-clean-100 Top-5 N-best（wav2vec2 + pyctcdecode, beam=32, 无 KenLM）=="
    # 全部使用本地模型目录：强制离线，避免 transformers 4.57 发 HEAD 请求假死
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    require_file "$MANIFEST_TRAIN"
    require_dir "$ASR_MODEL"
    run python route_a_ger/extract_nbest.py \
        --split train-clean-100 \
        --manifest "$MANIFEST_TRAIN" \
        --output_dir "$NBEST_DIR" \
        --model "$ASR_MODEL" \
        --n_best 5 \
        --beam_width 32

    local split
    for split in dev-clean test-clean test-other; do
        if [[ ! -f "${NBEST_DIR}/${split}.jsonl" ]]; then
            echo "提示: 缺少 ${NBEST_DIR}/${split}.jsonl —— 请从本地上传该 N-best 缓存（infer_ger 评测依赖）"
        fi
    done
}

# ---------------------------------------------------------------------------
# stage: qlora —— 1.7B QLoRA 微调 + test-clean/test-other 挂载评测（GPU）
# ---------------------------------------------------------------------------
stage_qlora() {
    echo "== [qlora] Qwen3-1.7B QLoRA（等效 batch=32, 2 epochs, template=plain）=="
    # 本地模型目录：强制离线，避免 transformers 4.57 发 HEAD 请求假死
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    require_file "${NBEST_DIR}/train-clean-100.jsonl"
    require_dir "$QWEN_17"
    detect_bf16

    run python route_a_ger/train_qlora.py \
        --nbest "${NBEST_DIR}/train-clean-100.jsonl" \
        --model "$QWEN_17" \
        --template plain \
        --epochs 2 \
        --batch_size 8 \
        --grad_accum 4 \
        --max_seq_len 512 \
        --lr 2e-4 \
        --output_dir "$CKPT_DIR" \
        "${BF16_ARG[@]}"

    local split
    for split in test-clean test-other; do
        require_file "${NBEST_DIR}/${split}.jsonl"
        echo "---- LoRA 挂载评测: $split ----"
        run python route_a_ger/infer_ger.py \
            --nbest "${NBEST_DIR}/${split}.jsonl" \
            --model "$QWEN_17" \
            --quant_mode nf4 \
            --lora_path "$CKPT_DIR" \
            --template plain \
            --resume \
            --write_summary \
            --max_vram_gb 22.0 \
            --out_dir "$OUTPUT_ROOT"
    done
}

# ---------------------------------------------------------------------------
# stage: sweep —— 1.7B/4B/8B × 精度 × split（GPU，按优先级）
# ---------------------------------------------------------------------------
SWEEP_FAIL=0
sweep_one() {
    local model_dir="$1" split="$2" quant="$3"
    local nbest="${NBEST_DIR}/${split}.jsonl"
    require_file "$nbest"
    echo "---- $(basename "$model_dir") | $split | $quant ----"
    if ! run python route_a_ger/infer_ger.py \
        --nbest "$nbest" \
        --model "$model_dir" \
        --backend hf_bnb \
        --quant_mode "$quant" \
        --template plain \
        --resume \
        --write_summary \
        --max_vram_gb 22.0 \
        --out_dir "$OUTPUT_ROOT"; then
        echo "!! 失败: $(basename "$model_dir") $split $quant（继续后续组合，可用 --resume 重跑）"
        SWEEP_FAIL=$((SWEEP_FAIL + 1))
    fi
}

stage_sweep() {
    echo "== [sweep] 24G 显存全组合评测（本地模型目录 + hf_bnb）=="
    # 本地模型目录：强制离线，避免 transformers 4.57 发 HEAD 请求假死
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    require_dir "$QWEN_17"
    require_dir "$QWEN_4B"
    require_dir "$QWEN_8B"

    # 优先级 1：1.7B × test-other
    local q
    for q in fp16 int8 nf4; do
        sweep_one "$QWEN_17" test-other "$q"
    done
    # 优先级 2：4B × test-clean/test-other
    local split
    for split in test-clean test-other; do
        for q in fp16 int8 nf4; do
            sweep_one "$QWEN_4B" "$split" "$q"
        done
    done
    # 优先级 3：8B × test-clean/test-other
    for split in test-clean test-other; do
        for q in fp16 int8 nf4; do
            sweep_one "$QWEN_8B" "$split" "$q"
        done
    done

    if [[ "$SWEEP_FAIL" -gt 0 ]]; then
        echo "== sweep 完成，失败 $SWEEP_FAIL 个组合（详见日志；重跑同一命令会 --resume 跳过已完成样本）=="
    else
        echo "== sweep 全部完成 =="
    fi
}

# ---------------------------------------------------------------------------
# stage: pack —— 汇总关键产物为 tar.gz
# ---------------------------------------------------------------------------
stage_pack() {
    echo "== [pack] 打包 summary.csv / LoRA adapter / 评测 JSONL / 日志 =="
    local archive="${OUTPUT_ROOT}/route_a_cloud_results.tar.gz"
    if $DRY_RUN; then
        echo "[dry-run] find ${OUTPUT_ROOT} -maxdepth 2 \\( -name predictions.jsonl -o -name cases.jsonl \\) -print0 > <manifest>"
        echo "[dry-run] tar -czf ${archive} -C ${WORKSPACE_ROOT} --null -T <manifest>"
        return 0
    fi

    local manifest
    manifest="$(mktemp)"
    {
        [[ -f "${OUTPUT_ROOT}/metrics/summary.csv" ]] && \
            printf '%s\0' "outputs/metrics/summary.csv"
        [[ -d "${OUTPUT_ROOT}/checkpoints" ]] && \
            printf '%s\0' "outputs/checkpoints"
        [[ -d "$LOG_DIR" ]] && printf '%s\0' "outputs/logs"
        ( cd "$WORKSPACE_ROOT" && find outputs -maxdepth 2 -type f \
            \( -name 'predictions.jsonl' -o -name 'cases.jsonl' \) -print0 )
    } > "$manifest"

    if [[ ! -s "$manifest" ]]; then
        echo "错误: 没有可打包的产物（请先跑评测/训练）" >&2
        rm -f "$manifest"
        exit 1
    fi

    # -C 必须先于 -T：GNU tar 的 -C 只影响其后的文件参数与 -T 清单
    tar -czf "$archive" -C "$WORKSPACE_ROOT" --null -T "$manifest"
    rm -f "$manifest"
    echo "已打包: $archive ($(du -h "$archive" | cut -f1))"
}

# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------
echo "stage=$STAGE | dry_run=$DRY_RUN | repo=$REPO_ROOT | workspace=$WORKSPACE_ROOT"
case "$STAGE" in
    setup)    stage_setup ;;
    download) stage_download ;;
    nbest)    stage_nbest ;;
    qlora)    stage_qlora ;;
    sweep)    stage_sweep ;;
    pack)     stage_pack ;;
    all)
        echo "注意: all 会在当前实例串行执行；GPU 实例跑 all 会浪费额度下载。"
        echo "      推荐 CPU 实例: --stage setup + download；GPU 实例: nbest/qlora/sweep。"
        stage_setup
        stage_download
        stage_nbest
        stage_qlora
        stage_sweep
        stage_pack
        ;;
esac
echo "stage '$STAGE' 完成。"
