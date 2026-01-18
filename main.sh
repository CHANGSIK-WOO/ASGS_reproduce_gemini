#!/usr/bin/env bash

#SBATCH --job-name=asgs_het_sem_3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=15G
#SBATCH --time=2-0
#SBATCH -o ./logs/%N_%x_%j.out
#SBATCH -e ./logs/%N_%x_%j.err

set -euo pipefail

# 포트 충돌 방지를 위한 랜덤 포트 생성
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 15000-65000 -n 1)
export WORLD_SIZE=2

# 작업 디렉토리 설정 (로그상의 경로 반영)
cd "/data/changsik/SOMA_gemini/" || exit 1

# --- [설정] ---
# 사용할 Config 파일 경로
CONFIG_FILE="configs/soma_aood_city_to_foggy_r50.yaml"
# 결과 저장 경로 (Task 별로 분리)
OUTPUT_DIR="./outputs/table1/het-sem-3"
# 체크포인트 경로
CHECKPOINT="${OUTPUT_DIR}/checkpoint.pth"

# 1. Resume 로직: 체크포인트가 있으면 이어서 학습
if [ -f "$CHECKPOINT" ]; then
    echo "Found checkpoint at $CHECKPOINT, resuming training..."
    # Config 오버라이드 형식으로 RESUME 경로 전달
    RESUME_ARG="RESUME $CHECKPOINT"
else
    echo "No checkpoint found, starting training from scratch..."
    RESUME_ARG=""
fi

echo "========================================================"
echo " Job: ASGS het-sem (Setting 1) / Unknown: 3"
echo " Config: ${CONFIG_FILE}"
echo " Output: ${OUTPUT_DIR}"
echo "========================================================"

# 2. 학습 실행 (torchrun 사용)
# [수정됨] --output_dir 삭제하고 --opts 내부로 이동

torchrun --nproc_per_node=1 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    main.py \
    --config_file ${CONFIG_FILE} \
#    --opts \
#    OUTPUT_DIR ${OUTPUT_DIR} \
#    DATASET.AOOD_SETTING 1 \
#    DATASET.AOOD_TASK 3 \
#    DATASET.NUM_CLASSES 4 \
#    TRAIN.EPOCHS 65 \
#    AOOD.ASGS.ENABLED True \
#    AOOD.MOTIF_ON False \
    ${RESUME_ARG}

echo ""
echo "Training completed!"
echo "To evaluate, check the logs or run evaluation mode."