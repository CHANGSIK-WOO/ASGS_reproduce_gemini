#!/usr/bin/env bash

#SBATCH --job-name=asgs_eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w vgi1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=20G
#SBATCH --time=2-0
#SBATCH -o ./logs/eval_%j.out
#SBATCH -e ./logs/eval_%j.err

set -euo pipefail

# 포트 설정 (충돌 방지)
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 15000-65000 -n 1)

# [설정] 평가할 모델 경로와 Config (사용자 환경에 맞게 수정 필요)
# 예: 학습된 best checkpoint 경로
CHECKPOINT="/data/changsik/SOMA_gemini/outputs/table1/het-sem-3/checkpoint.pth"
CONFIG_FILE="configs/soma_aood_city_to_foggy_r50.yaml"
OUTPUT_DIR="./outputs/table1/het-sem-3/eval_result"

echo "========================================================"
echo " Job: ASGS Evaluation"
echo " Checkpoint: ${CHECKPOINT}"
echo "========================================================"

# main_multi_eval.py를 사용하여 평가 실행
torchrun --nproc_per_node=1 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    main_multi_eval.py \
    --config_file ${CONFIG_FILE} \
    --opts \
    OUTPUT_DIR ${OUTPUT_DIR} \
    DATASET.AOOD_SETTING 1 \
    DATASET.AOOD_TASK 3 \
    DATASET.NUM_CLASSES 4 \
    EVAL True \
    RESUME ${CHECKPOINT}