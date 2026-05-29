# Change to the directory of the script
cd /data/zhenyangfan/EVAC
export OMP_NUM_THREADS=4

# Timestamped run name and per-run log directory
NAME_BASE="evac_robotwin_finetune"
TIMESTAMP=$(date +%Y-%m-%dT%H-%M-%S)
RUN_NAME="${NAME_BASE}_${TIMESTAMP}"
LOG_DIR="logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/training.log"

# change
GPU_NUMS=4

# Use torchrun to properly launch 8 distributed processes
torchrun --standalone --nproc_per_node=${GPU_NUMS} \
    trainer/trainer.py \
    --base /data/zhenyangfan/EVAC/configs/robotwin/train_config.yaml \
    --train \
    --devices ${GPU_NUMS} \
    --name "${RUN_NAME}" \
    lightning.trainer.num_nodes=1 2>&1 | tee "${LOG_FILE}"