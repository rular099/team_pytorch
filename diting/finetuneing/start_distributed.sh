#!/bin/bash
#SBATCH -o seist-pretrained-l4500.out
#SBATCH -J downstream
#SBATCH -p normal
#SBATCH -N 4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4

module unload compiler/rocm/2.9
module load compiler/rocm/dtk-23.04.1
module load apps/miniconda/3
source activate wnz_LSD

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


cd /public/home/zhengl/seismogram/wnz/code/LSD-Pretrain-PretextTask/finetuneing ; echo "[INFO] cd to: $(pwd)"

echo -e "\n[INFO] $(date '+%y-%m-%d %H:%M:%S')"
log_dir="/public/home/zhengl/seismogram/wnz/results/finetune/dis/pretrained-l4500_logs"
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
srun python -u main.py \
  --seed 0 \
  --mode "train_test" \
  --model-name "seist_l_dis" \
  --log-dir ${log_dir} \
  --data "/public/home/zhengl/SeisDataSets/LSD/LSD_dis" \
  --dataset-name "lsd" \
  --data-split true \
  --train-size 0.8 \
  --val-size 0.1 \
  --shuffle true \
  --workers 8 \
  --in-samples 6000 \
  --augmentation true \
  --epochs 200 \
  --patience 30 \
  --batch-size 256 \
  --warmup-steps 0.05 \
  --down-steps 0.05 \
  --pretrained "/public/home/zhengl/seismogram/wnz/results/pretrain/len_task/mocov2_2024-2-1-len-4500/checkpoint_pt_0029.pth.tar" >> ${log_dir}/log.txt