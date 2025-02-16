#!/bin/bash
#SBATCH --job-name=train_tiny_zero    # Name of the job
#SBATCH --output=/home/miw633/TinyZero/scripts/20250216_3_train_tiny_zero.log  # Output log file
#SBATCH --error=/home/miw633/TinyZero/scripts/20250216_3_train_tiny_zero.log   # Error log file
#SBATCH --time=96:00:00        		 # Max run time (hh:mm:ss)
#SBATCH --ntasks=1               	 # Number of tasks
#SBATCH --cpus-per-task=1        	 # Number of CPU cores per task
#SBATCH --mem=192GB           	 # Memory allocation
#SBATCH --partition=gpu_requeue 		 # Queue partition
#SBATCH --gres=gpu:a6000:4


module load gcc/9.2.0
module load cuda/12.1
module load miniconda3/23.1.0
source ~/.bashrc
conda activate "zero" 

echo "Job started at: $(date)"


export CUDA_LAUNCH_BLOCKING=1 # for debugging

N_GPUS=4
BASE_MODEL=Qwen/Qwen2-7B-Instruct
DATA_DIR=/home/miw633/TinyZero/data/countdown
EXPERIMENT_NAME=countdown-Qwen2-7B-Instruct
ROLLOUT_TP_SIZE=2

set -x

VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vTinyZero' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@

echo "Job ended at: $(date)"