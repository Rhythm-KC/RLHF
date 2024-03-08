#!/bin/bash -l 

#SBATCH --job-name=RKcRHLF-GPT2_Reward_model
#SBATCH --account pop-ml 
#SBATCH --partition=tier3

#SBATCH --output=sbatch_log/%x_%j.out		
#SBATCH --error=sbatch_log/%x_%j.err	

#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=320g

#SBATCH --time=1-6:00:00

conda activate finetuning
python ~/RLHF/src/finetuning/reward_model.py \
    --model_name_or_path="gpt2"\
    --output_dir="~/RLHF/output/reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=32 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=16 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to=none \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=1024 