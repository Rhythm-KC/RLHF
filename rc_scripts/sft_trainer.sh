#!/bin/bash -l 

#SBATCH --job-name=RKcRHLF_sft_GPT2
#SBATCH --account pop-ml 
#SBATCH --partition=debug

#SBATCH --output=sbatch_log/%x_%j.out		
#SBATCH --error=sbatch_log/%x_%j.err	

#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:p4:2
#SBATCH --mem=100g

#SBATCH --time=0-6:00:00

conda activate finetuning
python ~/RLHF/src/finetuning/SFT_trainer.py \
    --model_name_or_path="gpt2" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --report_to=none \
    --output_dir="~/RLHF/output/sft_gpt2" \
    --logging_steps=1 \
    --max_steps=-1 \
    --num_train_epochs=10 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16