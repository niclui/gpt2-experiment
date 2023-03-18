#!/bin/bash

# if [ $# -lt 2 ] means if the number of positional parameters set is less than 2
# if [[ $# -lt 2 ]]; then
#     echo "Main entry point for training."
#     echo "Usage:"
#     echo
#     echo "    $0 <dataset_name (e.g., openwebtext_wordlength)> <seed (e.g., 1)> [additional arguments]"
#     echo
#     echo "Additional arguments (see https://github.com/huggingface/transformers/blob/v4.12.5/src/transformers/training_args.py for the full list):"
#     echo ""
#     echo "    --max_steps: Total number of training steps to run."
#     echo "    --learning_rate: Highest learning rate in the learning rate schedule (by default, the learning rate schedule increases and decreases linearly, with the peak being this set learning rate)"
#     echo "    --warmup_steps: Number of steps to do a linear “warmup” from a small learning rate to the desired learning rate"
#     echo "    --save_steps: how many steps of training before saving (and evaluating on the val set)"

#     echo "    --per_device_train_batch_size: batch size per GPU. The total batch size is [number of GPUs] * per_device_train_batch_size"
#     echo "    --gradient_accumulation_steps: Allows for accumulating gradients across multiple sequential steps. This allows you to increase the batch size while trading off computation time. If this parameter is > 1, then the total batch size is [num GPUs] * per_device_train_batch_size * gradient_accumulation_steps"
#     echo "    --lr_scheduler_type: learning rate scheduler. The default is “linear” which does a linear increase and decrease to and from the learning_rate."
#     echo "    --adafactor: use this flag if you want to use the Adafactor optimizer instead of AdamW (default). Adafactor can save on GPU memory by only saving 2 copies of the model instead of 3 needed for AdamW (mean, variance, gradient)"
#     echo "    --from_scratch: use this flag if you want to scratch"
#     echo "    --fp16: use 16-bit floating point (to save memory)"
#     exit 1
# fi

set -x

CACHE=./cache
mkdir -p $CACHE

export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=100000000000
export TORCH_EXTENSIONS_DIR=$CACHE
export WANDB_DISABLED=true

set -x

# Dataset
dataset_name=/datadrive/openwebtext_wordlength # dataset
n_gpu=4 # number of GPUs

# Model training parameters
seed=123 # random seed
model_name_or_path=gpt2
per_device_train_batch_size=4
gradient_accumulation_steps=2
max_steps=50000
learning_rate=1e-6
warmup_steps=0
save_steps=2000

python -m torch.distributed.launch --nproc_per_node $n_gpu src/run_clm.py \
    --model_name_or_path $model_name_or_path \
    --block_size 1024 \
    --do_train \
    --do_eval \
    --logging_steps 2500 \
    --evaluation_strategy steps \
    --max_eval_samples 2 \
    --preprocessing_num_workers 8 \
    --tokenized_data_dir ${dataset_name}/tokenized_grouped \
    --output_dir ${dataset_name}_seed${seed} \
    --save_steps $save_steps \
    --lr_scheduler_type linear \
    --seed $seed \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_steps $max_steps \
    --learning_rate $learning_rate \
    --warmup_steps $warmup_steps \
    --save_steps $save_steps \
    --dataset_name $dataset_name \
    --seed $seed \
    --overwrite_output_dir \
    ${rest_args}
