#!/bin/bash

model=$1
unlearn_data_id=$2

master_port=16703

max_new_tokens=10 #max new tokens to be generated

ft_dir=ft_model_checkpoint/ft_${model}
reinforced_model_dir=checkpoint_whp_tv/reinforced_model/${model}/${unlearn_data_id}/checkpoint-10
save_dir=checkpoint_whp_tv/unlearning_checkpoint/whp/${model}/${unlearn_data_id}
mkdir -p $save_dir

cuda_devices="0,1"
CUDA_VISIBLE_DEVICES=${cuda_devices} torchrun \
        --nproc_per_node=2 \
        --master_port=$master_port \
        finetune_reinforced_model.py \
        --config-name=finetune_reinforced_model_per_fact.yaml \
        model_family=${model} \
        unlearn_data_id=${unlearn_data_id} \
        model_path=${ft_dir}

cuda_devices="0,1,2,3"
CUDA_VISIBLE_DEVICES=${cuda_devices} python whp.py \
    --curr_save_dir_top ${save_dir} \
    --model_family ${model} \
    --reinforced_model_dir ${reinforced_model_dir} \
    --model_dir ${ft_dir} \
    --unlearn_data_id ${unlearn_data_id} \
    --max_new_tokens $max_new_tokens
