#!/bin/bash
model=$1
unlearn_data_id=$2

master_port=16702

max_new_tokens=10 #max new tokens to be generated

ft_dir=ft_model_checkpoint/ft_${model}
reinforced_model_dir=/data/ruihan/llm_unlearning/checkpoint_whp_tv/reinforced_model/${model}/${unlearn_data_id}/checkpoint-10
save_dir=/data/ruihan/llm_unlearning/checkpoint_whp_tv/unlearning_checkpoint/tv/${model}/${unlearn_data_id}
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

CUDA_VISIBLE_DEVICES=${cuda_devices} python tv_run.py --unlearn_data_id=${unlearn_data_id} --reinforced_model_dir=${reinforced_model_dir} --model_family=${model} --ft_dir=${ft_dir} --out_dir=${save_dir}

for cur_save_dir in ${save_dir}/*/; do
    CUDA_VISIBLE_DEVICES=${cuda_devices} python vllm_eval.py --curr_save_dir $cur_save_dir --model_family $model --clean_cache true; 
done
