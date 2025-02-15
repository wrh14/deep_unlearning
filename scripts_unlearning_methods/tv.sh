#!/bin/bash
model=$1
unlearn_data_id=$2

master_port=16702
devices="2,3"

ft_dir=ft_model_checkpoint/ft_${model}
reinforced_model_up_dir=checkpoint_whp_tv/reinforced_model/${model}/${unlearn_data_id}
reinforced_model_dir=checkpoint_whp_tv/reinforced_model/${model}/${unlearn_data_id}/checkpoint-10
save_dir=checkpoint_whp_tv/unlearning_checkpoint/tv/${model}/${unlearn_data_id}
mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=${devices} torchrun \
        --nproc_per_node=2 \
        --master_port=$master_port \
        finetune_reinforced_model.py \
        --config-name=finetune_reinforced_model_per_fact.yaml \
        model_family=${model} \
        unlearn_data_id=${unlearn_data_id} \
        model_path=${ft_dir} \
        save_dir=${reinforced_model_up_dir}

CUDA_VISIBLE_DEVICES=${devices} python tv_run.py \
    --reinforced_model_dir=${reinforced_model_dir} \
    --model_family=${model} \
    --ft_dir=${ft_dir} \
    --out_dir=${save_dir}

for cur_save_dir in ${save_dir}/*/; do
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir $cur_save_dir --model_family $model --clean_cache false; 
    
    declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" ["gpt2-xl"]="openai-community/gpt2-xl" ["phi"]="microsoft/phi-1_5")
    model_id="${model_to_modelid[$model]}"
    CUDA_VISIBLE_DEVICES=${devices} lm_eval --model vllm \
        --model_args pretrained=${cur_save_dir},tokenizer=${model_id},tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
        --tasks piqa,race,mmlu \
        --batch_size auto \
        --output_path ${cur_save_dir}
    rm ${cur_save_dir}/*.safetensors
    rm ${cur_save_dir}/*.json
    rm ${cur_save_dir}/*.bin
done
