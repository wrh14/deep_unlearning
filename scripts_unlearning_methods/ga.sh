#!/bin/bash
master_port=18765;devices="0,1"
model=$1
unlearn_data_id=$2
model_path=ft_model_checkpoint/ft_${model}
forget_loss=ga

save_path=unlearning_checkpoint/ga/${model}/${unlearn_data_id}
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_family.yaml model_family=${model} unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path}; 

for cur_save_dir in ${save_path}/*/; do
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir ${cur_save_dir} --model_family $model --clean_cache false; 
    
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
