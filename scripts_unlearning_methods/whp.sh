#!/bin/bash
model=$1
unlearn_data_id=$2

master_port=16703
devices="5,6"


ft_dir=ft_model_checkpoint/ft_${model}
reinforced_model_up_dir=checkpoint_whp_tv/reinforced_model/${model}/${unlearn_data_id}
reinforced_model_dir=checkpoint_whp_tv/reinforced_model/${model}/${unlearn_data_id}/checkpoint-10
save_dir=checkpoint_whp_tv/unlearning_checkpoint/whp/${model}/${unlearn_data_id}
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


CUDA_VISIBLE_DEVICES=${devices} python whp_run.py \
    --curr_save_dir_top ${save_dir} \
    --model_family ${model} \
    --reinforced_model_dir ${reinforced_model_dir} \
    --model_dir ${ft_dir} \
    --unlearn_data_id ${unlearn_data_id}
    
declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" ["gpt2-xl"]="openai-community/gpt2-xl" ["phi"]="microsoft/phi-1_5")
model_id="${model_to_modelid[$model]}"
    
for alpha in 0.5 1.0 5.0 10.0 100.0 1000.0; do 
    CUDA_VISIBLE_DEVICES=${devices} lm_eval --model whp_hf \
        --tasks piqa,race,mmlu \
        --model_args parallelize=True,pretrained=${ft_dir},pretrained_reinforce=${reinforced_model_dir},alpha=${alpha},tokenizer=${model_id} \
        --batch_size 4 \
        --output_path ${save_dir}/checkpoint-${alpha}
done