#need model
master_port=16704;devices="0,1"
model=$1
unlearn_data_id=$2
model_path=ft_model_checkpoint/ft_${model}
forget_loss=npo

save_path=unlearning_checkpoint/${forget_los}/${model}/${unlearn_data_id}
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_family.yaml model_family=${model} unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path}; 
CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_family.yaml model_family=${model} unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path}; 
for cur_save_dir in ${save_path}/*/; do
    CUDA_VISIBLE_DEVICES=${devices} python vllm_eval.py --curr_save_dir $cur_save_dir --model_family $model --clean_cache true; 
done;
