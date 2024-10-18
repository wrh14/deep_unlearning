import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
from pathlib import Path
from utils import get_model_identifiers_from_yaml
import argparse
import gc
import numpy as np
from vllm.distributed.parallel_state import destroy_model_parallel
from datasets import Dataset
import argparse
from evaluate_util import eval_qa_vllm_whp


parser = argparse.ArgumentParser(description='evaluate whp by vllm')
parser.add_argument('--curr_save_dir_top', type=str, default=None, help="directory to save results")
parser.add_argument('--model_dir', type=str, default=None, help="pretrained model directory")
parser.add_argument('--reinforced_model_dir', type=str, default=None, help="finetuned model directory on the target fact")
parser.add_argument('--unlearn_data_id', type=int, default=None, help="id of the fact to unlearn")
parser.add_argument('--model_family', type=str, default=None, help="model family")
parser.add_argument('--max_new_tokens', type=int, default=10, help="max new tokens to be generated")

args = parser.parse_args()

torch.cuda.empty_cache()

model_family = args.model_family
model_dir= args.model_dir
reinforced_model_dir = args.reinforced_model_dir
unlearn_data_id = args.unlearn_data_id
curr_save_dir_top = args.curr_save_dir_top
max_new_tokens = args.max_new_tokens

model_cfg = get_model_identifiers_from_yaml(model_family)
model_id = model_cfg['model_id']

config = AutoConfig.from_pretrained(model_id)
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side="left")

model_eval1 = AutoModelForCausalLM.from_pretrained(model_dir, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16,trust_remote_code = True, token=os.environ['HF_TOKEN'], device_map=device_map)
model_eval1.eval()

model_eval2 = AutoModelForCausalLM.from_pretrained(reinforced_model_dir, config=config,use_flash_attention_2=model_cfg["flash_attention2"]=="true",  torch_dtype=torch.bfloat16,trust_remote_code = True, token=os.environ['HF_TOKEN'], device_map=device_map)
model_eval2.eval()

tokenizer.pad_token = tokenizer.eos_token

eval_dataset_list = [Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt")), Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt"))]
eval_dataset_name_list = ["relationships_", "biographies_"]

alphas_str_list = model_cfg["whp_alpha_list"].split(" ")
alphas = [float(alpha) for alpha in alphas_str_list]
# for alpha in alphas:
for eval_dataset, eval_dataset_name in zip(eval_dataset_list, eval_dataset_name_list):
    with torch.no_grad():
        print('Starting Dataset:', eval_dataset_name)
        
        correct_rephrase_list, responses_rephrase_list = eval_qa_vllm_whp(eval_dataset, model_eval1,model_eval2, tokenizer, alphas, max_new_tokens=max_new_tokens, qk="question4", ak="answer4", question_start_tag=model_cfg["question_start_tag"], question_end_tag=model_cfg["question_end_tag"], answer_tag="")

        for alpha, correct_rephrase, responses_rephrase in zip(alphas, correct_rephrase_list, responses_rephrase_list):
            print('Running for alpha:', alpha)
            curr_save_dir = curr_save_dir_top+f'/checkpoint-{alpha}'
            Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(correct_rephrase, f"{curr_save_dir}//{eval_dataset_name}correct.pt")
            torch.save(responses_rephrase, f"{curr_save_dir}//{eval_dataset_name}responses.pt")
            acc = np.asarray(correct_rephrase).astype(np.float32).mean()
            print(f"Accuracy: {acc}")

        gc.collect()
        torch.cuda.empty_cache()


destroy_model_parallel()
del model_eval1
del model_eval2
gc.collect()
torch.cuda.empty_cache()