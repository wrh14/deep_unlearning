import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
from pathlib import Path
from utils import get_model_identifiers_from_yaml
import argparse
import gc
import numpy as np
from datasets import Dataset
import argparse
from evaluate_util import eval_qa_whp
from whp import WHPModelForCausalLM


parser = argparse.ArgumentParser(description='evaluate whp')
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
device_map = "auto"

config = AutoConfig.from_pretrained(model_id)
kwargs = {"use_flash_attention_2":model_cfg["flash_attention2"]=="true", "torch_dtype":torch.bfloat16,"trust_remote_code":True, "token":os.environ['HF_TOKEN'], "device_map":device_map}

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

eval_dataset_list = [Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt")), Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt"))]
eval_dataset_name_list = ["relationships_", "biographies_"]

alphas_str_list = model_cfg["whp_alpha_list"].split(" ")
alphas = [float(alpha) for alpha in alphas_str_list]
for eval_dataset, eval_dataset_name in zip(eval_dataset_list, eval_dataset_name_list):
    with torch.no_grad():
        print('Starting Dataset:', eval_dataset_name)
        
        for alpha in alphas:
            curr_save_dir = curr_save_dir_top+f'/checkpoint-{alpha}'
            if os.path.exists(f"{curr_save_dir}//{eval_dataset_name}correct.pt"):
                correct_rephrase = torch.load(f"{curr_save_dir}//{eval_dataset_name}correct.pt")
                acc = np.asarray(correct_rephrase).astype(np.float32).mean()
                print(f"Load from checkpoint. Accuracy: {acc}")
                continue
            whp_model = WHPModelForCausalLM(model_dir, reinforced_model_dir, alpha=alpha, config=config, **kwargs)
            whp_model.eval()
            correct_rephrase, responses_rephrase = eval_qa_whp(eval_dataset, whp_model, tokenizer, max_new_tokens=max_new_tokens, qk="question4", ak="answer4", question_start_tag=model_cfg["question_start_tag"], question_end_tag=model_cfg["question_end_tag"], answer_tag=model_cfg["answer_tag"])
            
            print('Running for alpha:', alpha)
            Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(correct_rephrase, f"{curr_save_dir}//{eval_dataset_name}correct.pt")
            torch.save(responses_rephrase, f"{curr_save_dir}//{eval_dataset_name}responses.pt")
            acc = np.asarray(correct_rephrase).astype(np.float32).mean()
            print(f"Accuracy: {acc}")

        gc.collect()
        torch.cuda.empty_cache()


destroy_model_parallel()
del whp_model
gc.collect()
torch.cuda.empty_cache()