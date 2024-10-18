import argparse
import datasets
import gc
import torch
import numpy as np
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel
from pathlib import Path
from datasets import Dataset

from utils import get_model_identifiers_from_yaml
from evaluate_util import eval_qa_vllm

parser = argparse.ArgumentParser(description='evaluate llm by vllm')
parser.add_argument('--curr_save_dir', type=str, default=None)
parser.add_argument('--model_family', type=str, default="llama2-7b")
parser.add_argument('--clean_cache', type=str, default="false")
args = parser.parse_args()

curr_save_dir = args.curr_save_dir
model_cfg = get_model_identifiers_from_yaml(args.model_family)
model_id = model_cfg["model_id"]

#load vllm model
model_eval = LLM(curr_save_dir, tokenizer=model_id, device="auto")
# eval_dataset = datasets.load_from_disk(curr_save_dir+"/eval.hf")
eval_dataset_list = [Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt")), Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt"))]
eval_dataset_name_list = ["relationships_", "biographies_"]

#remove local model
if args.clean_cache == "true":
    import shutil
    shutil.rmtree(curr_save_dir)

Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

for eval_dataset, eval_dataset_name in zip(eval_dataset_list, eval_dataset_name_list):
    with torch.no_grad():
        correct, responses = eval_qa_vllm(eval_dataset, model_eval, qk="question4", ak="answer4", question_start_tag=model_cfg["question_start_tag"], question_end_tag=model_cfg["question_end_tag"], answer_tag=model_cfg["answer_tag"])
        torch.save(correct, f"{curr_save_dir}/{eval_dataset_name}correct.pt")
        torch.save(responses, f"{curr_save_dir}/{eval_dataset_name}responses.pt")
        acc = np.asarray(correct).astype(np.float32).mean()
        print(f"{eval_dataset}accuracy: {acc}")

destroy_model_parallel()
del model_eval
gc.collect()
torch.cuda.empty_cache()