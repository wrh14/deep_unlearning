import torch
import task_vector
import os
from utils import get_model_identifiers_from_yaml
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run TV unlearn with params.")
    
    parser.add_argument('--alpha_list', type=str, required=False,default=None, help="alphalist")
    parser.add_argument('--ft_dir', type=str, required=True, help="pretrained model directory")
    parser.add_argument('--reinforced_model_dir', type=str, required=True, help="finetuned model directory on the target fact")
    parser.add_argument('--out_dir', type=str, required=True, help="model directory for saving results")
    parser.add_argument('--model_family', type=str, required=True, help="model family")
    args = parser.parse_args()

    some_ft_model_dir = args.ft_dir
    model_dir = args.ft_dir
    some_reinforced_model_dir = args.reinforced_model_dir
    
    model_cfg = get_model_identifiers_from_yaml(args.model_family)
    alphas_str_list = model_cfg["tv_alpha_list"].split(" ")
    alphas = [float(alpha) for alpha in alphas_str_list]

    for alpha in alphas:
        out_dir = args.out_dir + f"/checkpoint-{alpha}"
        task_vector.unlearn(model_dir, out_dir, some_pt_model_dir=some_ft_model_dir,some_ft_model_dir=some_reinforced_model_dir, alpha=alpha)

if __name__ == "__main__":
    main()
