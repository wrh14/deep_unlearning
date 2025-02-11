# Evaluating Deep Unlearning in Large Language Model

This is the official github page for the paper [*Evaluating Deep Unlearning in Large Language Model*](https://arxiv.org/abs/2410.15153). This repository provides the code for using the benchmark EDU-RELAT, the evaluation code given any unlearning results, and the scripts of running four unlearning methods presented in the paper.

## Preparation
1. Install the environment
```
conda env create -f environment.yml
conda activate unlearning
pip install flash-attn --no-build-isolation
```
2. Install `lm-evaluation-harness` for general benchmark evaluation
```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
mv ../whp_huggingface.py lm_eval/models/
```

3. Download the model checkpoints from this [link](https://drive.google.com/drive/folders/1jZpmHHphXihdXvyD0xAhr3wjtO9qvJy-?usp=sharing) that is finetuned on our synthetic data `EDU-RELAT`. The layout would be
```
deep_unlearning/
    ft_model_checkpoint/
        ft_gpt2-xl
        ft_llama2-7b
        ft_llama3-8b
        ft_phi
```
3. Set up your huggingface key `HF_TOKEN` in the os environment.

## Load Synthetic Dataset EDU-RELAT
Here we provide the code to load the knowledge base and rule sets
1. Load the QA-form knowledge base as huggingfact datasets
```
from datasets import Dataset
dataset_relationships = Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt")) #load the facts in family relationships
dataset_biographies = Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt")) #load the facts in biographies
```
The question and answer are with the key `question4` and `answer4`.

2. Load the rule set
```
from calculate_recall_and_acc import Rule
rule_list = torch.load("synthetic_data/family_rule.pt")
```

3. Load the relationships as a list of tuple
```
from calculate_recall_and_acc import Person
(edge_list,relation_list, _, _) = torch.load("synthetic_data/family-200-graph.pt") #edge_list is a list of pairs of two people; relation_list is a list of relationthips in string, e.g. child.
```

## Evaluate Deep Unlearning on EDU-RELAT
We select a random subset of size 55 from the facts in family relationship to evaluate the deep unlearning. Given any `unlearn_target_data_id` in 0-54, the id of fact in relationships is
```
shuffled_edge_id_list = torch.load("synthetic_data/subsample.pt")
shuffled_unlearn_data_id = shuffled_edge_id_list[unlearn_target_data_id]
```
For any unlearning method, suppose `relationships_correct.pt` and `biographies_correct.pt` are two 0-1 vectors saved under the directory `input_dir`, which indicate the retained facts in family relationships and biographies after the unlearning the fact `unlearn_target_data_id`. The following script will calculate the recall and accuracy for this unlearning method to unlearn the target data.
```
python calculate_recall_and_acc.py --unlearn_data_id $unlearn_target_data_id --input_dir $input_dir
```
We provide `example_for_evaluation/relationships_correct.pt` and `example_for_evaluation/biographies_correct.pt` as an example for calculating the recall and accuracy, by running `python calculate_recall_and_acc.py --unlearn_data_id 1 --input_dir example_for_evaluation`.

## Reproducing the results of unlearning methods in the paper
In the paper, we tested with four unlearning methods: `gradient ascent (GA)`, `Negative preference optimization  (NPO)`, `task vector (TV)`, `who's harry potter (WHP)`. The hyperparameter list of each method is saved in `config/model_config.yaml`. The related scripts are saved in `./unlearning_methods`. By set any `unlearning_method` (`ga`, `npo`, `tv`, `whp`), any `target_model` (`phi`, `gpt2-xl`, `llama2-7b`, `llama3-8b`), and `unlearn_target_data_id` (0-54), the script is
```
bash unlearning_methods/${unlearning_methods}.sh $target_model $unlearn_target_data_id
```
After running unlearning methods, the code will save two 0-1 vectors `relationships_correct.pt` and `biographies_correct.pt` under the directory `scripts_unlearning_checkpoint/${unlearning_methods}/${target_model}/${unlearn_target_data_id}/checkpoint-${hyperparameter}`. Then run the script in the above section to calcualte the recall and accuracy.

## Citing Our Work

If you find our codebase and dataset beneficial, please cite our work:
```
@misc{wu2024evaluatingdeepunlearninglarge,
      title={Evaluating Deep Unlearning in Large Language Models}, 
      author={Ruihan Wu and Chhavi Yadav and Russ Salakhutdinov and Kamalika Chaudhuri},
      year={2024},
      eprint={2410.15153},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.15153}, 
}
```

## Acknowledgment
We would like to thank the authors of [TOFU](https://arxiv.org/abs/2401.06121) and [MUSE](https://arxiv.org/abs/2407.06460). Our code is built upon the github repository of them.
