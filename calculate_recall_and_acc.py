import argparse
import torch
import numpy as np

from utils_data_building import (
    Person, 
    Rule, 
)

from utils_metric import (
    check_if_in_deductive_closure, 
    get_minimal_nec_unlearn_and_not_included_unlearn, 
    get_prec_rec_acc,  
    get_valid_unlearn_general,
    get_edge_id,
    get_deductive_closure,
)

parser = argparse.ArgumentParser(description='calculate the recall and accucracy')
parser.add_argument('--unlearn_data_id', type=int, default=None, help="id of the fact to unlearn")
parser.add_argument('--input_dir', type=str, default=None, help="directory that saves the rettained knowledge base")
args = parser.parse_args()
        
(edge_list, edge_type_list, fixed_names, person_list) = torch.load("synthetic_data/family-200-graph.pt")
rule_list = torch.load("synthetic_data/family_rule.pt")
dc_edge_list, dc_edge_type_list = get_deductive_closure(edge_list, edge_type_list, rule_list, person_list)
shuffled_edge_id_list = torch.load("synthetic_data/subsample.pt")

shuffled_unlearn_data_id = shuffled_edge_id_list[args.unlearn_data_id]

if args.input_dir is None:
    print("pre-compute the minimal deep unlearning set only")
    precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(shuffled_unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, np.zeros(len(edge_list)), rule_list, num_seed=100)
    exit()
    
rel_ind = np.asarray(torch.load(f"{args.input_dir}/relationships_correct.pt")).astype(np.float32)
unlearn_ind = 1 - rel_ind
bio_ind = torch.load(f"{args.input_dir}/biographies_correct.pt")

precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(shuffled_unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, unlearn_ind, rule_list, num_seed=100)

rec = max(recall_list)
argmax = np.asarray(recall_list).argmax()
acc_rel = accuracy_list[argmax]
acc_bio = np.asarray(bio_ind).mean()

num_rel = len(rel_ind)
num_bio = len(bio_ind)
size_mul = len(list(minimal_unlearn_list)[argmax])
acc_all = ((acc_bio * num_bio) + accuracy_list[argmax] * ( num_rel - size_mul)) / (num_bio + num_rel - size_mul)
print(("recall", "accuracy of relationships", "accuracy of biographies", "accurcy of all knowledge base"))
print((rec, acc_rel, acc_bio, acc_all))
torch.save((rec, acc_rel, acc_bio, acc_all), f"{args.input_dir}/rec_acc.pt")