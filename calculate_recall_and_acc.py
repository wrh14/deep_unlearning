import argparse
import os
import torch
import numpy as np
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='calculate the recall and accucracy')
parser.add_argument('--unlearn_data_id', type=int, default=None, help="id of the fact to unlearn")
parser.add_argument('--input_dir', type=str, default=None, help="directory that saves the rettained knowledge base")
args = parser.parse_args()

class Person:
    def __init__(self):
        self.name = None
        self.gender = gender
        self.father = None
        self.mother = None
        self.children = None
        self.husband = None
        self.wife = None


from copy import deepcopy
class Rule:
    def __init__(self, left_tuples, right_tuple):
        self.left_tuples = left_tuples
        self.right_tuple = right_tuple
        self.num_var = max(max(tup[0], tup[2]) for tup in left_tuples + [right_tuple]) + 1
        
    def get_up_edges_list(self, edge_list, edge_type_list, unlearn_edge, unlearn_edge_type):
        source_type_dict = {}
        type_target_dict = {}
        for edge, edge_type in zip(edge_list, edge_type_list):
            source_type = (edge[0], edge_type)
            if source_type in source_type_dict.keys():
                source_type_dict[source_type].append(edge[1])
            else:
                source_type_dict[source_type] = [edge[1]]
                
            type_target = (edge_type, edge[1])
            if type_target in type_target_dict.keys():
                type_target_dict[type_target].append(edge[0])
            else:
                type_target_dict[type_target] = [edge[0]]
        
        var_value = -np.ones(self.num_var)
        var_value[self.right_tuple[0]] = unlearn_edge[0]
        var_value[self.right_tuple[2]] = unlearn_edge[1]
        
        dc_var_value_list = []
        
        def _get_up_edges_list(cur_var_value):
            if (cur_var_value == -1).sum() == 0:
                for tup in self.left_tuples + [self.right_tuple]:
                    if (cur_var_value[tup[0]], tup[1]) not in source_type_dict.keys():
                        return
                    if cur_var_value[tup[2]] not in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                        return
                
                if not any(np.array_equal(cur_var_value, unique_arr) for unique_arr in dc_var_value_list):
                    dc_var_value_list.append(cur_var_value)
                return
            
            for tup in self.left_tuples:
                if cur_var_value[tup[0]] == -1 and cur_var_value[tup[2]] != -1:
                    if (tup[1], cur_var_value[tup[2]]) in type_target_dict.keys():
                        for potential_tup0_val in type_target_dict[(tup[1], cur_var_value[tup[2]])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[0]] = potential_tup0_val
                            _get_up_edges_list(new_cur_var_value)
                elif cur_var_value[tup[2]] == -1 and cur_var_value[tup[0]] != -1:
                    if (cur_var_value[tup[0]], tup[1]) in source_type_dict.keys():
                        for potential_tup0_val in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[2]] = potential_tup0_val
                            _get_up_edges_list(new_cur_var_value)
        
        _get_up_edges_list(var_value)
        
        up_edges_list = []
        for dc_var_value in dc_var_value_list:
            up_edges = []
            for tup in self.left_tuples:
                up_edges.append((dc_var_value[tup[0]], tup[1], dc_var_value[tup[2]]))
            up_edges_list.append(up_edges)
            
        return up_edges_list
    
    def get_dc_edges_list(self, edge_list, edge_type_list):
        source_type_dict = {}
        type_target_dict = {}
        for edge, edge_type in zip(edge_list, edge_type_list):
            source_type = (edge[0], edge_type)
            if source_type in source_type_dict.keys():
                source_type_dict[source_type].append(edge[1])
            else:
                source_type_dict[source_type] = [edge[1]]
                
            type_target = (edge_type, edge[1])
            if type_target in type_target_dict.keys():
                type_target_dict[type_target].append(edge[0])
            else:
                type_target_dict[type_target] = [edge[0]]
        
        dc_var_value_list = []
        def _get_right_edges_list(cur_var_value):
            if (cur_var_value == -1).sum() == 0:
                for tup in self.left_tuples:
                    if (cur_var_value[tup[0]], tup[1]) not in source_type_dict.keys():
                        return
                    if cur_var_value[tup[2]] not in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                        return
                if not any(np.array_equal(cur_var_value, unique_arr) for unique_arr in dc_var_value_list):
                    dc_var_value_list.append(cur_var_value)
                return
            
            for tup in self.left_tuples:
                if cur_var_value[tup[0]] == -1 and cur_var_value[tup[2]] != -1:
                    if (tup[1], cur_var_value[tup[2]]) in type_target_dict.keys():
                        for potential_tup0_val in type_target_dict[(tup[1], cur_var_value[tup[2]])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[0]] = potential_tup0_val
                            _get_right_edges_list(new_cur_var_value)
                elif cur_var_value[tup[2]] == -1 and cur_var_value[tup[0]] != -1:
                    if (cur_var_value[tup[0]], tup[1]) in source_type_dict.keys():
                        for potential_tup0_val in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[2]] = potential_tup0_val
                            _get_right_edges_list(new_cur_var_value)
        
        for edge, edge_type in zip(edge_list, edge_type_list):
            if edge_type == self.left_tuples[0][1]:
                var_value = -np.ones(self.num_var)
                var_value[self.left_tuples[0][0]] = edge[0]
                var_value[self.left_tuples[0][2]] = edge[1]
                _get_right_edges_list(var_value)
        
        
        new_edge_list = []
        new_edge_type_list = []
        
        for dc_var_value in dc_var_value_list:
            new_edge = (dc_var_value[self.right_tuple[0]], dc_var_value[self.right_tuple[2]])
            new_edge_type = (self.right_tuple[1])
            
            if (dc_var_value[self.right_tuple[0]], self.right_tuple[1]) in source_type_dict.keys():
                if dc_var_value[self.right_tuple[2]] in source_type_dict[(dc_var_value[self.right_tuple[0]], self.right_tuple[1])]:
                    continue
            
            if self.right_tuple[1] in ["husband", "uncle", "father", "brother", "nephew"]:
                if person_list[int(dc_var_value[self.right_tuple[2]])].gender != "male":
                    continue
                if self.right_tuple[1] == "husband" and person_list[int(dc_var_value[self.right_tuple[0]])].gender != "female":
                    continue
                    
            elif self.right_tuple[1] in ["wife", "aunt", "mother", "sister", "niece"]:
                if person_list[int(dc_var_value[self.right_tuple[2]])].gender != "female":
                    continue
                if self.right_tuple[1] == "wife" and person_list[int(dc_var_value[self.right_tuple[0]])].gender != "male":
                    continue
            if dc_var_value[self.right_tuple[0]] == dc_var_value[self.right_tuple[2]]:
                continue
            
            new_edge_list.append(new_edge)
            new_edge_type_list.append(new_edge_type)
            
        return new_edge_list, new_edge_type_list


def check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
    cur_minimal_set = set(list(deepcopy(minimal_set)) + list(range(len(edge_list), len(dc_edge_list))))
    
    new_added_id_list = []
    t = 0
    while len(new_added_id_list) > 0 or t == 0:
        new_added_id_list = []
        t = t + 1
        for cur_unlearn_data_id in cur_minimal_set:
            unlearn_edge = dc_edge_list[cur_unlearn_data_id]
            unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id]
            rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]
            if_deducted = False
            for rule in rule_set_related:
                if if_deducted:
                    break
                up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type)
                for up_edges in up_edges_list:
                    up_edges_if_deducted = True
                    for up_edge in up_edges:
                        ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
                        if ind in cur_minimal_set:
                            up_edges_if_deducted = False
                            break
                    if up_edges_if_deducted:
                        if_deducted = True
                        new_added_id_list.append(cur_unlearn_data_id)
                        break
        for new_added_id in new_added_id_list:
            cur_minimal_set.remove(new_added_id)
    if unlearn_data_id in cur_minimal_set:
        return False
    else:
        return True              
                
    
def get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
    minimal_set = set([])
    minimal_set_unverified = set([unlearn_data_id])

    
    #Find a valid unlearning set expanded from the given unlearning result.
    while len(minimal_set_unverified) >= 1:
#         print(minimal_set_unverified)
        cur_unlearn_data_id = random.sample(sorted(minimal_set_unverified), 1)[0]
        minimal_set_unverified.remove(cur_unlearn_data_id)
        minimal_set.add(cur_unlearn_data_id)

        unlearn_edge = dc_edge_list[cur_unlearn_data_id]
        unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id]
        rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]

        for rule in rule_set_related:
            up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type)
            for up_edges in up_edges_list:
                if_suf = 0
                for up_edge in up_edges:
                    ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
                    if (ind in minimal_set) or (ind in minimal_set_unverified):
                        if_suf = 1
                        break
                if if_suf == 0:
                    rand_edge = random.sample(up_edges, 1)[0]
                    rand_ind = get_edge_id((rand_edge[0], rand_edge[2]), dc_edge_list)
                    minimal_set_unverified.add(rand_ind)
        
    minimal_set = set([i for i in minimal_set if i < len(edge_list)])
    #Prune the valid unlearning set by removing redundant element from the extended part
    
    C = []
    t = 0
    while len(C) != 0 or t==0:
        C = []
        t = t+1
        shuffled_minimal_set = np.asarray(list(minimal_set))[np.random.permutation(len(minimal_set))]
        for data_id in shuffled_minimal_set:
            minimal_set.remove(data_id)
            if not check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
                C.append(data_id)
            else:
                minimal_set.add(data_id)
    return minimal_set

    
def get_prec_rec_acc(minimal_set, unlearn_ind):
    minimal_set_ind = np.zeros(len(unlearn_ind))
    minimal_set_ind[list(minimal_set)] = 1
    prec = (minimal_set_ind * unlearn_ind).sum() / max(unlearn_ind.sum(), 1e-8)
    rec = (minimal_set_ind * unlearn_ind).sum() / minimal_set_ind.sum()
    acc = 1 - (unlearn_ind * (1 - minimal_set_ind)).sum() / (len(unlearn_ind) - len(minimal_set))
    return prec, rec, acc
    
    
def get_valid_unlearn_general(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, unlearn_ind, rule_list, num_seed=10):
    if os.path.exists(f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt"):
        minimal_unlearn_set = torch.load(f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt")
    else:
        minimal_unlearn_list = []
        for seed in tqdm(range(num_seed)):
            minimal_set = get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed)
            minimal_unlearn_list.append(minimal_set)
        minimal_unlearn_set = set([frozenset(minimal_set) for minimal_set in minimal_unlearn_list])
        torch.save(minimal_unlearn_set, f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt")
    precision_list = []
    recall_list = []
    acc_list = []
    for minimal_set in minimal_unlearn_set:
        prec, rec, acc = get_prec_rec_acc(minimal_set, unlearn_ind)
        precision_list.append(prec)
        recall_list.append(rec)
        acc_list.append(acc)
    
    return precision_list, recall_list, acc_list, minimal_unlearn_set

def get_edge_id(edge, edge_list):
    for i, _edge in enumerate(edge_list):
        if _edge == edge:
            return i
        
        
def get_deductive_closure(edge_list, edge_type_list, rule_list):
    dc_edge_list, dc_edge_type_list = deepcopy(edge_list), deepcopy(edge_type_list)
    new_edge_list = []
    new_edge_type_list = []
    cur_iter=0
    while len(new_edge_list) > 0 or cur_iter == 0:
        new_edge_list = []
        new_edge_type_list = []
        for rule in rule_list:
            _new_edge_list, _new_edge_type_list = rule.get_dc_edges_list(dc_edge_list, dc_edge_type_list)
            dc_edge_list = dc_edge_list + _new_edge_list
            dc_edge_type_list = dc_edge_type_list + _new_edge_type_list
            
            new_edge_list = new_edge_list + _new_edge_list
            new_edge_type_list = new_edge_type_list + _new_edge_type_list
            
        cur_iter += 1
    return dc_edge_list, dc_edge_type_list
        
(edge_list, edge_type_list, fixed_names, person_list) = torch.load("synthetic_data/family-200-graph.pt")
rule_list = torch.load("synthetic_data/family_rule.pt")
dc_edge_list, dc_edge_type_list = get_deductive_closure(edge_list, edge_type_list, rule_list)
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