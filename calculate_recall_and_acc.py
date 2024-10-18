import argparse
import os
import torch
import numpy as np
import random

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
    
def check_valid_unlearn(minimal_set, unlearn_data_id, edge_list, edge_type_list, rule_list):
    def _check_valid_unlearn(cur_unlearn_data_id, minimal_set_verified, rec=0):
#         print(rec, minimal_set_verified)
        unlearn_edge = edge_list[cur_unlearn_data_id]
        unlearn_edge_type = edge_type_list[cur_unlearn_data_id]
        rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]

        for rule in rule_set_related:
            up_edges_list = rule.get_up_edges_list(edge_list, edge_type_list, unlearn_edge, unlearn_edge_type)
            for up_edges in up_edges_list:
                if_suf = False
                for up_edge in up_edges:
                    ind = get_edge_id((up_edge[0], up_edge[2]))
                    if ind in minimal_set:
                        if ind in minimal_set_verified:
                            if_suf = True
                            break
                        else:
                            minimal_set_verified.add(cur_unlearn_data_id)
                            if _check_valid_unlearn(ind, minimal_set_verified, rec=rec+1):
                                minimal_set_verified.remove(cur_unlearn_data_id)
                                if_suf = True
                                break
                            minimal_set_verified.remove(cur_unlearn_data_id)
                if not if_suf:
                    return False
        minimal_set_verified.add(cur_unlearn_data_id)
        return True
    if unlearn_data_id not in minimal_set:
        return False
    return _check_valid_unlearn(unlearn_data_id, set([]))
    

def check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, rule_list):
    cur_minimal_set = set(deepcopy(minimal_set))
    
    new_added_id_list = []
    t = 0
    while len(new_added_id_list) > 0 or t == 0:
        new_added_id_list = []
        t = t + 1
        for cur_unlearn_data_id in cur_minimal_set:
            unlearn_edge = edge_list[cur_unlearn_data_id]
            unlearn_edge_type = edge_type_list[cur_unlearn_data_id]
            rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]
            if_deducted = False
            for rule in rule_set_related:
                if if_deducted:
                    break
                up_edges_list = rule.get_up_edges_list(edge_list, edge_type_list, unlearn_edge, unlearn_edge_type)
                for up_edges in up_edges_list:
                    up_edges_if_deducted = True
                    for up_edge in up_edges:
                        ind = get_edge_id((up_edge[0], up_edge[2]))
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
                
    
def get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, rule_list, seed=0, search_unlearn_ind=None):
    np.random.seed(seed)
    random.seed(seed)
    nec_unlearn_ind = np.zeros(len(unlearn_ind))
    not_included_unlearn_ind = np.zeros(len(unlearn_ind))
    
    minimal_set = set([])
    if search_unlearn_ind[unlearn_data_id] == 1:
        nec_unlearn_ind[unlearn_data_id] = 1;
    else:
        not_included_unlearn_ind[unlearn_data_id] = 1
    minimal_set_unverified = set([unlearn_data_id])
    
    #Find a valid unlearning set expanded from the given unlearning result.
    while len(minimal_set_unverified) >= 1:
        cur_unlearn_data_id = random.sample(sorted(minimal_set_unverified), 1)[0]
        minimal_set_unverified.remove(cur_unlearn_data_id)
        minimal_set.add(cur_unlearn_data_id)

        unlearn_edge = edge_list[cur_unlearn_data_id]
        unlearn_edge_type = edge_type_list[cur_unlearn_data_id]
        rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]

        for rule in rule_set_related:
            up_edges_list = rule.get_up_edges_list(edge_list, edge_type_list, unlearn_edge, unlearn_edge_type)
            for up_edges in up_edges_list:
                if_suf = 0
                for up_edge in up_edges:
                    ind = get_edge_id((up_edge[0], up_edge[2]))
                    if search_unlearn_ind[ind] == 1:
                        nec_unlearn_ind[ind] = 1;
                        if_suf = 1
                        if (ind not in minimal_set) and (ind not in minimal_set_unverified):
                            minimal_set_unverified.add(ind)
                        break
                    elif not_included_unlearn_ind[ind] == 1:
                        if_suf = 1
                        if (ind not in minimal_set) and (ind not in minimal_set_unverified):
                            minimal_set_unverified.add(ind)
                        break
                if if_suf == 0:
                    rand_edge = random.sample(up_edges, 1)[0]
                    rand_ind = get_edge_id((rand_edge[0], rand_edge[2]))
                    if (rand_ind not in minimal_set) and (rand_ind not in minimal_set_unverified):
                        minimal_set_unverified.add(rand_ind)
                    not_included_unlearn_ind[rand_ind] = 1
        
    #Prune the valid unlearning set by removing redundant element from the extended part
    minimal_set = set(minimal_set)
    
    C = []
    t = 0
    while len(C) != 0 or t==0:
        C = []
        t = t+1
        shuffled_minimal_set = np.asarray(list(minimal_set))[np.random.permutation(len(minimal_set))]
        for data_id in shuffled_minimal_set:
            minimal_set.remove(data_id)
            if not check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, rule_list):
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
    
    
def get_valid_unlearn_general(unlearn_data_id, edge_list, edge_type_list, unlearn_ind, rule_list, num_seed=100):
    os.makedirs("synthetic_data/unlearn_minimal_set", exist_ok=True)
    
    if os.path.exists(f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt"):
        minimal_unlearn_set = torch.load(f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt")
    else:
        minimal_unlearn_list = []
        search_unlearn_ind = np.zeros(len(unlearn_ind))
        for seed in range(num_seed):
            minimal_set = get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, rule_list, seed, search_unlearn_ind)
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

def get_edge_id(edge):
    for i, _edge in enumerate(edge_list):
        if _edge == edge:
            return i
        
(edge_list, edge_type_list, fixed_names, person_list) = torch.load("synthetic_data/family-200-graph.pt")
rule_list = torch.load("synthetic_data/family_rule.pt")
shuffled_edge_id_list = torch.load("synthetic_data/subsample.pt")

shuffled_unlearn_data_id = shuffled_edge_id_list[args.unlearn_data_id]

rel_ind = np.asarray(torch.load(f"{args.input_dir}/relationships_correct.pt")).astype(np.float32)
unlearn_ind = 1 - rel_ind
bio_ind = torch.load(f"{args.input_dir}/biographies_correct.pt")

precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(shuffled_unlearn_data_id, edge_list, edge_type_list, unlearn_ind, rule_list, num_seed=100)

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