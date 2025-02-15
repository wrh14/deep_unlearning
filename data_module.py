import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index
import os

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
        
    encoded_answer = tokenizer(
        new_answer, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
        
        
    #change label to -100 for question tokens
#     print(encoded['input_ids'][num_question_tokens], label[num_question_tokens])
    for i in range(num_question_tokens): label[i] = -100
    
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    

class FamilyForgetDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_configs, max_length=512,  unlearn_data_id=0, question_key=None, answer_key=None):
        super(FamilyForgetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.Dataset.from_dict(torch.load(data_path))
        self.data = add_dataset_index(self.data)
        self.qk = question_key
        self.ak = answer_key
        if isinstance(unlearn_data_id, int):
            unlearn_data_id = np.asarray([unlearn_data_id]).astype(np.int32)
        self.unlearn_data_id = unlearn_data_id
            
        self.model_configs = model_configs
        self.world_size = int(os.environ.get('WORLD_SIZE', 1)) 

    def __len__(self):
        return len(self.unlearn_data_id) * self.world_size

    def __getitem__(self, idx):
        data_id = int(self.unlearn_data_id[int(idx/self.world_size)])
        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        question = self.data[data_id][self.qk]
        answers = self.data[data_id][self.ak]
        indices = self.data[data_id]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)
    
def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

