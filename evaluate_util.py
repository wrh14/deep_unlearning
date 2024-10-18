from vllm import SamplingParams
import torch.nn.functional as F
import torch
from tqdm import tqdm

def eval_qa_vllm(dataset, model_eval, qk="question", ak="answer", question_start_tag="[INST] ", question_end_tag=" [/INST]", answer_tag=""):
    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    sampling_params = SamplingParams(temperature=0, top_p=0.6, max_tokens=10)
    responses = model_eval.generate(prompts, sampling_params)
    outputs = [response.outputs[0].text for response in responses]
    correct = [data[ak].lower() in output.lower() for data, output in zip(dataset, outputs)]
    return correct, responses

def eval_qa_vllm_whp(dataset, model_eval1,model_eval2, tokenizer,alpha_list, max_new_tokens=3, qk="question", ak="answer", question_start_tag = "[INST] ", question_end_tag = " [/INST]", answer_tag=""):
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    
    model_eval1.generation_config.pad_token_id = tokenizer.pad_token_id
    model_eval2.generation_config.pad_token_id = tokenizer.pad_token_id

    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    outputs_list = [[] for alpha in alpha_list]
    
    for i,prompt in enumerate(tqdm(prompts)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model_eval1.device)
        out1 = model_eval1.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_logits=True, output_scores=True)
        out2 = model_eval2.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_logits=True, output_scores=True)

        logits_list = []

        len1 = len(out1.logits)
        len2 = len(out2.logits)
        
        length = max(len1, len2)
        vocab_len = out1.logits[0].shape[1]
        if len1 < length:
            n_pads = length-len1
            zero_tensors = tuple(torch.zeros((1, vocab_len)) for _ in range(n_pads))
            out1.logits = out1.logits + zero_tensors
        elif len2 < length:
            n_pads = length-len2
            zero_tensors = tuple(torch.zeros((1, vocab_len)) for _ in range(n_pads))
            out2.logits = out2.logits + zero_tensors

        out2.logits = tuple(logit.to(model_eval1.device) for logit in out2.logits)
        out1.logits = tuple(logit.to(model_eval1.device) for logit in out1.logits)
        
        prob1_batch = F.softmax(torch.cat(list(out1.logits)), dim=-1)
        prob2_batch = F.softmax(torch.cat(list(out2.logits)), dim=-1)


        for i_alpha, alpha in enumerate(alpha_list):
            logits = prob1_batch - alpha * F.relu(prob2_batch-prob1_batch)
            predicted_token_ids = torch.argmax(logits, dim=-1)  # Shape should be [batch_size, seq_length]
            predicted_texts = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
            outputs_list[i_alpha].append(predicted_texts)
        
    correct_list = [[data[ak].lower() in output.lower() for data, output in zip(dataset, outputs)] for outputs in outputs_list]
    return correct_list, outputs_list
