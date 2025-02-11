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


def eval_qa_whp(dataset, whp_model, tokenizer, max_new_tokens=10, qk="question", ak="answer", question_start_tag = "[INST] ", question_end_tag = " [/INST]", answer_tag=""):
    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    output_list = []
    
    for i,prompt in enumerate(tqdm(prompts)):
        inputs = tokenizer(prompt, return_tensors="pt").to(whp_model.device)
        outputs = whp_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_list.append(predicted_text)
        
    correct= [data[ak].lower() in output.lower() for data, output in zip(dataset, output_list)]
    return correct, output_list