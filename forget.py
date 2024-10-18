import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
from datasets import Dataset
import os
import gc
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from data_module import FamilyForgetDataset, custom_data_collator
from unlearn_trainer import CustomFamilyTrainerForgetting
from utils import get_model_identifiers_from_yaml

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["model_id"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    #get the the unlearn_data_i in shuffled id
    subsample = torch.load(cfg.subsample_path)
    shuffled_unlearn_data_id = int(subsample[cfg.unlearn_data_id])
    torch_format_dataset = FamilyForgetDataset(cfg.data_path, tokenizer=tokenizer, model_configs=model_cfg, max_length=500, unlearn_data_id=shuffled_unlearn_data_id, question_key='question4', answer_key='answer4')
    
    if cfg.forget_loss == "ga":
        lr = float(model_cfg["ga_lr"])
        num_epochs = model_cfg["ga_num_epochs"]
    elif cfg.forget_loss == "npo":
        lr = float(model_cfg["npo_lr"])
        num_epochs = model_cfg["npo_num_epochs"]
    
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    max_steps = int(num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    
    
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=max(1,max_steps//20),
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_strategy="no",
        ddp_find_unused_parameters= False,
        deepspeed='config/ds_config.json',
        weight_decay = cfg.weight_decay,
        eval_steps = 1,
        evaluation_strategy = "steps",
        seed=cfg.seed,
    )
    
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break


    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], trust_remote_code = True)
    else:
        print("checkpoint not found")
        exit()
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

        
    trainer = CustomFamilyTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator,
        forget_loss = cfg.forget_loss,
        save_step_pattern=cfg.save_step_pattern,
        save_dir=cfg.save_dir
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    
    if cfg.forget_loss == "npo":
        outputs_f_ref_dir = f"{cfg.save_dir}/outputs_f_ref.pt"
        if not os.path.exists(outputs_f_ref_dir):
            ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], trust_remote_code = True)
            ref_model.eval()
            ref_model = trainer.e_prepare_deepspeed(ref_model)
            with torch.no_grad():
                inputs = trainer.train_dataset[0]
                input_ids, labels, attention_mask = inputs[0], inputs[1], inputs[2]
                input_ids, labels, attention_mask = input_ids.unsqueeze(0).to(local_rank), labels.unsqueeze(0).to(local_rank), attention_mask.unsqueeze(0).to(local_rank)
                outputs_f_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
            ref_model.destroy()
            del ref_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.save(outputs_f_ref, outputs_f_ref_dir)
            exit()
        trainer.outputs_f_ref_logits = torch.load(outputs_f_ref_dir).logits.to(local_rank)
#         trainer.outputs_f_ref.logits = trainer.outputs_f_ref.logits.to(local_rank)
        
    # trainer.train()
    trainer.train()

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()

