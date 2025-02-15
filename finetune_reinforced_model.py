from data_module import custom_data_collator, FamilyForgetDataset
from unlearn_trainer import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
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

@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family, config_path=cfg.config_path)
    model_id = model_cfg["model_id"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file
    #if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    subsample = torch.load(cfg.subsample_path)
    if cfg.unlearn_data_id != -1:
        shuffled_unlearn_data_id = int(subsample[cfg.unlearn_data_id])
    else:
        shuffled_unlearn_data_id = subsample
    torch_format_dataset = FamilyForgetDataset(cfg.data_path, tokenizer=tokenizer, model_configs=model_cfg,max_length=500, unlearn_data_id=shuffled_unlearn_data_id,question_key='question4', answer_key='answer4') 

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    print("max_steps calc parmas : len(torch_format_dataset)", len(torch_format_dataset), "num_epochs:", cfg.num_epochs, "batch_size:", batch_size, "gradient_accumulation_steps:",gradient_accumulation_steps, "num_devices:", num_devices, "steps_per_epoch:",steps_per_epoch)
    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    
    lr = float(model_cfg["reinforce_lr"])
    
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=lr,
            lr_scheduler_type=cfg.lr_scheduler_type,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps,
            save_strategy="steps",
            save_only_model=True,
            ddp_find_unused_parameters= False,
            evaluation_strategy="no",
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search(r"pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search(r"model-*\.safetensors", file):
            path_found = True
            break

    if path_found:
        print("INSIDE PATTH FOUND")
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], trust_remote_code = True)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()
