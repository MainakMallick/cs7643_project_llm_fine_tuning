import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt  # Add this import
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import IA3Config, get_peft_model
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from huggingface_hub import notebook_login
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

MODEL_TO_OUTPUT = {
    "facebook/opt-350m": "lora-opt-350m-finetuned", 
    "meta-llama/Meta-Llama-3-8B": "lora-meta-llama-3-8b-finetuned",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "IA3-deepseek-ai-1-5b-finetuned"
}
# 1. Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
OUTPUT_DIR = MODEL_TO_OUTPUT[MODEL_NAME]
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 1
LEARNING_RATE = 2e-4  # LoRA often trains well with slightly higher lr
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 512
LR_SCHEDULER_TYPE = "cosine"
SAVE_STEPS = 500
EVAL_STEPS = 10

def tokenize_example(example):
        text = format_example(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )
        return tokenized

def format_example(example):
    # If there's no 'input' (it might be empty), we can handle that gracefully
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output_text = example["output"].strip()

    if input_text:
        prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}"
        )
    return prompt
if __name__ == "__main__":
    # 2. Load dataset
    #    The iamtarun/python_code_instructions_18k_alpaca dataset likely has
    #    the following fields for each sample: "instruction", "input", and "output".
    dataset = load_dataset(DATASET_NAME)

    # You can verify the column names via dataset['train'].column_names
    # E.g., print(dataset['train'].column_names)

    # 3. Prompt Formatting
    #    We'll create a function to turn each example into a text prompt (prompt + expected answer).
    #    For Alpaca-style data, a recommended prompt structure is:
    #    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"

    

    # 4. Preprocess the dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    # For OPT models, the tokenizer might have a prefix like '</s>' as an EOS token
    # If needed, set pad_token if it's not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_datasets = dataset["train"]
    # Set a random seed for reproducibility
    seed = 42
    all_datasets = all_datasets.shuffle(seed=seed)
    
    # Split off 1000 examples for evaluation
    eval_dataset = all_datasets.select(range(1000))
    
    # Split another 1000 examples for testing
    test_dataset = all_datasets.select(range(1000, 2000))
    test_dataset.to_json("dataset/python_code_instructions_18k_alpaca_test_2.json")
    
    # Remove eval and test examples from the training dataset
    train_dataset = all_datasets.select(range(2000, len(all_datasets)))
    # Use map to preprocess
    train_dataset = train_dataset.map(tokenize_example, batched=False)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_dataset = eval_dataset.map(tokenize_example, batched=False)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 5. Load Model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 2. Define IA³ configuration
    peft_config = IA3Config(
        task_type="CAUSAL_LM",        # Required for causal language modeling
        target_modules=["k_proj", "v_proj"]  # Typical modules for IA³; adjust per model architecture
    )

    # 3. Apply PEFT
    IA3_model = get_peft_model(model, peft_config)
    IA3_model.print_trainable_parameters()

    # 8. Data Collator
    #    We can use a DataCollatorForLanguageModeling to handle dynamic LM masking/padding.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 9. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        fp16=True,  # if you have a GPU with fp16
        optim="adamw_torch",
        report_to="tensorboard",  # or "wandb"/"tensorboard"
    )

    # 10. Define Trainer
    trainer = Trainer(
        model=IA3_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 11. Train!
    train_result = trainer.train()

    # 12. Save final model (LoRA adapters)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 13. Plot loss vs iterations
    # Extract training loss values
    train_loss_values = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    train_steps = list(range(len(train_loss_values)))

    # Extract validation loss values
    eval_loss_values = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    eval_steps = [i for i, log in enumerate(trainer.state.log_history) if "eval_loss" in log]

    # --- Training Loss Plot ---
    plt.figure()
    plt.plot(train_steps, train_loss_values, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    train_plot_path = os.path.join(OUTPUT_DIR, f"training_loss_plot_{MODEL_NAME.replace('/', '_')}.png")
    plt.savefig(train_plot_path)
    plt.close()

    # --- Validation Loss Plot ---
    plt.figure()
    plt.plot(eval_steps, eval_loss_values, label="Validation Loss", color="orange", marker='o')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Validation Loss over Time")
    plt.legend()
    eval_plot_path = os.path.join(OUTPUT_DIR, f"validation_loss_plot_{MODEL_NAME.replace('/', '_')}.png")
    plt.savefig(eval_plot_path)
    plt.close()
        # You now have a LoRA-adapted model in OUTPUT_DIR.
    # To use it for inference, load the base model again and apply the saved LoRA weights:
    #
    # from peft import PeftModel
    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    # tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=False)
    # prompt = "Your prompt here"
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # outputs = model.generate(**inputs, max_new_tokens=100)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))