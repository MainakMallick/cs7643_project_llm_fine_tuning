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

# 1. Configuration
MODEL_NAME = "facebook/opt-350m"
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
OUTPUT_DIR = "lora-opt-350m-finetuned"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 1
LEARNING_RATE = 2e-4  # LoRA often trains well with slightly higher lr
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 512
LR_SCHEDULER_TYPE = "cosine"
SAVE_STEPS = 500
EVAL_STEPS = 500

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

# 4. Preprocess the dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# For OPT models, the tokenizer might have a prefix like '</s>' as an EOS token
# If needed, set pad_token if it's not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_example(example):
    text = format_example(example)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length"
    )
    return tokenized

# Use map to preprocess
train_dataset = dataset["train"].map(tokenize_example, batched=False)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
# If you have a validation set in the dataset, do the same:
if "validation" in dataset:
    eval_dataset = dataset["validation"].map(tokenize_example, batched=False)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    # Or split off some portion of train if needed
    eval_dataset = train_dataset.select(range(1000))  # quick hack for illustration

# 5. Load Model
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 6. Configure LoRA
#    Adjust parameters such as r, alpha, and dropout depending on your use case.
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # typical for OPT/Bloom, might be different for some models
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 7. Attach LoRA to the base model
lora_model = get_peft_model(base_model, lora_config)

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
    logging_steps=100,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    fp16=True,  # if you have a GPU with fp16
    optim="adamw_torch",
    report_to="none",  # or "wandb"/"tensorboard"
)

# 10. Define Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 11. Train!
trainer.train()

# 12. Save final model (LoRA adapters)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

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
