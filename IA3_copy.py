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

# Create output directories
output_dir = "./saved_models"
log_dir = "./logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"{log_dir}/run_{timestamp}")

# Load and preprocess dataset
try:
    # Attempt to load from Hugging Face datasets
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
    print("Successfully loaded dataset from Hugging Face")
except Exception as e:
    print(f"Error loading dataset from Hugging Face: {e}")
    print("Assuming dataset is available locally or through another method")
    # If not available, you might need to provide a local path or different access method
    # This is a placeholder - replace with actual loading method if needed
    # ds = load_dataset("path/to/local/dataset")
    raise

# Split dataset
ds = ds["train"].train_test_split(test_size=0.1)
ds["validation"] = ds["test"]
del ds["test"]

# Check data structure
print(f"Dataset info: {ds}")
print(f"Dataset columns: {list(ds['train'].features)}")
print(f"Dataset size: {len(ds['train'])} training, {len(ds['validation'])} validation examples")

print(f"Sample datapoint: {ds['train'][0]}")

# Let's examine the first sample to determine the column structure
print(f"Sample datapoint: {ds['train'][0]}")

# Adjust column names based on the actual dataset structure
# These are likely column names for instruction-based datasets
text_column = "instruction"  # Input prompt/instruction
label_column = "output"      # Expected output/completion
max_length = 128

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set padding token to: {tokenizer.pad_token}")

def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    
    # Print example to understand structure
    if len(inputs) > 0:
        print(f"Sample input: {inputs[0]}")
        print(f"Sample target: {targets[0]}")
    
    # Process inputs
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    
    # Process targets/labels
    labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100  # Set padding tokens to -100 to ignore in loss
    
    model_inputs["labels"] = labels
    return model_inputs

processed_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=ds["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_ds = processed_ds["train"]
eval_ds = processed_ds["validation"]

batch_size = 8

train_dataloader = DataLoader(
    train_ds, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_ds, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# Load base model - Llama 3.1 is a causal language model, not a seq2seq model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
peft_config = IA3Config(task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training parameters
lr = 8e-3
num_epochs = 2
logging_steps = 100  # Log every 100 steps
save_steps = 500     # Save checkpoint every 500 steps

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
device = "cuda"
model = model.to(device)

# Function to calculate accuracy
def compute_accuracy(preds, labels):
    true_labels = [tokenizer.decode(label[label != -100], skip_special_tokens=True) for label in labels]
    return np.mean([p.strip() == l.strip() for p, l in zip(preds, true_labels)])

# Training and evaluation loop
global_step = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        global_step += 1
        
        # Log training metrics
        if global_step % logging_steps == 0:
            avg_loss = total_loss / (step + 1)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)
            progress_bar.set_postfix(loss=avg_loss.item(), lr=lr_scheduler.get_last_lr()[0])
        
        # Save checkpoint
        if global_step % save_steps == 0:
            checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Saved checkpoint at step {global_step} to {checkpoint_dir}")
    
    # Evaluation at the end of each epoch
    model.eval()
    eval_loss = 0
    eval_preds = []
    all_labels = []
    
    for step, batch in enumerate(tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        loss = outputs.loss
        eval_loss += loss.detach().float()
        
        predictions = tokenizer.batch_decode(
            torch.argmax(outputs.logits, -1).detach().cpu().numpy(), 
            skip_special_tokens=True
        )
        eval_preds.extend(predictions)
        all_labels.extend(batch["labels"].detach().cpu().numpy())
    
    # Calculate and log evaluation metrics
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    
    accuracy = compute_accuracy(eval_preds, all_labels)
    
    writer.add_scalar("eval/loss", eval_epoch_loss.item(), global_step)
    writer.add_scalar("eval/perplexity", eval_ppl.item(), global_step)
    writer.add_scalar("eval/accuracy", accuracy, global_step)
    
    print(f"Epoch {epoch}: train_ppl={train_ppl:.4f}, train_loss={train_epoch_loss:.4f}, "
          f"eval_ppl={eval_ppl:.4f}, eval_loss={eval_epoch_loss:.4f}, accuracy={accuracy:.4f}")

# Save the final model
final_model_dir = f"{output_dir}/final-model"
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print(f"Saved final model to {final_model_dir}")

# Close the TensorBoard writer
writer.close()

# Push to HuggingFace Hub (optional)
account = "mmallick7"
peft_model_id = f"{account}/mt0-large-ia4"
model.push_to_hub(peft_model_id)

