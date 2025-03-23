from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

# Load fine-tuned model from Hugging Face Hub
model = AutoPeftModelForSeq2SeqLM.from_pretrained("mmallick7/mt0-large-ia3").to("cuda")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")

# Load dataset from Hugging Face
ds = load_dataset("financial_phrasebank", "sentences_allagree")

# Split dataset into train & validation sets
ds = ds["train"].train_test_split(test_size=0.1)
ds["validation"] = ds["test"]
del ds["test"]

# Get class labels (e.g., 'positive', 'negative', 'neutral')
classes = ds["validation"].features["label"].names

# Map numerical labels to their string values
ds = ds.map(lambda x: {"text_label": [classes[label] for label in x["label"]]}, batched=True, num_proc=1)

# Select a sentence for inference
text_column = "sentence"
i = 15  # Index of sentence for testing
sample_text = ds["validation"][text_column][i]
print("Input Sentence:", sample_text)

# Tokenize input sentence
inputs = tokenizer(sample_text, return_tensors="pt").to("cuda")

# Perform inference
with torch.no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

# Decode and print the predicted label
decoded_output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
print("Predicted Label:", decoded_output)
