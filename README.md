## Project Overview

This project demonstrates fine-tuning a causal language model using LoRA (Low-Rank Adaptation) on a dataset of Python code instructions. It also includes evaluation scripts to assess the fine-tuned model's performance using metrics like CodeBLEU and correctness on datasets such as HumanEval.

---

## Requirements

### Python Packages

To run this project, you need the following Python packages:
- `torch` (PyTorch)
- `transformers` (Hugging Face Transformers)
- `datasets` (Hugging Face Datasets)
- `peft` (Parameter-Efficient Fine-Tuning)
- `matplotlib` (for plotting training/validation loss)
- `tqdm` (for progress bars)
- `numpy` (for numerical operations)
- `argparse` (for command-line argument parsing)
- `codebleu` (for CodeBLEU evaluation)

You can install the required packages using:
```bash
pip install torch transformers datasets peft matplotlib tqdm numpy codebleu
```

---

## Fine-Tuning the Model

### Steps to Fine-Tune

1. **Dataset Preparation:**
   - The dataset used is `iamtarun/python_code_instructions_18k_alpaca`.
   - Ensure this dataset is available via Hugging Face Datasets.
   - The dataset is split into training, evaluation, and testing subsets.

2. **Run the Fine-Tuning Script:**

   Use the `fine_tune_lora.py` script to fine-tune the model with LoRA.

   Example command:
   ```bash
   python fine_tune_lora.py
   ```
   Use the `fine_tune_ia3.py` script to fine-tune the model with LoRA.

   Example command:
   ```bash
   python fine_tune_ia3.py
   ```

   This script:
   - Loads the dataset.
   - Preprocesses the data into instruction-following format.
   - Fine-tunes the model using LoRA.
   - Saves the fine-tuned model and generates training/validation loss plots.

   **Output:**
   - The fine-tuned model is saved in the `OUTPUT_DIR` specified in the script.
   - Training and validation loss plots are saved as PNG files in the same directory.

---

## Evaluating the Model

### Steps to Evaluate

**HumanEval Evaluation:**

Use the `evaluate.py` and `evaluate_passk.py` script to evaluate the model on CodeLBLEU and Pass@k benchmark respectively:
```bash
python evaluate.py
```


```bash
python evaluate_passk.py
```


**Output:**
- Evaluation results are saved in the output directory as JSON files.
- Metrics and correctness are printed to the console.

---

## Example Usage

**Fine-Tuning:**
```bash
python fine_tune_lora.py
```

**Evaluating on CodeBLEU:**
```bash
python evaluate.py
```


---

## Notes

- Ensure you have access to a GPU for faster training and evaluation.
- Modify hyperparameters like `BATCH_SIZE`, `LEARNING_RATE`, and `EPOCHS` in the scripts as needed.
- For inference, load the fine-tuned model using the following code snippet:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = PeftModel.from_pretrained(model, "lora-deepseek-ai-1-5b-finetuned")
tokenizer = AutoTokenizer.from_pretrained("lora-deepseek-ai-1-5b-finetuned", use_fast=False)

prompt = "Explain the difference between a list and a tuple in Python."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
