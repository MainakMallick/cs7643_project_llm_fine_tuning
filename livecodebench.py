#!/usr/bin/env python3
"""
Streamlined LoRA Fine-tuning and LiveCodeBench Evaluation Script

This script:
1. Fine-tunes a model using LoRA on the Python code instructions dataset
2. Evaluates the fine-tuned model against the base model using LiveCodeBench
"""

import os
import json
import torch
import random
import tempfile
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
OUTPUT_DIR = "lora-deepseek-finetuned"
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
EPOCHS = 1
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100
MAX_SEQ_LENGTH = 512
LR_SCHEDULER_TYPE = "cosine"
SAVE_STEPS = 500
EVAL_STEPS = 50
SUBSET_SIZE = 20  # Number of problems for evaluation

def format_example(example):
    """Format a dataset example into instruction-following format."""
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
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

def tokenize_example(example, tokenizer, max_length):
    """Tokenize a formatted example."""
    text = format_example(example)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    return tokenized

def fine_tune_model(
    base_model_name=MODEL_NAME,
    dataset_name=DATASET_NAME,
    output_dir=OUTPUT_DIR,
    batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_seq_length=MAX_SEQ_LENGTH,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    seed=42,
    use_fp16=True
):
    """Fine-tune a model with LoRA."""
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Load tokenizer
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    tokenize_func = lambda example: tokenize_example(example, tokenizer, max_seq_length)
    train_dataset = dataset["train"].map(tokenize_func, batched=False)
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create validation set
    if "validation" in dataset:
        eval_dataset = dataset["validation"].map(tokenize_func, batched=False)
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        # Split off some portion of train if needed
        eval_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
    
    # Load base model
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map="auto"
    )
    
    # Apply LoRA
    print("Applying LoRA adapter")
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=10,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type,
        fp16=use_fp16,
        optim="adamw_torch",
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Define trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting fine-tuning...")
    train_result = trainer.train()
    
    # Save final model
    print(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Plot training results
    print("Generating training plots...")
    
    # Extract training loss values
    train_loss_values = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    train_steps = list(range(len(train_loss_values)))
    
    # Extract validation loss values
    eval_loss_values = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]
    eval_steps = [i for i, log in enumerate(trainer.state.log_history) if "eval_loss" in log]
    
    # Training Loss Plot
    plt.figure()
    plt.plot(train_steps, train_loss_values, label="Training Loss", color="blue")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_loss_plot.png"))
    plt.close()
    
    # Validation Loss Plot
    if eval_loss_values:
        plt.figure()
        plt.plot(eval_steps, eval_loss_values, label="Validation Loss", color="orange", marker='o')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Validation Loss over Time")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "validation_loss_plot.png"))
        plt.close()
    
    print(f"Fine-tuning complete. Model saved to {output_dir}")
    return output_dir

def evaluate_on_livebench(
    base_model_name,
    model_path=None,
    output_dir=OUTPUT_DIR,
    subset_size=SUBSET_SIZE,
    num_samples=5,
    temperature=0.2,
    timeout=30,
    max_new_tokens=1024,
    seed=42
):
    """Evaluate models on LiveCodeBench."""
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load LiveCodeBench dataset
    try:
        print("Loading LiveCodeBench problems...")
        lcb_dataset = load_dataset("livecodebench/code_generation_lite", trust_remote_code=True, version_tag="release_v2")
        
        # Get problems from the dataset
        problems = []
        # Find problems in any available split
        for split in lcb_dataset:
            if len(lcb_dataset[split]) > 0:
                problems = list(lcb_dataset[split])
                print(f"Found {len(problems)} problems in '{split}' split")
                break
        
        # Sample a subset if needed
        if subset_size and len(problems) > subset_size:
            problems = random.sample(problems, subset_size)
            print(f"Selected {len(problems)} problems for evaluation")
        
    except Exception as e:
        print(f"Error loading LiveCodeBench: {e}")
        print("Cannot proceed without LiveCodeBench dataset")
        return
    
    # Initialize results dictionary
    results = {
        "base_model": {
            "model_name": base_model_name,
            "pass@1": 0.0,
            "pass@k": 0.0,
            "problems_evaluated": len(problems)
        },
        "lora_model": {
            "model_name": base_model_name,
            "model_path": model_path,
            "pass@1": 0.0,
            "pass@k": 0.0,
            "problems_evaluated": len(problems)
        }
    }
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Evaluate base model
    print(f"Evaluating base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.eval()
    
    base_results = evaluate_model_on_problems(
        base_model,
        tokenizer,
        problems,
        num_samples=num_samples,
        temperature=temperature,
        timeout=timeout,
        max_new_tokens=max_new_tokens
    )
    
    results["base_model"].update(base_results)
    
    # Free memory
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate fine-tuned model if path provided
    if model_path:
        print(f"Evaluating LoRA model from {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        lora_model = PeftModel.from_pretrained(base_model, model_path)
        lora_model.eval()
        
        lora_results = evaluate_model_on_problems(
            lora_model,
            tokenizer,
            problems,
            num_samples=num_samples,
            temperature=temperature,
            timeout=timeout,
            max_new_tokens=max_new_tokens
        )
        
        results["lora_model"].update(lora_results)
        
        # Generate comparison
        generate_comparison(
            results["base_model"],
            results["lora_model"],
            output_dir,
            num_samples
        )
        
        # Free memory
        del lora_model
        del base_model
        torch.cuda.empty_cache()
    
    # Save results
    results_file = os.path.join(output_dir, "livebench_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {results_file}")
    return results

def evaluate_model_on_problems(
    model,
    tokenizer,
    problems,
    num_samples=5,
    temperature=0.2,
    timeout=30,
    max_new_tokens=1024
):
    """Evaluate a model on the given problems."""
    results = {
        "problem_results": [],
        "correct_count": 0
    }
    
    # Track metrics
    total_correct_at_1 = 0
    total_correct_at_k = 0
    
    for i, problem in enumerate(tqdm(problems, desc="Evaluating problems")):
        problem_id = problem.get("question_id", f"prob_{i}")
        problem_title = problem.get("title", f"Problem {i}")
        
        print(f"\nProblem {i+1}/{len(problems)}: {problem_title}")
        
        try:
            # Generate prompt from the problem
            prompt = problem.get("prompt", "")
            if not prompt:
                # If prompt not available, generate a simple one
                title = problem.get("title", "")
                description = problem.get("description", "")
                examples = problem.get("examples", [])
                
                # Format examples as string
                examples_text = ""
                for j, example in enumerate(examples):
                    examples_text += f"Example {j+1}:\n"
                    examples_text += f"Input: {example.get('input', '')}\n"
                    examples_text += f"Output: {example.get('output', '')}\n\n"
                
                prompt = f"""Below is a coding problem. Write a Python function to solve it.

Problem: {title}

Description: {description}

{examples_text}
Write your solution:

def """
            
            # Generate solutions
            solutions = []
            for sample_idx in range(num_samples):
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if sample_idx > 0 else 0.0,  # First one is greedy
                        top_p=0.95,
                        do_sample=sample_idx > 0  # Only sample after first solution
                    )
                
                # Decode
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract code after prompt
                if generated_text.startswith(prompt):
                    solution = generated_text[len(prompt):].strip()
                else:
                    solution = generated_text.strip()
                
                # Ensure solution starts with "def"
                if not solution.startswith("def "):
                    solution = "def " + solution
                
                solutions.append(solution)
            
            # Evaluate solutions
            solution_results = []
            correct_count = 0
            
            for sample_idx, solution in enumerate(solutions):
                print(f"  Evaluating solution {sample_idx+1}/{num_samples}...")
                is_correct = evaluate_solution(solution, problem, timeout)
                solution_results.append({"correct": is_correct})
                
                if is_correct:
                    correct_count += 1
                    print(f"  ✓ Solution {sample_idx+1} is correct")
                else:
                    print(f"  ✗ Solution {sample_idx+1} is incorrect")
            
            # Update metrics
            pass_at_1 = 1 if solution_results[0]["correct"] else 0
            pass_at_k = 1 if correct_count > 0 else 0
            
            total_correct_at_1 += pass_at_1
            total_correct_at_k += pass_at_k
            
            # Record problem results
            problem_result = {
                "id": problem_id,
                "title": problem_title,
                "pass@1": pass_at_1,
                "pass@k": pass_at_k,
                "solutions": solution_results
            }
            results["problem_results"].append(problem_result)
            
        except Exception as e:
            print(f"Error on problem {problem_id}: {e}")
            continue
    
    # Calculate final metrics
    if len(problems) > 0:
        results["pass@1"] = total_correct_at_1 / len(problems)
        results["pass@k"] = total_correct_at_k / len(problems)
        results["correct_count"] = total_correct_at_1
    
    print(f"Pass@1: {results['pass@1']:.4f}")
    print(f"Pass@{num_samples}: {results['pass@k']:.4f}")
    
    return results

def evaluate_solution(solution, problem, timeout=30):
    """Evaluate a solution against LiveCodeBench test cases."""
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w+", delete=False) as f:
        f.write(solution)
        f.write("\n\n# Test cases\n")
        
        # Extract function name
        import re
        match = re.search(r"def\s+(\w+)\s*\(", solution)
        if not match:
            return False
        
        func_name = match.group(1)
        
        # Add test cases
        for i, test_case in enumerate(problem.get("test_cases", [])):
            try:
                # Parse input and expected output from LiveCodeBench format
                tc_input = test_case.get("input", "")
                tc_output = test_case.get("output", "")
                
                # Write test code
                f.write(f"try:\n")
                f.write(f"    test_result = {func_name}({tc_input})\n")
                f.write(f"    expected = {tc_output}\n")
                f.write(f"    assert str(test_result) == str(expected), ")
                f.write(f"f\"Test {i+1} failed: {{test_result}} != {{expected}}\"\n")
                f.write(f"    print(f\"Test {i+1} passed\")\n")
                f.write(f"except Exception as e:\n")
                f.write(f"    print(f\"Test {i+1} failed: {{e}}\")\n")
                f.write(f"    raise\n")
            except Exception as e:
                print(f"Error setting up test case {i}: {e}")
                continue
        
        temp_file = f.name
    
    try:
        # Run the solution
        result = subprocess.run(
            ["python", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Solution is correct if all tests pass (return code 0)
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        print(f"Evaluation timed out after {timeout} seconds")
        return False
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass

def generate_comparison(base_results, lora_results, output_dir, num_samples=5):
    """Generate comparison visualizations between base and LoRA model."""
    # Prepare data
    models = ["Base Model", "LoRA Model"]
    pass_at_1 = [base_results["pass@1"], lora_results["pass@1"]]
    pass_at_k = [base_results["pass@k"], lora_results["pass@k"]]
    
    # Calculate improvements
    abs_improve_1 = lora_results["pass@1"] - base_results["pass@1"]
    rel_improve_1 = (abs_improve_1 / base_results["pass@1"]) * 100 if base_results["pass@1"] > 0 else float('inf')
    
    abs_improve_k = lora_results["pass@k"] - base_results["pass@k"]
    rel_improve_k = (abs_improve_k / base_results["pass@k"]) * 100 if base_results["pass@k"] > 0 else float('inf')
    
    # Pass@1 plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, pass_at_1, color=["gray", "blue"])
    plt.title("LiveCodeBench Pass@1 Comparison")
    plt.ylabel("Pass@1 Score")
    plt.ylim(0, max(1.0, max(pass_at_1) * 1.2))
    
    # Add value labels
    for i, v in enumerate(pass_at_1):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
    
    # Add improvement annotation
    if abs_improve_1 > 0:
        plt.annotate(
            f"+{abs_improve_1:.4f} ({rel_improve_1:.1f}%)",
            xy=(1, pass_at_1[1]),
            xytext=(0.5, (pass_at_1[0] + pass_at_1[1])/2),
            arrowprops=dict(facecolor='green', shrink=0.05, width=2),
            ha='center',
            fontweight='bold',
            color='green'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pass_at_1_comparison.png"))
    plt.close()
    
    # Pass@k plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, pass_at_k, color=["gray", "blue"])
    plt.title(f"LiveCodeBench Pass@{num_samples} Comparison")
    plt.ylabel(f"Pass@{num_samples} Score")
    plt.ylim(0, max(1.0, max(pass_at_k) * 1.2))
    
    # Add value labels
    for i, v in enumerate(pass_at_k):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
    
    # Add improvement annotation
    if abs_improve_k > 0:
        plt.annotate(
            f"+{abs_improve_k:.4f} ({rel_improve_k:.1f}%)",
            xy=(1, pass_at_k[1]),
            xytext=(0.5, (pass_at_k[0] + pass_at_k[1])/2),
            arrowprops=dict(facecolor='green', shrink=0.05, width=2),
            ha='center',
            fontweight='bold',
            color='green'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pass_at_{num_samples}_comparison.png"))
    plt.close()
    
    # Print summary
    print("\n----- COMPARISON RESULTS -----")
    print(f"Base Model Pass@1: {pass_at_1[0]:.4f}")
    print(f"LoRA Model Pass@1: {pass_at_1[1]:.4f}")
    print(f"Absolute improvement: +{abs_improve_1:.4f}")
    print(f"Relative improvement: +{rel_improve_1:.1f}%")
    print()
    print(f"Base Model Pass@{num_samples}: {pass_at_k[0]:.4f}")
    print(f"LoRA Model Pass@{num_samples}: {pass_at_k[1]:.4f}")
    print(f"Absolute improvement: +{abs_improve_k:.4f}")
    print(f"Relative improvement: +{rel_improve_k:.1f}%")

def run_end_to_end(args):
    """Run the complete fine-tuning and evaluation pipeline."""
    # Step 1: Fine-tune the model using LoRA
    print("\n=== Step 1: Fine-tuning the model with LoRA ===\n")
    model_path = fine_tune_model(
        base_model_name=args.base_model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_seq_length=args.max_seq_length,
        lr_scheduler_type=args.lr_scheduler_type,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        use_fp16=not args.no_fp16
    )
    
    # Step 2: Evaluate on LiveCodeBench
    print("\n=== Step 2: Evaluating on LiveCodeBench ===\n")
    results = evaluate_on_livebench(
        base_model_name=args.base_model,
        model_path=model_path,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        timeout=args.timeout,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed
    )
    
    return results

def main():
    """Parse arguments and run script."""
    parser = argparse.ArgumentParser(description="Fine-tune with LoRA and evaluate on LiveCodeBench")
    
    # General arguments
    parser.add_argument("--base_model", type=str, default=MODEL_NAME,
                      help=f"Base model name (default: {MODEL_NAME})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                      help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed (default: 42)")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="end_to_end", 
                      choices=["end_to_end", "fine_tune_only", "evaluate_only"],
                      help="Operation mode (default: end_to_end)")
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to fine-tuned model (for evaluate_only mode)")
    
    # Fine-tuning parameters
    parser.add_argument("--dataset", type=str, default=DATASET_NAME,
                      help=f"Dataset name for fine-tuning (default: {DATASET_NAME})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                      help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS,
                      help=f"Gradient accumulation steps (default: {GRADIENT_ACCUMULATION_STEPS})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                      help=f"Number of epochs (default: {EPOCHS})")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                      help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS,
                      help=f"Warmup steps (default: {WARMUP_STEPS})")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH,
                      help=f"Maximum sequence length (default: {MAX_SEQ_LENGTH})")
    parser.add_argument("--lr_scheduler_type", type=str, default=LR_SCHEDULER_TYPE,
                      help=f"Learning rate scheduler type (default: {LR_SCHEDULER_TYPE})")
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS,
                      help=f"Save checkpoint every X steps (default: {SAVE_STEPS})")
    parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS,
                      help=f"Evaluate every X steps (default: {EVAL_STEPS})")
    parser.add_argument("--no_fp16", action="store_true",
                      help="Disable mixed precision training")
    
    # Evaluation parameters
    parser.add_argument("--subset_size", type=int, default=SUBSET_SIZE,
                      help=f"Number of problems to evaluate (default: {SUBSET_SIZE})")
    parser.add_argument("--temperature", type=float, default=0.2,
                      help="Temperature for generation (default: 0.2)")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples per problem for Pass@k (default: 5)")
    parser.add_argument("--timeout", type=int, default=30,
                      help="Timeout for code execution in seconds (default: 30)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                      help="Maximum number of new tokens to generate (default: 1024)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in the specified mode
    if args.mode == "fine_tune_only":
        fine_tune_model(
            base_model_name=args.base_model,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_seq_length=args.max_seq_length,
            lr_scheduler_type=args.lr_scheduler_type,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
            use_fp16=not args.no_fp16
        )
    
    elif args.mode == "evaluate_only":
        if not args.model_path:
            parser.error("--model_path is required for evaluate_only mode")
        
        evaluate_on_livebench(
            base_model_name=args.base_model,
            model_path=args.model_path,
            output_dir=args.output_dir,
            subset_size=args.subset_size,
            num_samples=args.num_samples,
            temperature=args.temperature,
            timeout=args.timeout,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed
        )
    
    else:  # end_to_end
        run_end_to_end(args)

if __name__ == "__main__":
    main()
