import json
import time
import re
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from google.colab import drive
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import math

# ANSI color codes for terminal output
BLUE = '\033[94m'     # For sample information
GREEN = '\033[92m'    # For correct predictions and success messages
RED = '\033[91m'      # For incorrect predictions
YELLOW = '\033[93m'   # For predictions and headers
CYAN = '\033[96m'     # For progress information
PURPLE = '\033[95m'   # For model responses
BOLD = '\033[1m'      # Bold text
ENDC = '\033[0m'      # End color

def load_dataset(file_path, dataset_name, limit=150):
    """Load samples from various datasets with appropriate format handling"""
    print(f"{CYAN}Loading {dataset_name} dataset from {file_path}...{ENDC}")
    data = []

    if not os.path.exists(file_path):
        print(f"{RED}File not found: {file_path}{ENDC}")
        return []

    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
                    if len(data) >= limit:
                        break
    else:  # .json files
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if isinstance(json_data, list):
                data = json_data[:limit]
            else:
                # Handle nested structures if needed
                if 'data' in json_data:
                    data = json_data['data'][:limit]
                else:
                    print(f"{YELLOW}Warning: Unexpected JSON structure in {file_path}{ENDC}")
                    data = [json_data]  # Just use the whole object as one sample

    print(f"{CYAN}Loaded {len(data)} samples from {dataset_name}.{ENDC}")
    return data

def create_prompt(sample, dataset_name):
    """Create appropriate prompts based on dataset type"""

    if dataset_name == "VitaminC" or dataset_name == "FEVER" or dataset_name == "FEVEROUS":
        # Fact verification datasets
        if dataset_name == "VitaminC":
            claim = sample["claim"]
            evidence = sample["evidence"]
        elif dataset_name == "FEVER":
            claim = sample["claim"]
            evidence = sample.get("evidence", "") or sample.get("context", "")
        elif dataset_name == "FEVEROUS":
            claim = sample["claim"]
            evidence = sample.get("evidence", "") or sample.get("context", "")

        return create_fact_verification_prompt(claim, evidence)

    elif dataset_name == "HotpotQA" or dataset_name == "2WikiMultihopQA":
        # Multi-hop QA datasets
        question = sample.get("question", "") or sample.get("query", "")
        context = sample.get("context", "")
        if not context and "original_context" in sample:
            context = sample["original_context"]

        return create_qa_prompt(question, context)

    elif dataset_name == "SVAMP":
        # Math word problem dataset
        question = sample.get("question", "") or sample.get("body", "")
        return create_math_prompt(question)

    elif dataset_name == "Bamboogle":
        # Bamboogle dataset - assume it's a QA task
        question = sample.get("question", "")
        context = sample.get("context", "")

        if not question and "answer" in sample:
            # If no question is provided but there's an answer, create a generic prompt
            return

        return create_qa_prompt(question, context)

    else:
        # Generic prompt for unknown datasets
        return f"Please analyze this data and think step-by-step:\n\n{json.dumps(sample, indent=2)}\n\nAfter your thinking, provide your answer in a latex boxed format:\n$\\boxed{{<finalAnswer>}}$"

def create_fact_verification_prompt(claim, evidence):
    """Create a prompt for fact checking"""
    prompt = f"""Claim: {claim}

Evidence: {evidence}

Think step-by-step to determine if the evidence SUPPORTS, REFUTES, or provides NOT ENOUGH INFO for the claim.

After your thinking, end your answer with one of these in a latex boxed format:
$\\boxed{{SUPPORTS}}$ or $\\boxed{{REFUTES}}$ or $\\boxed{{NOT ENOUGH INFO}}$

Always end with your answer in a latex boxed format:
$\\boxed{{<finalAnswer>}}$"""
    return prompt

def create_qa_prompt(question, context):
    """Create a prompt for question answering tasks"""
    prompt = f"""

Question: {question}

Think step-by-step to answer the question based on the context.

After your thinking, provide your final answer in a latex boxed format:
$\\boxed{{<finalAnswer>}}$
"""
    return prompt

def create_math_prompt(question):
    """Create a prompt for math word problems"""
    prompt = f"""Problem: {question}

Think step-by-step to solve this math problem.

After your thinking, provide your final numeric answer in a latex boxed format:
$\\boxed{{<finalAnswer>}}$
"""
    return prompt

def run_inference(model, tokenizer, prompt, device="cuda"):
    """Generate response using the chat template format"""
    # Create a messages array with user prompt
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Apply the chat template
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenized_chat,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=True,
        )

    # Decode the response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response by removing the prompt
    # This depends on how the specific model formats its responses
    # You might need to adjust this based on the model's output format
    response_parts = full_response.split("Assistant: ")
    if len(response_parts) > 1:
        return response_parts[-1].strip()  # Get the assistant's response
    else:
        return full_response  # Return full response if we can't split

    return response

import re

def extract_prediction(output, dataset_name, sample=None):
    """
    Extract the final boxed answer for any dataset.
    If we only find an empty \boxed{}, treat it as no valid box.
    """
    # For now, ignoring dataset_name / sample, just returning the last \boxed{...} if not empty
    pattern = r'\\boxed\s*\{([^}]*)\}'
    all_matches = re.findall(pattern, output)

    # Filter out empty or whitespace-only matches
    non_empty_matches = [m.strip() for m in all_matches if m.strip()]

    # If no valid non-empty box found, return None
    if not non_empty_matches:
        return None

    # Return the last valid boxed expression
    return non_empty_matches[-1]

def evaluate_correctness(prediction, ground_truth, dataset_name):
    """Evaluate if the prediction is correct based on dataset type"""

    if dataset_name == "VitaminC" or dataset_name == "FEVER" or dataset_name == "FEVEROUS":
        # Fact verification - direct comparison
        return prediction == ground_truth

    elif dataset_name == "HotpotQA" or dataset_name == "2WikiMultihopQA" or dataset_name == "Bamboogle":
        # QA evaluation - normalize and compare
        return normalize_qa_answers(prediction, ground_truth)

    elif dataset_name == "SVAMP":
        # Math evaluation - numeric comparison
        return evaluate_math_correctness(prediction, ground_truth)

    else:
        # Generic comparison for unknown datasets
        return prediction == ground_truth

def normalize_qa_answers(prediction, ground_truth):
    """Normalize and compare QA answers with flexible matching"""
    if not prediction or not ground_truth:
        return False

    # Handle list or dictionary ground truths
    if isinstance(ground_truth, list):
        ground_truth = " ".join([str(item) for item in ground_truth])
    elif isinstance(ground_truth, dict):
        if "answer" in ground_truth:
            ground_truth = ground_truth["answer"]
        else:
            ground_truth = str(ground_truth)

    # Normalize both strings
    pred_norm = prediction.lower().strip()
    truth_norm = str(ground_truth).lower().strip()

    # Remove punctuation and extra spaces
    pred_norm = re.sub(r'[^\w\s]', '', pred_norm).strip()
    truth_norm = re.sub(r'[^\w\s]', '', truth_norm).strip()

    # Check if prediction contains ground truth or vice versa
    return pred_norm in truth_norm or truth_norm in pred_norm

def evaluate_math_correctness(prediction, ground_truth):
    """Evaluate correctness of math answers with tolerance"""
    try:
        # Convert to numeric values
        if isinstance(prediction, str):
            prediction = float(re.search(r'(-?[\d.]+)', prediction.replace(',', '')).group(1))

        if isinstance(ground_truth, str):
            ground_truth = float(re.search(r'(-?[\d.]+)', ground_truth.replace(',', '')).group(1))

        # Compare with tolerance
        tolerance = 0.01
        return abs(float(prediction) - float(ground_truth)) < tolerance
    except (ValueError, TypeError, AttributeError):
        return False

def get_ground_truth(sample, dataset_name):
    """Extract ground truth from sample based on dataset type"""

    if dataset_name == "VitaminC":
        return sample.get("label", "")

    elif dataset_name == "FEVER" or dataset_name == "FEVEROUS":
        return sample.get("label", "")

    elif dataset_name == "HotpotQA" or dataset_name == "2WikiMultihopQA":
        return sample.get("answer", "")

    elif dataset_name == "SVAMP":
        return sample.get("answer", None)

    elif dataset_name == "Bamboogle":
        return sample.get("answer", "")

    else:
        return None
    
def evaluate_dataset(model, tokenizer, data, dataset_name):
    """Evaluate model on a specific dataset"""
    print(f"{BOLD}{CYAN}Starting evaluation of {dataset_name} dataset...{ENDC}")

    results = []
    start_time = time.time()

    for i, sample in enumerate(data):
        # Create appropriate prompt
        prompt = create_prompt(sample, dataset_name)

        # Print the prompt being sent to the model
        print(f"\n{BOLD}{'='*80}{ENDC}")
        print(f"{BLUE}{BOLD}SAMPLE {i+1}/{len(data)} - DATASET: {dataset_name}{ENDC}")
        print(f"{BLUE}{'-'*80}{ENDC}")
        print(f"{GREEN}PROMPT:{ENDC}")
        print(f"{GREEN}{prompt}{ENDC}")
        print(f"{BLUE}{'-'*80}{ENDC}")

        # Run inference
        output = run_inference(model, tokenizer, prompt)

        # Print the model's response
        print(f"{PURPLE}MODEL RESPONSE:{ENDC}")
        print(f"{PURPLE}{output}{ENDC}")

        # Extract prediction and ground truth
        prediction = extract_prediction(output, dataset_name, sample)
        true_label = get_ground_truth(sample, dataset_name)

        # Evaluate correctness
        correct = evaluate_correctness(prediction, true_label, dataset_name)
        correct_color = GREEN if correct else RED

        print(f"{BLUE}{'-'*80}{ENDC}")
        print(f"{YELLOW}PREDICTION: {prediction}{ENDC}")
        print(f"{YELLOW}TRUE LABEL: {true_label}{ENDC}")
        print(f"{correct_color}CORRECT: {correct}{ENDC}")
        print(f"{BOLD}{'='*80}{ENDC}")

        # Store result
        result = {
            "input": prompt,
            "output": output,
            "prediction": prediction,
            "true_label": true_label,
            "correct": correct
        }
        results.append(result)

        # Print progress
        if (i+1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i+1)
            remaining = avg_time * (len(data) - i - 1)
            print(f"{CYAN}Processed {i+1}/{len(data)} samples - "
                  f"Avg time per sample: {avg_time:.2f}s - "
                  f"Estimated time remaining: {remaining/60:.1f} minutes{ENDC}")
            results_df = pd.DataFrame(results)
            accuracy = results_df['correct'].mean()
            print(f"{BOLD}Current accuracy: {accuracy:.2f}{ENDC}")

    # Calculate overall accuracy
    results_df = pd.DataFrame(results)
    accuracy = results_df['correct'].mean()

    print(f"\n{BOLD}Results for {dataset_name}:{ENDC}")
    print(f"{BOLD}Overall accuracy: {accuracy:.2f}{ENDC}")

    # Save results


    # Define base output directory on Google Drive
    DRIVE_OUTPUT_DIR = "./SFT_test_results"
    output_dir = f"{DRIVE_OUTPUT_DIR}"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/{dataset_name}_results.csv", index=False)

    # Create visualizations
    # visualize_results(results_df, dataset_name)

    print(f"{GREEN}Results saved to {output_dir}/{ENDC}")

    # save dataset_name, accuracy, avg_time to a file
    with open(f"{output_dir}/summary.txt", "a") as f:
        f.write(f"{dataset_name}: {accuracy:.2f}\n")
        f.write(f"Avg time per sample: {avg_time:.2f}s\n\n")
    

    return accuracy, results_df

# Load model and tokenizer
print("Loading model and tokenizer...")
# model_path = "./checkpoint/llama3-1b-cpo-1epoch-mix_data"
model_path = "./checkpoint/llama3-1b-sft_mix_data"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# Ensure the tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

print(f"{GREEN}Model loaded successfully{ENDC}")

data_dir = "../data"

# Define dataset paths
datasets = {
    "VitaminC": f"{data_dir}/VitaminC/test.jsonl",
    "2WikiMultihopQA": f"{data_dir}/2WikiMultihopQA/truncated_first_150.json",
    "Bamboogle": f"{data_dir}/Bamboogle/test.json",
    "FEVER": f"{data_dir}/FEVER/fever_test.jsonl",
    "FEVEROUS": f"{data_dir}/FEVEROUS/feverous_test.jsonl",
    "HotpotQA": f"{data_dir}/HotpotQA/truncated_first_150.json",
    "SVAMP": f"{data_dir}/SVAMP/test.json"
}

# Display available datasets
print(f"{CYAN}Available datasets for evaluation:{ENDC}")
for i, dataset_name in enumerate(datasets.keys()):
    print(f"{i+1}. {dataset_name}")
    

# Choose which datasets to evaluate
# You can modify this list to evaluate specific datasets
# datasets_to_evaluate = ["VitaminC", "FEVER"]  # Uncomment to evaluate specific datasets
datasets_to_evaluate = [
    "VitaminC",
    # "FEVER",
    # "FEVEROUS",
    "HotpotQA",
    "2WikiMultihopQA",
    "SVAMP",
    "Bamboogle"
]  # Evaluate all available datasets
# Initialize results summary
summary = {}

# Process each selected dataset
for dataset_name in datasets_to_evaluate:
    if dataset_name not in datasets:
        print(f"{RED}Dataset {dataset_name} not found in available datasets{ENDC}")
        continue

    # Load dataset
    file_path = datasets[dataset_name]
    data = load_dataset(file_path, dataset_name, limit=150)  # Limit to 150 samples

    if data:
        # Evaluate on the dataset
        accuracy, _ = evaluate_dataset(model, tokenizer, data, dataset_name)
        summary[dataset_name] = accuracy

# print summary
print(summary)