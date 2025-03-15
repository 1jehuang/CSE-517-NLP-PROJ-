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
import pickle
from transformers import pipeline


def load_dataset(file_path, dataset_name, limit=150):
    """Load samples from various datasets with appropriate format handling"""
    print(f"Loading {dataset_name} dataset from {file_path}...")
    data = []

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
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
                    print(f"Warning: Unexpected JSON structure in {file_path}")
                    data = [json_data]  # Just use the whole object as one sample

    print(f"Loaded {len(data)} samples from {dataset_name}.")
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

def run_inference(pipe, prompt, device="cuda"):
    """Run inference on the model"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]
    # add chat template
    # messages = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
    outputs = pipe(
        messages,
        max_new_tokens=2048,
        temperature=0.2,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    # print(outputs)
    return outputs[0]["generated_text"][-1]['content']

    # # Tokenize input with attention mask
    # encoding = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128000)

    # inputs = {
    #     'input_ids': encoding.input_ids.to(device),
    #     'attention_mask': encoding.attention_mask.to(device)
    # }

    # with torch.no_grad():
    #     outputs = model.generate(
    #         input_ids=inputs['input_ids'],
    #         attention_mask=inputs['attention_mask'],
    #         max_new_tokens=2048,  # Adjust based on desired output length
    #         temperature=0.2,      # Lower temperature for focused output
    #         do_sample=True,
    #     )

    # return tokenizer.decode(outputs[0][encoding.shape[1]:], skip_special_tokens=True)

import re
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
        return ""

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



def generate_reasoning_evaluate_prompt(question, y):
    prompt = f"""
Question: {question}
Reasoning: {y}
Evaluate if the given reasoning can lead to the right solution to the question. Choose one word from (correct, likely, impossible) to indicate your evaluation of the reasoning's quality. Do not output anything else.
Evaluation (choose from correct/likely/impossible): """
    return prompt

def get_new_ys(x, ys, step, n_generate_sample, pipe):
    '''
    x is the question prompt
    ys is the current output candidates (from step 1 to step {step-1})
    step is the current step
    n_generate_sample is the number of new output candidates to generate
    return the new output candidates (from step 1 to step {step})
    '''
    new_ys = []
    for y in ys:
        for _ in range(n_generate_sample):
            prompt = x + y + "\nStep " + str(step) + ", "
            # print(prompt)
            new_y = run_inference(pipe, prompt)
            # clean new_y and only get the thought of this step
            ## cut if there's Step {step+1} or The answer is
            new_y = new_y.split("Step " + str(step+1))[0]
            # new_y = new_y.split("The answer is ")[0]
            new_ys.append(y + "\nStep " + str(step) + ", " + new_y)
    return new_ys


def get_new_ys_last_step(x, ys, n_generate_sample, pipe):
    '''
    x is the question prompt
    ys is the current output candidates (from step 1 to step {step-1})
    step is the current step
    n_generate_sample is the number of new output candidates to generate
    return the new output candidates (from step 1 to step {step})
    '''
    new_ys = []
    for y in ys:
        for _ in range(n_generate_sample):
            prompt = x + y + "\nANSWER: "
            # print(prompt)
            new_y = run_inference(pipe, prompt)
            # clean new_y and only get the thought of this step
            ## cut if there's Step {step+1} or The answer is

            new_ys.append(y + "\nANSWER: " + new_y)
    return new_ys


def evaluate_state(pipe, x, y):
    prompt = generate_reasoning_evaluate_prompt(x, y)
    response = run_inference(pipe, prompt).lower()
    # print(response)
    if 'impossible' in response or 'incorrect' in response or 'unlikely' in response or 'never' in response:
        return 0
    elif 'likely' in response:
        return 0.5
    elif 'correct' in response:
        return 1
    else:
        return 0.01

def solve(pipe, prompt, n_generate_sample=6, n_select_sample=3, depth=3, verbose=False):
    x = prompt
    ys = ["", ""]
    infos = []
    for step in range(1, depth + 1):
        if step < depth:
            new_ys = get_new_ys(x, ys, step, n_generate_sample, pipe)
        else:
            new_ys = get_new_ys_last_step(x, ys, n_generate_sample, pipe)
        if verbose:
            print(f"Step {step}: {new_ys}")
        ids = list(range(len(new_ys)))
        
        # evaluation
        values = [evaluate_state(pipe, x, y) for y in new_ys]
        if np.sum(values) == 0:
            values = [v + 1e-6 for v in values]
        # print(values)
        # select top n_select_sample
        ps = np.array(values) / np.sum(values)
        ## if the number of impossible is more than n_generate_sample - n_select_sample, we sample with replacement
        if step == depth:
            n_select_sample = 1
        if np.sum(np.array(values) == 0) > n_generate_sample - n_select_sample:
            # values = np.array(values) + 1e-6
            values = [v + 1e-6 for v in values]
            ps = np.array(values) / np.sum(values)
            selected_ids = np.random.choice(ids, size=n_select_sample, p=ps, replace=True)
        else:
            selected_ids = np.random.choice(ids, size=n_select_sample, p=ps, replace=False)
        select_new_ys = [new_ys[i] for i in selected_ids]
        
        # log
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        if verbose:
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
        
    return ys[0], infos    

def evaluate_dataset(pipe, data, dataset_name, verbose=False, n_generate_sample=6, n_select_sample=3, depth=3):
    """Evaluate model on a specific dataset"""
    print(f"Starting evaluation of {dataset_name} dataset...")

    results = []
    start_time = time.time()

    for i, sample in enumerate(data):
        # Create appropriate prompt
        prompt = create_prompt(sample, dataset_name)

        # Print the prompt being sent to the model
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{len(data)} - DATASET: {dataset_name}")
        print(f"{'-'*80}")
        print(f"PROMPT:")
        print(f"{prompt}")
        print(f"{'-'*80}")

        # Run inference
        output, infos = solve(pipe, prompt, n_generate_sample=n_generate_sample, n_select_sample=n_select_sample, depth=depth, verbose=verbose)

        # Print the model's response
        print(f"MODEL RESPONSE:")
        print(f"{output}")

        # Extract prediction and ground truth
        prediction = extract_prediction(output, dataset_name, sample)
        true_label = get_ground_truth(sample, dataset_name)

        # Evaluate correctness
        correct = evaluate_correctness(prediction, true_label, dataset_name)

        print(f"{'-'*80}")
        print(f"PREDICTION: {prediction}")
        print(f"TRUE LABEL: {true_label}")
        print(f"CORRECT: {correct}")
        print(f"{'='*80}")

        temp_time = time.time()
        elapsed = temp_time - start_time
        avg_time = elapsed / (i+1)
        # Store result
        result = {
            "input": prompt,
            "output": output,
            "prediction": prediction,
            "true_label": true_label,
            "correct": correct,
            "infos": infos,
            "avg_time": avg_time
        }
        results.append(result)

        # Print progress
        if (i+1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i+1)
            remaining = avg_time * (len(data) - i - 1)
            print(f"Processed {i+1}/{len(data)} samples - "
                  f"Avg time per sample: {avg_time:.2f}s - "
                  f"Estimated time remaining: {remaining/60:.1f} minutes")


    # Calculate overall accuracy
    accuracy = sum(r['correct'] for r in results) / len(results)

    print(f"\nResults for {dataset_name}:")
    print(f"Overall accuracy: {accuracy:.2f}")
    
    latency = results[-1]['avg_time']
    print(f"Average latency: {latency:.2f}s")

    # Save results
    output_dir = f"results_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # save to pickle
    with open(f"{output_dir}/results_tot.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # save to json
    with open(f"{output_dir}/results_tot.json", "w") as f:
        json.dump(results, f, indent=2)


    print(f"Results saved to {output_dir}/")

    return accuracy, results



import argparse
parser = argparse.ArgumentParser(description='Run inference on a dataset')
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help='Model name or path')
# for each dataset there's a parameter for action
parser.add_argument('--vitaminc', action='store_true', help='Evaluate on VitaminC dataset')
parser.add_argument('--fever', action='store_true', help='Evaluate on FEVER dataset')
parser.add_argument('--feverous', action='store_true', help='Evaluate on FEVEROUS dataset')
parser.add_argument('--hotpotqa', action='store_true', help='Evaluate on HotpotQA dataset')
parser.add_argument('--wikimultihopqa', action='store_true', help='Evaluate on 2WikiMultihopQA dataset')
parser.add_argument('--svamp', action='store_true', help='Evaluate on SVAMP dataset')
parser.add_argument('--bamboogle', action='store_true', help='Evaluate on Bamboogle dataset')
# device
parser.add_argument('--device', type=str, default="cuda:0", help='Device to run inference on')
args = parser.parse_args()

# Load model and tokenizer
print("Loading model and tokenizer...")
model_path = "meta-llama/Llama-3.2-1B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
pipe = pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map=args.device)

# # Ensure the tokenizer has a pad_token
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
#     model.resize_token_embeddings(len(tokenizer))

print(f"Model loaded successfully")

# try a simple prompt
simple_prompt = "What is the sum of 2 and 3?"
simple_output = run_inference(pipe, simple_prompt)
print(f"Simple prompt: {simple_prompt}")
print(f"Simple output: {simple_output}")


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
print(f"Available datasets for evaluation:")
for i, dataset_name in enumerate(datasets.keys()):
    print(f"{i+1}. {dataset_name}")

datasets_to_evaluate = []
if args.vitaminc:
    datasets_to_evaluate.append("VitaminC")
if args.fever:
    datasets_to_evaluate.append("FEVER")
if args.feverous:
    datasets_to_evaluate.append("FEVEROUS")
if args.hotpotqa:
    datasets_to_evaluate.append("HotpotQA")
if args.svamp:
    datasets_to_evaluate.append("SVAMP")
if args.bamboogle:
    datasets_to_evaluate.append("Bamboogle")
if args.wikimultihopqa:
    datasets_to_evaluate.append("2WikiMultihopQA")
    

# datasets_to_evaluate = [
#     # "VitaminC",
#     # "FEVER",
#     "FEVEROUS",
#     "HotpotQA",
#     "2WikiMultihopQA",
#     "SVAMP",
#     "Bamboogle"
# ]  # Evaluate all available datasets
# Initialize results summary
summary = {}

# Process each selected dataset
for dataset_name in datasets_to_evaluate:
    if dataset_name not in datasets:
        print(f"Dataset {dataset_name} not found in available datasets")
        continue

    # Load dataset
    file_path = datasets[dataset_name]
    data = load_dataset(file_path, dataset_name, limit=150)  # Limit to 150 samples

    if data:
        # Evaluate on the dataset
        accuracy, _ = evaluate_dataset(pipe, data, dataset_name, verbose=False, n_generate_sample=4, n_select_sample=2, depth=3)
        summary[dataset_name] = accuracy

# Print summary of results
print(f"\nSummary of Results:")
for dataset, accuracy in summary.items():
    print(f"{dataset}: {accuracy:.4f}")
