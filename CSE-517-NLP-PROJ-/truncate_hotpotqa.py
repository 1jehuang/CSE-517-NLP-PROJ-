import json
import random
import pandas as pd
import os
from google.colab import drive

def truncate_hotpotqa_dataset():
    """
    Load HotpotQA dataset from Google Drive, truncate to 150 samples, and save the truncated dataset.
    Provides both sequential and random sampling options.
    """
    print("Processing HotpotQA dataset...")

    # Mount Google Drive
    print("Mounting Google Drive...")
    drive.mount('/content/drive')

    # Path to the dataset
    file_path = '/content/drive/Shareddrives/517 nlp project/data/HotpotQA/dev.json'

    # Load the dataset
    print(f"Loading dataset from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Original dataset loaded: {len(data)} samples")

    # Display a sample to verify structure
    print("\nFirst item structure:")
    keys = data[0].keys()
    print(f"Keys in data: {list(keys)}")

    # Analyze question types if available
    if 'type' in data[0]:
        question_types = {}
        for item in data:
            q_type = item.get('type', 'unknown')
            if q_type not in question_types:
                question_types[q_type] = 0
            question_types[q_type] += 1

        print("\nQuestion types distribution:")
        for q_type, count in question_types.items():
            print(f"  {q_type}: {count}")

    # Create truncated versions
    truncated_first_150 = data[:150]

    # Create a randomly sampled version with fixed seed for reproducibility
    random.seed(42)
    truncated_random_150 = random.sample(data, min(150, len(data)))

    # Save paths
    output_dir = '/content/drive/Shareddrives/517 nlp project/data/HotpotQA'
    first_output_path = os.path.join(output_dir, 'truncated_first_150.json')
    random_output_path = os.path.join(output_dir, 'truncated_random_150.json')

    # Save the truncated datasets
    with open(first_output_path, 'w', encoding='utf-8') as f:
        json.dump(truncated_first_150, f, ensure_ascii=False, indent=2)

    with open(random_output_path, 'w', encoding='utf-8') as f:
        json.dump(truncated_random_150, f, ensure_ascii=False, indent=2)

    print("\nTruncated datasets saved to:")
    print(f"- {first_output_path}")
    print(f"- {random_output_path}")

    # Compare statistics on answer length
    def analyze_dataset(name, dataset):
        question_lengths = [len(item['question'].split()) for item in dataset]
        answer_lengths = [len(str(item['answer']).split()) for item in dataset]

        stats = {
            "avg_question_len": sum(question_lengths) / len(question_lengths),
            "max_question_len": max(question_lengths),
            "min_question_len": min(question_lengths),
            "avg_answer_len": sum(answer_lengths) / len(answer_lengths),
            "max_answer_len": max(answer_lengths),
            "min_answer_len": min(answer_lengths)
        }

        print(f"\n{name} Statistics:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.2f}")

        return stats

    # Analyze all datasets
    orig_stats = analyze_dataset("Original Dataset", data)
    first_stats = analyze_dataset("First 150 Samples", truncated_first_150)
    random_stats = analyze_dataset("Random 150 Samples", truncated_random_150)

    print("\nProcess completed successfully!")

    return {
        "original": data,
        "first_150": truncated_first_150,
        "random_150": truncated_random_150,
        "stats": {
            "original": orig_stats,
            "first_150": first_stats,
            "random_150": random_stats
        }
    }

if __name__ == "__main__":
    result = truncate_hotpotqa_dataset()
