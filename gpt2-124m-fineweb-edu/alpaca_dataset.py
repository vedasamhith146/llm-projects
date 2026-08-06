from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Load your dataset (replace with your actual dataset loading code)
# For example: dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = load_dataset("tatsu-lab/alpaca",split="train")# <-- PUT YOUR DATASET VARIABLE HERE

# 3. Define the formatting function
def get_length(example):
    # This must match EXACTLY what you will feed the SFTTrainer
    text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"length": len(tokenizer(text)["input_ids"])}

# 4. Map the function to the dataset (this might take 10-30 seconds)
print("Calculating token lengths...")
dataset_with_lengths = dataset.map(get_length)

# 5. Extract all lengths and calculate statistics
lengths = dataset_with_lengths["length"]

print("\n--- DATASET LENGTH STATISTICS ---")
print(f"Total examples: {len(lengths)}")
print(f"Minimum length: {min(lengths)} tokens")
print(f"Maximum length: {max(lengths)} tokens")
print(f"Mean (Average) length: {np.mean(lengths):.1f} tokens")
print(f"Median length: {np.median(lengths):.1f} tokens")
print(f"90th Percentile: {np.percentile(lengths, 90):.1f} tokens")
print(f"95th Percentile: {np.percentile(lengths, 95):.1f} tokens")
print("---------------------------------")

# 6. See how many you lose if you cap at 512
over_512 = sum(1 for l in lengths if l > 512)
print(f"\nIf you set max_seq_length=512, you will truncate/drop {over_512} examples ({(over_512/len(lengths))*100:.2f}% of the dataset).")
    