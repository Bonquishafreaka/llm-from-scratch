from datasets import load_dataset

# Download the dataset
dataset = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k")

# See what's in it
print("Dataset structure:")
print(dataset)
print("\n----- ONE EXAMPLE -----")
print(dataset["train_sft"][0])