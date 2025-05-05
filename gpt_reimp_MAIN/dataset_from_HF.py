from datasets import load_dataset
#
# use name="sample-10BT" to use the 10BT sample

#
# Load the dataset
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True)

# Open a file to write the entries
with open("input.txt", "w", encoding="utf-8") as f:
    for i, entry in enumerate(dataset):
        if i >= 3000:
            break
        # Assuming each entry has a 'text' field
        f.write(entry["text"] + "\n")

print("First 3,000 entries have been saved to input.txt")
