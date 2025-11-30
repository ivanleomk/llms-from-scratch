import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import time


def text_iterator(directory_path):
    """
    An iterator that yields the content of each .txt file in a directory.
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            full_path = os.path.join(directory_path, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                yield f.read()


# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'data' directory, relative to the script's location
data_directory = os.path.join(script_dir, "data")

# 1. Initialize a new BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 2. Configure the trainer
# We set a vocabulary size and define the special tokens we want to use
trainer = BpeTrainer(vocab_size=1000, special_tokens=["<|endoftext|>"])


# 3. Train the tokenizer on our dataset iterator
start = time.time_ns()
tokenizer.train_from_iterator(text_iterator(data_directory), trainer=trainer)
end = time.time_ns()

# 4. Save the tokenizer's vocabulary and merge rules
output_dir = "bpe_tokenizer"
os.makedirs(output_dir, exist_ok=True)
tokenizer.model.save(output_dir)

print(f"Tokenizer trained and saved to '{output_dir}' directory.")
print(f"Training time: {(end - start) / 1_000_000:.2f} ms")
