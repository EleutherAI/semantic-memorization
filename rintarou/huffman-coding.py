from datasets import load_dataset
import numpy as np
from collections import Counter
import pickle

dataset = load_dataset("EleutherAI/pythia-memorized-evals")


for k, v in enumerate(dataset):
    print(v, len(dataset[v]))


def calculate_huffman_code_length(tokens):

    """
    Calculate the length of the huffman code for the given tokens.
    """
    # Get the number of times each character appears in the text.
    char_counts = np.array(list(Counter(tokens).values()))
    # Get the probability of each character appearing in the text.
    char_probs = char_counts / len(tokens)
    # Calculate the length of the huffman code for each character.
    char_code_lengths = np.ceil(-np.log2(char_probs))
    # Calculate the average length of the huffman code for the text.
    huffman_code_length = np.sum(char_code_lengths * char_probs)
    return huffman_code_length


def calculate_for_all_models():
    entropy_dict = {}
    for k,v in enumerate(dataset):
        hc_memorized = []
        for i in dataset[v]['tokens']:
            hc_memorized.append(calculate_huffman_code_length(i))
        entropy_dict[v] = np.mean(hc_memorized)


    pickle.dump(entropy_dict, open("entropy_dict.p", "wb"))

if __name__ == '__main__':
    calculate_for_all_models()
