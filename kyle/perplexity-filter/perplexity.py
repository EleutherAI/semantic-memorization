from transformers import AutoTokenizer, GPTNeoXForCausalLM
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from datasets import load_dataset, ReadInstruction
from tqdm import tqdm
from datetime import datetime
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import os

class PileDataset(Dataset):
    is_dataframe = False

    def __init__(self, memories, tokenizer):
        self.tokenizer = tokenizer
        self.memories = memories

    def __getitem__(self, index):
        tokens = self.memories.iloc[index]["tokens"][:64]
        decoded_text = self.tokenizer.decode(tokens)
        return decoded_text

    def __len__(self):
        return len(self.memories["index"])


def load_tokenizer(split_name):
    isDeduped = split_name.startswith("deduped")
    model = split_name.split("duped.")[-1]
    corresponding_model = f"EleutherAI/pythia-{model}{'-deduped' if isDeduped else ''}"
    tokenizer =  AutoTokenizer.from_pretrained(corresponding_model)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(split_name):
    isDeduped = split_name.startswith("deduped")
    model = split_name.split("duped.")[-1]
    corresponding_model = f"EleutherAI/pythia-{model}{'-deduped' if isDeduped else ''}"
    # device_map = { "": 0, "gpt_neox.embed_in": 1, "gpt_neox.final_layer_norm": 1}
    return GPTNeoXForCausalLM.from_pretrained(corresponding_model, device_map="auto")


def calculate_perplexity(logits, labels):
    # Store the probabilities for each token. These will be summed later, but having the
    # individual probabilities is helpful for debugging.
    token_probs = []

    # Don't include the final token logits. There are no labels for
    # these since the sequence has ended.
    num_special_tokens = len(labels[labels == 0])
    num_normal_tokens = len(labels) - num_special_tokens

    for token_index in range(num_normal_tokens - 1):
        # Map the logits to probabilities.
        predicted_probs = torch.softmax(logits[token_index], dim=0, dtype=torch.float64)
        # Get the probability of the correct label.
        label_prob = predicted_probs[labels[token_index + 1]]

        # Check if the label probability is 0. This is likely due to a special token. If this is the case,
        # break and stop calculating the perplexity for this sequence.
        if label_prob == 0:
            break

        # Store the probability for this token.
        token_probs.append(label_prob.detach())

    # Caluclate the log-likelyhood of the sequence by summing the probabilities
    # of each token and then taking the log.
    log_likelihood = torch.log(torch.stack(token_probs)).sum()

    # Caluclate the cross entropy by dividing the negative log-likelihood by the number of tokens.
    cross_entropy = -log_likelihood / len(token_probs)

    # Calculate the perplexity by taking the exponential of the cross entropy.
    perplexity = torch.exp(cross_entropy).item()
    # assert perplexity != float("inf"), "Perplexity is infinite. This is probably due to a token that has a probability of 0."
    return perplexity


def get_batch_size(split_name):
    size_batch_map = {
        "70m": 512,
        "160m": 512,
        "410m": 512,
        "1b": 256,
        "1.4b": 256,
        "2.8b": 128,
        "6.9b": 64,
        "12b": 64
    }
    model_size = ".".join(split_name.split(".")[1:])
    return size_batch_map[model_size]


def get_dataset(dataset_name, split_name, sample=None):
    dataset = None

    if dataset_name.split("-")[0] == "pile":
        scheme = split_name.split(".")[0]
        pile_path = f"EleutherAI/pile-{scheme}-pythia-random-sampled"
        lower_index = 0 if dataset_name == "pile-1" else 50
        upper_index = 50 if dataset_name == "pile-1" else 100
        print(f"Loading {pile_path} {lower_index}-{upper_index}%")
        pile_tokens = load_dataset(pile_path, split=ReadInstruction("train", from_=lower_index, to=upper_index, unit="%", rounding="pct1_dropremainder")).to_pandas()[["index", "tokens"]]
        if dataset is None:
            dataset = pile_tokens
        else:
            dataset = pd.concat([dataset, pile_tokens])
    else:
        dataset = load_dataset("EleutherAI/pythia-memorized-evals")[split_name].to_pandas()

    return dataset if sample is None else dataset.sample(sample).reset_index(drop=True)


def get_model_perplexities(split_name, run_id, dataset):
    pile_sequences = get_dataset(dataset, split_name, sample=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(split_name)
    pythia_model = load_model(split_name)
    pile_dataset = PileDataset(pile_sequences, tokenizer)
    batch_size = get_batch_size(split_name)
    data_loader = DataLoader(pile_dataset, batch_size=batch_size)
    all_perplexities = np.array([])

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Calculating {dataset} perplexities for {split_name}"):
            tokenized_batch = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)
            tokenized_batch.to(device)
            labels = tokenized_batch["input_ids"]

            outputs = pythia_model(**tokenized_batch, labels=tokenized_batch["input_ids"])
            logits = outputs.logits.detach()

            perplexities = [calculate_perplexity(logits[i], labels[i]) for i in range(len(logits))]
            all_perplexities = np.append(all_perplexities, perplexities)

    perplexities_df = pd.DataFrame({
        "index": pile_sequences["index"],
        "perplexity": all_perplexities
    })
    file_name = split_name.replace(".", "_")
    perplexities_df.to_csv(f"./datasets/{run_id}/{dataset}_{file_name}.csv", index=False)
    print(perplexities_df)

def main():
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"./datasets/{experiment_timestamp}", exist_ok=True)

    for model_size in ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]:
        for data_scheme in ["deduped", "duped"]:
            for dataset in ["pile-1", "pile-2", "memories"]:
                split_name = f"{data_scheme}.{model_size}"
                # split_name = "deduped.12b"
                # split_name = "deduped.160m"
                # split_name = "deduped.6.9b"
                # split_name = "deduped.1b"
                # split_name = "deduped.410m"
                print(f"Calculating perplexities for {split_name} on {dataset} dataset")
                get_model_perplexities(split_name, experiment_timestamp, dataset)

if __name__ == "__main__":
    main()