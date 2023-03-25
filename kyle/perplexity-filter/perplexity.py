from transformers import AutoTokenizer, GPTNeoXForCausalLM
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from natsort import natsorted
from tqdm import tqdm
import plotly.express as px
import pandas as pd
import torch

class HFMemoriesDataset(Dataset):
    is_dataframe = False

    def __init__(self, memories, tokenizer, sample=None):
        self.tokenizer = tokenizer
        self.memories = memories
        if sample is not None:
            self.memories = self.memories.to_pandas().sample(sample)
            self.is_dataframe = True

    def __getitem__(self, index):
        memory_record = (
            self.memories.iloc[index] if self.is_dataframe else self.memories[index]
        )
        decoded_text = self.tokenizer.decode(memory_record["tokens"])
        return decoded_text

    def __len__(self):
        return len(self.memories)


def load_tokenizer(split_name):
    isDeduped = split_name.startswith("deduped")
    model = split_name.split("duped.")[-1]
    corresponding_model = f"EleutherAI/pythia-{model}{'-deduped' if isDeduped else ''}"
    tokenizer =  AutoTokenizer.from_pretrained(corresponding_model, load_in_8bit=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(split_name):
    isDeduped = split_name.startswith("deduped")
    model = split_name.split("duped.")[-1]
    corresponding_model = f"EleutherAI/pythia-{model}{'-deduped' if isDeduped else ''}"
    return GPTNeoXForCausalLM.from_pretrained(corresponding_model, device_map="sequential", load_in_8bit=True).eval()


def calculate_perplexity(logits, labels):
    # Store the probabilities for each token. These will be summed later, but having the
    # individual probabilities is helpful for debugging.
    token_probs = []

    # Don't include the final token logits. There are no labels for
    # these since the sequence has ended.
    shifted_logits = logits.detach()[:-1, :]

    for token_index in range(len(shifted_logits)):
        # Map the logits to probabilities.
        predicted_probs = torch.softmax(shifted_logits[token_index], dim=0)
        # Get the probability of the correct label.
        label_prob = predicted_probs[labels[token_index + 1]]
        # Store the probability for this token.
        token_probs.append(label_prob.detach())

    # Caluclate the log-likelyhood of the sequence by summing the probabilities
    # of each token and then taking the log.
    log_likelihood = torch.log(torch.stack(token_probs)).sum()

    # Caluclate the cross entropy by dividing the negative log-likelihood by the number of tokens.
    cross_entropy = -log_likelihood / len(shifted_logits)

    # Calculate the perplexity by taking the exponential of the cross entropy.
    perplexity = torch.exp(cross_entropy).item()
    return perplexity


def get_model_perplexities(split_name):
    memories = load_dataset("EleutherAI/pythia-memorized-evals")[split_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(split_name)
    pythia_model = load_model(split_name)
    memories_dataset = HFMemoriesDataset(memories, tokenizer)
    data_loader = DataLoader(memories_dataset, batch_size=64)
    all_perplexities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Calculating perplexities for {split_name}"):
            tokenized_batch = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)
            tokenized_batch.to(device)
            labels = tokenized_batch["input_ids"]

            outputs = pythia_model(**tokenized_batch, labels=tokenized_batch["input_ids"])
            logits = outputs.logits.detach()

            perplexities = [calculate_perplexity(logits[i], labels[i]) for i in range(len(logits))]
            all_perplexities += perplexities

    perplexities_df = memories.to_pandas()[["index"]]
    perplexities_df["perplexity"] = all_perplexities
    file_name = split_name.replace(".", "_")
    perplexities_df.to_csv(f"./datasets/memories_{file_name}.csv", index=False)
    print(perplexities_df)


if __name__ == "__main__":
    all_memories_splits = load_dataset("EleutherAI/pythia-memorized-evals")
    model_sizes = (
        natsorted(set([split_name.split("uped.")[-1] for split_name in all_memories_splits if "m" in split_name]))
        # Omit the billions of parameters model because it's too big to fit in memory at the moment
        # natsorted(set(([split_name.split("uped.")[-1] for split_name in all_memories_splits if "b" in split_name])))
    )

    for pile_dataset in ["deduped", "duped"]:
        ordered_splits = [f"{pile_dataset}.{model_size}" for model_size in model_sizes]
        for split_name in ordered_splits:
            # split_name = "deduped.12b"
            # split_name = "deduped.410m"
            get_model_perplexities(split_name)