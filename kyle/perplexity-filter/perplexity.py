from transformers import AutoTokenizer, GPTNeoXForCausalLM
from torch.utils.data import Dataset, DataLoader
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
    return GPTNeoXForCausalLM.from_pretrained(corresponding_model, device_map="auto", load_in_8bit=True).eval()


def calculate_perplexity(logits, labels):
    shift_logits = logits.detach()[:-1, :].contiguous()
    shift_labels = labels[1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss()
    cross_entropy = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(cross_entropy)
    return perplexity


def get_model_perplexities(split_name):
    memories = load_dataset("EleutherAI/pythia-memorized-evals")[split_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(split_name)
    pythia_model = load_model(split_name)
    memories_dataset = HFMemoriesDataset(memories, tokenizer)
    data_loader = DataLoader(memories_dataset, batch_size=128)
    all_perplexities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Calculating perplexities for {split_name}"):
            tokenized_batch = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)
            tokenized_batch.to(device)
            labels = tokenized_batch["input_ids"][:, 1:].contiguous()

            outputs = pythia_model(**tokenized_batch, labels=tokenized_batch["input_ids"])
            logits = outputs.logits.detach()

            labels = tokenized_batch["input_ids"]
            perplexities = [calculate_perplexity(logits[i], labels[i]) for i in range(len(logits))]
            all_perplexities += [perplexity.item() for perplexity in perplexities]

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