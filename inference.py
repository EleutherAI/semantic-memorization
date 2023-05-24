"""
This file contains the code for running batch inference on the Pythia models. We can save results from
the inferences to a CSV file for later analysis. This is useful for calculating perplexity, entropy,
and other metrics.

Example Usage: python inference.py --models=410m,1b,12b --schemes=duped --datasets=memories,pile --sample-size=100000
"""

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, ReadInstruction
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import os


class PileDataset(Dataset):
    """
    The wrapped around the Pile-derived pandas dataframe. This allows us to use the
    PyTorch DataLoader to load the data in batches.
    """
    def __init__(self, memories, tokenizer):
        self.tokenizer = tokenizer
        self.memories = memories

    def __getitem__(self, index):
        tokens = self.memories.iloc[index]["tokens"][:64]
        decoded_text = self.tokenizer.decode(tokens)
        return self.memories.iloc[index]["index"], decoded_text

    def __len__(self):
        return len(self.memories["index"])


def load_tokenizer(split_name: str) -> AutoTokenizer:
    """Get the HuggingFace tokenizer for the current model"""
    isDeduped = split_name.startswith("deduped")
    model = split_name.split("duped.")[-1]
    corresponding_model = f"EleutherAI/pythia-{model}{'-deduped' if isDeduped else ''}"
    tokenizer = AutoTokenizer.from_pretrained(corresponding_model)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(split_name):
    """Get the HuggingFace model for the current model"""
    isDeduped = split_name.startswith("deduped")
    model = split_name.split("duped.")[-1]
    corresponding_model = f"EleutherAI/pythia-{model}{'-deduped' if isDeduped else ''}"
    return GPTNeoXForCausalLM.from_pretrained(corresponding_model, device_map="auto")


def calculate_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> torch.float64:
    """
    Clauclate the perplexity of a sequence given the logits and labels

    Args:
        logits (torch.Tensor): The logits for the model's generation
        labels (torch.Tensor): The true tokens for the sequence

    Returns:
        torch.float64: The model's perplexity for the given sequence
    """
    # Store the probabilities for each token. These will be summed later, but having the
    # individual probabilities is helpful for debugging.
    token_probs = []

    # Don't include the final token logits. There are no labels for
    # these since the sequence has ended.
    num_special_tokens = len(labels[labels == 0])
    num_normal_tokens = len(labels) - num_special_tokens

    for token_index in range(num_normal_tokens - 1):
        # Map the logits to probabilities.
        predicted_probs = torch.softmax(logits[token_index], dim=0, dtype=torch.float16)
        # Get the probability of the correct label.
        label_prob = predicted_probs[labels[token_index + 1]]

        # Check if the label probability is 0. This is likely due a rounding error. Recalculate
        # the probability using double precision.
        if label_prob == 0:
            predicted_probs = torch.softmax(logits[token_index], dim=0, dtype=torch.float64)
            label_prob = predicted_probs[labels[token_index + 1]]

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


def get_batch_size(model_name: str) -> int:
    """
    Get the optimal batch size for the current model. This is based on the model's size
    where the batch size is the largest that can fit on our GPUs. Yiu may need to adjust
    this if you have a different GPU.

    Args:
        model_name (str): The model name

    Returns:
        int: The batch size to use for inference
    """
    size_batch_map = {
        "70m": 512,
        "160m": 512,
        "410m": 512,
        "1b": 256,
        "1.4b": 256,
        "2.8b": 128,
        "6.9b": 64,
        "12b": 64,
    }
    model_size = ".".join(model_name.split(".")[1:])
    return size_batch_map[model_size]


def get_dataset(dataset_name: str, split_name: str, sample: int = None) -> pd.DataFrame:
    """
    Read the given dataframe from HuggingFace, process, and return the sequences as
    a pandas dataframe.

    Args:
        dataset_name (str): The dataset path
        split_name (str): The split within the dataset we're interested in
        sample (int, optional): The number of samples to take from the dataset. Defaults to None.

    Returns:
        pd.DataFrame: The pandas dataframe storing the dataset
    """
    dataset = None
    if dataset_name.split("-")[0] == "pile":
        scheme = split_name.split(".")[0]
        pile_path = f"EleutherAI/pile-{scheme}-pythia-random-sampled"
        lower_index = 0 if dataset_name == "pile-1" else 50
        upper_index = 50 if dataset_name == "pile-1" else 100

        print(f"Loading {pile_path} {lower_index}-{upper_index}%")
        pile_tokens = load_dataset(
            pile_path,
            split=ReadInstruction(
                "train",
                from_=lower_index,
                to=upper_index,
                unit="%",
                rounding="pct1_dropremainder",
            ),
        ).to_pandas()[["index", "tokens"]]

        if dataset is None:
            dataset = pile_tokens
        else:
            dataset = pd.concat([dataset, pile_tokens])
    else:
        dataset = load_dataset("EleutherAI/pythia-memorized-evals")[split_name].to_pandas()

    return dataset if sample is None else dataset.sample(sample).reset_index(drop=True)


def run_model_inferences(split_name: str, run_id: str, dataset: str, sample_size: int = None):
    """
    Run inference for the given model and dataset. Save the results to a CSV file.

    Args:
        split_name (str): The model+scheme used to determine the tokenizer and model
        run_id (str): The timestamp for this run
        dataset (str): The dataset to run inference on
        sample_size (int, optional): The maximum number of random samples run inference on. Defaults to None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(split_name)
    pythia_model = load_model(split_name)
    pile_sequences = get_dataset(dataset, split_name, sample=sample_size)
    pile_dataset = PileDataset(pile_sequences, tokenizer)
    batch_size = get_batch_size(split_name)
    data_loader = DataLoader(pile_dataset, batch_size=batch_size)

    with torch.no_grad():
        desc = f"Collecting {dataset} inference responses for {split_name}"
        for batch in tqdm(data_loader, desc=desc):
            batch_sequences = batch[1]
            tokenized_batch = tokenizer(
                batch_sequences,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            tokenized_batch.to(device)
            labels = tokenized_batch["input_ids"]

            outputs = pythia_model(
                **tokenized_batch,
                labels=tokenized_batch["input_ids"],
                output_attentions=True,
            )
            save_inference_log(split_name, run_id, dataset, batch, labels, outputs)


def save_inference_log(split_name: str, run_id: str, dataset: pd.DataFrame, batch: tuple, labels: torch.Tensor, outputs: CausalLMOutputWithPast):
    """
    Extract the desired data from the model response and save it to a CSV file.

    Args:
        split_name (str): The model+scheme used to determine the tokenizer and model
        run_id (str): The timestamp for this run
        dataset (str): The dataset to run inference on
        batch (tuple): The input batch containing the sequence ids and sequences
        labels (torch.Tensor): The labels for the batch. Used to calculate perplexity
        outputs (CausalLMOutputWithPast): The response from the Pythia model
    """
    logits = outputs.logits.detach()
    perplexities = [calculate_perplexity(logits[i], labels[i]) for i in range(len(logits))]
    all_perplexities = np.append(all_perplexities, perplexities)
    inference_logs = []

    batch_sequence_ids = batch[0]
    for index, id_tensor in enumerate(batch_sequence_ids):
        inference_log = {
            "index": id_tensor.detach().item(),
            "perplexity": perplexities[index],
            "mean_loss": outputs.loss.detach().item() / len(labels[index]),
        }
        for layer_index, attention_layer in enumerate(outputs.attentions):
            sequence_attention = attention_layer[index].detach().tolist()
            inference_log[f"attn_{layer_index}"] = sequence_attention
        inference_logs.append(inference_log)

    file_name = split_name.replace(".", "_")
    inference_logs_df = pd.DataFrame(inference_logs)
    inference_logs_df.to_csv(f"datasets/{run_id}/{dataset}_{file_name}.csv", index=False, mode="a")


def parse_cli_args():
    parser = ArgumentParser()
    models_arg_help = "The Pythia model to get the perplexities for. Valid options are: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b"
    models_args_default = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    parser.add_argument(
        "--models",
        type=str,
        help=models_arg_help,
        choices=models_args_default,
        default=models_args_default,
    )

    schemes_args_help = "The data scheme to get the perplexities for. Valid options are: deduped, duped"
    schemes_args_default = ["deduped", "duped"]
    parser.add_argument(
        "--schemes",
        type=str,
        help=schemes_args_help,
        choices=schemes_args_default,
        default=schemes_args_default,
    )

    dataset_arg_help = "The dataset in which to get inference responses for. Valid options are: memories, pile."
    datasets_args_default = ["pile", "memories"]
    parser.add_argument(
        "--datasets",
        type=str,
        help=dataset_arg_help,
        choices=datasets_args_default,
        default=datasets_args_default,
    )

    sample_size_arg_help = "The number of samples to take from the dataset. Defaults to None."
    parser.add_argument(
        "--sample-size",
        type=int,
        help=sample_size_arg_help,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_cli_args()
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"./datasets/{experiment_timestamp}", exist_ok=True)

    print("---------------------------------------------------------------------------")
    print("Starting inference run with the following parameters:")
    print(f"Timestamp: {experiment_timestamp}")
    print(f"Models: {args.models}")
    print(f"Schemes: {args.schemes}")
    print(f"Datasets: {args.datasets}")
    if args.sample_size is not None:
        print(f"Sample size: {args.sample_size}")
    print("---------------------------------------------------------------------------")

    for model_size in args.models if isinstance(args.models, list) else args.models.split(","):
        for data_scheme in args.schemes if isinstance(args.schemes, list) else args.schemes.split(","):
            for dataset in args.datasets if isinstance(args.datasets, list) else args.datasets.split(","):
                split_name = f"{data_scheme}.{model_size}"
                print(f"Collecting inferences for {split_name} on {dataset} dataset")
                run_model_inferences(split_name, experiment_timestamp, dataset, args.sample_size)


if __name_ == "__main__":
    main()
