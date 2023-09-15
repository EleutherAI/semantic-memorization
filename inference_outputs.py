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
import multiprocessing
import torch
import os


class PileDataset(Dataset):
    """
    The wrapped around the Pile-derived pandas dataframe. This allows us to use the
    PyTorch DataLoader to load the data in batches.
    """

    def __init__(self, memories, tokenizer):
        self.tokenizer = tokenizer
        self.memories = memories.rename(columns={"index": "Index", "tokens": "Tokens"}) if "index" in memories.columns else memories

    def __getitem__(self, index):
        tokens = self.memories.iloc[index]["Tokens"][:64]
        decoded_text = self.tokenizer.decode(tokens)
        return self.memories.iloc[index]["Index"], decoded_text

    def __len__(self):
        return len(self.memories["Index"])


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
        "160m": 256,
        "410m": 256,
        "1b": 128,
        "1.4b": 128,
        "2.8b": 64,
        "6.9b": 64,
        "12b": 16,
    }
    return size_batch_map[model_name]


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
        dataset = load_dataset(pile_path, split="train").to_pandas()[["Index", "Tokens"]]
    else:
        dataset = load_dataset("EleutherAI/pythia-memorized-evals")[split_name].to_pandas()

    return dataset if sample is None else dataset.sample(sample).reset_index(drop=True)


def run_model_inferences(split_name: str, run_id: str, dataset: str, batch_size: int, sample_size: int = None):
    """
    Run inference for the given model and dataset. Save the results to a CSV file.

    Args:
        split_name (str): The model+scheme used to determine the tokenizer and model
        run_id (str): The timestamp for this run
        dataset (str): The dataset to run inference on
        sample_size (int, optional): The maximum number of random samples run inference on. Defaults to None.
    """
    tokenizer = load_tokenizer(split_name)
    pythia_model = load_model(split_name)
    pile_sequences = get_dataset(dataset, split_name, sample=sample_size)
    pile_dataset = PileDataset(pile_sequences, tokenizer)
    data_loader = DataLoader(pile_dataset, batch_size=batch_size)

    with torch.multiprocessing.Pool(processes=4) as p:

        with torch.no_grad():
            desc = f"Collecting {dataset} inference responses for {split_name}"
            for batch in tqdm(data_loader, desc=desc):
                batch_sequences = batch[1]
                tokenized_batch = tokenizer(
                    batch_sequences,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                    padding=True,
                )
                tokenized_batch.to(pythia_model.device)
                labels = tokenized_batch["input_ids"]

                outputs = pythia_model(
                    **tokenized_batch,
                    labels=tokenized_batch["input_ids"],
                    output_attentions=True,
                )



                # results = p.map(parse_attn, [t.detach().cpu() for t in outputs.attentions])
                # print(results)

                # attentions_table = {}
                for i in tqdm(range(len(batch[0]))):
                    current_example_id = batch[0][i]
                    current_example_attentions = torch.stack(outputs.attentions)[:, i, :]
                    # attentions_table[current_example_id] = current_example_attentions
                    torch.save(current_example_attentions, f"datasets/{run_id}/{dataset}_attentions_{current_example_id}.pt")
                    # print(current_example_attentions.shape)

                # inference_logs = pd.DataFrame({
                #     "Loss": outputs.loss.detach().cpu().tolist(),
                #     "Logits": outputs.logits.detach().cpu().tolist(),
                #     "Attentions": [attn_tensor.detach().cpu().tolist() for attn_tensor in outputs.attentions],
                # })
                # save_inference_log(split_name, run_id, dataset, inference_logs)
                # torch.cuda.empty_cache()


def save_inference_log(split_name: str, run_id: str, dataset: str, inference_logs_df: pd.DataFrame):
    """Saves the accumilated inference log in a pandas dataframe

    Args:
        split_name (str): The model+scheme used to determine the tokenizer and model
        run_id (str): The timestamp for this run
        dataset (str): The dataset to run inference on
        inference_logs (list): Accumilated inference logs
    """
    file_name = split_name.replace(".", "_")
    inference_logs_df.to_csv(f"datasets/{run_id}/{dataset}_{file_name}.csv", index=False, mode="a")


def parse_attn(attn_t):
    return attn_t.tolist()

def parse_cli_args():
    parser = ArgumentParser()
    models_arg_help = "The Pythia model to get the perplexities for. Valid options are: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b"
    models_args_default = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    parser.add_argument(
        "--models",
        type=str,
        help=models_arg_help,
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

    features_arg_help = "The features to extract from the model response. Valid options are: attn, loss, perplexity"
    features_arg_default = ["attn", "loss", "ppl"]
    parser.add_argument(
        "--features",
        type=str,
        help=features_arg_help,
        choices=features_arg_default,
        default=features_arg_default,
    )

    sample_size_arg_help = "The number of samples to take from the dataset. Defaults to None."
    parser.add_argument(
        "--sample_size",
        type=int,
        help=sample_size_arg_help,
        default=None,
    )

    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for inference")

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
                batch_size = args.batch_size if args.batch_size is not None else get_batch_size(model_size)
                run_model_inferences(split_name, experiment_timestamp, dataset, batch_size, args.sample_size)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
