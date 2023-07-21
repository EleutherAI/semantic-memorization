import os
from argparse import ArgumentParser
from datetime import datetime
from collections import Counter

import pandas as pd
from datasets import load_dataset

from filters.base import PIPELINE_SINGLETON as PIPELINE
from filters.highly_duplicated_filter import generate_sequence_histogram, get_highly_duplicated_filter_func


def parse_cli_args():
    """
    Parse the command line arguments for the script.
    """
    parser = ArgumentParser()

    run_id_args_help = "The ID for this run. Defaults to current date and time."
    run_id_args_default = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument(
        "--run_id",
        type=str,
        help=run_id_args_help,
        default=run_id_args_default,
    )

    models_args_help = "The Pythia model to get the perplexities for. Valid options are: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b"
    models_args_default = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    parser.add_argument(
        "--models",
        type=str,
        help=models_args_help,
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

    dataset_args_help = "The dataset in which to get inference responses for. Valid options are: memories, pile."
    datasets_args_default = ["pile", "memories"]
    parser.add_argument(
        "--datasets",
        type=str,
        help=dataset_args_help,
        choices=datasets_args_default,
        default=datasets_args_default,
    )

    sample_size_args_help = "The number of samples to take from the dataset. Defaults to None."
    parser.add_argument(
        "--sample_size",
        type=int,
        help=sample_size_args_help,
        default=None,
    )

    return parser.parse_args()

def get_dataset(dataset_name: str, split_name: str, sample_size: int = None) -> pd.DataFrame:
    """
    Get the dataset for the given dataset name and split name.

    Args:
        dataset_name (str): The name of the dataset to get.
        split_name (str): The name of the split to get.
        sample_size (int, optional): The number of samples to take from the dataset. Defaults to None.

    Returns:
        pd.DataFrame: The dataset.
    """
    required_columns = ["index", "tokens"]
    dataset = None

    if dataset_name.split("-")[0] == "pile":
        scheme = split_name.split(".")[0]
        pile_path = f"EleutherAI/pile-{scheme}-pythia-random-sampled"
        dataset = load_dataset(pile_path, split="train").to_pandas()[required_columns]
    else:
        dataset = load_dataset("EleutherAI/pythia-memorized-evals")[split_name].to_pandas()[required_columns]

    return dataset if sample_size is None else dataset.sample(sample_size).reset_index(drop=True)

def register_args_based_filters(histogram: Counter[str, int]) -> None:
    """
    Register argument-based filters, e.g. highly duplicated filters, to the pipeline singleton instance.

    Args:
        histogram (Counter[str, int]): The histogram of the dataset's tokens.
    """
    duplicated_filter_func = get_highly_duplicated_filter_func(histogram)

    @PIPELINE.register_filter(output_column='is_duplicated')
    def highly_duplicated_filter(row: pd.Series) -> bool:
        return duplicated_filter_func(row)

def run_pipeline(run_id: str, dataset_name: str, split_name: str, sample_size: int = None) -> None:
    """
    Run the pipeline on the given dataset and split.

    Args:
        run_id (str): The ID of the run.
        dataset_name (str): The name of the dataset to get.
        split_name (str): The name of the split to get.
        sample_size (int, optional): The number of samples to take from the dataset. Defaults to None.

    Returns:
        None
    """
    dataset = get_dataset(dataset_name, split_name, sample_size=sample_size)
    tokens = dataset["tokens"]
    token_histogram = generate_sequence_histogram(tokens)

    register_args_based_filters(token_histogram)
    transformed_dataset = PIPELINE.transform(dataset)

    file_name = split_name.replace(".", "_")
    transformed_dataset.to_csv(f"datasets/{run_id}/{dataset_name}_{file_name}.csv", index=False)


def main():
    """
    The main function of the script.
    """
    args = parse_cli_args()
    os.makedirs(f"./datasets/{args.run_id}", exist_ok=True)

    print("---------------------------------------------------------------------------")
    print("Starting metric calculation run with the following parameters:")
    print(f"Run ID: {args.run_id}")
    print(f"Models: {args.models}")
    print(f"Schemes: {args.schemes}")
    print(f"Datasets: {args.datasets}")
    if args.sample_size is not None:
        print(f"Sample size: {args.sample_size}")
    print("---------------------------------------------------------------------------")

    for model_size in args.models if isinstance(args.models, list) else args.models.split(","):
        for data_scheme in args.schemes if isinstance(args.schemes, list) else args.schemes.split(","):
            for dataset_name in args.datasets if isinstance(args.datasets, list) else args.datasets.split(","):
                split_name = f"{data_scheme}.{model_size}"
                print(f"Calculating metrics for {split_name} on dataset {dataset_name}...")
                run_pipeline(args.run_id, dataset_name, split_name, args.sample_size)


if __name__ == "__main__":
    main()