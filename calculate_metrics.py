import logging
import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from filters.base import PIPELINE_SINGLETON as PIPELINE

FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S %z")

STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
STDOUT_HANDLER.setLevel(logging.DEBUG)
STDOUT_HANDLER.setFormatter(FORMATTER)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(STDOUT_HANDLER)


class Schema(Enum):
    DEDUPLICATED = "deduped"
    DUPLICATED = "duped"


@dataclass
class Dataset:
    data: pd.DataFrame
    schema: Schema


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

    schemes_args_help = "The data scheme used for Pythia model training. Valid options are: deduped, duped"
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


def get_dataset(dataset_name: str, split_name: str, sample_size: int = None) -> Dataset:
    """
    Get the dataset for the given dataset name and split name.

    Args:
        dataset_name (str): The name of the dataset to get.
        split_name (str): The name of the split to get.
        sample_size (int, optional): The number of samples to take from the dataset. Defaults to None.

    Returns:
        pd.DataFrame: The dataset.
    """
    required_columns = ["sequence_id", "tokens"]
    schema_name, model_size = split_name.split(".")
    schema = Schema.DEDUPLICATED if schema_name == "deduped" else Schema.DUPLICATED

    if dataset_name.split("-")[0] == "pile":
        pile_path = f"EleutherAI/pile-{schema_name}-pythia-random-sampled"
        dataset = load_dataset(pile_path, split="train").to_pandas()
        # The original dataset has a different capitalization for the column names, so we'll rename them along
        # with other columns for clarity and consistency.
        dataset = dataset.rename(
            columns={
                "Index": "sequence_id",
                "Tokens": "tokens",
                "70M": "70m",
                "160M": "160m",
                "410M": "410m",
                "1B": "1b",
                "1.4B": "1.4b",
                "2.8B": "2.8b",
                "6.9B": "6.9b",
                "12B": "12b",
            }
        )
        # This dataset already contains the memorization score, we'll fetch it by the model parameter size.
        required_columns.append(model_size)
        # We'll also rename the memorization score column for consistency.
        dataset = dataset[required_columns].rename(columns={model_size: "memorization_score"})
    else:
        dataset = load_dataset("EleutherAI/pythia-memorized-evals")[split_name].to_pandas().rename(columns={"index": "sequence_id"})
        dataset = dataset[required_columns]
        # This dataset already indicates all sequences are memorized.
        dataset["memorization_score"] = 1.0

    dataset = dataset if sample_size is None else dataset.sample(sample_size).reset_index(drop=True)

    return Dataset(data=dataset, schema=schema)


def get_precomputed_signals(dataset: Dataset) -> Dict[str, Any]:
    """
    Get the precomputed signals for the given dataset.

    Args:
        dataset (Dataset): The dataset to get the precomputed signals for.
    """
    schema = dataset.schema.value
    # (sequence_id, frequency)
    sequence_duplicates = (
        load_dataset(f"usvsnsp/{schema}-num-duplicates")["train"].to_pandas().rename(columns={"Index": "sequence_id", "Counts": "frequency"})
    )
    # (token_id, frequency)
    memorized_frequencies = (
        load_dataset(f"usvsnsp/{schema}-num-frequencies")["memorized"].to_pandas().rename(columns={"TokenID": "token_id", "Frequency": "frequency"})
    )
    # (token_id, frequency)
    non_memorized_frequencies = (
        load_dataset(f"usvsnsp/{schema}-num-frequencies")["non_memorized"]
        .to_pandas()
        .rename(columns={"TokenID": "token_id", "Frequency": "frequency"})
    )

    signals = {
        "sequence_duplicates": sequence_duplicates,
        "memorized_frequencies": memorized_frequencies,
        "non_memorized_frequencies": non_memorized_frequencies,
    }

    return signals


def register_args_based_filters(signals: Dict[str, Any]) -> None:
    """
    Register signal-based filters, e.g. highly duplicated filters, to the pipeline singleton instance.

    Args:
        signals (Dict[str, Any]): The precomputed signals to use for the filters.
    """
    sequence_duplicates = signals["sequence_duplicates"]
    memorized_frequencies = signals["memorized_frequencies"]
    non_memorized_frequencies = signals["non_memorized_frequencies"]

    @PIPELINE.register_filter(output_column="num_duplicates")
    def num_sequence_duplicate_filter(row: pd.Series) -> int:
        sequence_index = row["sequence_id"]

        try:
            num_duplicates = sequence_duplicates[sequence_duplicates["sequence_id"] == sequence_index].iloc[0]["frequency"]
        except:
            LOGGER.warning(f"Sequence index {sequence_index} not found in the sequence duplication dataset. Defaulting to -1.")
            # If the sequence is not in the dataset, then we'll set
            # the number of duplicates to -1 as an invalidation signal.
            num_duplicates = -1

        return num_duplicates

    @PIPELINE.register_filter(output_column="memorized_token_frequencies")
    def memorized_token_frequency_filter(row: pd.Series) -> List[int]:
        tokens = row["tokens"]
        token_frequencies = list(map(lambda _id: memorized_frequencies[memorized_frequencies["token_id"] == _id].iloc[0]["frequency"], tokens))

        return token_frequencies

    @PIPELINE.register_filter(output_column="non_memorized_token_frequencies")
    def non_memorized_token_frequency_filter(row: pd.Series) -> List[int]:
        tokens = row["tokens"]
        token_frequencies = list(
            map(lambda _id: non_memorized_frequencies[non_memorized_frequencies["token_id"] == _id].iloc[0]["frequency"], tokens)
        )

        return token_frequencies


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
    signals = get_precomputed_signals(dataset)
    register_args_based_filters(signals)

    transformed_dataset = PIPELINE.transform(dataset.data)
    file_name = split_name.replace(".", "_")
    transformed_dataset.to_csv(f"datasets/{run_id}/{dataset_name}_{file_name}.csv", index=False)


def main():
    """
    The main function of the script.
    """
    args = parse_cli_args()

    os.makedirs(f"./datasets/{args.run_id}", exist_ok=True)
    file_handler = logging.FileHandler(f"./datasets/{args.run_id}/run.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FORMATTER)
    LOGGER.addHandler(file_handler)

    LOGGER.info("---------------------------------------------------------------------------")
    LOGGER.info("Starting metric calculation run with the following parameters:")
    LOGGER.info(f"Run ID: {args.run_id}")
    LOGGER.info(f"Models: {args.models}")
    LOGGER.info(f"Schemes: {args.schemes}")
    LOGGER.info(f"Datasets: {args.datasets}")
    if args.sample_size is not None:
        LOGGER.info(f"Sample size: {args.sample_size}")
    LOGGER.info("---------------------------------------------------------------------------")

    for model_size in args.models if isinstance(args.models, list) else args.models.split(","):
        for data_scheme in args.schemes if isinstance(args.schemes, list) else args.schemes.split(","):
            for dataset_name in args.datasets if isinstance(args.datasets, list) else args.datasets.split(","):
                split_name = f"{data_scheme}.{model_size}"
                LOGGER.info(f"Calculating metrics for {split_name} on dataset {dataset_name}...")
                run_pipeline(args.run_id, dataset_name, split_name, args.sample_size)


if __name__ == "__main__":
    main()
