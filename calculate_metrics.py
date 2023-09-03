import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Optional

import pyspark.pandas as ps
from datasets import load_dataset
from pyspark.sql import SparkSession, DataFrame

from utils import initialize_logger, initialize_formatter
from utils import initialize_spark
from filters import PIPELINE
from filters.constants import PrecomputedFeatureName

LOGGER: logging.Logger = initialize_logger()
SPARK: SparkSession = initialize_spark()


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

    sample_seed_args_help = "The seed to use for sampling the dataset. Defaults to None."
    parser.add_argument(
        "--sample_seed",
        type=int,
        help=sample_seed_args_help,
        default=None,
    )

    spark_cache_args_help = "The directory to store cached Spark datasets to speed up the data loading. Default to spark_cache/."
    parser.add_argument(
        "--spark_cache_dir",
        type=str,
        default="spark_cache",
        help=spark_cache_args_help,
    )

    return parser.parse_args()


def load_dataset(dataset_name: str, scheme: str, model_size: str, cache_dir: str) -> DataFrame:
    """
    Load the dataset from HuggingFace datasets. If the dataset is not locally available, then
    download it from HuggingFace datasets and cache it as a Spark DataFrame in Parquet format.

    Args:
        dataset_name (str): Name of the dataset to download.
        scheme (str): Data scheme used for Pythia model training.
        model_size (str): Pythia model size.
        cache_dir (str): Directory to store cached Spark datasets to speed up the data loading.

    Returns:
        DataFrame: Spark DataFrame containing the dataset.
    """
    split_name = f"{scheme}.{model_size}"
    required_columns = ["sequence_id", "tokens"]
    is_pile = dataset_name.split("-")[0] == "pile"

    if is_pile:
        hf_dataset_name = f"EleutherAI/pile-{scheme}-pythia-random-sampled"
    else:
        hf_dataset_name = f"EleutherAI/pythia-memorized-evals"

    cache_name = hf_dataset_name if is_pile else f"{hf_dataset_name}-{split_name}"
    cache_path = f"{cache_dir}/{cache_name}"

    if os.path.isdir(cache_path):
        LOGGER.info(f"Dataset {hf_dataset_name} already exists, skipping the download.")
        return SPARK.read.parquet(cache_path)

    LOGGER.info(f"Downloading dataset {hf_dataset_name}...")
    if is_pile:
        # The original dataset has a different capitalization for the column names, so we'll rename them along
        # with other columns for clarity and consistency.
        dataset = (
            load_dataset(hf_dataset_name, split="train")
            .to_pandas()
            .rename(
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
        )
        dataset.tokens = dataset.tokens.map(lambda x: x.tolist())
        # This dataset already contains the memorization score, we'll fetch it by the model parameter size.
        required_columns.append(model_size)
        # We'll also rename the memorization score column for consistency.
        dataset = dataset[required_columns].rename(columns={model_size: "memorization_score"})
    else:
        dataset = load_dataset(hf_dataset_name, split=split_name).to_pandas().rename(columns={"index": "sequence_id"})
        dataset.tokens = dataset.tokens.map(lambda x: x.tolist())
        dataset = dataset[required_columns]
        # This dataset already indicates all sequences are memorized.
        dataset["memorization_score"] = 1.0

    LOGGER.info(f"Converting and caching the dataset {hf_dataset_name} as Spark DataFrame in {cache_path}...")
    # Convert the Pandas DataFrame dataset to Spark DataFrame in Parquet
    ps.from_pandas(dataset).to_spark().write.parquet(cache_path)

    return SPARK.read.parquet(cache_path)


def load_precomputed_features(schema: str, cache_dir: str) -> Dict[PrecomputedFeatureName, DataFrame]:
    """
    Load the pre-computed features from HuggingFace datasets. If the features are not locally available, then
    download them from HuggingFace datasets and cache them as Spark DataFrames in Parquet format.

    Args:
        schema (str): Data scheme used for Pythia model training.
        cache_dir (str): Directory to store cached Spark datasets to speed up the data loading.

    Returns:
        Dict[PrecomputedFeatureName, DataFrame]: Dictionary of pre-computed features.
    """
    features = {}
    hf_dataset_names = [
        (PrecomputedFeatureName.SEQUENCE_FREQUENCIES, f"usvsnsp/{schema}-num-duplicates", "train", {"Index": "sequence_id", "Counts": "frequency"}),
        (
            PrecomputedFeatureName.MEMORIZED_TOKEN_FREQUENCIES,
            f"usvsnsp/{schema}-num-frequencies",
            "memorized",
            {"TokenID": "token_id", "Frequency": "frequency"},
        ),
        (
            PrecomputedFeatureName.NON_MEMORIZED_TOKEN_FREQUENCIES,
            f"usvsnsp/{schema}-num-frequencies",
            "non_memorized",
            {"TokenID": "token_id", "Frequency": "frequency"},
        ),
    ]

    for enum, name, split, column_mapping in hf_dataset_names:
        cache_path = f"{cache_dir}/{name}-{split}"

        if os.path.isdir(cache_path):
            LOGGER.info(f"Dataset {name}-{split} already exists, skipping the download.")
            features[enum] = SPARK.read.parquet(cache_path)
            continue

        LOGGER.info(f"Downloading dataset {name}-{split}...")
        dataset = load_dataset(name, split=split).to_pandas().rename(columns=column_mapping)

        LOGGER.info(f"Converting and caching the dataset {name}-{split} as Spark DataFrame {cache_path}...")
        # Convert the Pandas DataFrame dataset to Spark DataFrame in Parquet
        ps.from_pandas(dataset).to_spark().write.parquet(cache_path)

        features[enum] = SPARK.read.parquet(cache_path)

    return features


def run_pipeline(
    dataset: DataFrame,
    dataset_name: str,
    split_name: str,
    run_id: str,
    sample_size: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> None:
    if sample_size is not None:
        dataset = dataset.sample(1.0, seed=sample_seed).limit(sample_size)

    transformed_dataset = PIPELINE.transform(dataset)
    file_name = split_name.replace(".", "_")
    transformed_dataset.write.parquet(f"datasets/{run_id}/{dataset_name}_{file_name}")


def main():
    """
    The main function of the script.
    """
    args = parse_cli_args()

    os.makedirs(f"./datasets/{args.run_id}", exist_ok=True)
    file_handler = logging.FileHandler(f"./datasets/{args.run_id}/run.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(initialize_formatter())
    LOGGER.addHandler(file_handler)

    LOGGER.info("---------------------------------------------------------------------------")
    LOGGER.info("Starting metric calculation run with the following parameters:")
    LOGGER.info(f"Run ID: {args.run_id}")
    LOGGER.info(f"Models: {args.models}")
    LOGGER.info(f"Schemes: {args.schemes}")
    LOGGER.info(f"Datasets: {args.datasets}")
    LOGGER.info(f"Spark cache directory: {args.spark_cache_dir}/")
    if args.sample_size is not None:
        LOGGER.info(f"Sample size: {args.sample_size}")
    if args.sample_seed is not None:
        LOGGER.info(f"Sample seed: {args.sample_seed}")
    LOGGER.info("---------------------------------------------------------------------------")

    for model_size in args.models if isinstance(args.models, list) else args.models.split(","):
        for data_scheme in args.schemes if isinstance(args.schemes, list) else args.schemes.split(","):
            LOGGER.info("Loading pre-computed features...")
            precomputed_features = load_precomputed_features(data_scheme, args.spark_cache_dir)
            PIPELINE.register_features(precomputed_features)

            for dataset_name in args.datasets if isinstance(args.datasets, list) else args.datasets.split(","):
                split_name = f"{data_scheme}.{model_size}"
                LOGGER.info(f"Loading dataset {dataset_name} and split {split_name}...")
                dataset = load_dataset(dataset_name, data_scheme, model_size, args.spark_cache_dir)
                LOGGER.info(f"Calculating metrics for {split_name} on dataset {dataset_name}...")
                run_pipeline(dataset, dataset_name, split_name, args.run_id, args.sample_size, args.sample_seed)


if __name__ == "__main__":
    main()
