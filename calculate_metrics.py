import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Optional, List

from datasets import load_dataset as hf_load_dataset
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from utils import initialize_logger, initialize_formatter
from filters import PIPELINE
from filters.constants import PrecomputedFeatureName
from spark.constants import NUM_SPARK_PARTITIONS, NUM_OUTPUT_PARTITIONS, SPARK_CACHE_DIR
from spark.utils import initialize_spark

LOGGER: logging.Logger = initialize_logger()
SPARK: SparkSession = initialize_spark()
PIPELINE.register_spark_session(SPARK)


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
    datasets_args_default = ["pile", "memories", "pile_test"]
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

    return parser.parse_args()


def load_pile_dataset(scheme: str) -> DataFrame:
    """
    Load the Pile dataset from HuggingFace. If the dataset is not locally available, then
    download it from HuggingFace datasets and cache it as a Spark DataFrame in Parquet format.

    Args:
        scheme (str): Data scheme used for Pythia model training.

    Returns:
        DataFrame: Spark DataFrame containing the dataset.
    """
    hf_dataset_name = f"EleutherAI/pile-{scheme}-pythia-random-sampled"
    cache_path = f"{SPARK_CACHE_DIR}/{hf_dataset_name}"

    if os.path.isdir(cache_path):
        LOGGER.info(f"Dataset {hf_dataset_name} already exists, skipping the download.")
        return SPARK.read.parquet(cache_path)

    LOGGER.info(f"Downloading dataset {hf_dataset_name}...")

    # The original dataset has a different capitalization for the column names, so we'll rename them along
    # with other columns for clarity and consistency.
    dataset = (
        hf_load_dataset(hf_dataset_name, split="train")
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

    LOGGER.info(f"Converting and caching the dataset {hf_dataset_name} as Spark DataFrame in {cache_path}...")
    # Convert the Pandas DataFrame dataset to Spark DataFrame in Parquet
    SPARK.createDataFrame(dataset).repartition(NUM_SPARK_PARTITIONS).write.parquet(cache_path)

    return SPARK.read.parquet(cache_path)


def load_non_pile_dataset(dataset_name: str, scheme: str, model_size: str) -> DataFrame:
    """
    Load the non-Pile dataset from HuggingFace. If the dataset is not locally available, then
    download it from HuggingFace datasets and cache it as a Spark DataFrame in Parquet format.

    Args:
        dataset_name (str): Name of the dataset to download.
        scheme (str): Data scheme used for Pythia model training.
        model_size (str): Pythia model size.

    Returns:
        DataFrame: Spark DataFrame containing the dataset.
    """
    split_name = f"{scheme}.{model_size}"
    required_columns = ["sequence_id", "tokens"]
    is_test = dataset_name == "pile_test"
    is_memorized = dataset_name == "memories"

    if is_memorized:
        hf_dataset_name = f"EleutherAI/pythia-memorized-evals"
    elif is_test:
        hf_dataset_name = f"usvsnsp/pile-test-sampled"

    cache_name = hf_dataset_name if is_test else f"{hf_dataset_name}-{split_name}"
    cache_path = f"{SPARK_CACHE_DIR}/{cache_name}"

    if os.path.isdir(cache_path):
        LOGGER.info(f"Dataset {hf_dataset_name} already exists, skipping the download.")
        return SPARK.read.parquet(cache_path)

    LOGGER.info(f"Downloading dataset {hf_dataset_name}...")
    if is_memorized:
        dataset = hf_load_dataset(hf_dataset_name, split=split_name).to_pandas().rename(columns={"index": "sequence_id"})
        dataset.tokens = dataset.tokens.map(lambda x: x.tolist())
        dataset = dataset[required_columns]
    elif is_test:
        dataset = hf_load_dataset(hf_dataset_name, split="train").to_pandas()
        dataset.tokens = dataset.tokens.map(lambda x: x.tolist())

    LOGGER.info(f"Converting and caching the dataset {hf_dataset_name} as Spark DataFrame in {cache_path}...")
    # Convert the Pandas DataFrame dataset to Spark DataFrame in Parquet
    SPARK.createDataFrame(dataset).repartition(NUM_SPARK_PARTITIONS).write.parquet(cache_path)

    return SPARK.read.parquet(cache_path)


def load_precomputed_features(scheme: str, is_test=False) -> Dict[PrecomputedFeatureName, DataFrame]:
    """
    Load the pre-computed features from HuggingFace datasets. If the features are not locally available, then
    download them from HuggingFace datasets and cache them as Spark DataFrames in Parquet format.

    Args:
        scheme (str): Data scheme used for Pythia model training.
        is_test (bool): Load a sampled versions if required in case of testing

    Returns:
        Dict[PrecomputedFeatureName, DataFrame]: Dictionary of pre-computed features.
    """
    features = {}
    hf_dataset_names = [
        (PrecomputedFeatureName.SEQUENCE_FREQUENCIES, f"usvsnsp/{scheme}-num-duplicates", "train", {"Index": "sequence_id", "Counts": "frequency"}),
        (
            PrecomputedFeatureName.MEMORIZED_TOKEN_FREQUENCIES,
            f"usvsnsp/{scheme}-num-frequencies",
            "memorized",
            {"TokenID": "token_id", "Frequency": "frequency"},
        ),
        (
            PrecomputedFeatureName.NON_MEMORIZED_TOKEN_FREQUENCIES,
            f"usvsnsp/{scheme}-num-frequencies",
            "non_memorized",
            {"TokenID": "token_id", "Frequency": "frequency"},
        ),
    ]

    for enum, name, split, column_mapping in hf_dataset_names:
        cache_path = f"{SPARK_CACHE_DIR}/{name}-{split}"

        # If we're testing, then control the number of rows to load
        num_test_rows = 3000
        adjusted_split = f"{split}-test" if is_test else split
        adjusted_hf_split = f"{split}[:{num_test_rows}]" if is_test else split
        adjusted_cache_path = f"{cache_path}-test" if is_test else cache_path

        if os.path.isdir(adjusted_cache_path):
            LOGGER.info(f"Dataset {name}-{adjusted_split} already exists, skipping the download.")
            features[enum] = SPARK.read.parquet(adjusted_cache_path)
            continue

        LOGGER.info(f"Downloading dataset {name}-{adjusted_split}...")
        dataset = hf_load_dataset(name, split=adjusted_hf_split).to_pandas().rename(columns=column_mapping)

        LOGGER.info(f"Converting and caching the dataset {name}-{adjusted_split} as Spark DataFrame {adjusted_cache_path}...")
        # Convert the Pandas DataFrame dataset to Spark DataFrame in Parquet
        SPARK.createDataFrame(dataset).repartition(NUM_SPARK_PARTITIONS).write.parquet(adjusted_cache_path)

        features[enum] = SPARK.read.parquet(adjusted_cache_path).cache()

    return features


def run_non_pile_pipeline(
    dataset: DataFrame,
    dataset_name: str,
    split_name: str,
    run_id: str,
    sample_size: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> None:
    """
    Run the pipeline for non-Pile datasets.

    Args:
        dataset (DataFrame): Spark DataFrame containing the dataset.
        dataset_name (str): Name of the dataset.
        split_name (str): Name of the split.
        run_id (str): ID of the run.
        sample_size (Optional[int]): Number of samples to take from the dataset.
        sample_seed (Optional[int]): Seed to use for sampling the dataset.

    Returns:
        None
    """
    if sample_size is not None:
        dataset = dataset.sample(1.0, seed=sample_seed).limit(sample_size)

    transformed_dataset = PIPELINE.transform(dataset)
    # Non-pile datasets already indicate that all sequences are memorized.
    transformed_dataset = transformed_dataset.withColumn("memorization_score", F.lit(1.0))
    LOGGER.info(f"Transformed Dataset {dataset_name}-{split_name} Schema:")
    transformed_dataset.printSchema()
    LOGGER.info(f"{transformed_dataset.schema.simpleString()}")
    file_name = split_name.replace(".", "_", 1)
    transformed_dataset.coalesce(NUM_OUTPUT_PARTITIONS).write.parquet(f"datasets/{run_id}/{dataset_name}_{file_name}")


def run_pile_pipeline(
    dataset: DataFrame,
    dataset_name: str,
    data_scheme: str,
    model_sizes: List[str],
    run_id: str,
    sample_size: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> None:
    """
    Run the pipeline for Pile datasets.

    Args:
        dataset (DataFrame): Spark DataFrame containing the dataset.
        dataset_name (str): Name of the dataset.
        data_scheme (str): Data scheme used for Pythia model training.
        model_sizes (List[str]): List of Pythia model sizes.
        run_id (str): ID of the run.
        sample_size (Optional[int]): Number of samples to take from the dataset.
        sample_seed (Optional[int]): Seed to use for sampling the dataset.

    Returns:
        None
    """
    if sample_size is not None:
        dataset = dataset.sample(1.0, seed=sample_seed).limit(sample_size)

    main = dataset.alias("main")
    no_scores = main.select("sequence_id", "tokens")
    transformed_dataset = PIPELINE.transform(no_scores).alias("transformed")

    # Memorization score already exists per model size, we'll perform the join to export
    # each dataset by model size separately.
    for model_size in model_sizes:
        memorization_scores = main.select(
            "main.sequence_id",
            F.col(f"main.{model_size}").alias("memorization_score"),
        ).alias("score")
        joined_dataset = transformed_dataset.join(memorization_scores, on="sequence_id", how="left").select(
            "transformed.*",
            "score.memorization_score",
        )
        split_name = f"{data_scheme}.{model_size}"
        LOGGER.info(f"Transformed Dataset {dataset_name}-{split_name} Schema:")
        joined_dataset.printSchema()
        LOGGER.info(f"{joined_dataset.schema.simpleString()}")
        file_name = split_name.replace(".", "_", 1)
        joined_dataset.coalesce(NUM_OUTPUT_PARTITIONS).write.parquet(f"datasets/{run_id}/{dataset_name}_{file_name}")


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
    if args.sample_size is not None:
        LOGGER.info(f"Sample size: {args.sample_size}")
    if args.sample_seed is not None:
        LOGGER.info(f"Sample seed: {args.sample_seed}")
    LOGGER.info("---------------------------------------------------------------------------")

    model_sizes = args.models if isinstance(args.models, list) else args.models.split(",")
    dataset_names = args.datasets if isinstance(args.datasets, list) else args.datasets.split(",")
    data_schemes = args.schemes if isinstance(args.schemes, list) else args.schemes.split(",")

    for dataset_name in dataset_names:
        is_test = dataset_name == "pile_test"
        is_memorized = dataset_name == "memories"
        is_pile = dataset_name == "pile"

        for data_scheme in data_schemes:
            LOGGER.info("Loading pre-computed features...")
            precomputed_features = load_precomputed_features(data_scheme, is_test=is_test)
            PIPELINE.register_features(precomputed_features)

            if is_memorized or is_test:
                # The memorized dataset has multiple splits by the model size
                for model_size in model_sizes:
                    split_name = f"{data_scheme}.{model_size}"
                    LOGGER.info(f"Loading dataset {dataset_name} and split {split_name}...")
                    dataset = load_non_pile_dataset(dataset_name, data_scheme, model_size)
                    LOGGER.info(f"Calculating metrics for {split_name} on dataset {dataset_name}...")
                    run_non_pile_pipeline(dataset, dataset_name, split_name, args.run_id, args.sample_size, args.sample_seed)
            elif is_pile:
                LOGGER.info(f"Loading dataset {dataset_name}...")
                # The pile dataset contains all model sizes in a single split
                dataset = load_pile_dataset(data_scheme)
                LOGGER.info(f"Calculating metrics for {data_scheme} on dataset {dataset_name}...")
                run_pile_pipeline(dataset, dataset_name, data_scheme, model_sizes, args.run_id, args.sample_size, args.sample_seed)

            # Clear the cache because pre-computed features are differentiated based on the data scheme
            SPARK.catalog.clearCache()


if __name__ == "__main__":
    main()
