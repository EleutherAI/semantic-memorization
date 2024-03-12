import itertools
import json
import logging
import os
import pickle
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats
from scipy.stats import pearsonr as pearson_correlation
from scipy.stats import spearmanr as spearman_correlation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xicorpy import compute_xi_correlation as xi_correlation  # https://swarnakumar.github.io/xicorpy/xi/

from model_parameters import (
    CATEGORICAL_FEATURE_COLUMNS,
    CLASS_WEIGHT_METHOD,
    CONTINUOUS_FEATURE_COLUMNS,
    DATA_SCHEME,
    EXPERIMENT_ROOT,
    FIT_INTERCEPT,
    GLOBAL_SEED,
    MAX_MODEL_ITERATIONS,
    MODEL_SIZE,
    GENERATION_HF_DATASET_NAME,
    REG_NAME,
    REG_STRENGTH,
    TAXONOMIES,
    TAXONOMY_QUANTILES,
    TAXONOMY_SEARCH_FEATURES,
    TEST_SIZE,
    TRAIN_SIZE,
    VALIDATION_SIZE,
    derive_is_templating_feature,
    taxonomy_function,
)

LOGGER = logging.getLogger("experiments")
LOGGER.setLevel(logging.INFO)

Dataset = Tuple[np.ndarray, np.ndarray]


@dataclass
class ModelResult:
    model: LogisticRegression
    train_roc_auc: float
    train_pr_auc: float
    test_roc_auc: float
    test_pr_auc: float
    validation_roc_auc: float
    validation_pr_auc: float
    wald_stats: List[float]
    wald_pvalue: List[float]
    lrt_pvalue: Optional[float]
    baseline_roc_auc: Optional[float]
    baseline_pr_auc: Optional[float]


def parse_cli_args() -> Namespace:
    """
    Parse the command line arguments for the script.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--run_id",
        type=str,
        help="The ID for this run. Defaults to current date and time.",
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    parser.add_argument(
        "--taxonomy_search_start_index",
        type=int,
        help="The starting index for the list of taxonomy search candidates",
        default=None,
    )

    parser.add_argument(
        "--taxonomy_search_end_index",
        type=int,
        help="The ending index for the list of taxonomy search candidates",
        default=None,
    )

    parser.add_argument(
        "--sequence_duplication_threshold",
        type=int,
        help="The threshold to classify a sample as recitation",
        default=10,
    )

    return parser.parse_args()


def load_hf_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the HF datasets from the HuggingFace Hub. Then merge the generation and entropy datasets based on the sequence ID.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The pile dataset and the memories dataset.
    """
    pile_dataset = load_dataset(GENERATION_HF_DATASET_NAME, split=f"pile_{DATA_SCHEME}_{MODEL_SIZE}").to_pandas()
    memories_dataset = load_dataset(GENERATION_HF_DATASET_NAME, split=f"memories_{DATA_SCHEME}_{MODEL_SIZE}").to_pandas()

    LOGGER.info(f"Pile generation dataset shape: {pile_dataset.shape}")
    LOGGER.info(f"Memories generation dataset shape: {memories_dataset.shape}")

    LOGGER.info("Merging generation and entropy datasets...")

    # Drop duplicate sequence IDs
    # Observation -- Only sequence ID `101275048` was duplicated. Some columns have different values, e.g. perplexity statistics.
    # TODO: Investigate data generation pipeline
    memories_dataset.drop_duplicates("sequence_id", keep="first", inplace=True)
    pile_dataset.drop_duplicates("sequence_id", keep="first", inplace=True)

    LOGGER.info(f"Final pile dataset shape: {pile_dataset.shape}")
    LOGGER.info(f"Final memories dataset shape: {memories_dataset.shape}")

    return pile_dataset, memories_dataset


def construct_derived_features(pile_dataset: pd.DataFrame, memories_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct the derived features for the pile and memories datasets.

    Args:
        pile_dataset (pd.DataFrame): The pile dataset.
        memories_dataset (pd.DataFrame): The memories dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The pile dataset and the memories dataset with the derived features.
    """
    LOGGER.info("Constructing `is_templating` feature...")
    pile_dataset["is_templating"] = pile_dataset.apply(derive_is_templating_feature, axis=1)
    memories_dataset["is_templating"] = memories_dataset.apply(derive_is_templating_feature, axis=1)

    return pile_dataset, memories_dataset


def preprocess_dataset(pile_dataset: pd.DataFrame, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize the dataset and return the features and labels.

    Args:
        pile_dataset (pd.DataFrame): The pile dataset.
        normalize: (bool): Whether to normalize dataset

    Returns:
        Tuple[np.ndarray, np.ndarray]: The features and labels.
    """
    # Scale the continuous features to be zero-mean and unit-variance
    if normalize:
        feature_scaler = StandardScaler().fit(pile_dataset[CONTINUOUS_FEATURE_COLUMNS])
        continuous_features = feature_scaler.transform(pile_dataset[CONTINUOUS_FEATURE_COLUMNS])
    else:
        continuous_features = pile_dataset[CONTINUOUS_FEATURE_COLUMNS]
    categorical_features = pile_dataset[CATEGORICAL_FEATURE_COLUMNS]
    features = np.hstack((continuous_features, categorical_features))
    labels = (pile_dataset.memorization_score == 1.0).astype(int).values

    processed_df = pd.DataFrame(
        continuous_features,
        columns=CONTINUOUS_FEATURE_COLUMNS,
        index=categorical_features.index,
    )
    processed_df = processed_df.join(categorical_features)

    return features, labels, processed_df


def split_dataset(features, labels) -> Optional[Tuple[Dataset, Dataset, Dataset]]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.

    Returns:
        Optional[Tuple[Dataset, Dataset, Dataset]]: The training, validation, and test sets.
    """

    try:
        one_features, two_features, one_labels, two_labels = train_test_split(
            features,
            labels,
            test_size=1 - TRAIN_SIZE,
            random_state=GLOBAL_SEED,
            stratify=labels,
        )

        return (
            (one_features, one_labels),
            (two_features, two_labels),
        )
    except Exception as e:
        LOGGER.error(f"Dataset splitting failed with Exception {e}")
        return None


def calculate_label_priors(labels: np.ndarray):
    """
    Calculate the label priors.

    Args:
        labels (np.ndarray): The labels.

    Returns:
        None
    """
    prob_negatives = (labels == 0).astype(int).sum() / len(labels)
    prob_positives = 1 - prob_negatives

    LOGGER.info("Class Priors")
    LOGGER.info(f"Memorized (+): {prob_positives * 100:.4f}%")
    LOGGER.info(f"Non-memorized (-): {prob_negatives * 100:.4f}%")
    LOGGER.info("=" * 30)


def likelihood_ratio_test(baseline_predictions: np.ndarray, taxonomic_predictions: np.ndarray, labels: np.ndarray, dof: int = 1) -> np.ndarray:
    """
    Perform the likelihood ratio test to compare the baseline model with the taxonomic model.

    Args:
        baseline_predictions (np.ndarray): The baseline model predictions.
        taxonomic_predictions (np.ndarray): The taxonomic model predictions.
        labels (np.ndarray): The labels.
        dof (int, optional): The degrees of freedom. Defaults to 1.

    Returns:
        np.ndarray: The p-value of the test.
    """
    # Reference: https://stackoverflow.com/questions/48185090/how-to-get-the-log-likelihood-for-a-logistic-regression-model-in-sklearn
    # H_0 (Null Hypothesis)
    baseline_log_likelihood = -log_loss(labels, baseline_predictions, normalize=False)
    # H_1 (Alternative Hypothesis)
    taxonomic_log_likelihood = -log_loss(labels, taxonomic_predictions, normalize=False)

    # References
    # - https://stackoverflow.com/questions/38248595/likelihood-ratio-test-in-python
    # - https://rnowling.github.io/machine/learning/2017/10/07/likelihood-ratio-test.html
    likelihood_ratio = -2 * (baseline_log_likelihood - taxonomic_log_likelihood)
    pvalue = stats.chi2.sf(likelihood_ratio, df=dof)

    return pvalue


def wald_test(model: LogisticRegression, features: np.ndarray) -> Tuple[float, List[float]]:
    """
    Perform the Wald Test to determine the significance of the coefficients.

    Args:
        model (LogisticRegression): The model.
        features (np.ndarray): The features.

    Returns:
        Tuple[List[float], List[float]]: The Wald statistics and p-values.
    """
    probs = model.predict_proba(features)
    num_samples = len(probs)
    # Include the intercept term as the first feature
    num_features = len(model.coef_[0]) + 1
    coefficients = np.concatenate([model.intercept_, model.coef_[0]])
    full_features = np.matrix(np.insert(np.array(features), 0, 1, axis=1))

    # References
    # - https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
    ans = np.zeros((num_features, num_features))
    for i in range(num_samples):
        ans = ans + np.dot(np.transpose(full_features[i, :]), full_features[i, :]) * probs[i, 1] * probs[i, 0]
    var_covar_matrix = np.linalg.inv(np.matrix(ans))
    std_err = np.sqrt(np.diag(var_covar_matrix))
    wald = (coefficients**2) / (std_err**2)
    pvalues = np.array([2 * (1 - stats.norm.cdf(np.abs(w))) for w in np.sqrt(wald)])

    return wald.tolist(), pvalues.tolist()


def calculate_correlation_coefficients(features: np.ndarray, labels: np.ndarray) -> Tuple[List, List, List]:
    """
    Calculate the correlation coefficients for each feature.

    Args:
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.

    Returns:
        Tuple[List, List, List]: The Pearson, Spearman, and Xi correlation coefficients with p-values.
    """
    pearsons, spearmans, xis = [], [], []

    for i in range(features.shape[1]):
        LOGGER.info(f"Calculating correlation coefficients on feature index {i}...")
        feature = features[:, i]
        pearson_result = pearson_correlation(feature, labels, alternative="two-sided")
        spearman_result = spearman_correlation(feature, labels, alternative="two-sided")
        xi_result = xi_correlation(feature, labels, get_modified_xi=False, get_p_values=True)
        xi_statistic, xi_pvalue = float(xi_result[0][0, 0]), float(xi_result[1][0, 0])

        pearsons.append((pearson_result.statistic, pearson_result.pvalue))
        spearmans.append((spearman_result.statistic, spearman_result.pvalue))
        xis.append((xi_statistic, xi_pvalue))

    return pearsons, spearmans, xis


def calculate_all_correlation_coefficients(features: np.ndarray, labels: np.ndarray, taxonomy_categories: pd.Series, args: Namespace) -> DefaultDict:
    """
    Calculate the correlation coefficients for each feature and taxonomy.

    Args:
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.
        taxonomy_categories (pd.Series): The taxonomy categories.
        args (Namespace): The command line arguments.

    Returns:
        DefaultDict: The correlation coefficients for each feature and taxonomy.
    """
    coefficients = defaultdict(dict)
    coefficients["metadata"]["sequence_duplication_threshold"] = args.sequence_duplication_threshold

    baseline_pearson, baseline_spearman, baseline_xi = calculate_correlation_coefficients(features, labels)
    coefficients["baseline"]["pearson"] = baseline_pearson
    coefficients["baseline"]["spearman"] = baseline_spearman
    coefficients["baseline"]["xi"] = baseline_xi

    for taxonomy in TAXONOMIES:
        sample_indices = taxonomy_categories.index[taxonomy_categories == taxonomy]
        taxonomic_features, taxonomic_labels = features[sample_indices, :], labels[sample_indices]
        taxonomic_pearson, taxonomic_spearman, taxonomic_xi = calculate_correlation_coefficients(taxonomic_features, taxonomic_labels)
        coefficients[taxonomy]["pearson"] = taxonomic_pearson
        coefficients[taxonomy]["spearman"] = taxonomic_spearman
        coefficients[taxonomy]["xi"] = taxonomic_xi

    return coefficients


def save_correlation_coefficients(base_path: str, data_scheme: str, model_size: str, coefficients: DefaultDict):
    """
    Save the correlation coefficients to a JSON file.

    Args:
        base_path (str): The base path to save the coefficients.
        data_scheme (str): The data scheme.
        model_size (str): The model size.
        coefficients (DefaultDict): The correlation coefficients.

    Returns:
        None
    """
    full_path = f"{base_path}/{data_scheme}/{model_size}"

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    metadata_path = os.path.join(full_path, "correlation_coefficients.json")
    with open(metadata_path, "w") as file:
        json.dump(coefficients, file)


def save_lr_models(base_path: str, data_scheme: str, taxonomy: str, model_size: str, model: LogisticRegression, model_metadata: Dict[Any, Any]):
    """
    Save the LR model and metadata to a pickle file.

    Args:
        base_path (str): The base path to save the coefficients.
        data_scheme (str): The data scheme.
        taxonomy (str): The taxonomy.
        model_size (str): The model size.
        model (LogisticRegression): The model.
        model_metadata (Dict[Any, Any]): The model metadata.

    Returns:
        None
    """
    full_path = f"{base_path}/{data_scheme}/{model_size}/{taxonomy}"

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    model_path = os.path.join(full_path, "lr.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    metadata = {
        "data_scheme": data_scheme,
        "taxonomy": taxonomy,
        "model_size": model_size,
        **model_metadata,
    }

    metadata_path = os.path.join(full_path, "metadata.json")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)


def train_lr_model(
    train_features: np.ndarray, 
    train_labels: np.ndarray, 
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray, 
    test_labels: np.ndarray,
) -> Tuple[LogisticRegression, np.ndarray, 
    Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Train the LR model.

    Args:
        train_features (np.ndarray): The training features.
        train_labels (np.ndarray): The training labels.
        test_features (np.ndarray): The evaluation features.
        test_labels (np.ndarray): The evaluation labels.
        validation_features (np.ndarray): The evaluation features.
        validation_labels (np.ndarray): The evaluation labels.

    Returns:
        Tuple[LogisticRegression, np.ndarray, float, float, float, float]: The trained model and test/evaluation metrics.
    """
    # Training with fixed parameters
    model = LogisticRegression(
        fit_intercept=FIT_INTERCEPT,
        random_state=GLOBAL_SEED,
        max_iter=MAX_MODEL_ITERATIONS,
        penalty=REG_NAME,
        C=REG_STRENGTH,
        class_weight=CLASS_WEIGHT_METHOD,
    )
    model.fit(train_features, train_labels)

    # Calculate classification metrics
    train_predictions = model.predict_proba(train_features)[:, 1]
    train_roc_auc = roc_auc_score(train_labels, train_predictions)
    train_pr_auc = average_precision_score(train_labels, train_predictions)

    valid_predictions = model.predict_proba(validation_features)[:, 1]
    validation_roc_auc = roc_auc_score(validation_labels, valid_predictions)
    validation_pr_auc = average_precision_score(validation_labels, valid_predictions)

    test_predictions = model.predict_proba(test_features)[:, 1]
    test_roc_auc = roc_auc_score(test_labels, test_predictions)
    test_pr_auc = average_precision_score(test_labels, test_predictions)

    LOGGER.info(f"Training ROC AUC: {train_roc_auc:.4f} | Training PR AUC: {train_pr_auc:.4}")
    LOGGER.info(f"Test ROC AUC: {test_roc_auc:.4f} | Test PR AUC: {test_pr_auc:.4}")
    LOGGER.info(f"Validation ROC AUC: {validation_roc_auc:.4f} | Validation PR AUC: {validation_pr_auc:.4}")

    return (
        model, 
        test_predictions,
        (train_roc_auc, train_pr_auc),
        (validation_roc_auc, validation_pr_auc),
        (test_roc_auc, test_pr_auc),
    )


def train_baseline_model(
    features: np.ndarray,
    labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray
) -> Optional[ModelResult]:
    """
    Train the baseline model.

    Args:
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.
        test_features (np.ndarray): The test features.
        test_labels (np.ndarray): The test labels.

    Returns:
        Optional[ModelResult]: The baseline model result.
    """
    calculate_label_priors(labels)

    datasets = split_dataset(features, labels)
    if datasets is None:
        LOGGER.error("Dataset splitting failed, returning a null model...")
        return None

    train, validation = datasets
    train_features, train_labels = train
    validation_features, validation_labels = validation

    model, test_predictions, train_metrics, validation_metrics, test_metrics = train_lr_model(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    )

    try:
        LOGGER.info("Performing Wald Test...")
        wald_stats, wald_pvalue = wald_test(model, test_features)
    except Exception as e:
        LOGGER.info(f"Wald Test failed with Exception {e}")
        wald_stats, wald_pvalue = [], []

    train_roc_auc, train_pr_auc = train_metrics
    test_roc_auc, test_pr_auc = test_metrics
    validation_roc_auc, validation_pr_auc = validation_metrics


    return ModelResult(
        model=model,
        train_roc_auc=train_roc_auc,
        train_pr_auc=train_pr_auc,
        validation_roc_auc=validation_roc_auc,
        validation_pr_auc=validation_pr_auc,
        test_roc_auc=test_roc_auc,
        test_pr_auc=test_pr_auc,
        wald_stats=wald_stats,
        wald_pvalue=wald_pvalue,
        # Not applicable since there are no alternative models to compare with, this is the baseline
        lrt_pvalue=None,
        baseline_roc_auc=None,
        baseline_pr_auc=None,
    )


def train_taxonomic_model(
    features: np.ndarray,
    labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    baseline_model: LogisticRegression,
) -> Optional[ModelResult]:
    """
    Train the taxonomic model.

    Args:
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.
        test_features (np.ndarray): The test features.
        test_labels (np.ndarray): The test labels.
        baseline_model (LogisticRegression): The baseline model.

    Returns:
        ModelResult: The taxonomic model result.
    """
    calculate_label_priors(labels)

    datasets = split_dataset(features, labels)
    if datasets is None:
        LOGGER.error("Dataset splitting failed, returning a null model...")
        return None

    train, validation = datasets
    train_features, train_labels = train
    validation_features, validation_labels = validation

    model, test_predictions, train_metrics, validation_metrics, test_metrics = train_lr_model(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        test_features,
        test_labels,
    )

    train_roc_auc, train_pr_auc = train_metrics
    test_roc_auc, test_pr_auc = test_metrics
    validation_roc_auc, validation_pr_auc = validation_metrics

    LOGGER.info("Getting baseline predictions on the test set...")
    baseline_test_predictions = baseline_model.predict_proba(test_features)[:, 1]

    baseline_roc_auc = roc_auc_score(test_labels, baseline_test_predictions)
    baseline_pr_auc = average_precision_score(test_labels, baseline_test_predictions)

    lrt_pvalue = None
    wald_stats, wald_pvalue = [], []

    LOGGER.info("Performing Likelihood Ratio Test...")
    lrt_pvalue = likelihood_ratio_test(baseline_test_predictions, test_predictions, test_labels)

    try:
        LOGGER.info("Performing Wald Test...")
        wald_stats, wald_pvalue = wald_test(model, test_features)
    except Exception as e:
        LOGGER.info(f"Wald Test failed with Exception {e}")

    return ModelResult(
        model=model,
        train_roc_auc=train_roc_auc,
        train_pr_auc=train_pr_auc,
        validation_roc_auc=validation_roc_auc,
        validation_pr_auc=validation_pr_auc,
        test_roc_auc=test_roc_auc,
        test_pr_auc=test_pr_auc,
        baseline_roc_auc=baseline_roc_auc,
        baseline_pr_auc=baseline_pr_auc,
        wald_stats=wald_stats,
        wald_pvalue=wald_pvalue,
        lrt_pvalue=lrt_pvalue,
    )

def train_and_save_baseline_and_taxonomic_models(
    experiment_base: str,
    features: np.ndarray,
    labels: np.ndarray,
    taxonomy_categories_train: pd.Series,
    taxonomy_categories_test: pd.Series,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    args: Namespace,
) -> Optional[Tuple[ModelResult, List[Tuple[str, ModelResult]]]]:
    """
    Train and save the baseline and taxonomic models.

    Args:
        experiment_base (str): The experiment base path.
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.
        taxonomy_categories_train (pd.Series): The train taxonomy categories.
        taxonomy_categories_test (pd.Series): The test taxonomy categories.
        args (Namespace): The command line arguments.

    Returns:
        Optional[Tuple[ModelResult, List[Tuple[str, ModelResult]]]]: The baseline model result and the taxonomic model results.
    """
    LOGGER.info("Training the baseline model with all data...")

    baseline_result = train_baseline_model(features, labels, test_features, test_labels)

    if baseline_result is None:
        LOGGER.error("Baseline model is null, skipping...")
        return None

    LOGGER.info("Saving baseline model results...")
    baseline_model = baseline_result.model

    # hack to remove a field
    baseline_metadata = {
        "sequence_duplication_threshold": args.sequence_duplication_threshold,
        **asdict(baseline_result, dict_factory=lambda x: {k: v for (k, v) in x if k != 'model'})
    }
    save_lr_models(experiment_base, DATA_SCHEME, "baseline", MODEL_SIZE, baseline_model, baseline_metadata)

    taxonomic_results = []
    for taxonomy in TAXONOMIES:
        sample_indices = taxonomy_categories_train.index[taxonomy_categories_train == taxonomy]
        taxonomic_features, taxonomic_labels = features[sample_indices, :], labels[sample_indices]

        sample_indices = taxonomy_categories_test.index[taxonomy_categories_test == taxonomy]
        taxonomic_features_test = test_features[sample_indices, :]
        taxonomic_labels_test = test_labels[sample_indices]

        LOGGER.info(f"Training {taxonomy}-partitioned model...")
        taxonomic_model_result = train_taxonomic_model(
            taxonomic_features,
            taxonomic_labels,
            taxonomic_features_test,
            taxonomic_labels_test,
            baseline_result.model,
        )

        if taxonomic_model_result is None:
            LOGGER.error(f"{taxonomy}-partitioned model is null, skipping...")
            continue

        LOGGER.info(f"Saving {taxonomy}-partitioned model results...")

        baseline_predictions = baseline_result.model.predict_proba(taxonomic_features_test)[:, 1]
        
        taxonomic_model = taxonomic_model_result.model
        taxonomic_model_metadata = {
            "sequence_duplication_threshold": args.sequence_duplication_threshold,
            **asdict(taxonomic_model_result, dict_factory=lambda x: {k: v for (k, v) in x if k != 'model'})
        }
        save_lr_models(experiment_base, DATA_SCHEME, taxonomy, MODEL_SIZE, taxonomic_model, taxonomic_model_metadata)
        taxonomic_results.append((taxonomy, taxonomic_model_result))

    return baseline_result, taxonomic_results


def save_taxonomy_search_models(
    base_path: str,
    data_scheme: str,
    model_size: str,
    taxonomy_1_name: str,
    taxonomy_1_threshold_quantile: float,
    taxonomy_2_name: str,
    taxonomy_2_threshold_quantile: float,
    model: Any,
    model_metadata: Dict[str, Any],
) -> None:
    """
    Save the taxonomy search models to the specified location.

    Args:
        base_path (str): The base path where the models will be saved.
        data_scheme (str): The data scheme.
        model_size (str): The model size.
        taxonomy_1_name (str): The name of the first taxonomy feature.
        taxonomy_1_threshold_quantile (float): The threshold quantile for the first taxonomy feature.
        taxonomy_2_name (str): The name of the second taxonomy feature.
        taxonomy_2_threshold_quantile (float): The threshold quantile for the second taxonomy feature.
        model (Any): The trained model to be saved.
        model_metadata (Dict[str, Any]): Additional metadata to be saved along with the model.

    Returns:
        None
    """
    full_path = f"{base_path}/{data_scheme}/{model_size}/taxonomy_search/{taxonomy_1_name}-{taxonomy_1_threshold_quantile}/{taxonomy_2_name}-{taxonomy_2_threshold_quantile}"

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    model_path = os.path.join(full_path, "lr.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    metadata = {
        "data_scheme": data_scheme,
        "model_size": model_size,
        "taxonomy_1_feature_name": taxonomy_1_name,
        "taxonomy_1_threshold_quantile": taxonomy_1_threshold_quantile,
        "taxonomy_2_feature_name": taxonomy_2_name,
        "taxonomy_2_threshold_quantile": taxonomy_2_threshold_quantile,
        **model_metadata,
    }

    metadata_path = os.path.join(full_path, "metadata.json")
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)


def generate_taxonomy_quantile_thresholds(memories_dataset: pd.DataFrame) -> DefaultDict:
    """
    Generate the taxonomy quantile thresholds for each feature.

    Args:
        memories_dataset (pd.DataFrame): The memories dataset.

    Returns:
        DefaultDict: The taxonomy quantile thresholds for each feature.
    """
    taxonomy_thresholds = defaultdict(dict)

    for feature in tqdm(TAXONOMY_SEARCH_FEATURES):
        for quantile in TAXONOMY_QUANTILES:
            threshold = memories_dataset[feature].quantile(quantile)
            taxonomy_thresholds[feature][quantile] = threshold

    LOGGER.info(f"Taxonomy Search Quantile Thresholds {taxonomy_thresholds}")

    return taxonomy_thresholds


def generate_optimal_taxonomy_candidate(feature_1, threshold_1, feature_2, threshold_2) -> Callable[[pd.Series], str]:
    """
    Generate the optimal taxonomy candidate function.

    Args:
        feature_1 (str): The name of the first feature.
        threshold_1 (float): The threshold for the first feature.
        feature_2 (str): The name of the second feature.
        threshold_2 (float): The threshold for the second feature.

    Returns:
        Callable[[pd.Series], str]: The optimal taxonomy candidate function.
    """

    def classify_row(row: pd.Series):
        has_taxonomy_1 = row[feature_1] > threshold_1
        has_taxonomy_2 = row[feature_2] > threshold_2

        if has_taxonomy_1:
            return "taxonomy_1"

        if has_taxonomy_2:
            return "taxonomy_2"

        return "taxonomy_3"

    return classify_row


def check_training_eligibility(labels: np.ndarray, sample_indices: np.ndarray) -> bool:
    """
    Check if model training is availabile based on the label prior. The threshold
    could be extreme where they have no samples or only one class.

    Args:
        labels (np.ndarray): The labels.
        sample_indices (np.ndarray): The sample indices.

    Returns:
        bool: True if training is available, False otherwise.
    """
    if len(sample_indices) == 0:
        LOGGER.info("Taxonomy candidate has no samples")
        return False

    sample_labels = labels[sample_indices]
    num_samples = len(sample_labels)
    num_positives = (sample_labels == 1).astype(int).sum()
    num_negatives = num_samples - num_positives

    if num_positives == 0 or num_negatives == 0:
        LOGGER.info(f"Taxonomy candidate has no samples for one of the classes: {num_positives} positives | {num_negatives} negatives")
        return False

    return True


def train_and_save_all_taxonomy_pairs(
    experiment_base: str,
    features: np.ndarray,
    labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    baseline_model: LogisticRegression,
    taxonomy_thresholds: DefaultDict,
    processed_df_train: pd.DataFrame,
    processed_df_test: pd.DataFrame,
    start_index: int = None,
    end_index: int = None,
) -> None:
    """
    Trains models for all pairs of features to find the optimal taxonomy.

    Args:
        experiment_base (str): The experiment base path.
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.
        baseline_model (LogisticRegression): The baseline model.
        taxonomy_thresholds (DefaultDict): The taxonomy thresholds.
        processed_df_train (pd.DataFrame): The processed pile dataset, train split.
        processed_df_test (pd.DataFrame): The processed pile dataset, test split.
        start_index (int, optional): The starting index for the list of taxonomy search candidates. Defaults to None.
        end_index (int, optional): The ending index for the list of taxonomy search candidates. Defaults to None.

    Returns:
        None
    """
    LOGGER.info("Generating all pairs of features for the optimal taxonomy")
    features_with_quantiles = list(itertools.product(TAXONOMY_SEARCH_FEATURES, TAXONOMY_QUANTILES))
    # We drop candidates where they are the same feature.
    # Future work may include exploring features with interesting value regimes.
    optimal_taxonomy_candidates = sorted([t for t in list(itertools.permutations(features_with_quantiles, 2)) if t[0][0] != t[1][0]])
    LOGGER.info(f"Generated {len(optimal_taxonomy_candidates)} pairs of optimal taxonomy candidates")

    if start_index is not None or end_index is not None:
        is_start_index_valid = start_index >= 0 and start_index < len(optimal_taxonomy_candidates)
        is_end_index_valid = end_index >= 0 and end_index <= len(optimal_taxonomy_candidates)
        are_both_indices_valid = is_start_index_valid and is_end_index_valid and end_index >= start_index

        if are_both_indices_valid:
            LOGGER.info(f"Training a subset of {end_index - start_index} taxonomy candidates...")
            LOGGER.info(f"Start Index: {start_index} | End Index: {end_index}")
            optimal_taxonomy_candidates = optimal_taxonomy_candidates[start_index:end_index]
        else:
            LOGGER.info("Subset indices are not valid, training all taxonomy candidates...")

    for i, (candidate_1, candidate_2) in enumerate(optimal_taxonomy_candidates):
        candidate_1_name, candidate_1_threshold_quantile = candidate_1
        candidate_2_name, candidate_2_threshold_quantile = candidate_2
        candidate_1_threshold = taxonomy_thresholds[candidate_1_name][candidate_1_threshold_quantile]
        candidate_2_threshold = taxonomy_thresholds[candidate_2_name][candidate_2_threshold_quantile]
        LOGGER.info(f"Training [{i + 1}/{len(optimal_taxonomy_candidates)}] taxonomy candidate model...")
        LOGGER.info(f"Candidate 1 feature - {candidate_1_name} | Candidate 2 feature - {candidate_2_name}")
        LOGGER.info(f"Candidate 1 value - {candidate_1_threshold} | Candidate 2 value - {candidate_2_threshold}")

        LOGGER.info("Generating taxonomy categories...")
        taxonomy_func = generate_optimal_taxonomy_candidate(candidate_1_name, candidate_1_threshold, candidate_2_name, candidate_2_threshold)
        taxonomy_categories_train = processed_df_train.apply(taxonomy_func, axis=1)
        taxonomy_categories_test = processed_df_test.apply(taxonomy_func, axis=1)

        for taxonomy in ["taxonomy_1", "taxonomy_2", "taxonomy_3"]:
            LOGGER.info(f"Training {taxonomy} model...")

            sample_indices = taxonomy_categories_train.index[taxonomy_categories_train == taxonomy]
            if not check_training_eligibility(labels, sample_indices):
                continue

            taxonomic_features, taxonomic_labels = features[sample_indices, :], labels[sample_indices]
            
            sample_indices = taxonomy_categories_test.index[taxonomy_categories_test == taxonomy]
            if not check_training_eligibility(test_labels, sample_indices):
                continue

            taxonomic_features_test, taxonomic_labels_test = features[sample_indices, :], labels[sample_indices]


            taxonomic_model_result = train_taxonomic_model(
                taxonomic_features,
                taxonomic_labels,
                taxonomic_features_test,
                taxonomic_labels_test,
                baseline_model,
            )

            if taxonomic_model_result is None:
                LOGGER.error(f"{taxonomy} model is null, skipping...")
                continue

            LOGGER.info(f"Saving {taxonomy} model results...")

            taxonomic_model = taxonomic_model_result.model
            taxonomic_model_metadata = {
                **asdict(taxonomic_model_result, dict_factory=lambda x: {k: v for (k, v) in x if k != 'model'})
            }
            save_taxonomy_search_models(
                experiment_base,
                DATA_SCHEME,
                MODEL_SIZE,
                candidate_1_name,
                candidate_1_threshold_quantile,
                candidate_2_name,
                candidate_2_threshold_quantile,
                taxonomic_model,
                taxonomic_model_metadata,
            )


def main():
    """
    The main function of the script.
    """
    args = parse_cli_args()

    experiment_base = f"{EXPERIMENT_ROOT}/{args.run_id}"
    os.makedirs(f"{experiment_base}", exist_ok=True)
    file_handler = logging.FileHandler(f"{EXPERIMENT_ROOT}/{args.run_id}/run.log")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    LOGGER.info("---------------------------------------------------------------------------")
    LOGGER.info("Starting model training with the following parameters:")
    LOGGER.info(f"Run ID: {args.run_id}")
    if args.taxonomy_search_start_index is not None:
        LOGGER.info(f"Taxonomy Search Start Index: {args.taxonomy_search_start_index}")
    if args.taxonomy_search_end_index is not None:
        LOGGER.info(f"Taxonomy Search End Index: {args.taxonomy_search_end_index}")
    LOGGER.info(f"Sequence Duplication Threshold: {args.sequence_duplication_threshold}")
    LOGGER.info("---------------------------------------------------------------------------")

    LOGGER.info("Loading HF datasets...")
    pile_dataset, memories_dataset = load_hf_dataset()

    LOGGER.info("Constructing derived features...")
    pile_dataset, memories_dataset = construct_derived_features(pile_dataset, memories_dataset)

    LOGGER.info("Generating taxonomy categories for each sample...")
    taxonomy_func = taxonomy_function(args.sequence_duplication_threshold)
    taxonomy_categories = pile_dataset.apply(taxonomy_func, axis=1)

    features, labels, processed_df = preprocess_dataset(pile_dataset, normalize=True)

    LOGGER.info("Calculating correlation coefficients of base + taxonomic features...")
    correlation_results = calculate_all_correlation_coefficients(features, labels, taxonomy_categories, args)
    save_correlation_coefficients(experiment_base, DATA_SCHEME, MODEL_SIZE, correlation_results)

    LOGGER.info("Training baseline and taxonomic models...")

    ds = pd.DataFrame({
        'features': features.tolist(),
        'labels': labels,
        'taxonomy': taxonomy_categories
    })
    ds['strat'] = ds['labels'].astype('str') + ds['taxonomy'].astype('str')

    train, test = split_dataset(ds, ds['strat'])
    features, labels = np.array(train[0]['features'].to_list()), train[0]['labels'].to_numpy()
    test_features, test_labels = np.array(test[0]['features'].to_list()), test[0]['labels'].to_numpy()
    taxonomy_categories_train = train[0]['taxonomy'].reset_index(drop=True)
    taxonomy_categories_test = test[0]['taxonomy'].reset_index(drop=True)
    processed_df_train = processed_df.loc[train[0].index].reset_index(drop=True)
    processed_df_test = processed_df.loc[test[0].index].reset_index(drop=True)

    model_results = train_and_save_baseline_and_taxonomic_models(
        experiment_base, features, labels, 
        taxonomy_categories_train, taxonomy_categories_test,
        test_features, test_labels, args
    )
    if model_results is None:
        LOGGER.error("Model results are null, exiting...")
        return

    baseline_result, _ = model_results

    LOGGER.info("Generating taxonomy quantile thresholds...")
    taxonomy_thresholds = generate_taxonomy_quantile_thresholds(memories_dataset)

    LOGGER.info("Starting to train all taxonomy pairs...")
    train_and_save_all_taxonomy_pairs(
        experiment_base, features, labels, test_features, test_labels,
        baseline_result.model, taxonomy_thresholds,
        processed_df_train, processed_df_test,
        start_index=args.taxonomy_search_start_index,
        end_index=args.taxonomy_search_end_index,
    )


if __name__ == "__main__":
    main()
