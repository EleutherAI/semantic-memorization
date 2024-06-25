import itertools
import json
import logging
import os
import pickle
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats
from scipy.stats import pearsonr as pearson_correlation
from scipy.stats import spearmanr as spearman_correlation
from sklearn.model_selection import KFold

from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    log_loss, precision_score, 
    recall_score,
    precision_recall_curve
)
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
    NATURAL_LANGUAGE_SCORE_THRESHOLDS,
    NATURAL_LANGUAGE_SCORE_COLUMN,
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

from model_utils import PredictionModel, expected_calibration_error

LOGGER = logging.getLogger("experiments")
LOGGER.setLevel(logging.INFO)

Dataset = Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]


@dataclass
class ModelResult:
    model: PredictionModel
    train_roc_auc: float
    train_pr_auc: float
    test_roc_auc: float
    test_pr_auc: float
    validation_roc_auc: float
    validation_pr_auc: float
    wald_stats: List[float]
    wald_pvalue: List[float]
    wald_columns: List[str]
    lrt_pvalue: Optional[float]
    baseline_test_roc_auc: Optional[float]
    baseline_test_pr_auc: Optional[float]
    expected_calibration_error: float


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
        default=6,
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


def preprocess_dataset(pile_dataset: pd.DataFrame, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Normalize the dataset and return the features and labels.

    Args:
        pile_dataset (pd.DataFrame): The pile dataset.
        normalize: (bool): Whether to normalize dataset

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]: The features, labels, and the processed dataset.
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
        index=pile_dataset.index,
    )
    processed_df = processed_df.join(categorical_features)

    meta_features = pile_dataset[['ds_type', NATURAL_LANGUAGE_SCORE_COLUMN]]
    processed_df = processed_df.join(meta_features)

    return features, labels, processed_df


def split_dataset(features: Union[np.ndarray, pd.DataFrame], labels: Union[np.ndarray, pd.DataFrame]) -> Optional[Tuple[Dataset, Dataset]]:
    """
    Split the dataset into training and testing sets.

    Args:
        features (Union[np.ndarray, pd.DataFrame]): The features.
        labels (Union[np.ndarray, pd.DataFrame]): The labels.

    Returns:
        Optional[Tuple[Dataset, Dataset]]: The training and testing datasets.
    """

    try:
        train_features, test_features, train_labels, test_labels = train_test_split(
            features,
            labels,
            test_size=1 - TRAIN_SIZE,
            random_state=GLOBAL_SEED,
            stratify=labels,
        )

        return (
            (train_features, train_labels),
            (test_features, test_labels),
        )
    except Exception as e:
        LOGGER.error(f"Dataset splitting failed with Exception {e}")
        return None


def calculate_label_priors(labels: pd.Series):
    """
    Calculate the label priors.

    Args:
        labels (pd.Series): The labels.

    Returns:
        None
    """
    prob_negatives = (labels == 0).mean()
    prob_positives = 1 - prob_negatives

    samples_negative = (labels == 0).sum()
    samples_positive = len(labels) - samples_negative

    LOGGER.info("Class Priors")
    LOGGER.info(f"Memorized (+): {prob_positives * 100:.4f}%, {samples_positive} samples")
    LOGGER.info(f"Non-memorized (-): {prob_negatives * 100:.4f}%, {samples_negative} samples")
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


def wald_test(model: PredictionModel, features: np.ndarray) -> Tuple[float, List[float]]:
    """
    Perform the Wald Test to determine the significance of the coefficients.

    Args:
        model (PredictionModel): The model.
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


def calculate_correlation_coefficients(features: pd.DataFrame, labels: pd.Series) -> Tuple[List, List, List]:
    """
    Calculate the correlation coefficients for each feature.

    Args:
        features (np.ndarray): The features.
        labels (np.ndarray): The labels.

    Returns:
        Tuple[List, List, List]: The Pearson, Spearman, and Xi correlation coefficients with p-values.
    """
    pearsons, spearmans, xis = [], [], []

    for col in features.columns:
        LOGGER.info(f"Calculating correlation coefficients on feature index {col}...")
        pearson_result = pearson_correlation(features[col], labels, alternative="two-sided")
        spearman_result = spearman_correlation(features[col], labels, alternative="two-sided")
        xi_result = xi_correlation(features[col].to_numpy(), labels.to_numpy(), get_modified_xi=False, get_p_values=True)
        xi_statistic, xi_pvalue = float(xi_result[0][0, 0]), float(xi_result[1][0, 0])

        pearsons.append((col, pearson_result.statistic, pearson_result.pvalue))
        spearmans.append((col, spearman_result.statistic, spearman_result.pvalue))
        xis.append((col, xi_statistic, xi_pvalue))

    return pearsons, spearmans, xis


def calculate_all_correlation_coefficients(
    dataset: pd.DataFrame,
    args: Namespace,
) -> DefaultDict:
    """
    Calculate the (baseline, code/nl, taxonomy) correlation coefficients for each feature.

    Args:
        dataset: Dataset of features
        args (Namespace): The command line arguments.

    Returns:
        DefaultDict: The (baseline, code/nl, taxonomy) correlation coefficients for each feature.
    """
    coefficients = defaultdict(dict)
    coefficients["metadata"]["sequence_duplication_threshold"] = args.sequence_duplication_threshold

    feature_cols = CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS

    baseline_pearson, baseline_spearman, baseline_xi = calculate_correlation_coefficients(dataset[feature_cols], dataset["labels"])
    coefficients["baseline"] = defaultdict(dict)
    coefficients["baseline"]["all"]["pearson"] = baseline_pearson
    coefficients["baseline"]["all"]["spearman"] = baseline_spearman
    coefficients["baseline"]["all"]["xi"] = baseline_xi

    nl_ds = dataset[dataset[NATURAL_LANGUAGE_SCORE_COLUMN] >= NATURAL_LANGUAGE_SCORE_THRESHOLDS[0]]
    nl_pearson, nl_spearman, nl_xi = calculate_correlation_coefficients(nl_ds[feature_cols], nl_ds["labels"])
    coefficients["baseline"]["natural_language"]["pearson"] = nl_pearson
    coefficients["baseline"]["natural_language"]["spearman"] = nl_spearman
    coefficients["baseline"]["natural_language"]["xi"] = nl_xi

    code_ds = dataset[dataset[NATURAL_LANGUAGE_SCORE_COLUMN] <= NATURAL_LANGUAGE_SCORE_THRESHOLDS[-1]]
    code_pearson, code_spearman, code_xi = calculate_correlation_coefficients(code_ds[feature_cols], code_ds["labels"])
    coefficients["baseline"]["code"]["pearson"] = code_pearson
    coefficients["baseline"]["code"]["spearman"] = code_spearman
    coefficients["baseline"]["code"]["xi"] = code_xi

    for taxonomy in TAXONOMIES:
        tax_all_ds = dataset[dataset["base_taxonomy"] == taxonomy]
        taxonomic_pearson, taxonomic_spearman, taxonomic_xi = calculate_correlation_coefficients(tax_all_ds[feature_cols], tax_all_ds["labels"])
        coefficients[taxonomy] = defaultdict(dict)
        coefficients[taxonomy]["all"]["pearson"] = taxonomic_pearson
        coefficients[taxonomy]["all"]["spearman"] = taxonomic_spearman
        coefficients[taxonomy]["all"]["xi"] = taxonomic_xi

        tax_nl_ds = tax_all_ds[tax_all_ds[NATURAL_LANGUAGE_SCORE_COLUMN] >= NATURAL_LANGUAGE_SCORE_THRESHOLDS[0]]
        tax_nl_pearson, tax_nl_spearman, tax_nl_xi = calculate_correlation_coefficients(tax_nl_ds[feature_cols], tax_nl_ds["labels"])
        coefficients[taxonomy]["natural_language"]["pearson"] = tax_nl_pearson
        coefficients[taxonomy]["natural_language"]["spearman"] = tax_nl_spearman
        coefficients[taxonomy]["natural_language"]["xi"] = tax_nl_xi

        tax_code_ds = tax_all_ds[dataset[NATURAL_LANGUAGE_SCORE_COLUMN] <= NATURAL_LANGUAGE_SCORE_THRESHOLDS[-1]]
        tax_code_pearson, tax_code_spearman, tax_code_xi = calculate_correlation_coefficients(tax_code_ds[feature_cols], tax_code_ds["labels"])
        coefficients[taxonomy]["code"]["pearson"] = tax_code_pearson
        coefficients[taxonomy]["code"]["spearman"] = tax_code_spearman
        coefficients[taxonomy]["code"]["xi"] = tax_code_xi

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


def save_models(save_path: str, model: PredictionModel, model_metadata: Dict[Any, Any]):
    """
    Save the LR model and metadata to a pickle file.

    Args:
        save_path (str): Path to save model data
        model (PredictionModel): The model.
        model_metadata (Dict[Any, Any]): The model metadata.

    Returns:
        (str) Path to folder of saved results
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_path = os.path.join(save_path, f"lr.pkl")
    with open(model_path, "wb") as file:
        pickle.dump(model, file)


    metadata_path = os.path.join(save_path, f"metadata.json")
    with open(metadata_path, "w") as file:
        json.dump(model_metadata, file)


def train_lr_model(
    train_features: np.ndarray, 
    train_labels: np.ndarray, 
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray, 
    test_labels: np.ndarray,
) -> Tuple[PredictionModel,
    Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
    """
    Train the LR model.

    Args:
        train_features (np.ndarray): The training features.
        train_labels (np.ndarray): The training labels.
        test_features (np.ndarray): The test features.
        test_labels (np.ndarray): The test labels.
        validation_features (np.ndarray): The validation features.
        validation_labels (np.ndarray): The validation labels.


    Returns:
        Tuple[PredictionModel, (float, float), (float, float), (float, float), float]: The trained model, 
            test/evaluation metrics and expected calibration error
    """
    # Training with fixed parameters
    model = PredictionModel(
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

    validation_predictions = model.predict_proba(validation_features)[:, 1]
    validation_roc_auc = roc_auc_score(validation_labels, validation_predictions)
    validation_pr_auc = average_precision_score(validation_labels, validation_predictions)
    # valid_precision, valid_recall, valid_thresholds = precision_recall_curve(validation_labels, validation_predictions)
    # valid_f1_score = (1/(1/valid_precision + 1/valid_recall)).tolist()
    # threshold = valid_thresholds[np.argmax(valid_f1_score)]
    # model.set_threshold(threshold)

    test_predictions = model.predict_proba(test_features)
    test_roc_auc = roc_auc_score(test_labels, test_predictions[:, 1])
    test_pr_auc = average_precision_score(test_labels, test_predictions[:, 1])

    LOGGER.info(f"Training ROC AUC: {train_roc_auc:.4f} | Training PR AUC: {train_pr_auc:.4}")
    LOGGER.info(f"Test ROC AUC: {test_roc_auc:.4f} | Test PR AUC: {test_pr_auc:.4}")
    LOGGER.info(f"Validation ROC AUC: {validation_roc_auc:.4f} | Validation PR AUC: {validation_pr_auc:.4}")
    
    ece = expected_calibration_error(test_predictions, test_labels, M=10)
    return (
        model, 
        (train_roc_auc, train_pr_auc),
        (validation_roc_auc, validation_pr_auc),
        (test_roc_auc, test_pr_auc),
        ece,
    )


def train_baseline_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Optional[ModelResult]:
    """
    Train the baseline model.

    Args:
        train_df (pd.DataFrame): Pandas dataframe with train samples
        test_df (pd.DataFrame): Pandas dataframe with test samples

    Returns:
        Optional[ModelResult]: The baseline model result.
    """
    
    train_feature_df = train_df[CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    train_label_df = train_df['labels']
    test_feature_df = test_df[CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    test_label_df = test_df['labels']
    calculate_label_priors(train_label_df)

    datasets = split_dataset(train_feature_df, train_label_df)
    if datasets is None:
        LOGGER.error("Dataset splitting failed, returning a null model...")
        return None

    train, validation = datasets
    train_feature_df, train_labels = train
    validation_feature_df, validation_labels = validation

    model, train_metrics, validation_metrics, test_metrics, ece = train_lr_model(
        train_feature_df,
        train_labels,
        validation_feature_df,
        validation_labels,
        test_feature_df,
        test_label_df,
    )

    try:
        LOGGER.info("Performing Wald Test...")
        wald_stats, wald_pvalue = wald_test(model, test_feature_df)
        wald_columns = [column for column in test_feature_df]
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
        wald_columns=wald_columns,
        # Not applicable since there are no alternative models to compare with, this is the baseline
        lrt_pvalue=None,
        baseline_test_roc_auc=None,
        baseline_test_pr_auc=None,
        expected_calibration_error=ece
    )

def train_taxonomic_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    taxonomy: str,
    baseline_model: PredictionModel,
) -> Optional[ModelResult]:
    """
    Train the taxonomic model.

    Args:
        train_df (pd.DataFrame): Pandas dataframe with train samples
        test_df (pd.DataFrame): Pandas dataframe with test samples
        taxonomy (str): Name of taxonomy being saved
        baseline_model (PredictionModel): The baseline model.

    Returns:
        Tuple[ModelResult, dict]: The taxonomic model result and model predictions
    """
    train_feature_df = train_df[CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    train_label_df = train_df['labels']

    test_feature_df = test_df[test_df['curr_taxonomy'] == taxonomy][CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    test_label_df = test_df.loc[test_feature_df.index]['labels']

    datasets = split_dataset(train_feature_df, train_label_df)
    if datasets is None:
        LOGGER.error("Dataset splitting failed, returning a null model...")
        return None, None

    train, validation = datasets
    train_feature_df, train_labels = train
    validation_feature_df, validation_labels = validation

    model, train_metrics, validation_metrics, test_metrics, ece = train_lr_model(
        train_feature_df,
        train_labels,
        validation_feature_df,
        validation_labels,
        test_feature_df,
        test_label_df,
    )

    train_roc_auc, train_pr_auc = train_metrics
    test_roc_auc, test_pr_auc = test_metrics
    validation_roc_auc, validation_pr_auc = validation_metrics

    LOGGER.info("Running baseline model on the test taxonomy set...")
    test_feature_df = test_df[CONTINUOUS_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS]
    test_label_df = test_df['labels']
    baseline_test_predictions = baseline_model.predict(test_feature_df)
    test_predictions = model.predict(test_feature_df)
    predictions = pd.DataFrame()
    predictions['labels'] = test_label_df
    predictions['taxonomy'] = test_df['curr_taxonomy']
    predictions['base_taxonomy'] = test_df['base_taxonomy']
    predictions['model_predictions'] = test_predictions
    predictions['baseline_predictions'] = baseline_test_predictions
    predictions['model_prediction_probs'] = model.predict_proba(test_feature_df).tolist()
    predictions['base_prediction_probs'] = baseline_model.predict_proba(test_feature_df).tolist()

    predictions_curr = predictions[predictions['taxonomy'] == taxonomy]

    baseline_test_taxonomy_roc_auc = roc_auc_score(
        predictions_curr['labels'], predictions_curr['baseline_predictions'])
    baseline_test_taxonomy_pr_auc = average_precision_score(
        predictions_curr['labels'], predictions_curr['baseline_predictions'])

    lrt_pvalue = None
    wald_stats, wald_pvalue, wald_columns = [], [], []

    LOGGER.info("Performing Likelihood Ratio Test...")
    lrt_pvalue = likelihood_ratio_test(
        predictions_curr['baseline_predictions'], 
        predictions_curr['model_predictions'],
        predictions_curr['labels'],
    )

    try:
        LOGGER.info("Performing Wald Test...")
        uniques = test_feature_df.nunique().reset_index()
        # We remove all features with same value across data, to avoid singular matrix on wald test
        wald_columns = []
        bad_columns = []
        for row in uniques.itertuples():
            if row._2 > 1:
                wald_columns.append(row.index)
            else:
                bad_columns.append(row.index)
        if len(bad_columns) != 0:
            LOGGER.info(f"Ignoring features {bad_columns} while performing wald test, as their values are all the same")
        wald_stats, wald_pvalue = wald_test(model, test_feature_df[wald_columns])
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
        baseline_test_roc_auc=baseline_test_taxonomy_roc_auc,
        baseline_test_pr_auc=baseline_test_taxonomy_pr_auc,
        wald_stats=wald_stats,
        wald_pvalue=wald_pvalue,
        wald_columns=wald_columns,
        lrt_pvalue=lrt_pvalue,
        expected_calibration_error=ece
    ), predictions


def train_and_save_taxonomic_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    taxonomies: List[str],
    save_path: str,
    baseline_model: PredictionModel,
    args: Namespace,
    metadata: Dict[str, Any] = {},
) -> Optional[List[Tuple[str, ModelResult]]]:
    """
    Train and save the taxonomic models.

    Args:
        train_df (pd.DataFrame): Pandas dataframe with train samples
        test_df (pd.DataFrame): Pandas dataframe with test samples
        taxonomies (List[str]): List of taxonomic categories
        save_path (str): Path for saving the model to
        baseline_model (PredictionModel): Baseline model
        args (Namespace): The command line arguments.

    Returns:
        Optional[Tuple[ModelResult, List[Tuple[str, ModelResult]]]]: The baseline model result and the taxonomic model results.
    """

    taxonomic_results = []
    if not check_training_eligibility(train_df, taxonomies):
        LOGGER.info(f"Training not eligible for current taxonomy, exiting")
        return None
    
    predictions = pd.DataFrame()
    for taxonomy in taxonomies:
        taxonomy_train_df = train_df[train_df['curr_taxonomy'] == taxonomy]
        tax_save_path = os.path.join(save_path, taxonomy)

        LOGGER.info(f"Training {taxonomy}-partitioned model...")
        LOGGER.info(f"Training size: {len(taxonomy_train_df)}")
        calculate_label_priors(taxonomy_train_df['labels'])
        LOGGER.info(f"Test size: {len(test_df[test_df['curr_taxonomy'] == taxonomy])}")
        calculate_label_priors(test_df[test_df['curr_taxonomy'] == taxonomy]['labels'])
        taxonomic_model_result, split_preds = train_taxonomic_model(
            taxonomy_train_df,
            test_df,
            taxonomy,
            baseline_model,
        )

        if taxonomic_model_result is None:
            LOGGER.error(f"{taxonomy}-partitioned model is null, skipping...")
            continue

        predictions[f'{taxonomy}_predictions'] = split_preds['model_predictions']
        predictions[f'{taxonomy}_prediction_probs'] = split_preds['model_prediction_probs']
        predictions['baseline_predictions'] = split_preds['baseline_predictions']
        predictions['baseline_prediction_probs'] = split_preds['base_prediction_probs']
        predictions['labels'] = split_preds['labels']
        predictions['model_taxonomy'] = split_preds['taxonomy']
        predictions['base_taxonomy'] = split_preds['base_taxonomy']
        LOGGER.info(f"Saving {taxonomy}-partitioned model results...")
        
        taxonomic_model = taxonomic_model_result.model
        taxonomic_model_metadata = {
            **metadata,
            "sequence_duplication_threshold": args.sequence_duplication_threshold,
            "taxonomy": taxonomy,
            **asdict(taxonomic_model_result, dict_factory=lambda x: {k: v for (k, v) in x if k != 'model'}),
            **metadata
        }
        save_models(tax_save_path, taxonomic_model, taxonomic_model_metadata)

        taxonomic_results.append((taxonomy, taxonomic_model_result))

    predictions['model_predictions'] = predictions.apply(lambda x: x[f'{x["model_taxonomy"]}_predictions'], axis=1)
    predictions['model_prediction_probs'] = predictions.apply(lambda x: x[f'{x["model_taxonomy"]}_prediction_probs'], axis=1)
    taxonomic_preds = {}
    taxonomic_preds['recitation'] = predictions[predictions['base_taxonomy'] == 'recitation']
    taxonomic_preds['reconstruction'] = predictions[predictions['base_taxonomy'] == 'reconstruction']
    taxonomic_preds['recollection'] = predictions[predictions['base_taxonomy'] == 'recollection']
    taxonomic_preds['aggregate'] = predictions

    LOGGER.info("Calculating Taxonomic metrics")
    taxonomic_prediction_metrics_agg = defaultdict(list)
    for prediction in taxonomic_preds:
        preds = taxonomic_preds[prediction]
        k_fold = KFold(n_splits=100, shuffle=True)
        for indicies, _ in k_fold.split(preds):
            split_preds = preds.iloc[indicies]
            if split_preds['labels'].nunique() == 1:
                continue
            for model_type in ['model', 'baseline']:
                taxonomic_prediction_metrics_agg[f'{model_type}_{prediction}_roc_auc'].append(roc_auc_score(
                    split_preds['labels'], split_preds[f'{model_type}_predictions'])
                )
                taxonomic_prediction_metrics_agg[f'{model_type}_{prediction}_pr_auc'].append(average_precision_score(
                    split_preds['labels'], split_preds[f'{model_type}_predictions'])
                )
                taxonomic_prediction_metrics_agg[f'{model_type}_{prediction}_precision_'].append(precision_score(
                    split_preds['labels'], split_preds[f'{model_type}_predictions'])
                )
                taxonomic_prediction_metrics_agg[f'{model_type}_{prediction}_recall_'].append(recall_score(
                    split_preds['labels'], split_preds[f'{model_type}_predictions'])
                )

                taxonomic_prediction_metrics_agg[f'{model_type}_{prediction}_ece'].append(expected_calibration_error(
                    split_preds[f'{model_type}_prediction_probs'], split_preds[f'labels'])
                )
    
    taxonomic_prediction_metrics = {}
    for metric, value in taxonomic_prediction_metrics_agg.items():
        taxonomic_prediction_metrics[metric] = (np.mean(value), np.std(value))

    # Saving taxonomic predictions
    with open(os.path.join(save_path,"taxonomic_prediction_metrics.json"), 'w') as f:
        json.dump(taxonomic_prediction_metrics, f)

    predictions.to_parquet(os.path.join(save_path, "predictions.parquet"))
    return taxonomic_results

def train_and_save_baseline_and_taxonomic_models(
    experiment_base: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: Namespace,
) -> Optional[Tuple[ModelResult, List[Tuple[str, ModelResult]]]]:
    """
    Train and save the baseline and taxonomic models.

    Args:
        experiment_base (str): The experiment base path.
        train_df (pd.DataFrame): Pandas dataframe with train samples
        test_df (pd.DataFrame): Pandas dataframe with test samples
        args (Namespace): The command line arguments.

    Returns:
        Optional[Tuple[ModelResult, List[Tuple[str, ModelResult]]]]: The baseline model result and the taxonomic model results.
    """
    LOGGER.info("Training the baseline model with all data...")

    baseline_result = train_baseline_model(train_df, test_df)
    if baseline_result is None:
        LOGGER.error("Baseline model is null, skipping...")
        return None
    baseline_model = baseline_result.model

    metadata = {
        "data_scheme": DATA_SCHEME,
        "model_size": MODEL_SIZE,
    }
    # hack to remove a field
    baseline_metadata = {
        **metadata,
        "taxonomy": "baseline",
        "sequence_duplication_threshold": args.sequence_duplication_threshold,
        **asdict(baseline_result, dict_factory=lambda x: {k: v for (k, v) in x if k != 'model'}),
    }
    LOGGER.info("Saving baseline model results...")
    baseline_save_path = f"{experiment_base}/{DATA_SCHEME}/{MODEL_SIZE}/baseline/"
    save_models(baseline_save_path, baseline_model, baseline_metadata)

    train_df['curr_taxonomy'] = train_df['base_taxonomy']
    test_df['curr_taxonomy'] = test_df['base_taxonomy']

    save_path = f"{experiment_base}/{DATA_SCHEME}/{MODEL_SIZE}/model_taxonomy/"
    taxonomic_results = train_and_save_taxonomic_models(
        train_df,
        test_df,
        TAXONOMIES,
        save_path,
        baseline_result.model,
        args,
        metadata=metadata,
    )
    return baseline_result, taxonomic_results
    

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
        has_taxonomy_1 = row[feature_1] >= threshold_1
        has_taxonomy_2 = row[feature_2] >= threshold_2

        if has_taxonomy_1:
            return "taxonomy_1"

        if has_taxonomy_2:
            return "taxonomy_2"

        return "taxonomy_3"

    return classify_row


def check_training_eligibility(
    train_df: pd.DataFrame, 
    taxonomies: List[str],
) -> bool:
    """
    Check if model training is availabile based on the label prior. The threshold
    could be extreme where they have no samples or only one class.

    Args:
        train_df (pd.DataFrame): Pandas dataframe with train samples

    Returns:
        bool: True if training is available, False otherwise.
    """
    LOGGER.info("Checking Training eligibility")
    for taxonomy in taxonomies:
        tax_train_df = train_df[train_df['curr_taxonomy'] == taxonomy]
        if len(tax_train_df) == 0:
            LOGGER.info(f"{taxonomy} as no samples. Current permutation is not trainable")
            return False
        label_true = tax_train_df[tax_train_df['labels'] == 1]
        label_false = tax_train_df[tax_train_df['labels'] == 0]
        if len(label_true) == 0:
            LOGGER.info(f"{taxonomy} has no positive samples. Current permutation is not trainable")
            return False
        if len(label_false) == 0:
            LOGGER.info(f"{taxonomy} has no negative samples. Current permutation is not trainable")
            return False

    return True


def train_and_save_all_taxonomy_pairs(
    experiment_base: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    unnormalized_train_df: pd.DataFrame,
    unnormalized_test_df: pd.DataFrame,
    baseline_model: PredictionModel,
    taxonomy_thresholds: DefaultDict,
    args: Namespace,
    start_index: int = None,
    end_index: int = None,
) -> None:
    """
    Trains models for all pairs of features to find the optimal taxonomy.

    Args:
        experiment_base (str): The experiment base path.
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The test dataset.
        unnormalized_train_df (pd.DataFrame): Train un-normalized dataset
        unnormalized_test_df (pd.DataFrame): Test un-normalized dataset
        baseline_model (PredictionModel): Baseline model
        taxonomy_thresholds (DefaultDict): The taxonomy thresholds.
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
        train_df['curr_taxonomy'] = unnormalized_train_df.apply(taxonomy_func, axis=1)
        test_df['curr_taxonomy'] = unnormalized_test_df.apply(taxonomy_func, axis=1)
        save_path = f"{experiment_base}/{DATA_SCHEME}/{MODEL_SIZE}/taxonomy_search/{candidate_1_name}"
        save_path += f"-{candidate_1_threshold_quantile}/{candidate_2_name}-{candidate_2_threshold_quantile}/"
        metadata = {
            "data_scheme": DATA_SCHEME,
            "model_size": MODEL_SIZE,
            "taxonomy_1_feature_name": candidate_1_name,
            "taxonomy_1_threshold_quantile": candidate_1_threshold,
            "taxonomy_2_feature_name": candidate_2_name,
            "taxonomy_2_threshold_quantile": candidate_2_threshold,
        }
        train_and_save_taxonomic_models(
            train_df,
            test_df,
            ["taxonomy_1", "taxonomy_2", "taxonomy_3"],
            save_path,
            baseline_model,
            args,
            metadata=metadata,
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

    pile_dataset['ds_type'] = 'representative'
    memories_dataset['ds_type'] = 'memories'
    
    combined_dataset = pd.concat([pile_dataset, memories_dataset], ignore_index=True).drop_duplicates('sequence_id')
    LOGGER.info("Pre-processing the dataset...")
    features , labels, data_df = preprocess_dataset(combined_dataset, normalize=True)

    LOGGER.info("Splitting the dataset into train and test aggregate sets...")
    data_df['labels'] = labels
    data_df['base_taxonomy'] = combined_dataset.apply(taxonomy_func, axis=1)
    data_df['stratify_labels'] = data_df['labels'].astype('str') + data_df['base_taxonomy'].astype('str')

    LOGGER.info("Calculating correlation coefficients of base + taxonomic features...")
    correlation_results = calculate_all_correlation_coefficients(data_df, args)
    save_correlation_coefficients(experiment_base, DATA_SCHEME, MODEL_SIZE, correlation_results)

    data_df_rep = data_df[data_df['ds_type'] == 'representative']
    data_df_mem = data_df[data_df['ds_type'] == 'memories']
    train_data, test_data = split_dataset(data_df_rep, data_df_rep['stratify_labels'])
    train_df, test_df = train_data[0], test_data[0]

    # Combine memories in train subset
    train_df = pd.concat([train_df, data_df_mem])

    unnormalized_train_df = combined_dataset.loc[train_df.index]
    unnormalized_test_df = combined_dataset.loc[test_df.index]
    

    LOGGER.info("Training baseline and taxonomic models...")
    baseline_model_results, tax_model_results = train_and_save_baseline_and_taxonomic_models(
        experiment_base,
        train_df,
        test_df,
        args,
    )
    if baseline_model_results is None or tax_model_results is None:
        LOGGER.error("Model results are null, exiting...")
        return

    LOGGER.info("Generating taxonomy quantile thresholds...")
    taxonomy_thresholds = generate_taxonomy_quantile_thresholds(memories_dataset)

    LOGGER.info("Starting to train all taxonomy pairs...")
    train_and_save_all_taxonomy_pairs(
        experiment_base,
        train_df,
        test_df,
        unnormalized_train_df,
        unnormalized_test_df,
        baseline_model_results.model,
        taxonomy_thresholds,
        args,
        start_index=args.taxonomy_search_start_index,
        end_index=args.taxonomy_search_end_index,
    )


if __name__ == "__main__":
    main()

