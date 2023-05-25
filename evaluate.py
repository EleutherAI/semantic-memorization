"""
The purpose of this file is to evaluate the classification performance of the filters.
You can register your filter by adding the entry to the dicitonary at the bottom
of the file. The key is the name of the category that the filter is supposed to detect,
and the value is the function that implements the filter. You should import your filtering
method from the filters module.
"""
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from filters.pattern_incrementing import incrementing_sequences_filter

tqdm.pandas()


def log_mistakes_report(mistakes: pd.DataFrame, category: str, eval_timestamp: str, eval_directory: str):
    """
    Saves the mistakes made by the filter to a CSV file. This allows for manual inspection of the mistakes
    for debugging purposes.

    Args:
        mistakes (pd.DataFrame): The entries in the evaluation dataset that were misclassified by the filter
        category (str): The name of the category that the filter targets
        eval_timestamp (str): The execution timestamp which acts as a unique identifier for the evaluation run
        eval_directory (str): The directory where the evaluation results are stored
    """
    if not os.path.exists("eval_results/"):
        os.makedirs("eval_results/")
    if not os.path.exists(eval_directory):
        os.makedirs(eval_directory)

    mistakes.to_csv(f"{eval_directory}/mistakes_{eval_timestamp}_{category}.csv", index=False)


def evaluate_filter(category: str, filter_function: function, dataset: pd.DataFrame, eval_timestamp: str) -> dict:
    """
    Evaluate the classification performance of the provided filter

    Args:
        category (str): The name of the category that the filter targets
        filter_function (function): The method reference of the filter
        dataset (pd.DataFrame): The evaluation dataset
        eval_timestamp (str): The execution timestamp which acts as a unique identifier for the evaluation run

    Returns:
        dict: The classification report of the filter
    """
    filter_judgments = dataset["shortened_text"].progress_apply(filter_function)
    filter_labels = dataset["Category"].progress_apply(lambda c: c == category)
    report_dict = classification_report(filter_labels, filter_judgments, output_dict=True)
    evaluation_log = {
        "Category": category,
        "Method": filter_function.__name__,
        "Accuracy": report_dict["accuracy"],
        "Precision": report_dict["True"]["precision"],
        "Recall": report_dict["True"]["recall"],
        "F1": report_dict["True"]["f1-score"],
        "Support": report_dict["True"]["support"],
    }

    mistakes = dataset[filter_judgments != filter_labels]
    mistakes["Judgment"] = filter_judgments
    log_mistakes_report(mistakes, category, eval_timestamp, f"eval_results/{eval_timestamp}")
    print(classification_report(filter_labels, filter_judgments))
    return evaluation_log


def evaluate(filters: dict):
    """
    Evaluate the classification performance for each filter

    Args:
        filters (dict): The filters to evaluate. The key is the name of the category and value is the filter function.
    """
    dataset = pd.read_csv("datasets/eval/Pythia_70m_Deduped_Low_Perplexity_Labeling_Formatted.csv")
    eval_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_results = []
    for category, filter_function in filters.items():
        print(f"Evaluating Filter: {category}")
        eval_results.append(evaluate_filter(category, filter_function, dataset, eval_timestamp))

    eval_results_df = pd.DataFrame(eval_results)
    eval_results_df.to_csv(f"eval_results/{eval_timestamp}/report_{eval_timestamp}.csv", index=False)

    print("\n--------------------------------------------------------------------------------------------------------------")
    print(eval_results_df)
    print("--------------------------------------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    # Register additional filters here. The key is the name of the category that the filter is
    # supposed to detect, and the value is the function that implements the filter.
    evaluate({"pattern-incrementing": incrementing_sequences_filter})
