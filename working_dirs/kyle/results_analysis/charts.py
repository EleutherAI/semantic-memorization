import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, get_dataset_split_names, DatasetDict
from tqdm import tqdm

sns.set_theme(style="darkgrid")
sns.set_context("talk")
tqdm.pandas()

split_sample_size = None

memories_path = "usvsnsp/memories-semantic-memorization-filter-results"
get_dataset_split_names(memories_path)
memories_dataset = DatasetDict()

# get splits that have deduped in the name
splits = [split for split in get_dataset_split_names(memories_path) if "deduped" in split]
for split in tqdm(splits):
    memories_dataset[split] = load_dataset(memories_path, split=f"{split}[:{split_sample_size}]" if split_sample_size else split)

print(memories_dataset)

pile_path = "usvsnsp/pile-semantic-memorization-filter-results"
get_dataset_split_names(pile_path)
pile_dataset = DatasetDict()

splits = [split for split in get_dataset_split_names(pile_path) if "deduped" in split]
for split in tqdm(splits):
    pile_dataset[split] = load_dataset(pile_path, split=f"{split}[:{split_sample_size}]" if split_sample_size else split)

print(pile_dataset)

split_to_param_count = {
    "70m": 70000000,
    "160m": 160000000,
    "410m": 410000000,
    "1b": 1000000000,
    "1.4b": 1400000000,
    "2.8b": 2800000000,
    "6.9b": 6900000000,
    "12b": 12000000000,
}

combined_dataframe = None
for split in tqdm(memories_dataset, desc="Loading Memories"):
    current_frame = memories_dataset[split].to_pandas()
    current_frame.drop(columns=["text", "frequencies", "tokens"], inplace=True)
    current_frame["Model"] = ".".join(split.split(".")[2:])
    current_frame["Param Count"] = split_to_param_count[current_frame["Model"].iloc[0]]
    current_frame["Deduped"] = "deduped" in split
    current_frame["Memorized"] = True
    if combined_dataframe is None:
        combined_dataframe = current_frame
    else:
        combined_dataframe = pd.concat([combined_dataframe, current_frame])

for split in tqdm(pile_dataset, desc="Loading Pile"):
    current_frame = pile_dataset[split].to_pandas()
    current_frame.drop(columns=["text", "frequencies", "tokens"], inplace=True)
    current_frame["Model"] = ".".join(split.split(".")[2:])
    current_frame["Param Count"] = split_to_param_count[current_frame["Model"].iloc[0]]
    current_frame["Deduped"] = "deduped" in split
    current_frame["Memorized"] = False
    combined_dataframe = pd.concat([combined_dataframe, current_frame])

combined_dataframe = combined_dataframe.sort_values("Param Count").dropna(subset=["sequence_perplexity"])
display(combined_dataframe.shape)
combined_dataframe.head()

# drop cases where generation_perplexity is -1
before_count = combined_dataframe.shape[0]
combined_dataframe = combined_dataframe[combined_dataframe["generation_perplexity"] != -1]
after_count = combined_dataframe.shape[0]
print(f"Dropped {before_count - after_count} rows with -1 generation_perplexity")

# set num_repeating = 0 if -1
combined_dataframe.loc[combined_dataframe["num_repeating"] == -1, "num_repeating"] = 0
display(combined_dataframe.value_counts("num_repeating").head())

def get_category(row):
    if row["Memorized"] == False:
        return "Not Memorized"
    if row["sequence_duplicates"] >= 200:
        return "Recitation"
    if row["is_incrementing"] or row["num_repeating"] != 0:
        return "Reconstruction"

    return "Recollection"

combined_dataframe["category"] = combined_dataframe.progress_apply(lambda row: get_category(row), axis=1)
combined_dataframe.value_counts(["Model", "Deduped", "category"])

# Plot Graphs
deduped_plotting_frame = combined_dataframe[combined_dataframe["Deduped"] == True]
deduped_memories = deduped_plotting_frame[deduped_plotting_frame["Memorized"] == True]

"""
['sequence_id', 'sequence_duplicates', 'max_frequency', 'avg_frequency',
'min_frequency', 'median_frequency', 'p25_frequency', 'p75_frequency',
'is_incrementing', 'repeating_offset', 'num_repeating',
'smallest_repeating_chunk', 'memorization_score',
'templating_frequency_0.9', 'templating_frequency_0.8',
'prompt_perplexity', 'generation_perplexity', 'sequence_perplexity',
'Model', 'Param Count', 'Deduped', 'Memorized', 'category']
"""

titles = {
    # Categorical
    "category": "Count of Memories by Taxonomical Category",
    # "sequence_duplicates": "Mean Duplication Per Example",
    # "is_incrementing": "Percent of Sequences That Are Incrementing",

    # # Length of repeating subsequences
    # "num_repeating": "Mean Token Length For Repeating Subsequences",

    # # Cosine Similarities
    # "templating_frequency_0.9": "Mean Number of Examples 0.9 Cosime Similarity To Each Example",
    # "templating_frequency_0.8": "Mean Number of Examples 0.8 Cosime Similarity To Each Example",

    # # Perplexity
    # "prompt_perplexity": "Mean Prompt Perplexity",
    # "sequence_perplexity": "Mean Sequence Perplexity",
    # "generation_perplexity": "Mean Generation Perplexity",

    # # Token frequencies
    # "token_frequency": "Mean Token Frequency Statistics",
    # "median_frequency": "Mean Median Frequency for All Unique Tokens in Each Sequence",
    # "avg_frequency": "Mean Average Frequency for All Unique Tokens in Each Sequence",
    # "p25_frequency": "Mean 25th Percentile Frequency for All Unique Tokens in Each Sequence",
    # "min_frequency": "Mean Minimum Frequency for All Unique Tokens in Each Sequence",

    "null": "null"
}

# create subplots where each metric is on its own row. The first column is fo rmemorized overall and the second is broken down by category.
fig, axes = plt.subplots(len(titles), 2, figsize=(30, 15 * len(titles)))

for metric in tqdm(titles):
    if metric == "null":
        continue

    for column in [0, 1]:
        title_text = titles[metric]

        if metric == "token_frequency":
            sns.boxplot(
                data=deduped_plotting_frame,
                y="avg_frequency",
                x="Model",
                ax=axes[list(titles.keys()).index(metric), column],
                gap=0.5,
                hue="category" if column == 1 else "Memorized",
            )

        elif metric == "category":
            plotting_frame = deduped_plotting_frame[deduped_plotting_frame["Memorized"] == True]
            if column == 0:
                sns.histplot(
                    data=plotting_frame,
                    x="Model",
                    hue="category",
                    ax=axes[list(titles.keys()).index(metric), column],
                    multiple="stack",
                    stat="count",
                    common_norm=False,
                )
            else:
                title_text = title_text.replace("Count", "Percent")
                all_percents = []
                for param_count in tqdm(split_to_param_count.values()):
                    model_examples = plotting_frame[plotting_frame["Param Count"] == param_count]
                    model_percents = model_examples.value_counts("category", normalize=True).to_dict()
                    for category in model_percents:
                        all_percents.append({
                            "Param Count": param_count,
                            "category": category,
                            "percent": model_percents[category],
                        })

                sns.lineplot(
                    data=pd.DataFrame(all_percents),
                    x="Param Count",
                    y="percent",
                    hue="category",
                    ax=axes[list(titles.keys()).index(metric), column],
                    markers=True,
                    marker="o",
                )

                axes[list(titles.keys()).index(metric), column].set_xscale("log")

                # make y axis percents and scale values by 100 and have %
                axes[list(titles.keys()).index(metric), column].set_yticklabels([f"{int(tick * 100)}%" for tick in axes[list(titles.keys()).index(metric), column].get_yticks()])


        else:
            sns.lineplot(
                data=deduped_plotting_frame.reset_index(),
                x="Param Count",
                y=metric,
                ax=axes[list(titles.keys()).index(metric), column],
                markers=True,
                hue="category" if column == 1 else "Memorized",
                marker="o",
            )

        # log x axis if line plot
        if metric not in ["category", "token_frequency"]:
            axes[list(titles.keys()).index(metric), column].set_xscale("log")

        # set title
        axes[list(titles.keys()).index(metric), column].set_title(title_text)

        # make title bold
        axes[list(titles.keys()).index(metric), column].title.set_weight("bold")

        # set x label based off the title
        quant_metic = title_text.split()[0]
        axes[list(titles.keys()).index(metric), column].set_ylabel(quant_metic)

        # don't use scientific notation on y axis
        try:
            axes[list(titles.keys()).index(metric), column].get_yaxis().get_major_formatter().set_scientific(False)
        except:
            print(f"Failed to set scientific notation for {metric}")


# add margins between rows
plt.subplots_adjust(hspace=0.25)

# save fig
plt.savefig("metrics_analysis_memories.png", dpi=300, bbox_inches="tight")
fig.show()