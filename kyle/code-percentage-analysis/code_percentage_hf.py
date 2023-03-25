import torch
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from natsort import natsorted
from datasets import load_dataset
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HFMemoriesDataset(Dataset):
    is_dataframe = False

    def __init__(self, memories, tokenizer, sample=None):
        self.tokenizer = tokenizer
        self.memories = memories
        if sample is not None:
            self.memories = self.memories.to_pandas().sample(sample)
            self.is_dataframe = True

    def __getitem__(self, index):
        memory_record = (
            self.memories.iloc[index] if self.is_dataframe else self.memories[index]
        )
        decoded_text = self.tokenizer.decode(memory_record["tokens"])
        return decoded_text

    def __len__(self):
        return len(self.memories)


def calculate_nl_percent(dataset, pythia_tokenizer):
    memories = HFMemoriesDataset(
        dataset,
        pythia_tokenizer,
        # sample=100000,
    )
    data_loader = DataLoader(memories, batch_size=64)

    device = torch.device("cuda:7")
    model_path = "usvsnsp/code-vs-nl"
    code_tokenizer = AutoTokenizer.from_pretrained(model_path)
    code_classifier = AutoModelForSequenceClassification.from_pretrained(model_path)
    code_classifier.to(device).eval()

    running_nl_count = 0
    dataset_length = len(memories)

    with torch.no_grad():
        for text in tqdm(data_loader, desc=f"Evaluating {dataset.split}"):
            tokenized_batch = code_tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True, padding=True
            )
            tokenized_batch.to(device)
            outputs = code_classifier(**tokenized_batch)
            probabilities = softmax(outputs.logits.detach())
            natural_language_count = (probabilities[:, 0] > 0.457414).sum().item()
            running_nl_count += natural_language_count

    create_pie_chart(dataset.split, running_nl_count, dataset_length)
    nl_percent = 100 * running_nl_count / dataset_length
    print(f"Natural language makes up {nl_percent}% of {dataset.split}")

    return nl_percent


def create_pie_chart(pythia_model_name, running_nl_count, dataset_length):
    pile_varient = str(pythia_model_name).split(".")[0]

    results_frame = pd.DataFrame(
        {
            "Type": ["Natural Language", "Code/Numbers"],
            "Counts": [running_nl_count, dataset_length - running_nl_count],
        }
    )
    fig = px.pie(
        results_frame,
        names="Type",
        values="Counts",
        title=f"Pythia {pythia_model_name} Memory Surface Forms",
    )
    fig.write_image(f"./code-percentage-analysis/{pile_varient}/{pythia_model_name}_surface_forms.png", scale=6, width=1080, height=1080)


def get_tokenizer(split_name):
    isDeduped = split_name.startswith("deduped")
    model = split_name.split("duped.")[-1]
    corresponding_model = f"EleutherAI/pythia-{model}{'-deduped' if isDeduped else ''}"
    return AutoTokenizer.from_pretrained(corresponding_model)


def main():
    all_memories_splits = load_dataset("EleutherAI/pythia-memorized-evals")
    model_sizes = natsorted(set([split_name.split("uped.")[-1] for split_name in all_memories_splits if "m" in split_name])) + natsorted(set(([split_name.split("uped.")[-1] for split_name in all_memories_splits if "b" in split_name])))
    pile_dataset_types = []
    all_nl_percents = []

    for pile_dataset in ["deduped", "duped"]:
        nl_percents = []
        ordered_splits = [f"{pile_dataset}.{model_size}" for model_size in model_sizes]

        for split_name in ordered_splits:
            dataset = all_memories_splits[split_name]
            pythia_tokenizer = get_tokenizer(split_name)
            nl_percent = calculate_nl_percent(dataset, pythia_tokenizer)

            nl_percents.append(nl_percent)
            pile_dataset_types.append(pile_dataset)

        chart_date = pd.DataFrame(
            {
                "Models": model_sizes,
                "Percent Natural Language": nl_percents,
            }
        )
        title = f"{pile_dataset.title()}: Percent of Memories Which Are Natural Language"
        fig = px.line(
            chart_date, x="Models", y="Percent Natural Language", text="Percent Natural Language", title=title
        )
        fig.update_traces(textposition="bottom right")
        fig.update_traces(text= [f'{val}%' for val in chart_date["Percent Natural Language"]])
        fig.write_image(
            f"./code-percentage-analysis/all_{pile_dataset}_nl_percents.png",
            scale=6, width=1080, height=1080
        )
        all_nl_percents += nl_percents

    create_combined_line_chart(model_sizes, pile_dataset_types, all_nl_percents)


def create_combined_line_chart(model_sizes, pile_dataset_types, all_nl_percents):
    chart_date = pd.DataFrame(
        {
            "Models": model_sizes * 2,
            "Percent Natural Language": all_nl_percents,
            "Pile Type": pile_dataset_types,
        }
    )
    title = f"Percent of Memories Which Are Natural Language by Model & Pile Varient"
    fig = px.line(
        chart_date, x="Models", y="Percent Natural Language", text="Percent Natural Language", color="Pile Type", title=title
    )
    fig = px.line(
        chart_date, x="Models", y="Percent Natural Language", text="Percent Natural Language", color="Pile Type", title=title, # template='plotly_dark'
    )
    fig.update_traces(textposition="bottom right")
    fig.update_layout(yaxis_ticksuffix = "%", )
    fig.write_image(
            f"./code-percentage-analysis/all_combined_nl_percents.png", width=1080, height=1080, scale=6
            )


if __name__ == "__main__":
    main()
