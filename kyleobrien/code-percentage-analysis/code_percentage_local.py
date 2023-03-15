import torch
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MemoriesDataset(Dataset):
    def __init__(self, memories_path, tokenizer_path, is_hdf=True, sample_size=None):
        print(f"\nReading memorization dataset: {memories_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.memories = (
            pd.read_hdf(memories_path) if is_hdf else pd.read_csv(memories_path)
        )
        if sample_size is not None:
            self.memories = self.memories.sample(sample_size)

    def __getitem__(self, index):
        memory_record = self.memories.iloc[index]
        natural_language = self.tokenizer.decode(memory_record["tokens"])
        return natural_language

    def __len__(self):
        return len(self.memories)


def calculate_nl_percent(memory_file_path):
    is_hdf = memory_file_path.endswith(".hdf")
    pythia_path = "EleutherAI/pythia-70m-deduped"
    memories = MemoriesDataset(
        # "/home/mchorse/kyleobrien/semantic-memorization/memorized-data/memories-pythia-duplicated/125m.hdf",
        memory_file_path,
        pythia_path,
        is_hdf,
    )
    data_loader = DataLoader(memories, batch_size=128)

    device = torch.device("cuda:5")
    model_path = "usvsnsp/code-vs-nl"
    code_tokenizer = AutoTokenizer.from_pretrained(model_path)
    code_classifier = AutoModelForSequenceClassification.from_pretrained(model_path)
    code_classifier.to(device).eval()

    running_nl_count = 0
    dataset_length = len(memories)

    with torch.no_grad():
        for text in tqdm(data_loader, desc=f"Evaluating {memory_file_path}"):
            tokenized_batch = code_tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True, padding=True
            )
            tokenized_batch.to(device)
            outputs = code_classifier(**tokenized_batch)
            probabilities = softmax(outputs.logits.detach())
            natural_language_count = (probabilities[:, 0] > 0.457414).sum().item()
            running_nl_count += natural_language_count

    create_pie_chart(memory_file_path, running_nl_count, dataset_length)
    nl_percent = 100 * running_nl_count / dataset_length
    print(f"Natural language makes up {nl_percent}% of {memory_file_path}")

    return nl_percent


def create_pie_chart(memory_file_path, running_nl_count, dataset_length):
    results_frame = pd.DataFrame(
        {
            "Type": ["Natural Language", "Code/Numbers"],
            "Counts": [running_nl_count, dataset_length - running_nl_count],
        }
    )
    pythia_model_name = memory_file_path.split("/")[-1].split(".hdf")[0]
    fig = px.pie(
        results_frame,
        names="Type",
        values="Counts",
        title=f"Pythia {pythia_model_name} Memory Surface Forms",
    )
    fig.write_image(
        f"/home/mchorse/kyleobrien/semantic-memorization/code-percentage-analysis/{pythia_model_name}_surface_forms.png"
    )


def main():
    memories_path = "memorized-data/memories-pythia-duplicated/"
    models = [
        "19m",
        "125m",
        "350m",
        "800m",
        "1.3b",
        "2.7b",
        "6.7b",
        "13b",
        "all_unique_memorized_sequences",
    ]
    nl_percents = []

    for model in models:
        memory_file_path = f"{memories_path}/{model}.hdf"
        nl_percent = calculate_nl_percent(memory_file_path)
        nl_percents.append(nl_percent)

    chart_date = pd.DataFrame({"Models": models[:-1], "NL Percents": nl_percents[:-1]})
    title = f"Natural Language Percents: Overall={nl_percents[-1]}"
    fig = px.line(
        chart_date, x="Models", y="NL Percents", text="NL Percents", title=title
    )
    fig.update_traces(textposition="bottom right")
    fig.write_image(
        f"/home/mchorse/kyleobrien/semantic-memorization/code-percentage-analysis/all_nl_percents.png"
    )


if __name__ == "__main__":
    main()
