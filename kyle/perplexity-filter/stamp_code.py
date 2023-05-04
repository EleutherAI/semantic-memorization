import torch
import pandas as pd
from pandarallel import pandarallel
from datasets import load_dataset, ReadInstruction
from torch.nn.functional import softmax
from torch.multiprocessing import Pool, Process, set_start_method
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
tqdm.pandas()


def is_code(text):
    try:
        tokens = code_tokenizer(text=text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        outputs = code_classifier(**tokens)
        probabilities = softmax(outputs.logits.detach())
        return (probabilities[:, 0] > 0.457414).item()
    except Exception as e:
        print(f"Error: {e} for text: {text}")
        return None


# def main(dataset_name, model, tokenizer):
#     Add a text column to the Pile dataset
#     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
#     memories_tokens = load_dataset("EleutherAI/pythia-memorized-evals", split="deduped.70m").to_pandas()[["index", "tokens"]]
#     memories_ppls = pd.read_csv("70m_4_27/memories_deduped_70m.csv")
#     all_memories_data = memories_ppls.join(memories_tokens.set_index("index"), on="index", how="inner")
#     all_memories_data["text"] = all_memories_data["tokens"].progress_apply(lambda x: tokenizer.decode(x))
#     pile_tokens = pd.read_csv("70m_4_27/enriched_pile-1_deduped_70m.csv")


if __name__ == "__main__":
    set_start_method('spawn')
    # pandarallel.initialize(progress_bar=True)

    # Init combined dataset
    combined_data = pd.read_csv(
        "/home/kyle/repos/semantic-memorization/kyle/perplexity-filter/70m_4_27/combined_data_with_text.csv",
        # quoting=3,
        # on_bad_lines="skip",
        # lineterminator="\n",
        # low_memory=False,
        engine="python",
        # nrows=100000
        )
    print(f"Loaded {len(combined_data)} rows of combined data")
    print(combined_data)
    # Init code model
    device = torch.device("cuda")
    code_classifier = AutoModelForSequenceClassification.from_pretrained("usvsnsp/code-vs-nl").to(device).eval()
    code_tokenizer = AutoTokenizer.from_pretrained("usvsnsp/code-vs-nl")

    combined_data["is_code"] = combined_data["text"].progress_apply(lambda x: is_code(x))
    combined_data.to_csv("70m_4_27/combined_data_with_text_and_code.csv", index=False)
    print(combined_data)