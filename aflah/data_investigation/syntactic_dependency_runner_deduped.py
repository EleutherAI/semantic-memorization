from datasets import load_dataset, ReadInstruction
import spacy
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

df_deduped_pythia = pd.read_pickle("../data/deduped_pythia.pkl")

def get_spacy_doc(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    return doc

def get_avg_dependency_length(doc):
    ls_lengths = []
    for token in doc:
        for child in token.children:
            distance = child.i - token.i
            # print(f"{token.text} --{child.dep_}--> {child.text}  (distance: {distance})")
            ls_lengths.append(abs(distance))
    return np.mean(ls_lengths)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")

def get_string(token_indices):
    # token_indices = list(token_indices)
    return tokenizer.decode(token_indices)

df_deduped_pythia['first_64_tokens'] = df_deduped_pythia['tokens'].parallel_apply(lambda x: x[:64])
df_deduped_pythia['first_64_tokens_string'] = df_deduped_pythia['first_64_tokens'].parallel_apply(get_string)
ls_first_64_tokens_string = df_deduped_pythia['first_64_tokens_string'].to_list()

ls_docs = []
# # use a thread pool to parallelize the computation of docs
# with ThreadPoolExecutor(max_workers=os.cpu_count()//4) as executor:
#     futures = [executor.submit(get_spacy_doc, sent) for sent in ls_first_64_tokens_string]

#     # use tqdm to display a progress bar while the threads are running
#     for future in tqdm(as_completed(futures), total=len(futures)):
#         ls_docs.append(future.result())

for sent in tqdm(ls_first_64_tokens_string):
    ls_docs.append(get_spacy_doc(sent))

ls_dependency_lengths_deduped = []
# # use a thread pool to parallelize the computation of dependency lengths
# with ThreadPoolExecutor(max_workers=os.cpu_count()//4) as executor:
#     futures = [executor.submit(get_avg_dependency_length, doc) for doc in ls_docs]

#     # use tqdm to display a progress bar while the threads are running
#     for future in tqdm(as_completed(futures), total=len(futures)):
#         ls_dependency_lengths_deduped.append(future.result())

for doc in tqdm(ls_docs):
    ls_dependency_lengths_deduped.append(get_avg_dependency_length(doc))

# print the list of average dependency lengths for each sentence
print(ls_dependency_lengths_deduped)

import pickle
# save docs
with open("../data/ls_docs_full_deduped.pkl", "wb") as f:
    pickle.dump(ls_docs, f)
# save the list of average dependency lengths for each sentence
with open("../data/ls_dependency_lengths_full_deduped.pkl", "wb") as f:
    pickle.dump(ls_dependency_lengths_deduped, f)

df_deduped_pythia['dependency_length'] = ls_dependency_lengths_deduped
df_deduped_pythia['docs'] = ls_docs

df_deduped_pythia.to_pickle("../data/df_deduped_pythia_full_with_dependency_length.pkl")