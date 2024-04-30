# plot graph showing pearsonr and p-value for each pair of features
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
# generate data for it

from sentence_transformers import SentenceTransformer
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("EleutherAI/pythia-memorized-evals")

for k, v in enumerate(dataset):
    print(v, len(dataset[v]))


model_sizes = ['70m','160m','410m','1b','1.4b','2.8b','6.9b','12b']
"""
dataset: pile
duped.6.9b 2120969
duped.70m 463953
deduped.2.8b 1355211
duped.1b 1256141
deduped.1.4b 1048097
deduped.160m 581195
duped.2.8b 1675077
duped.160m 689673
deduped.12b 1871215
deduped.410m 811039
deduped.6.9b 1680294
duped.12b 2382326
duped.410m 970341
deduped.1b 1032865
deduped.70m 411448
duped.1.4b 1373722
"""

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

for k,v in enumerate(dataset):
    #v has substring 'deduped'
    sentences = []
    if 'deduped' in v:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b-deduped")
    else:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
    for i in dataset[v]['tokens']:
        text = tokenizer.decode(i)
        sentences.append(text)
    corpus_embeddings = model.encode(sentences, batch_size= 128)
    # open(f"embeddings_{v}.pickle","wb")
    # create pickle file & save embeddings & sentences
    with open(f"embeddings_{v}.pickle","wb") as f:
        pickle.dump({'embeddings':corpus_embeddings,
                     'sentences':sentences},
                     f)
    print(f"saved embeddings_{v}.pickle")
