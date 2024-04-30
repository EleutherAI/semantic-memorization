from sentence_transformers import SentenceTransformer
import pickle
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
import scipy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# from Levenshtein import distance as levenshtein_distance
dataset = load_dataset("EleutherAI/pythia-memorized-evals")

for k, v in enumerate(dataset):
    print(v, len(dataset[v]))


# model_sizes = ['70m','160m','410m','1b','1.4b','2.8b','6.9b','12b']
model_sizes = ['12b']
time_steps = [23000,43000,63000,83000,103000,123000]
wandb.init(
    # set the wandb project where this run will be logged
    project="snowclones")

import torch

def normalize_embeddings(embeddings, batch_size = 10000):
    normalized_embeddings = []
    for start_idx in range(0, embeddings.size(0), batch_size):
        end_idx = min(start_idx + batch_size, embeddings.size(0))
        batch = embeddings[start_idx:end_idx]
        norm = batch.norm(p=2, dim=1, keepdim=True)
        normalized_batch = batch.div(norm)
        normalized_embeddings.append(normalized_batch)

    return torch.cat(normalized_embeddings, 0)


def number_of_snowclone(corpus_embeddings,queries, threshold,model_name):
    frequencies = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    corpus_embeddings = torch.Tensor(corpus_embeddings).to(device)
    queries = torch.Tensor(queries).to(device)
    n_samples = len(queries)
    step_size =100
    corpus_embeddings_norm = normalize_embeddings(corpus_embeddings)
    # Wrap the range object with tqdm for a progress bar
    for start_idx in tqdm(range(0, n_samples, step_size)):
        end_idx = min(start_idx + step_size, n_samples)

        # Multi query vector
        query_embeddings = queries[start_idx:end_idx]

        # Compute cosine distances
        # distances = scipy.spatial.distance.cdist(query_embeddings, corpus_embeddings, "cosine")

        # Normalize the embeddings for cosine similarity
        query_embeddings_norm = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)

        # Calculate cosine similarity
        distances = torch.mm(query_embeddings_norm, corpus_embeddings_norm.t())
        # Count the number of distances that are smaller or equal to (1-threshold)
        count = torch.sum(distances >= (threshold), axis=1)
        # Append counts to frequencies
        frequencies.extend(count.cpu().numpy().tolist())
    with open(str(threshold)+"_"+str(model_name)+"corpus.pickle", "wb") as f:
        pickle.dump({"frequencies": frequencies},f)
    assert n_samples == len(frequencies)
    return frequencies




for k,v in enumerate(dataset):
    # todo here
    prompt = []
    exit = []
    full_sentences = []
    if 'deduped' in v:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b-deduped")
        corpus_filename = "embeddings_deduped.pickle"  # Name of the deduped corpus embeddings file
    else:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
        corpus_filename = "embeddings_duped.pickle"  # Name of the duped corpus embeddings file

    with open("embeddings_"+str(v)+".pickle","rb") as f:
        data = pickle.load(f)
    query_embeddings = data['embeddings']
    sentences = data['sentences']
    print("done")
    
    with open(corpus_filename,"rb") as f:
        data = pickle.load(f)
    corpus_embeddings = data['embeddings']
    print("done")


    d = number_of_snowclone(corpus_embeddings,query_embeddings,0.8,v)
    # save above dict with key as model size and value as dict
    # print(f"model : {v}, dic : {d} ")
    wandb.log({"model": v, "dic": d})
                                 
