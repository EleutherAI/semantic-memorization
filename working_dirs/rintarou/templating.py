import pickle
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
import scipy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
import torch
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="em")


with open("embeddings_deduped.pickle","rb") as f:
    data = pickle.load(f)
    corpus_embeddings = data['embeddings']
    sentences = data['index']
print("done")



with open("queries_deduped.pickle", "rb") as f:
    d = pickle.load(f)

queries = d['queries']

def number_of_snowclones(corpus_embeddings, queries, threshold=0.9):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    corpus_embeddings = torch.Tensor(corpus_embeddings).to(device)
    
    n_samples = len(corpus_embeddings)
    batch_size = 100
    counts = []
    corpus_embeddings_norm = corpus_embeddings / corpus_embeddings.norm(dim=1, keepdim=True)    
    for start_idx in tqdm(range(0, n_samples, batch_size)):
        end_idx = min(start_idx + batch_size, n_samples)
        
        # Extracting batch of embeddings
        batch_embeddings = corpus_embeddings_norm[start_idx:end_idx]

        # Calculate cosine similarity in batches
        batch_similarities = torch.mm(batch_embeddings, corpus_embeddings_norm.t())
        
        for i, similarities in enumerate(batch_similarities):
            # Convert similarities to a PyTorch tensor
            similarities_tensor = similarities
            if any(ord(char) > 127 for char in queries[start_idx + i]):
                levenshtein_counts = 0
                counts.append(levenshtein_counts)
                continue
    
            # Find the indices that cross the threshold using PyTorch
            indices = torch.nonzero(similarities_tensor >= threshold).squeeze()
            indices_over_threshold = [indices.item()] if indices.dim() == 0 else indices.tolist()
            '''
            if len(indices_over_threshold) > 1: 
                print(indices_over_threshold)
            '''
            # Calculate the Levenshtein distance using the actual query strings and count them
            levenshtein_counts = 0
            for idx in indices_over_threshold:
                if levenshtein_distance(queries[start_idx + i], queries[idx]) < 20 and levenshtein_distance(queries[start_idx + i], queries[idx]) > 0:
                    levenshtein_counts += 1
                # if (levenshtein_counts): print(levenshtein_counts ,queries[idx])
            counts.append(levenshtein_counts)
    return counts



frequencies = number_of_snowclones(corpus_embeddings,queries,0.8)



with open("0.8_deduped_snowclones_edit.pickle", "wb") as f:
    pickle.dump({"frequencies": frequencies, "index": sentences},f)
