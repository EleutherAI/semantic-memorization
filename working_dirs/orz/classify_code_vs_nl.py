import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm.auto import tqdm
import torch.distributed as dist
import argparse
import socket
import os

def init_distributed(rank: int, world_size: int):
    """Initializes torch distributed group

    Args:
        rank (int): Rank of current process
        world size (int): Total number of processes
    """
    dist.init_process_group(backend = "nccl", rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)

if __name__ == '__main__':
    DS_TYPE = 'deduped'
    parser = argparse.ArgumentParser(
        prog = 'Classify Pythia tokens into cude or nl',
    )
    parser.add_argument(
        '--offsets_path',
        default = f'results/zero_offsets_{DS_TYPE}.npy',
        help = 'Path to Sequences of tokens'
    )
    parser.add_argument(
        '--num_sequences',
        default = 143000*1024,
        help = 'Number of sequences to calculate this upon'
    )
    parser.add_argument(
        '--batch_size',
        default = 128,
        help = 'Batch size while decoding and calculating embeddings'
    )
    parser.add_argument(
        '--save_path',
        default = f'results/classification_scores/{DS_TYPE}/',
        help = 'Path to save scores on on'
    )
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    args.rank, args.world_size = rank, world_size
    init_distributed(rank, world_size)

    if args.rank == 0:
        print(socket.gethostname())

    assert args.num_sequences % (args.world_size * args.batch_size)  == 0, "Number of sequences must be a multiple of world size"
    args.num_sequences //= args.world_size
    args.start = args.rank*args.num_sequences

    model = AutoModelForSequenceClassification.from_pretrained('usvsnsp/code-vs-nl').half().eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    class_tok = AutoTokenizer.from_pretrained("usvsnsp/code-vs-nl")
    batches = np.lib.format.open_memmap(args.offsets_path)
    num_tokens = batches.shape[-1]

    batches = batches[args.start:args.start + args.num_sequences].reshape((-1, args.batch_size, num_tokens))

    if args.rank == 0:
        batches = tqdm(batches)
    
    nl_scores = []
    indicies = np.arange(args.start, args.start + args.num_sequences)
    for idx, batch in enumerate(batches):
        detokenized = tokenizer.batch_decode(batch)
        tokens = class_tok(detokenized, return_tensors = 'pt', padding = True, truncation = True)
        with torch.no_grad():
            outputs = model(input_ids = tokens['input_ids'].cuda(), attention_mask = tokens['attention_mask'].cuda())
            probabilities = torch.nn.functional.softmax(outputs.logits, dim = 1).cpu().numpy()
            nl_scores.append(probabilities)
        dist.barrier()
    
    nl_scores = np.concatenate(nl_scores)
    if rank == 0:
        os.makedirs(args.save_path, exist_ok = True)
    dist.barrier()
    with open(f"{args.save_path}scores_{rank}.npz", 'wb') as f:
        np.savez(f, nl_scores = nl_scores, indicies = indicies)