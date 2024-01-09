import boto3
from botocore.config import Config
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from mpi4py import MPI
import argparse
import random
import string
import time
import os
import gc
import logging
import pickle

def iterate(args):
    s3 = args.s3
    rank = args.rank
    args.part_number = args.rank + 1
    num_records = args.save_every*1024
    if args.total_iters % args.save_every != 0:
        raise ValueError((
            "Make sure that arguments of `total_iters` and `save_every` are same`"
            "across all executed files"
        ))
        
    total_files = args.total_iters // args.save_every

    if total_files % args.world_size != 0:
        raise ValueError((
            "Make sure that the number of processes are divisible "
            f"by the files generated, {total_files}"))
    
    # hash_path = os.path.join(args.temp_local_folder, args.dataset_type + ".pkl")
    hash_path = f"/admin/home-orz/{args.dataset_type}/{args.dataset_type}.pkl"
    with open(hash_path, "rb") as fp:
        all_hashes = pickle.load(fp)
    
    it = range(args.part_number, total_files + 1, args.world_size)
    if args.rank == 0:
        it = tqdm(it)
    
    
    for part in it:
        args.s3.download_file(args.bucket, args.cache_key + f"/{part}.csv", args.temp_local_file + ".csv")
        df = pd.read_csv(args.temp_local_file + ".csv",
            names = ["Index", "Offset", "Hash"],
            dtype = {
                "Index": np.int64,
                "Offset": np.int32,
                "Hash": np.uint64
            }
        )
        df = df[df["Hash"].apply(lambda x: x in all_hashes)]
        save_path = os.path.join(args.temp_local_folder, f"{args.dataset_type}_{part}.npz")
        np.savez(
            save_path,
            index = df['Index'].to_numpy().astype(np.uint32),
            offset = df['Offset'].to_numpy().astype(np.uint16),
            hash = df['Hash'].to_numpy().astype(np.uint64),
        )
        args.comm.Barrier()
    

if __name__ == '__main__':
    NGRAMS = 32
    DS_TYPE = "standard"
    parser = argparse.ArgumentParser(
        prog = 'Calculate Duplicates in a large dataset',
        description = 'Calculates 64-gram duplicates in ~billion token large datasets.'
    )
    parser.add_argument(
        '--bucket',
        default = 's-eai-neox-west',
        help = 'Name of bucket to use. (s3://bucket-name/path/to/file)'
    )
    parser.add_argument(
        '--cache_key',
        default = f'orz/pile_{NGRAMS}_gram_hashes/{DS_TYPE}',
        help = 'Key to an empty file. Will be used as cache for saving semi-processed data'
    )
    parser.add_argument(
        '--pile_key',
        default = f'orz/pile/{DS_TYPE}/document.bin',
        help = 'Key to the location of preshuffled MemMapped dataset bin file'
    )
    parser.add_argument(
        '--save_every',
        default = 20,
        help = 'Determines that data was saved to s3 after every nth batch of sequences',
        type = int
    )
    parser.add_argument(
        '--total_iters',
        default = 143000,
        help = ('Total number of batches of sequences to iteratated over while calculating '
                'duplicates. Each batch comprises of 1024 sequences'),
        type = int
    )
    parser.add_argument(
        '--temp_local_folder',
        default = f'/fsx/orz/temp/{DS_TYPE}' ,
        help = 'Temporary cache folder to help upload parts to s3'
    )
    parser.add_argument(
        '--ngrams',
        default = NGRAMS,
        type = int,
        help = 'Ngrams to calculate duplicates out of'
    )

    logging.getLogger('boto3').setLevel(logging.DEBUG)
    logging.getLogger('botocore').setLevel(logging.DEBUG)
    logging.getLogger('s3transfer').setLevel(logging.DEBUG)

    tqdm.pandas()
    # Parse args and get current process rank
    args = parser.parse_args()
    args.comm = MPI.COMM_WORLD
    args.rank = args.comm.Get_rank()
    args.world_size = args.comm.Get_size()
    if args.rank == 0:
        print("*"*10 + "INPUT CONFIG:" + "*"*10)
        for arg in vars(args):
            print(arg, getattr(args, arg))
        print("*"*28)
        os.makedirs(args.temp_local_folder, exist_ok = True)
        for file_name in os.listdir(args.temp_local_folder):
            if file_name.endswith(".pkl"):
                continue
            os.remove(os.path.join(args.temp_local_folder, file_name))

    
    random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
    args.temp_local_file = os.path.join(args.temp_local_folder, f'{random_str}_{args.rank}')
    args.dataset_type = args.pile_key.split("/")[-2]

    # Initialize s3, create a multipart upload
    time.sleep(args.rank/80)
    args.s3 = boto3.client('s3', config=Config(retries={'max_attempts': 10}))

    # Barrier
    args.comm.Barrier()
    iterate(args)
    gc.collect()
    
    if args.rank == 0:
        merged_dataset = []
        total_files = args.total_iters // args.save_every
        indicies = []
        hashes = []
        offsets = []
        for part in tqdm(range(1, total_files + 1)):
            part_path = os.path.join(args.temp_local_folder, f"{args.dataset_type}_{part}.npz")
            df = np.load(part_path)
            indicies.append(df['index'])
            hashes.append(df['hash'])
            offsets.append(df['offset'])
            os.remove(part_path)

        os.makedirs("results", exist_ok = True)
        indicies = np.concatenate(indicies)
        hashes = np.concatenate(hashes)
        offsets = np.concatenate(offsets)
        np.savez(
            f'results/{args.dataset_type}.npz',
            index = indicies,
            hash = hashes,
            offset = offsets
        )
