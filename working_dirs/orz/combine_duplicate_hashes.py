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
import logging
import pickle

def iterate(args):
    s3 = args.s3
    rank = args.rank
    args.part_number = args.rank + 1
    num_records = args.save_every*1024
    if args.total_iters % args.save_every != 0:
        raise ValueError((
            "Make sure your arguments for both `calculate_duplicates`"
            "and `combine_duplicates` are the same"
        ))
        
    total_files = args.total_iters // args.save_every

    if total_files % args.world_size != 0:
        raise ValueError((
            "Make sure that the number of processes are divisible "
            f"by the files generated, {total_files}"))
    
    all_hashes = set()
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
        df = df[df["Offset"] == 0]
        for idx, seq_hash in df["Hash"].items():
            all_hashes.add(seq_hash)

        args.comm.Barrier()
    
    with open(os.path.join(args.temp_local_folder, f"{args.rank}" + ".pkl"), "wb") as fp:
        pickle.dump(all_hashes, fp)
    

if __name__ == '__main__':
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
        default = 'orz/semantic-memorization-duplicates/standard',
        help = 'Key to an empty file. Will be used as cache for saving semi-processed data'
    )
    parser.add_argument(
        '--pile_key',
        default = 'orz/pile/standard/document.bin',
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
        default = '/fsx/orz/temp/',
        help = 'Temporary cache folder to help upload parts to s3'
    )

    logging.getLogger('boto3').setLevel(logging.DEBUG)
    logging.getLogger('botocore').setLevel(logging.DEBUG)
    logging.getLogger('s3transfer').setLevel(logging.DEBUG)

    # Parse args and get current process rank
    args = parser.parse_args()
    args.comm = MPI.COMM_WORLD
    args.rank = args.comm.Get_rank()
    if args.rank == 0:
        print("*"*10 + "INPUT CONFIG:" + "*"*10)
        for arg in vars(args):
            print(arg, getattr(args, arg))
        print("*"*28)
    args.world_size = args.comm.Get_size()
    random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
    args.temp_local_file = os.path.join(args.temp_local_folder, f'{random_str}_{args.rank}')
    args.dataset_type = args.pile_key.split("/")[-2]

    # Initialize s3, create a multipart upload
    time.sleep(args.rank/80)
    args.s3 = boto3.client('s3', config=Config(retries={'max_attempts': 10}))

    # Barrier
    args.comm.Barrier()
    iterate(args)
    

    if args.rank == 0:
        all_hashes = set()
        for rank in range(args.world_size):
            path = os.path.join(args.temp_local_folder, f"{rank}" + ".pkl")
            with open(os.path.join(path) , 'rb') as f:
                hashes = pickle.load(f)
                all_hashes = all_hashes.union(hashes)
            
            os.remove(path)
            
        
        with open(os.path.join(args.temp_local_folder, args.dataset_type + ".pkl"), "wb") as fp:
            pickle.dump(all_hashes, fp)

