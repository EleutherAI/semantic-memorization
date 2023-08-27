import boto3
from botocore.config import Config
import numpy as np
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

def pow_mod(x, y, z):
    "Calculate (x ** y) % z efficiently."
    number = int(1)
    x = int(x)
    while y:
        if y & 1:
            number = number * x % z
        y >>= 1
        x = x * x % z
    return number


def save_results(args):
    file_size = os.path.getsize(args.temp_local_file)
    if  file_size > 4:
        resp = args.s3.upload_file(
            Filename = args.temp_local_file,
            Key = args.cache_key + f'/{args.part_number}.csv',
            Bucket = args.bucket
        )
    return



def gethash(arr, seq_idx, start, MOD1 = int(int(1e18) + 3), MOD2 = 59981):
    ans = 0
    for i in range(0, 64, 8):
        ans += pow_mod(arr[seq_idx, start + i], (i+1)//8, MOD1)
        ans %= MOD1
    
    ans += int(np.sum(arr[seq_idx, start:start + 64]))
    ans %= MOD1
    return ans
def iterate(args):
    s3 = args.s3
    rank = args.rank
    args.part_number = args.rank + 1
    iters_per_rank = args.total_iters // args.world_size

    if args.total_iters % args.world_size != 0:
        raise ValueError((
            "Make sure that the number of processes are divisible "
            "by total iters."))
        
    
    if args.total_iters % args.save_every != 0:
        raise ValueError((
            "Make sure that checkpointing iters are divisible "
            "by total iters."))

    # Batch size of uint_16 arrays in bytes
    batch_size = 2049*1024*2
    results = open(args.temp_local_file, 'w')
    args.curr_rank = iters_per_rank*1024*rank

    if args.rank == 0:
        it = tqdm(range(1, iters_per_rank + 1))
    else:
        it = range(1, iters_per_rank + 1)
    for doc_idx in it:
        try:
            dataset = args.s3.get_object(
                Bucket = args.bucket, 
                Key = args.pile_key,
                Range = f'bytes={args.curr_rank*2049*2}-{args.curr_rank*2049*2 + batch_size}'
            )
            data = dataset['Body'].read(batch_size)
            data = np.frombuffer(data, dtype = np.uint16).reshape(-1, 2049)
        except ValueError as e:
            print("Error in rank: ", args.rank, args.curr_rank)
            print(e)
        for seq_idx in range(len(data)):
            for i in range(2049 - 64):
                hashed_arr = gethash(data,seq_idx, i)
                results.write(f'{args.curr_rank},{i},{hashed_arr}\n')
            args.curr_rank += 1
        # Barrier
        args.comm.Barrier()
        
        if (doc_idx % args.save_every == 0):
            results.flush()
            results.close()
            save_results(args)
            args.part_number += args.world_size
            results = open(args.temp_local_file, 'w')

    save_results(args)

    

            

    
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
        help = 'Save data to s3 after every nth batch of sequences',
        type = int
    )
    parser.add_argument(
        '--total_iters',
        default = 143000,
        help = ('Total number of batches of sequences to iterate over for calculating duplicates'
                'Each batch comprises of 1024 sequences'),
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
    args.world_size = args.comm.Get_size()
    args.rank = args.comm.Get_rank() 
    
    if args.rank == 0:
        print("*"*10 + "INPUT CONFIG:" + "*"*10)
        for arg in vars(args):
            print(arg, getattr(args, arg))
        print("*"*28)
    
    random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
    args.temp_local_file = os.path.join(args.temp_local_folder, f'{random_str}_{args.rank}.csv')

    # Initialize s3, create a multipart upload
    time.sleep(args.rank/80)
    args.s3 = boto3.client('s3', config=Config(retries={'max_attempts': 10}))

    # Barrier
    args.comm.Barrier()
    iterate(args)
    
