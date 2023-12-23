import boto3
s3 = boto3.client('s3')
from tqdm.auto import tqdm
import pandas as pd
from datasets import Dataset
import os
import re

if __name__ == '__main__':
    temp_path = '/fsx/orz/temp/temp.csv'
    os.makedirs('/fsx/orz/temp', exist_ok = True)

    models = ['70m', '160m', '410m', '1b', '1_4b', '2_8b', '6_9b', '12b']

    ds_types = ['memories', 'pile']
    df_types = ['duped', 'deduped']

    completed = set([
        re.compile("memories_*"),
        re.compile("pile_duped_*"),
        re.compile("pile_deduped_70m"),
        re.compile("pile_deduped_160m"),
        re.compile("pile_deduped_410m"),
        re.compile("pile_deduped_1b"),
        re.compile("pile_deduped_1_4b"),
        re.compile("pile_deduped_2_8b"),
        re.compile("pile_deduped_6_9b")
    ])
    for ds_type in ds_types:
        for df_type in df_types:
            for model in tqdm(models):
                is_seen = False
                for pattern in completed:
                    if pattern.match(f'{ds_type}_{df_type}_{model}'):
                        is_seen = True
                        break
                if is_seen:
                    continue
                key = f'orz/semantic-memorization/entropies/{ds_type}_{df_type}_{model}.csv'
                bucket = 's-eai-neox-west'
                file_path = temp_path
                print(f"Downloading file... {model}")
                s3.download_file(bucket, key, file_path)
                with open(file_path, 'r') as f:
                    data = f.read().splitlines()
                    print(len(data))
                    data = [data[0]] + [i for i in data[1:] if i[0] != 'i']
                with open(file_path, 'w') as f:
                    f.write('\n'.join(data))
                
                print("Processing file...")
                data = pd.read_csv(file_path, index_col = 'index')
                
                print("Total Number of sequences: ", len(data))
                array_cols = [i for i in data.columns.values if i.startswith('gini') or i.startswith('entropy')]
                for col in tqdm(array_cols, desc = 'Cleaning data in ' + model):
                    col_new = col.split('_')
                    col_new = '_'.join([col_new[0], 'heads', col_new[2]])
                    data[col] = data[col].map(lambda x:[float(c) for c in x.strip('[]').split()])
                    if col != col_new:
                        data.rename(columns = {col: col_new}, inplace = True)
                
                ds = Dataset.from_pandas(data)
                ds = ds.rename_column("index", "sequence_id")
                ds.push_to_hub('usvsnsp/semantic-memorization-entropies', split = ds_type, revision = f'{df_type}_{model}', max_shard_size = '25GB')
                

