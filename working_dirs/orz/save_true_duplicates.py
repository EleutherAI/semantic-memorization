import pandas as pd
import argparse
from tqdm.auto import tqdm
tqdm.pandas()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type = str, default = 'standard')
    parser.add_argument('--num_parts', type = int, default = 20)
    args = parser.parse_args()

    data = {}
    for i in range(args.num_parts):
        ds = pd.read_csv(f"build/true_counts_{i}.csv")
        for row in tqdm(ds.itertuples(), total = len(ds)):
            data[row._1] = row.Count
    
    true_counts = pd.DataFrame()
    true_counts['Index'] = [i for i in range(1024*143000)]
    true_counts['Counts'] = true_counts['Index'].progress_map(lambda x:data[x] if x in data else 1)
    true_counts.sort_values("Index", inplace = True)
    true_counts.to_parquet(f"{args.dataset_type}_true_counts.parquet")

    