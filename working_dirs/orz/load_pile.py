from mmap_dataset import MMapIndexedDataset
from tqdm import trange

if __name__ == '__main__':
    DATASET = 'deduped' # Choose between 'standard' and 'deduped'

    dataset_path = f'/mnt/ssd-1/pile_preshuffled/{DATASET}/document'
    dataset = MMapIndexedDataset(dataset_path, skip_warmup = True)

    # You can iterate over the full dataset in about 30 seconds

    batch_size = 10240
    for i in trange(0, 143000*1024, batch_size):
        batch = dataset[i:i + batch_size]


