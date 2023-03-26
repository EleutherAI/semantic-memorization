
import sys
import os
import numpy as np

megatron_dir = "/home/mchorse/kyleobrien/gpt-neox/"
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), megatron_dir))
sys.path.append(folder_path)
from megatron.data.data_utils import build_the_dataset

dataset = build_the_dataset(
        data_prefix = '/mnt/ssd-1/data/pile_20B_tokenizer/pile_20B_tokenizer_text_document', # Replace with the path of pile document
        name = 'train_0',
        data_impl='mmap',
        num_samples=500,
        seq_length=2048,
        seed=1234,
        skip_warmup=True,
        build_index_mappings=False
    )

idx_path = '/mnt/ssd-1/data/pile_20B_tokenizer/pile_20B_tokenizer_text_document_train_0_indexmap_143213ns'
dataset.doc_idx = np.load(f"{idx_path}_2048sl_1234s_doc_idx.npy")
dataset.sample_idx = np.load(f"{idx_path}_2048sl_1234s_sample_idx.npy")
dataset.shuffle_idx = np.load(f"{idx_path}_2048sl_1234s_shuffle_idx.npy")
# dataset.shuffle_idx_len = dataset.shuffle_idx.shape[0] - 1
# dataset.sample_idx_len = dataset.sample_idx.shape[0] - 1

print(dataset)
