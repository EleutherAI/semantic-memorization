import pandas as pd
import os
import pickle
from lrs import longestRepeatedSublist
from mrs import find_most_occuring_substring
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import multiprocessing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=multiprocessing.cpu_count())

files = os.listdir('../data')
# choose files without 'lrs' or 'mrs' in the name
files = [file for file in files if 'lrs' not in file and 'mrs' not in file]

def preprocess_tokens(toks):
    str_arr = toks.astype(str)
    str_lst = ''.join(str_arr)
    str_lst = str_lst.replace(' ', '').replace(',', '')
    return str_lst

def get_df_with_mrs(file):
    df = pickle.load(open('../data/' + file, 'rb'))
    df['preprocessed_tokens'] = df['tokens'].parallel_apply(preprocess_tokens)
    df['mrs'] = df['preprocessed_tokens'].parallel_apply(find_most_occuring_substring)
    df['mrs_len'] = df['mrs'].parallel_apply(len)
    # # save the df as a pickle file
    pickle.dump(df, open('../data/' + file + '_with_mrs', 'wb'))
    return df

def plot_and_save_mrs(df, file):
    plt.figure(figsize=(10, 8))
    sns.distplot(df['mrs_len'], kde=False)
    plt.title('Distribution of MRS Lengths for ' + file)
    plt.xlabel('MRS Length')
    plt.ylabel('Count')
    plt.savefig('../plots/' + file + '_mrs_dist.png')

for file in files:
    df = get_df_with_mrs(file)
    plot_and_save_mrs(df, file)
    break

# Stitch together the plots into a pdf
import glob
from fpdf import FPDF
from PIL import Image

pdf = FPDF()
# imagelist is the list with all image filenames
files = glob.glob('../plots/*.png')
# choose files with mrs in the name
files = [file for file in files if 'mrs' in file]
for image in files:
    pdf.add_page()
    pdf.image(image, 0, 0, 210, 297)
pdf.output("../plots/mrs_dists.pdf", "F")
