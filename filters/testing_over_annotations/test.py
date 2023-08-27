import pandas as pd
import os
import sys
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from pattern_incrementing import incrementing_sequences_filter
from highly_repetitive import break_and_compare_wrapper

df = pd.read_csv('filters/testing_over_annotations/test_data_full.csv')
df = df[['shortened_text', 'Category']]

all_categories = ['code', 'nl', 'pattern-incrementing', 'pattern-repeating', 'duplicated',
 'template', 'code+nl', 'empty/blank', 'other', 'random', 'structured']

target_category = "pattern-repeating"
use_tokenizer = True

df.dropna(inplace=True)

text = df['shortened_text'].to_list()
category = df['Category'].to_list()

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
)

ls = []
ls_true = [0]*len(category)
for t in text:
    # remove newlines
    # replace all non alphanumeric characters with spaces
    t = t.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # replace multiple spaces with single space
    t = ' '.join(t.split())
    if use_tokenizer:
        t = tokenizer(t)['input_ids']
    resp = break_and_compare_wrapper(t, 2, 10)
    if resp[-1] != -1:
        ls.append(target_category)
    else:
        ls.append("")
print(ls)
print(category)

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(ls + category)

encoded_list1 = encoder.transform(ls)
encoded_list2 = encoder.transform(category)

print(encoded_list1)
print(encoded_list2)

# Print classification report after a train/test split:
print(classification_report(encoded_list2, encoded_list1))

df['predicted'] = ls
df.to_csv('filters/testing_over_annotations/test_data_full_pred.csv', index=False)