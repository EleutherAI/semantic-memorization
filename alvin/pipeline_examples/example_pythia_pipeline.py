import sys

from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

# Relative import from the root of the repository, this is a trade-off
# for flexibility between separate experiments.
#
# TODO: You'll need to change this based on your own directory structure.
# `os.path.dirname(os.path.abspath(__file__))` is viable, 
# but __file__ is not available in the interactive interpreter.
repo_root = '/home/alvin/research/semantic-memorization'
sys.path.append(repo_root)

from common.pipeline import MetricFilterPipeline


tokenizer = AutoTokenizer.from_pretrained(
  'EleutherAI/pythia-70m-deduped',
  revision='step3000',
)

def main():
    # Set up context
    dataset = load_dataset('EleutherAI/pythia-memorized-evals')
    raw_df = dataset['deduped.70m'].to_pandas()
    tokenizer = AutoTokenizer.from_pretrained(
        'EleutherAI/pythia-70m-deduped',
        revision='step3000',
    )

    # Create a pipeline instance
    pipeline = MetricFilterPipeline()

    # Define and register a filter using `register_filter`
    @pipeline.register_filter(output_column='text')
    def decode_token(row: pd.Series) -> str:
        return tokenizer.decode(row['tokens'])

    # Run the pipeline with registered filter(s) on any dataframe
    decoded_df = pipeline.run_pipeline(raw_df)
    print(decoded_df.head())

if __name__ == '__main__':
    main()