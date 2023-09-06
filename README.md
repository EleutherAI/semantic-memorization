# Semantic Memorization 

Additional information and workstream can be found in the ([Notion project](https://eleutherai.notion.site/Semantic-Memorization-eeba3b27f82e43f4b636d742f2914d4f)).

## Motivation

`Memorization` refers to language models' tendency to sometimes output entire training sequences verbatim. This phenomenon is not deeply understood but has implications for safely deploying language models. In particular, it is vital to minimize a model’s memorization of sensitive datapoints such as those containing personally identifiable information (PII) and trade secrets.

This project aims to challenge this traditional definition of memorization. We believe that it captures the spirit of the problem but that it is too broad. For example, the `k-elicitable` definition ( Carlini et al., 2022) treats highly repetitive text, code, and sequences with only a single true continuation as memorized and thus undesirable. We conjecture that traditional memorization definitions incorrectly capture too many of these benign memorizations and don't accurately reflect undesirable memorization.

![image](https://user-images.githubusercontent.com/17308542/225178468-e3014b13-513e-4d0d-a72b-26f900ec9932.png)

Archetypal examples of sequences from The Pile “memorized” by GPT-2, even though GPT-2 was not trained on The Pile. This implies that either there is training set overlap, or that there are sequences that most competent language models could predict without needing to see the sequence during training. Carlini et al., 2022)

## Potential Research/Paper Contributions

- We want to develop a robust taxonomy of types of memorization as well as the ability to analyze memorization across these categories. This may involve developing some metric for how likely a sequence is to be memorized, mapping a model's activations to memorization type, or another approach.
- A definition of memorization that better captures `adverse/harmful memorizations` while minimizing the inclusion of `spurious/benign memorizations` is an essential step in measuring this problem and taking action toward mitigating it.
- Can we assign a probability to whether a particular sequence will be memorized or not? This coupled with a taxonomy may help us begin to understand why LLMs memorize some data and not others.
- Can we develop a classifier than can filter out benign memorizations? This will allow us to measure harmful memorizations more closely.

## Datasets

We’re currently analyzing the data memorized by the Pythia models as a part of the [Emergent and Predictable Memorization in Large Language Models](https://cdn.discordapp.com/attachments/1029044901645652111/1084179731991244880/Interpretability-31.pdf) EleutherAI paper. Reading that paper this give a better understanding of where the data came from and what it means. The datasets can be found on Hugging Face. [EleutherAI/pythia-memorized-evals · Datasets at Hugging Face](https://huggingface.co/datasets/EleutherAI/pythia-memorized-evals)

![image](https://user-images.githubusercontent.com/17308542/225178619-98c2f26e-98f0-40b2-9034-1abf083e6329.png)

## Background

Having a basic grasp of the existing literature and problem area will be helpful for contributing to this project. You don’t need a super deep understand and there are opportunities for contributing across different levels of experience. **Please add any more papers/articles that you think are relevant as well as leave comments on existing articles.**

## Development Setup
1. Setup your Python (3.11.4) environment via [Conda](https://docs.conda.io/projects/miniconda/en/latest/)
2. Run `apt-get install -y openjdk-11-jdk` to install JDK for PySpark if you're on Ubuntu, otherwise feel free to use the appropriate package manager
3. Install Python packages via `pip install -r requirements.txt`

## Running metric pipeline
1. Run `python calculate_metrics.py`
2. To monitor the status of Spark jobs, go to `http://localhost:4040/jobs/`; Don't forget to port-forward `4040` if necessary
