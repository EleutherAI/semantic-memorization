from scipy.stats import entropy

def calculate_kl_divergence(all_memories_and_sample, memories_dataset, duplication_count):
    all_memories_bootstrap_sample = all_memories_and_sample[all_memories_and_sample["sequence_duplicates"] == duplication_count].sample(frac=1, replace=True)
    memories_at_duplication_count = memories_dataset[memories_dataset["sequence_duplicates"] == duplication_count]["generation_perplexity"]
    non_memories_at_duplication_count = all_memories_bootstrap_sample[all_memories_bootstrap_sample["sequence_duplicates"] == duplication_count]["generation_perplexity"]

    # downsample to calculate kl divergence
    balance_size = min(len(memories_at_duplication_count), len(non_memories_at_duplication_count))
    memories_at_duplication_count = memories_at_duplication_count.sample(balance_size)
    non_memories_at_duplication_count = non_memories_at_duplication_count.sample(balance_size)

    # generate kl divergence in perplexity with memories as the true distribution
    kl_divergence = entropy(memories_at_duplication_count, non_memories_at_duplication_count)
    return {
        "Duplicates": duplication_count,
        "KL Divergence": kl_divergence
    }