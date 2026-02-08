import torch
import random
from fuzzy_network_pytorch.bea.Input import InputBEA

def get_rnd_geneId_lists(gene_nr: int):
    """
    Separates the bacterial chromosome into smaller gene-lists (PyTorch version).

    Example:
    >>> get_rnd_geneId_lists(15)
    [tensor([ 0, 12,  3, 13]), tensor([10,  1,  2, 11]), tensor([ 4,  6, 14,  9]), tensor([8, 7, 5])]
    """
    tmp = torch.randperm(gene_nr)  # Random permutation of indices
    split_into = random.randint(1, max(3, gene_nr // 2))
    tmp_splits = torch.tensor_split(tmp, split_into)  # Split into sublists
    return tmp_splits


def generate_rand_indeces(inp: InputBEA):
    """
    Randomly selects a subset of observation indices based on SUBSAMPL_RATIO.

    Returns a Python list of integers (compatible with indexing).
    """
    nr_observ = inp.observations.shape[0]
    n_samples = round(nr_observ * inp.SUBSAMPL_RATIO)
    indices = random.sample(range(nr_observ), n_samples)
    return indices
