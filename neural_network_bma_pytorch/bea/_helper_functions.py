import random
import numpy.random as rnd
import numpy as np

from bea.Input import InputBEA


def get_rnd_geneId_lists(gene_nr, n_lists=None):
    '''
    Separates the bacterial chromosome into smaller gene-lists.

    Parameters
    ----------
    gene_nr : int
        Number of genes.
    n_lists : int, optional
        Desired number of gene lists (default is None, meaning random).

    Example
    -------
    >>> get_rnd_geneId_lists(gene_nr=15)
    >>> # Example output: [array([ 0, 12,  3, 13]), array([10,  1,  2, 11]),
    >>> #                  array([ 4,  6, 14,  9]), array([8, 7, 5])]
    '''

    tmp = np.arange(gene_nr)
    rnd.shuffle(tmp)
    
    if n_lists is None:
        split_into = rnd.randint(1, max(3, gene_nr // 2))
    else:
        split_into = max(1, min(n_lists, gene_nr))  # Ensure valid range

    tmp = np.array_split(tmp, split_into)
    return tmp

def generate_rand_indeces(inp:InputBEA):
  nr_observ = inp.observations.shape[0]
  return random.sample([i for i in range(nr_observ)], round(nr_observ * inp.SUBSAMPL_RATIO))
