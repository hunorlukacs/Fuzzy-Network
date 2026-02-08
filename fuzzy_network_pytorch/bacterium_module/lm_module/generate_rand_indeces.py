import random
from Input import Input



def generate_rand_indeces(inp:Input):
  nr_observ = inp.observations.shape[0]
  return random.sample([i for i in range(nr_observ)], round(nr_observ * inp.SUBSAMPL_RATIO))