import copy
import torch
from torch import nn
from tqdm import tqdm
from neural_network_bma_pytorch.bea.Input import InputBEA
from neural_network_bma_pytorch.bea.bacterium_modul.BacteriumAbstract import BacteriumAbstract
from neural_network_bma_pytorch.bea.model_save_load import save_model, load_model
from neural_network_bma_pytorch.bea.population_module.PopulationAbstract import PopulationAbstract

class Statistics:
    def __init__(self):
        self.Errors = []
        self.Evolution = []

class BEA_optimizer:
    """Bacterial evolutionary algorithm optimizer (PyTorch compatible)."""

    def __init__(self, model_phenotype: nn.Module, inp: InputBEA, MyBacteriumConcreteClass: BacteriumAbstract):
        self._inp = inp
        self.MyBacteriumConcreteClass = MyBacteriumConcreteClass
        self.MyPopulationConcreteClass = None
        self._solution = None
        self._population = None  # lazy initialization
        self._current_generation = 0
        self.model_phenotype = model_phenotype
        self.statistics = Statistics()

    @property
    def current_generation(self):
        return self._current_generation

    @current_generation.setter
    def current_generation(self, new_current_generation):
        self._current_generation = new_current_generation

    @property
    def inp(self):
        return self._inp

    @inp.setter
    def inp(self, new_inp):
        self._inp = new_inp

    @property
    def population(self):
        if self._population is None:
            if self.inp.observations is None or self.inp.desired_outputs is None or self.inp.observation_dim is None:
                raise ValueError(f'Cannot get population, input not set.')
            # Lazy initialization of the population
            if self.MyPopulationConcreteClass is None:
                self._population = PopulationAbstract(inp=self.inp, MyBacteriumConcreteClass=self.MyBacteriumConcreteClass,
                                                      model_phenotype=self.model_phenotype)
            else:
                self._population = self.MyPopulationConcreteClass(inp=self.inp, MyBacteriumConcreteClass=self.MyBacteriumConcreteClass)
        return self._population

    @population.setter
    def population(self, new_population):
        self._population = new_population

    @property
    def solution(self) -> BacteriumAbstract:
        if not self._solution:
            raise ValueError('No solution yet. Call fit() first!')
        return self._solution

    @solution.setter
    def solution(self, new_solution):
        self._solution = new_solution

    def apply_optimisation_operators(self) -> PopulationAbstract:
        inp = self.inp
        if inp.b_mut:
            self.population.mutation()
        if inp.b_gt:
            self.population.gene_transfer()
        return self.population
    
    def fit(self, observations: torch.Tensor, desired_outputs: torch.Tensor = None, verbose=False):
      """Trains the population using BEA."""
      if desired_outputs is not None:
          assert len(observations) == len(desired_outputs)
          self.inp.desired_outputs = desired_outputs

      self.inp.observations = observations
      self.inp.input_set_fitData()

      self.solution = self.population.population[0]
      self.current_generation = 0

      # Only create progress bar if verbose=True
      if verbose:
          progress_bar = tqdm(range(self.current_generation, self.inp.n_gen),
                              desc=f'Error: {self.solution.error}', leave=True)
      else:
          progress_bar = range(self.current_generation, self.inp.n_gen)

      for gen_num in progress_bar:
          self.population = self.apply_optimisation_operators()
          self.solution = min(self.population.population, key=lambda bacterium: bacterium.error)
          self.statistics.Errors.append(self.solution.error)
          self.statistics.Evolution.append(copy.deepcopy(self.population.population))
          if verbose:
              progress_bar.set_description(f'Error: {self.solution.error}')
              progress_bar.refresh()
          self.current_generation += 1

      return self.solution, self.population

    def predict(self, Xs: torch.Tensor):
        if not self.solution:
            raise ValueError('No solution yet. Call fit() first!')
        return self.solution.predict(Xs)

    def save(self, path, filename, append_time=True):
        if self.population is None or self.solution is None:
            raise NameError('Population or solution is None!')

        data = {
            'population': self.population,
            'solution': self.solution,
            'inp': copy.copy(self.inp),
            'current_generation': self.current_generation,
            'statistics': self.statistics
        }
        data['inp'].observations = None
        data['inp'].desired_outputs = None

        save_model(data=data, directory=path, filename=filename, append_time=append_time)
        return True

    def load(self, path, set_input=False):
        data = load_model(path=path)
        self.population = data['population']
        self.solution = data['solution']
        if set_input:
            self.inp = data['inp']
        self.current_generation = data['current_generation']
        self.statistics = data['statistics']
        print('[+] Model loaded successfully!')
        return True
