import copy
from tqdm import tqdm
import torch
from fuzzy_network_pytorch.bea.Input import InputBEA
from fuzzy_network_pytorch.bea.model_save_load import save_model, load_model
from fuzzy_network_pytorch.bea.Population import Population
from fuzzy_network_pytorch import levenberg_marquardt_pytorch as tlm

class Statistics():
    Errors: list = []
    Evolution: list = []

class BEA_optimizer():
    """ Bacterial Evolutionary Algorithm optimizer for PyTorch models. """
    
    def __init__(self, model_phenotype, inp: InputBEA, MyBacteriumConcreteClass, loss_fn=tlm.MSELoss()):
        """
        model_phenotype: PyTorch model (e.g., FuzzyLayer or FuzzyNetwork)
        inp: InputBEA instance
        MyBacteriumConcreteClass: Your Bacterium class (PyTorch version)
        """
        self._inp = inp
        self._model_phenotype = model_phenotype
        self.MyBacteriumConcreteClass = MyBacteriumConcreteClass
        self.MyPopulationConcreteClass = None
        self._solution = None
        self._population = None  # lazy initialization
        self._statistics = None
        self._current_generation = 0
        self.model_phenotype = model_phenotype
        self.statistics = Statistics()
        self.loss_fn = loss_fn

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
                raise ValueError(f'[-] Cannot get attribute population, because '
                                 f'observations={self.inp.observations is None}, '
                                 f'desired_outputs={self.inp.desired_outputs is None}, '
                                 f'observation_dim={self.inp.observation_dim is None}.')
            else:
                if self.MyPopulationConcreteClass is None:
                    self._population = Population(
                        inp=self.inp,
                        MyBacteriumConcreteClass=self.MyBacteriumConcreteClass,
                        model_phenotype=self.model_phenotype,
                        loss_fn=self.loss_fn
                    )
                    # print('LEN population: ', len(self._population.population))
                else:
                    self._population = self.MyPopulationConcreteClass(inp=self.inp, MyBacteriumConcreteClass=self.MyBacteriumConcreteClass, loss_fn=tlm.MSELoss())
        return self._population

    @population.setter
    def population(self, new_population):
        self._population = new_population

    @property
    def solution(self):
        """Returns the best individual."""
        if not self._solution:
            raise ValueError('[-] No solution yet. Call fit() first!')
        return self._solution
    
    @solution.setter
    def solution(self, new_solution):
        self._solution = new_solution

    def apply_optimisation_operators(self):
        inp = self.inp

        if inp.b_mut:
            self.population.mutation()

        if inp.b_gt:
            self.population.gene_transfer()

        return self.population

    def fit(self, observations, desired_outputs=None):
        """Run the BEA optimization loop."""
        if desired_outputs is not None:
            assert len(observations) == len(desired_outputs), \
                f'Mismatch lengths: observations={len(observations)}, desired_outputs={len(desired_outputs)}'
            self.inp.desired_outputs = desired_outputs

        self.inp.observations = observations
        self.inp.input_set_fitData()

        self.solution = self.population.population[0]
        self.current_generation = 0

        # t = tqdm(range(self.current_generation, self.inp.n_gen),
        #          desc=f'BEA-Error: {self.solution.error.item() if torch.is_tensor(self.solution.error) else self.solution.error}',
        #          leave=True)


        for gen_num in range(self.current_generation, self.inp.n_gen):
            self.population = self.apply_optimisation_operators()
            self.solution = min(self.population.population, key=lambda b: b.error)
            self.statistics.Errors.append(self.solution.error.item() if torch.is_tensor(self.solution.error) else self.solution.error)
            # t.set_description(f'Error: {self.solution.error.item() if torch.is_tensor(self.solution.error) else self.solution.error}')
            # t.refresh()
            self.current_generation += 1


        return self.solution, self.population

    def predict(self, Xs):
        """Return the best individual's prediction."""
        if not self.solution:
            raise ValueError('[-] No solution yet. Call fit() first!')
        return self.solution.predict(Xs)

    def save(self, path, filename, append_time=True):
        """Save optimizer state including population and best solution."""
        if self.population is None or self.solution is None:
            raise NameError('[-] population and/or solution is None!')

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
        """Load optimizer state."""
        data = load_model(path=path)
        self.population = data['population']
        self.solution = data['solution']
        if set_input:
            self.inp = data['inp']
        self.current_generation = data['current_generation']
        self.statistics = data['statistics']
        print('[+] Model loaded successfully!')
        return True
