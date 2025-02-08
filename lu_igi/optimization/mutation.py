import random
import numpy as np
from .problem import Problem, TRANSITION_PROBABILITIES_KEY
from pymoo.core.mutation import Mutation as PymooMutation

class Mutation(PymooMutation):

    def __init__(self, probability : float):
        super().__init__()
        self.probability = probability

    def _mutate_variable(self,i,value,problem):
        probabilities = problem.graph.nodes[problem.get_node(i)][TRANSITION_PROBABILITIES_KEY]
        if random.random()<=self.probability:
            value = random.choices(list(probabilities.keys()), list(probabilities.values()),k=1)[0]
        return value

    def mutate_solution(self,solution,problem):
        return [self._mutate_variable(i,v,problem) for i,v in enumerate(solution)]

    def _do(self, problem : Problem, X, **kwargs):
        return np.array([self.mutate_solution(solution,problem) for solution in X])