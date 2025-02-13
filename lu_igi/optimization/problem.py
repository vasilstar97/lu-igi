from enum import Enum
import networkx as nx
import numpy as np
from pymoo.core.problem import Problem as PymooProblem

LAND_USE_KEY = 'land_use'
AREA_KEY = 'area'
RATIO_KEY = 'ratio'
LENGTH_KEY = 'length'
TRANSITION_WEIGHTS_KEY = 'transition_weights'

class FitnessType(Enum):
    PROBABILITY = 'Вероятность перехода'
    SHARE_MSE = 'Близость к целевому состоянию'
    ADJACENCY_PENALTY = 'Конфликт размещения функциональных зон'

class OptimizationType(Enum):
    MAXIMIZE = 'Максимизация'
    MINIMIZE = 'Минимизация'

FITNESS_OPTIMIZATION_TYPES = {
    FitnessType.PROBABILITY : OptimizationType.MAXIMIZE,
    FitnessType.SHARE_MSE : OptimizationType.MINIMIZE,
    FitnessType.ADJACENCY_PENALTY : OptimizationType.MINIMIZE
}

class Problem(PymooProblem):

    def __init__(
            self, 
            graph : nx.DiGraph, 
            nodes : list[int], 
            target_shares : dict[int, float], 
            num_classes : int, 
            fitness_types : list[FitnessType],
            adjacency_rules_graph : nx.Graph,
        ):     
        self.graph = graph
        self.nodes = nodes
        self.target_shares = target_shares
        self.fitness_types = fitness_types
        self.adjacency_rules_graph = adjacency_rules_graph
        super().__init__(
            n_var=len(self.nodes),
            n_obj=len(fitness_types),
            n_constr=0,
            xl=0,
            xu=num_classes-1,
            type_var=int # Дискретные переменные (метки)
        )

    def get_node(self, i):
        return self.nodes[i]

    def _evaluate_share_mse(self, solution):
        total_area = sum(self.graph.nodes[self.get_node(i)][AREA_KEY] for i,_ in enumerate(solution))
        deviations = []
        for cls,target_share in self.target_shares.items():
            nodes = [self.get_node(i) for i,v in enumerate(solution) if round(v)==cls]
            area = sum([self.graph.nodes[n][AREA_KEY] for n in nodes], 0)
            share = area / total_area
            deviation = (share - target_share) ** 2
            deviations.append(deviation)
        return sum(deviations)

    def _get_transition_probability(self, i, v):
        node = self.get_node(i)
        v = round(v)
        return self.graph.nodes[node][TRANSITION_WEIGHTS_KEY][v] * self.graph.nodes[node][AREA_KEY]

    def _evaluate_probability(self, solution) -> float:
        probabilities = [self._get_transition_probability(i,v) for i,v in enumerate(solution)]
        return np.prod(probabilities)
    
    def _evaluate_adjacency_penalty(self, solution) -> float:

        def penalty(u,v,d):
            if u in self.nodes:
                u_cls = solution[self.nodes.index(u)]
            else:
                u_cls = self.graph.nodes[u][LAND_USE_KEY]
            if v in self.nodes:
                v_cls = solution[self.nodes.index(v)]
            else:
                v_cls = self.graph.nodes[u][LAND_USE_KEY]
            
            if u_cls is not None and v_cls is not None:
                if not self.adjacency_rules_graph.has_edge(u_cls, v_cls):
                    return self.graph.nodes[u][AREA_KEY] * self.graph.nodes[v][AREA_KEY]
            return 0

        return sum([penalty(u,v,d) for u,v,d in self.graph.edges(data=True) if u in self.nodes or v in self.nodes])

    
    def _get_fitness(self, fitness_type, solution):
        fitness_value = 0
        if fitness_type == FitnessType.PROBABILITY:
            fitness_value = self._evaluate_probability(solution)
        if fitness_type == FitnessType.SHARE_MSE:
            fitness_value = self._evaluate_share_mse(solution)
        if fitness_type == FitnessType.ADJACENCY_PENALTY:
            fitness_value = self._evaluate_adjacency_penalty(solution)
        if FITNESS_OPTIMIZATION_TYPES[fitness_type] == OptimizationType.MAXIMIZE:
            fitness_value *= -1
        return fitness_value

    def _get_fitnesses(self, solution):
        return {fitness_type : self._get_fitness(fitness_type, solution) for fitness_type in self.fitness_types}
    
    def _evaluate(self, solutions, out, *args, **kwargs):

        solutions_fitnesses = {fitness_type : [] for fitness_type in self.fitness_types}

        for solution in solutions:

            fitnesses = self._get_fitnesses(solution)

            for ft, value in fitnesses.items():
                solutions_fitnesses[ft].append(value)

        out["F"] = np.column_stack([v for v in solutions_fitnesses.values()])