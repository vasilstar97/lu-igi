import networkx as nx
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from pymoo.core.population import Population
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from .rules import ADJACENCY_RULES_GRAPH
from .transition_matrix import TRANSITION_MATRIX
from .mutation import Mutation
from .problem import (
    Problem, 
    FitnessType,
    LAND_USE_KEY,
    AREA_KEY, 
    RATIO_KEY, 
    TRANSITION_PROBABILITIES_KEY, 
    LENGTH_KEY,
)
from ..models.land_use import LandUse

GEOMETRY_KEY = 'geometry'

LAND_USE = list(LandUse)

ASSIGNED_LAND_USE_KEY = 'assigned_land_use'
N_EVAL = 1000
POPULATION_SIZE = 10
MUTATION_PROBABILITY = 0.1
SEED = 42

class Optimizer():

    def __init__(self, graph : nx.DiGraph):
        self.graph = self._preprocess_graph(graph)

    def best_solutions(self, results, fitness_types : list[FitnessType]):
        solutions = results.X
        fitnesses = results.F
        best = {}
        for j, fitness_type in enumerate(fitness_types):
            min_i = min([i for i,_ in enumerate(solutions)], key = lambda i : fitnesses[i][j])
            best[fitness_type] = {
                'X': solutions[min_i],
                'f': fitnesses[min_i][j]
            }
        return best

    def to_gdf(self, solution, blocks_ids : list[int], crs = None):
        if crs is None:
            assert 'crs' in self.graph.graph, 'CRS should be provided either in graph or as param'
            crs = self.graph.graph['crs']

        data = [self.graph.nodes[block_id] for block_id in blocks_ids]
        gdf = gpd.GeoDataFrame(data, crs=crs)
        gdf[LAND_USE_KEY] = gdf[LAND_USE_KEY].apply(lambda lu : None if lu is None or np.isnan(lu) else list(LandUse)[round(lu)])
        gdf[ASSIGNED_LAND_USE_KEY] = [LAND_USE[round(v)] for v in solution]
        return gdf
    
    def plot(self, gdf : gpd.GeoDataFrame, title : str = 'Визуализация', *args, **kwargs):
        ax = gdf.plot(column=ASSIGNED_LAND_USE_KEY, legend=True, *args, **kwargs)
        ax.set_axis_off()
        ax.set_title(title)

    @staticmethod
    def _preprocess_graph(graph):
        graph = graph.copy()

        for _,data in tqdm(graph.nodes(data=True)):
            geometry = data[GEOMETRY_KEY]
            land_use = data[LAND_USE_KEY]

            data[AREA_KEY] = geometry.area
            data[RATIO_KEY] = 1
            data[LAND_USE_KEY] = None if land_use is None else LAND_USE.index(land_use)

            if land_use is None:
                data[TRANSITION_PROBABILITIES_KEY] = {LAND_USE.index(lu) : 1/len(list(LandUse)) for lu in list(LandUse)}
            else:
                data[TRANSITION_PROBABILITIES_KEY] = {LAND_USE.index(lu) : TRANSITION_MATRIX.loc[land_use, lu] for lu in list(LandUse)}

        for u,v,data in tqdm(graph.edges(data=True)):
            geometry = data[GEOMETRY_KEY]
            data[LENGTH_KEY] = geometry.length

        return graph
    
    def _generate_initial_solution(self, problem : Problem):
        mutation = Mutation(1.0)
        solution = np.zeros(len(problem.nodes))
        return mutation.mutate_solution(solution, problem)
    
    def _generate_initial_population(self, problem : Problem, population_size : int):
        return Population.new(X=[self._generate_initial_solution(problem) for _ in range(population_size)])
    
    def run(
            self, 
            blocks_ids : list[int],
            target_lu_shares,
            fitness_types : list[FitnessType] = list(FitnessType),
            n_eval : int = N_EVAL,
            population_size : int = POPULATION_SIZE,
            mutation_probability : float = MUTATION_PROBABILITY,
            seed : int = SEED,
            verbose : bool = True,
        ):
        
        target_lu_shares = {LAND_USE.index(lu):share for lu,share in target_lu_shares.items()}

        problem = Problem(self.graph, blocks_ids, target_lu_shares, len(LAND_USE), fitness_types, adjacency_rules_graph=nx.relabel_nodes(ADJACENCY_RULES_GRAPH, LAND_USE.index))
        initial_population = self._generate_initial_population(problem, population_size)
        mutation = Mutation(mutation_probability)
        termination = get_termination("n_eval", n_eval)
        algorithm = NSGA2(
            pop_size=population_size,
            mutation = mutation
        )

        return minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            # save_history=True,
            verbose=verbose,
            X=initial_population
        )




