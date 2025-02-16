import networkx as nx
import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from pymoo.core.population import Population
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from .rules import ADJACENCY_RULES_GRAPH
from .transition_matrices import POSSIBILITY_MATRIX
from .mutation import Mutation
from .problem import (
    Problem, 
    FitnessType,
    OptimizationType,
    OPTIMIZATION_TYPES,
    LAND_USE_KEY,
    AREA_KEY, 
    RATIO_KEY, 
    TRANSITION_WEIGHTS_KEY, 
    LENGTH_KEY,
)
from ..models.land_use import LandUse

LAND_USE = list(LandUse)

GEOMETRY_KEY = 'geometry'
SOLUTION_COLUMN = 'solution'
FITNESS_COLUMN = 'fitness'
ASSIGNED_LAND_USE_KEY = 'assigned_land_use'
GDF_KEY = 'gdf'

N_EVAL = 1000
POPULATION_SIZE = 10
MUTATION_PROBABILITY = 0.1
SEED = 42

class Optimizer():

    def __init__(self, graph : nx.DiGraph):
        self.graph = self._preprocess_graph(graph)
    
    def plot(self, gdf : gpd.GeoDataFrame, *args, **kwargs):
        ax = gdf.plot(column=ASSIGNED_LAND_USE_KEY, legend=True, *args, **kwargs)
        ax.set_axis_off()

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
                data[TRANSITION_WEIGHTS_KEY] = {LAND_USE.index(lu) : 1 for lu in list(LandUse)}
            else:
                data[TRANSITION_WEIGHTS_KEY] = {LAND_USE.index(lu) : 1 if POSSIBILITY_MATRIX.loc[land_use, lu] else 0 for lu in list(LandUse)}

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
    
    def expand_result_df(self, result_df : pd.DataFrame, crs = None) -> list[dict]:
        if crs is None:
            assert 'crs' in self.graph.graph, 'CRS should be provided either in graph or as param'
            crs = self.graph.graph['crs']

        result = []
        for i,row in result_df.iterrows():
            row_dict = {ft.value : row[ft.value] for ft in list(FitnessType)}
            land_use_dict = row[ASSIGNED_LAND_USE_KEY]
            blocks_ids = land_use_dict.keys()
            # data = [{self.graph.nodes[block_id],} for block_id in blocks_ids]
            df = pd.DataFrame.from_dict(land_use_dict, orient='index', columns=[ASSIGNED_LAND_USE_KEY])
            df[LAND_USE_KEY] = [self.graph.nodes[i][LAND_USE_KEY] for i in df.index]
            df[LAND_USE_KEY] = df[LAND_USE_KEY].apply(lambda lu : None if np.isnan(lu) else LAND_USE[int(lu)])

            gdf = gpd.GeoDataFrame([{'id': block_id, 'geometry': self.graph.nodes[block_id][GEOMETRY_KEY]} for block_id in blocks_ids], crs=crs)
            gdf = gdf.set_crs(crs).set_index('id', drop=True)
            gdf = pd.concat([gdf, df[[LAND_USE_KEY, ASSIGNED_LAND_USE_KEY]]], axis=1)
            row_dict[GDF_KEY] = gdf
            # gdf[LAND_USE_KEY] = gdf[LAND_USE_KEY].apply(lambda lu : None if lu is None or np.isnan(lu) else list(LandUse)[round(lu)])
            # gdf[ASSIGNED_LAND_USE_KEY] = [LAND_USE[round(v)] for v in solution]
            result.append(row_dict)
        return result

    def result_to_df(self, result, blocks_ids) -> pd.DataFrame:
        data = {
            SOLUTION_COLUMN : list(result.X),
            FITNESS_COLUMN : list(result.F)
        }
        df = pd.DataFrame.from_dict(data)
        
        def explain_solution(solution):
            res = {}
            for i,v in enumerate(solution):
                v = round(v)
                lu = LAND_USE[v]
                block_id = blocks_ids[i]
                res[block_id] = lu
            return res            

        df[ASSIGNED_LAND_USE_KEY] = df[SOLUTION_COLUMN].apply(explain_solution)

        for i,fitness_type in enumerate(FitnessType):
            k = -1 if OPTIMIZATION_TYPES[fitness_type] == OptimizationType.MAXIMIZE else 1
            df[fitness_type.value] = df[FITNESS_COLUMN].apply(lambda f : k * f[i])

        return df.sort_values(FitnessType.SHARE_MSE.value)

    def run(
            self, 
            blocks_ids : list[int],
            target_lu_shares,
            n_eval : int = N_EVAL,
            population_size : int = POPULATION_SIZE,
            mutation_probability : float = MUTATION_PROBABILITY,
            seed : int = SEED,
            verbose : bool = True,
        ):
        
        target_lu_shares = {LAND_USE.index(lu):share for lu,share in target_lu_shares.items()}

        problem = Problem(self.graph, blocks_ids, target_lu_shares, len(LAND_USE), adjacency_rules_graph=nx.relabel_nodes(ADJACENCY_RULES_GRAPH, LAND_USE.index))
        initial_population = self._generate_initial_population(problem, population_size)
        mutation = Mutation(mutation_probability)
        termination = get_termination("n_eval", n_eval)
        algorithm = NSGA2(
            pop_size=population_size,
            mutation = mutation
        )

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=verbose,
            X=initial_population
        )

        return self.result_to_df(result, blocks_ids)




