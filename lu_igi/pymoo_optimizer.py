import random
from enum import Enum, IntEnum
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.population import Population
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.pntx import SinglePointCrossover
from functools import reduce
import geopandas as gpd
import networkx as nx
from lu_igi.model import Model
from lu_igi.land_use import LandUse
from lu_igi.rules import ADJACENCY_RULES_GRAPH, AREA_RANGES, ASPECT_RATIO_RANGES
from lu_igi.transition_matrix import TRANSITION_MATRIX

class OptimizationAlgorithm(Enum):
    NSGA2 = 'Non-dominated Sorting Genetic Algorithm'
    BRKGA = 'Biased Random Key Genetic Algorithm'

class FitnessType(IntEnum):
    SHARE_LU = 0
    PROBABILITY = 1
    ADJACENCY = 2
    AREA = 3
    RATIO = 4

class LandUseProblem(Problem):

    def __init__(self, model : Model, territory_gdf : gpd.GeoDataFrame, lu_shares : dict[LandUse, float]):        
        self.model = model
        self.blocks_gdf = self._get_blocks_subgdf(territory_gdf)
        self.adjacency_graph = self._get_adjacency_subgraph(self.blocks_gdf)
        self.land_use = list(LandUse)
        self.lu_shares = lu_shares
        super().__init__(n_var=len(self.blocks_gdf),
                         n_obj=5,
                         n_constr=0,
                         xl=1,
                         xu=len(self.land_use),
                         type_var=int)  # Дискретные переменные (метки)
        
    def _get_blocks_subgdf(self, territory_gdf : gpd.GeoDataFrame):
        if not territory_gdf.crs == self.model.crs:
            territory_gdf = territory_gdf.to_crs(self.model.crs)
        blocks_gdf = self.model.blocks_gdf.copy()
        return blocks_gdf[blocks_gdf.intersects(territory_gdf.union_all())]
    
    def _get_adjacency_subgraph(self, blocks_subgdf : gpd.GeoDataFrame):
        adjacency_graph = self.model.adjacency_graph
        nodes = {node for block_i in blocks_subgdf.index for node in adjacency_graph.neighbors(block_i)}
        return adjacency_graph.subgraph(nodes)
    
    def _get_adjacency_penalty(self, blocks_subgdf : gpd.GeoDataFrame, adjacency_subgraph : nx.Graph) -> float:
        blocks_gdf = self.model.blocks_gdf
        penalty = 0

        for u,v,d in adjacency_subgraph.edges(data=True):
            # get u land use
            if u in blocks_subgdf.index:
                lu_u = blocks_subgdf.loc[u,'assigned_land_use']
            else:
                lu_u = blocks_gdf.loc[u,'land_use']
            # get v land use
            if v in blocks_subgdf.index:
                lu_v = blocks_subgdf.loc[v,'assigned_land_use']
            else:
                lu_v = blocks_gdf.loc[v,'land_use']
            # if adjacency not allowed, penalty it by the length
            if not ADJACENCY_RULES_GRAPH.has_edge(lu_u, lu_v):
                border_length = d['border_length']
                penalty += border_length

        return penalty
    
    def _get_area_penalty(self, blocks_gdf : gpd.GeoDataFrame) -> float:

        def _get_penalty(area : float, lu : LandUse):
            min_area, max_area = AREA_RANGES[lu]
            if area < min_area:
                return min_area - area
            if area > max_area:
                return area - max_area
            return 0

        blocks_gdf['area_penalty'] = blocks_gdf.apply(lambda s : _get_penalty(s['area'], s['assigned_land_use']), axis=1)
        return blocks_gdf['area_penalty'].sum()
    
    def _get_ratio_penalty(self, blocks_gdf : gpd.GeoDataFrame) -> float:

        def _get_penalty(ratio : float, lu : LandUse):
            min_ratio, max_ratio = ASPECT_RATIO_RANGES[lu]
            if ratio < min_ratio:
                return min_ratio - ratio
            if ratio > max_ratio:
                return ratio - max_ratio
            return 0

        blocks_gdf['ratio_penalty'] = blocks_gdf.apply(lambda s : _get_penalty(s['aspect_ratio'], s['assigned_land_use']), axis=1)
        return blocks_gdf['ratio_penalty'].sum()
    
    def _get_transition_probability(self, blocks_gdf : gpd.GeoDataFrame) -> float:
        blocks_gdf['probability'] = blocks_gdf.apply(lambda s : s['probabilities'].get(s['assigned_land_use']), axis=1)
        # blocks_gdf['probability'] = blocks_gdf.apply(lambda s : TRANSITION_MATRIX.loc[s['land_use'], s['assigned_land_use']], axis=1)
        return reduce(lambda a,b : a*b, [p for p in blocks_gdf['probability']])
    
    def _get_share_least_squares(self, blocks_gdf : gpd.GeoDataFrame, target_lu_shares : dict[LandUse, float]) -> float:
        area = blocks_gdf.area.sum() # TODO overall or only territory ?
        deviations = []
        for lu,target_share in target_lu_shares.items():
            lu_area = blocks_gdf[blocks_gdf['assigned_land_use'] == lu].area.sum()
            actual_share = lu_area / area
            deviation = (actual_share - target_share) ** 2
            deviations.append(deviation)
        return sum(deviations)
    
    def _solution_to_blocks_gdf(self, solution):
        gdf = self.blocks_gdf.copy()
        gdf['assigned_land_use'] = [self.land_use[int(v)] for v in solution]
        return gdf

    def _evaluate(self, x, out, *args, **kwargs):

        F = {
            FitnessType.SHARE_LU : [],
            FitnessType.PROBABILITY : [],
            FitnessType.ADJACENCY : [],
            FitnessType.AREA : [],
            FitnessType.RATIO : [],
        }

        for solution in x:
            blocks_gdf = self._solution_to_blocks_gdf(solution)
            lu_shares = self.lu_shares

            f = {
                FitnessType.SHARE_LU :  self._get_share_least_squares(blocks_gdf, lu_shares),
                FitnessType.PROBABILITY : self._get_transition_probability(blocks_gdf),
                FitnessType.ADJACENCY : self._get_adjacency_penalty(blocks_gdf, self.adjacency_graph),
                FitnessType.AREA : self._get_area_penalty(blocks_gdf),
                FitnessType.RATIO : self._get_ratio_penalty(blocks_gdf),
            }

            for ft, value in f.items():
                F[ft].append(value)

        out["F"] = np.column_stack([v for v in F.values()])

class CustomMutation(Mutation):

    def __init__(self, probability : float =0.1):
        super().__init__()
        self.probability = probability

    def _do(self, problem : LandUseProblem, X, **kwargs):
        # Create random perturbation
        print(X)
        blocks_gdf = problem.blocks_gdf
        def mutate(i,x):
            probabilities = blocks_gdf.iloc[i]['probabilities']
            if random.random()<=self.probability:
                x = random.choices(list(probabilities.keys()), list(probabilities.values()),k=1)[0]
                x = problem.land_use.index(x)
            return x
        
        X = np.array([[mutate(i,x) for i,x in enumerate(solution)] for solution in X])
        return X 

class PymooOptimizer():
    
    def __init__(self, model : Model):
        self.model = model

    @staticmethod
    def get_best_result(results, fitness_type : FitnessType = FitnessType.PROBABILITY):
        X = results.X
        F = results.F
        min_i = min([i for i,_ in enumerate(X)], key = lambda i : F[i][fitness_type.value])
        return X[min_i], F[min_i]

    def run(self, territory_gdf : gpd.GeoDataFrame, lu_shares : dict[LandUse, float], n_eval : int = 1000, opt_alg : OptimizationAlgorithm = OptimizationAlgorithm.BRKGA):
        
        problem = LandUseProblem(self.model, territory_gdf, lu_shares)
        
        if opt_alg == OptimizationAlgorithm.NSGA2:
            algorithm = NSGA2(
                pop_size=10,
                mutation = CustomMutation(probability=0.1)
            )
        elif opt_alg == OptimizationAlgorithm.BRKGA:
            algorithm = BRKGA(
                n_elites = 2,
                n_offsprings = 7,
                n_mutants = 1,
            )
        else:
            raise ValueError('Invalid opt_alg value')
        
        initial_solution = [lu_i for lu_i in problem.blocks_gdf['land_use'].apply(lambda lu : problem.land_use.index(lu))]
        initial_population = Population.new(X=np.tile(initial_solution, (10,1)))
        
        termination = get_termination("n_eval", n_eval)
        results = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            save_history=True,
            verbose=True,
            X=initial_population
        )
        return results, {ft : problem._solution_to_blocks_gdf(self.get_best_result(results, ft)[0]) for ft in list(FitnessType)}
