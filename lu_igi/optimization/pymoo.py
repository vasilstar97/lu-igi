import random
import networkx as nx
import numpy as np
from lu_igi.models.land_use import LandUse
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from enum import IntEnum

LAND_USE = list(LandUse)

class FitnessType(IntEnum):
    SHARE_LU = 0
    PROBABILITY = 1

class LandUseProblem(Problem):

    def __init__(self, graph : nx.DiGraph, blocks_ids : list[int], target_shares : dict[LandUse, float]):     
        self.graph = graph
        self.blocks_ids = blocks_ids
        self.target_shares = target_shares
        super().__init__(
            n_var=len(self.blocks_ids),
            n_obj=1,
            n_constr=0,
            xl=0,
            xu=len(LAND_USE)-1,
            type_var=int # Дискретные переменные (метки)
        )

    def get_block_id(self, i):
        return problem.blocks_ids[i]
    
    def _get_share_mse(self, solution):
        blocks_area = sum(self.graph.nodes[block_id] for block_id in self.blocks_ids)

        deviations = []
        for lu,target_share in self.target_shares.items():
            lu_area = blocks_gdf[blocks_gdf['assigned_land_use'] == lu].area.sum()
            actual_share = lu_area / area
            deviation = (actual_share - target_share) ** 2
            deviations.append(deviation)
        return sum(deviations)

        for i,v in solution:
            block_id = self.get_block_id(i)


    def _get_transition_probability(self, i, v):
        block_id = self.get_block_id(i)
        land_use = LAND_USE[round(v)]
        return self.graph.nodes[block_id]['transition_probabilities'][land_use]

    def _transition_probability_fitness(self, solution) -> float:
        probabilities = [self._get_transition_probability(i,v) for i,v in enumerate(solution)]
        return np.prod(probabilities)
    
    def _evaluate(self, x, out, *args, **kwargs):

        F = {
            # FitnessType.SHARE_LU : [],
            FitnessType.PROBABILITY : [],
        }

        for solution in x:

            f = {
                # FitnessType.SHARE_LU :  self._get_share_least_squares(blocks_gdf, lu_shares),
                FitnessType.PROBABILITY : 1 / self._transition_probability_fitness(solution),
            }

            for ft, value in f.items():
                F[ft].append(value)

        out["F"] = np.column_stack([v for v in F.values()])