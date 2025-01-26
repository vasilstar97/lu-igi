import random
from functools import reduce
import numpy as np
import pygad as pg
import geopandas as gpd
import pandas as pd
import networkx as nx
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
from .model import Model
from .land_use import LandUse
from .rules import ADJACENCY_RULES_GRAPH, AREA_RANGES, ASPECT_RATIO_RANGES
from .transition_matrix import TRANSITION_MATRIX

NUM_GENERATIONS = 1000
SOL_PER_POP = 100
MUTATION_PROBABILITY = 0.1
TITLES = ['LU shares (least squares)', 'Probability', 'Adjacency penalty', 'Area penalty', 'Ratio penalty']

class PygadOptimizer():

    def __init__(self, model : Model):        
        self.model = model

    @staticmethod
    def plot_fitness(ga_instance : pg.GA, titles : list[str] = TITLES):

        _, axs = plt.subplots(len(titles), figsize=(15,len(titles)*2))  # 2 rows, 1 column
        solutions = ga_instance.solutions_fitness

        for k,title in enumerate(titles):
            data = [solution[k] for solution in solutions]
            data_df = pd.DataFrame(data)
            data_df.apply(lambda s : 1/s if k != 1 else s).plot(ax=axs[k], title=title, legend=False)

        plt.tight_layout()  # Adjust layout
        plt.show()

    def _get_blocks_subgdf(self, territory_gdf : gpd.GeoDataFrame):
        if not territory_gdf.crs == self.model.crs:
            logger.warning('Territory must have same CRS as model blocks. Assigning')
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
        # return blocks_gdf['probability'].sum()
        return np.prod(blocks_gdf['probability'])
        # return reduce(lambda a,b : a*b, [p*10_000 for p in blocks_gdf['probability']])
        # return np.sum([np.log(p) for p in blocks_gdf['probability']])
    
    def _get_share_least_squares(self, blocks_gdf : gpd.GeoDataFrame, target_lu_shares : dict[LandUse, float]) -> float:
        area = blocks_gdf.area.sum() # TODO overall or only territory ?
        deviations = []
        for lu,target_share in target_lu_shares.items():
            lu_area = blocks_gdf[blocks_gdf['assigned_land_use'] == lu].area.sum()
            actual_share = lu_area / area
            deviation = (actual_share - target_share) ** 2
            deviations.append(deviation)
        return sum(deviations)

    def run(self, territory_gdf : gpd.GeoDataFrame, lu_shares : dict[LandUse, float], saturation=10, num_generations = NUM_GENERATIONS, sol_per_pop = SOL_PER_POP, mutation_probability = MUTATION_PROBABILITY):
        blocks_gdf = self._get_blocks_subgdf(territory_gdf)
        adjacency_graph = self._get_adjacency_subgraph(blocks_gdf)
        land_use = list(LandUse)

        def solution_to_blocks_gdf(solution):
            gdf = blocks_gdf.copy()
            gdf['assigned_land_use'] = [land_use[int(v)] for v in solution]
            return gdf

        def fitness_func(ga_instance, solution, solution_idx): # -> max
            gdf = solution_to_blocks_gdf(solution)
            
            share_ls = self._get_share_least_squares(gdf, lu_shares)
            probability = self._get_transition_probability(gdf)
            adjacency_penalty = self._get_adjacency_penalty(gdf, adjacency_graph)
            area_penalty = self._get_area_penalty(gdf)
            ratio_penalty = self._get_ratio_penalty(gdf)

            return 1/share_ls, probability, 1/adjacency_penalty, 1/area_penalty, 1/ratio_penalty

        gene_space = [i for i,_ in enumerate(land_use)]

        def mutation(offspring, ga_instance):

            for j in range(offspring.shape[1]):  # Перебираем потомков
                probabilities = blocks_gdf.iloc[j]['probabilities']
                for i in range(offspring.shape[0]):
                    if random.random()<=mutation_probability:
                        lu = random.choices(list(probabilities.keys()), list(probabilities.values()),k=1)[0]
                        offspring[i,j] = land_use.index(lu)
        
            return offspring
        
        def generate_solution():

            solution = []

            for i in blocks_gdf.index:
                probabilities = blocks_gdf.loc[i]['probabilities']
                lu = random.choices(list(probabilities.keys()), list(probabilities.values()),k=1)[0]
                solution.append(land_use.index(lu))
            
            return solution

        initial_population = [generate_solution() for _ in range(0, sol_per_pop)]

        with tqdm(total=num_generations) as pbar:
            ga_instance = pg.GA(
                num_generations = num_generations,
                sol_per_pop = sol_per_pop,
                num_parents_mating = sol_per_pop//2,
                gene_space = gene_space,
                num_genes = len(blocks_gdf),
                fitness_func = fitness_func,
                on_generation = lambda _: pbar.update(1),
                stop_criteria = f'saturate_{saturation}',
                save_solutions=True,
                mutation_type=mutation,
                initial_population=initial_population
            )
            ga_instance.run()

        best_solution, _, _ = ga_instance.best_solution()
        
        return solution_to_blocks_gdf(best_solution), ga_instance

