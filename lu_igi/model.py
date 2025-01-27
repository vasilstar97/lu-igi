import shapely
import geopandas as gpd
import pandas as pd
import networkx as nx
from loguru import logger
from tqdm import tqdm
from .tolerance_matrix import TOLERANCE_MATRIX
from .land_use import LandUse

BORDER_LENGTH_NAME = 'border_length'
ASPECT_RATIO_NAME = 'aspect_ratio'
LAND_USE_NAME = 'land_use'
AREA_NAME = 'area'

def _get_aspect_ratio(block_geometry):
    rectangle = block_geometry.minimum_rotated_rectangle
    rectangle_coords = list(rectangle.exterior.coords)
    side_lengths = [
        ((rectangle_coords[i][0] - rectangle_coords[i - 1][0]) ** 2 + (rectangle_coords[i][1] - rectangle_coords[i - 1][1]) ** 2) ** 0.5
        for i in range(1, 5)
    ]
    length_1, length_2 = side_lengths[0], side_lengths[1]
    aspect_ratio = max(length_1, length_2) / min(length_1, length_2)
    return aspect_ratio

class Model():

    def __init__(self, blocks_gdf):
        logger.info('Preprocessing blocks')
        self.blocks_gdf = self._preprocess_blocks(blocks_gdf)
        logger.info('Building adjacency graph')
        self.adjacency_graph = self._get_adjacency_graph(blocks_gdf)
        logger.success('Initialization finished')

    @property
    def crs(self):
        return self.blocks_gdf.crs

    # def plot(self):
    #     edges_gdf = self._get_edges_gdf()
    #     ax = self.blocks_gdf.plot(figsize=(20,20), color='#ddd')
    #     ax = self.blocks_gdf.plot(ax=ax, edgecolor='#fff', column=LAND_USE_NAME, legend=True) #color='#888'
    #     edges_gdf[edges_gdf[BORDER_LENGTH_NAME]>0].plot(ax=ax, color='#000')
    #     ax.set_axis_off()

    # def _get_edges_gdf(self):
    #     def _edge_to_linestring(edge):
    #         geom_a = self.blocks_gdf.loc[edge[0],'geometry']
    #         geom_b = self.blocks_gdf.loc[edge[1],'geometry']
    #         return shapely.LineString([geom.representative_point() for geom in [geom_a, geom_b]])

    #     edges_gdf = gpd.GeoDataFrame([{'from': edge[0], 'to': edge[1], **edge[2], 'geometry': _edge_to_linestring(edge)}for edge in nx.to_edgelist(self.adjacency_graph)])
    #     return edges_gdf.set_crs(self.blocks_gdf.crs)

    @classmethod
    def _preprocess_blocks(cls, blocks_gdf):
        blocks_gdf = blocks_gdf[['geometry', LAND_USE_NAME]].copy()
        blocks_gdf[AREA_NAME] = blocks_gdf.area
        blocks_gdf[ASPECT_RATIO_NAME] = blocks_gdf.geometry.apply(_get_aspect_ratio)
        return blocks_gdf
        
    @staticmethod
    def _get_adjacency_graph(blocks_gdf):
        # sjoin blocks with blocks
        sjoin = blocks_gdf.sjoin(blocks_gdf, predicate='intersects')
        sjoin = sjoin[sjoin.index != sjoin['index_right']]
        sjoin[BORDER_LENGTH_NAME] = sjoin.apply(lambda s : s.geometry.intersection(blocks_gdf.loc[s.index_right].geometry).length, axis=1)
        # create graph
        adj_graph = nx.DiGraph()

        adj_graph.add_nodes_from(blocks_gdf.index)

        for i,row in tqdm(sjoin.iterrows()):
            data = {
                BORDER_LENGTH_NAME: row[BORDER_LENGTH_NAME]/blocks_gdf.loc[i].geometry.length
            }
            adj_graph.add_edge(i, row['index_right'], **data)

        return adj_graph