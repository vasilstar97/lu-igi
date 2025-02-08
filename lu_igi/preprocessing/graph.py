import shapely
import pandas as pd
import pandera as pa
import geopandas as gpd
import networkx as nx
import pickle
from tqdm import tqdm
from loguru import logger
from pandera.typing import Series
from ..models.land_use import LandUse
from ..models.schema import BaseSchema

SOURCE_COLUMN = 'source'
TARGET_COLUMN = 'target'

tqdm.pandas()

class BlocksSchema(BaseSchema):
    _geom_types = [shapely.Polygon]
    land_use : Series = pa.Field(isin=LandUse, nullable=True)
            
    @pa.parser('land_use')
    @classmethod
    def parse_land_use(cls, series : pd.Series):

        def parse(lu):
            if isinstance(lu, str):
                return LandUse[lu.lower()]
            return lu

        return series.apply(parse)

    # @pa.check('land_use')
    # @classmethod
    # def check_land_use(cls, series : pd.Series):

    #     def check(lu):
    #         if isinstance(lu, LandUse):
    #             return True
    #         if lu is None:
    #             return True
    #         return False

    #     return series.apply(check)

def _generate_adjacency_edges(blocks_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info('Generating edges')
    edges_gdf = blocks_gdf.sjoin(blocks_gdf, predicate='intersects')[['geometry', 'index_right']]
    edges_gdf = edges_gdf[edges_gdf.index != edges_gdf['index_right']]
    edges_gdf = edges_gdf.reset_index(drop=False).rename(columns={
        'index': SOURCE_COLUMN, 
        'index_right': TARGET_COLUMN, 
    })

    def get_intersection_geometry(series : gpd.GeoSeries):
        source_geom = blocks_gdf.loc[series[SOURCE_COLUMN], 'geometry']
        target_geom = blocks_gdf.loc[series[TARGET_COLUMN], 'geometry']
        return source_geom.intersection(target_geom)

    edges_gdf['geometry'] = edges_gdf.progress_apply(get_intersection_geometry, axis=1)
    return edges_gdf.set_index([SOURCE_COLUMN, TARGET_COLUMN])

def generate_adjacency_graph(blocks_gdf : gpd.GeoDataFrame) -> nx.DiGraph:
    
    logger.info('Validating input')
    blocks_gdf = BlocksSchema(blocks_gdf)

    # blocks_df = generate_node_features(blocks_gdf)
    # blocks_gdf = pd.concat([blocks_gdf, blocks_df], axis=1)

    adj_graph = nx.DiGraph(None, crs=blocks_gdf.crs) # 
    adj_graph.add_nodes_from([(i, row.to_dict()) for i,row in blocks_gdf.iterrows()])

    edges_gdf = _generate_adjacency_edges(blocks_gdf)
    # edges_gdf = generate_edges_features(edges_gdf, blocks_gdf)
    
    adj_graph.add_edges_from([(i[0], i[1], row.to_dict()) for i,row in edges_gdf.iterrows()])

    logger.success('Graph successfully generated')
    return adj_graph

def save_graph(graph : nx.DiGraph, path : str):
    with open(path, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(path : str) -> nx.DiGraph:
    with open(path, 'rb') as f:
        return pickle.load(f)