import shapely
import pygeoops
import pandas as pd
import pandera as pa
import geopandas as gpd
import networkx as nx
import featuretools as ft
from tqdm import tqdm
from loguru import logger
from pandera.typing import Series
from ..land_use import LandUse
from ..models.schema import BaseSchema
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data.data import Data
from sklearn.model_selection import train_test_split
from .gcn import CLASSES


tqdm.pandas()

BLOCK_AREA_COLUMN = 'block_area'
BLOCK_LENGTH_COLUMN = 'block_length'
BLOCK_CORNERS_COUNT_COLUMN = 'block_corners_count'
OUTER_RADIUS_COLUMN = 'outer_radius'
INNER_RADIUS_COLUMN = 'inner_radius'
CENTERLINE_LENGTH_COLUMN = 'centerline_length'
ASPECT_RATIO_COLUMN = 'aspect_ratio'
LAND_USE_COLUMN = 'land_use'

SOURCE_COLUMN = 'source'
TARGET_COLUMN = 'target'
BORDER_LENGTH_COLUMN = 'border_length'

class BlocksSchema(BaseSchema):
    _geom_types = [shapely.Polygon]
    land_use : Series[str] = pa.Field(nullable=True)

    @pa.parser('land_use')
    @classmethod
    def parse_land_use(cls, series : pd.Series):
        return series.apply(lambda lu : lu.value if isinstance(lu, LandUse) else lu)

    # @pa.check('land_use')
    # @classmethod
    # def check_land_use(cls, series : pd.Series):
    #     return series.apply(lambda lu : True if lu is None else isinstance(lu, LandUse))


def _calculate_outer_radius(polygon : shapely.Polygon):
    center = polygon.representative_point()
    corners = [shapely.Point(coord) for coord in polygon.exterior.coords]
    return max(center.distance(corner) for corner in corners)

def _calculate_inner_radius(polygon : shapely.Polygon):
    center = polygon.representative_point()
    corners = [shapely.Point(coord) for coord in polygon.exterior.coords]
    side_centers = [shapely.MultiPoint([corners[i], corners[i+1]]).centroid for i in range(len(corners)-1)]
    return min(center.distance(point) for point in side_centers)

def _calculate_aspect_ratio(polygon : shapely.Polygon):
    rectangle = polygon.minimum_rotated_rectangle
    rectangle_coords = list(rectangle.exterior.coords)
    side_lengths = [
        ((rectangle_coords[i][0] - rectangle_coords[i - 1][0]) ** 2 + (rectangle_coords[i][1] - rectangle_coords[i - 1][1]) ** 2) ** 0.5
        for i in range(1, 5)
    ]
    length_1, length_2 = side_lengths[0], side_lengths[1]
    aspect_ratio = max(length_1, length_2) / min(length_1, length_2)
    return aspect_ratio

def generate_node_features(blocks_gdf : gpd.GeoDataFrame) -> pd.DataFrame:
    logger.info('Generating nodes features')
    blocks_gdf = blocks_gdf.copy()

    logger.info('Calculating usual features')
    blocks_gdf[BLOCK_AREA_COLUMN] = blocks_gdf.area
    blocks_gdf[BLOCK_LENGTH_COLUMN] = blocks_gdf.length
    blocks_gdf[BLOCK_CORNERS_COUNT_COLUMN] = blocks_gdf.geometry.apply(lambda g : len(g.exterior.coords))

    logger.info('Calculating radiuses')
    blocks_gdf[OUTER_RADIUS_COLUMN] = blocks_gdf.geometry.progress_apply(_calculate_outer_radius)
    blocks_gdf[INNER_RADIUS_COLUMN] = blocks_gdf.geometry.progress_apply(_calculate_inner_radius)

    logger.info('Calculating centerlines')
    blocks_gdf[CENTERLINE_LENGTH_COLUMN] = blocks_gdf.geometry.progress_apply(pygeoops.centerline).length
    
    logger.info('Calculating aspect ratios')
    blocks_gdf[ASPECT_RATIO_COLUMN] = blocks_gdf.geometry.progress_apply(_calculate_aspect_ratio)

    logger.success('Features successfully calculated')
    return blocks_gdf[[
        BLOCK_AREA_COLUMN,
        BLOCK_LENGTH_COLUMN,
        BLOCK_CORNERS_COUNT_COLUMN,
        OUTER_RADIUS_COLUMN,
        INNER_RADIUS_COLUMN,
        CENTERLINE_LENGTH_COLUMN,
        ASPECT_RATIO_COLUMN
    ]]

def generate_combinations(df : pd.DataFrame):
    logger.info('Generating features combinations')
    df = df.copy()
    df['id'] = df.index
    es = ft.EntitySet(id="")
    es = es.add_dataframe(dataframe_name="", dataframe=df, index='id')
    df, _ = ft.dfs(
        entityset=es,
        target_dataframe_name="",
        max_depth=1,
        trans_primitives=["multiply_numeric", "divide_numeric"]
    )
    return df

def generate_adjacency_edges(blocks_gdf : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger.info('Generating edges')
    edges_gdf = blocks_gdf.sjoin(blocks_gdf, predicate='intersects')[['geometry', 'index_right']]
    edges_gdf = edges_gdf[edges_gdf.index != edges_gdf['index_right']]
    edges_gdf = edges_gdf.reset_index(drop=False).rename(columns={
        'index': SOURCE_COLUMN, 
        'index_right': TARGET_COLUMN, 
    })

    def get_intersection_geometry(series):
        source_geom = blocks_gdf.loc[series[SOURCE_COLUMN], 'geometry']
        target_geom = blocks_gdf.loc[series[TARGET_COLUMN], 'geometry']
        return source_geom.intersection(target_geom)

    edges_gdf['geometry'] = edges_gdf.progress_apply(get_intersection_geometry, axis=1)
    return edges_gdf.set_index([SOURCE_COLUMN, TARGET_COLUMN])

def generate_edges_features(edges_gdf : gpd.GeoDataFrame, blocks_gdf : gpd.GeoDataFrame) -> pd.DataFrame:
    logger.info('Generating edges features')
    edges_gdf = edges_gdf.copy()
    edges_gdf[BORDER_LENGTH_COLUMN] = edges_gdf.geometry.progress_apply(lambda g : g.length)
    return edges_gdf[[BORDER_LENGTH_COLUMN]]

def create_adjacency_graph(blocks_gdf : gpd.GeoDataFrame, combinations : bool = True) -> nx.DiGraph:
    
    logger.info('Validating input')
    blocks_gdf = BlocksSchema(blocks_gdf)

    blocks_df = generate_node_features(blocks_gdf)
    if combinations:
        blocks_df = generate_combinations(blocks_df)
    node_features = list(blocks_df.columns)
    blocks_df['land_use'] = blocks_gdf.land_use

    adj_graph = nx.DiGraph()
    adj_graph.add_nodes_from([(i, row.to_dict()) for i,row in blocks_df.iterrows()])

    edges_gdf = generate_adjacency_edges(blocks_gdf)
    edges_gdf = generate_edges_features(edges_gdf, blocks_gdf)
    edge_features = list(edges_gdf.columns)
    
    adj_graph.add_edges_from([(i[0], i[1], row.to_dict()) for i,row in edges_gdf.iterrows()])

    logger.success('Graph successfully generated')
    return adj_graph, node_features, edge_features

def graph_to_data(graph : nx.DiGraph, node_features : list[str], edge_features : list[str]) -> Data:
    data = from_networkx(graph, group_node_attrs=filter(lambda f : f != LAND_USE_COLUMN, node_features), group_edge_attrs=edge_features)
    data.y = torch.tensor([-1 if lu is None else CLASSES.index(lu) for lu in data[LAND_USE_COLUMN]]).long()
    data.remove_tensor(LAND_USE_COLUMN)
    return data