import geopandas as gpd
import pandas as pd
import shapely
from loguru import logger
from pandera.typing import Series
from .land_use import LandUse
from ..models.schema import BaseSchema

BLOCK_ID_COLUMN = 'block_id'
LAND_USE_COLUMN = 'land_use'
SHARES_COLUMN = 'shares'

class BlocksSchema(BaseSchema):
    _geom_types = [shapely.Polygon]


class ZonesSchema(BaseSchema):
    _geom_types = [shapely.Polygon, shapely.MultiPolygon]
    zone : Series[str]

def _validate_input(blocks : gpd.GeoDataFrame, zones : gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    logger.info('Validating input')
    blocks = BlocksSchema(blocks)
    zones = ZonesSchema(zones)
    assert blocks.crs == zones.crs, "Blocks CRS must match functional zones CRS"
    return blocks, zones

def _process_zones(zones : gpd.GeoDataFrame, land_use_mapping : dict[str, LandUse]):
    logger.info('Processing functional zones')
    zones = zones.copy()
    zones[LAND_USE_COLUMN] = zones['zone'].apply(land_use_mapping.get) 
    return zones[~zones[LAND_USE_COLUMN].isna()]

def process_land_use(blocks : gpd.GeoDataFrame, zones : gpd.GeoDataFrame, land_use_mapping : dict[str, LandUse], min_intersection_share : float = 0.5):
    blocks, zones = _validate_input(blocks, zones)
    zones = _process_zones(zones, land_use_mapping)
    
    logger.info('Intersecting geometries')

    sjoin_gdf = blocks.sjoin(zones, predicate='intersects')

    def _get_shares(series):
        block_i = series.name
        block_geometry = series.geometry
        block_area = block_geometry.area
        gdf = sjoin_gdf[sjoin_gdf.index == block_i]
        shares = {}
        for zone_i in gdf['index_right']:
            land_use = zones.loc[zone_i, LAND_USE_COLUMN]
            zone_geometry = zones.loc[zone_i, 'geometry']
            intersection_geometry = shapely.intersection(zone_geometry, block_geometry)
            intersection_area = intersection_geometry.area
            intersection_share = intersection_area/block_area
            if intersection_share >= min_intersection_share:
                shares[land_use] = intersection_area/block_area
        return shares
    
    logger.info('Calculating shares')
    
    blocks[SHARES_COLUMN] = blocks.apply(_get_shares, axis=1)
    blocks[LAND_USE_COLUMN] = blocks[SHARES_COLUMN].apply(lambda shares : max(shares, key=shares.get) if len(shares)>0 else None)
    logger.success('Shares calculated')
    return blocks