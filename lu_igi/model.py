import shapely
import geopandas as gpd
import networkx as nx

BORDER_LENGTH_NAME = 'border_length'
ASPECT_RATIO_NAME = 'aspect_ratio'
LAND_USE_NAME = 'land_use'
AREA_NAME = 'area'

class Model():

    def __init__(self, blocks_gdf):
        self.blocks_gdf = self._preprocess_blocks(blocks_gdf)
        self.adjacency_graph = self._get_adjacency_graph(blocks_gdf)

    @property
    def crs(self):
        return self.blocks_gdf.crs

    def plot(self):
        edges_gdf = self._get_edges_gdf()
        ax = self.blocks_gdf.plot(figsize=(20,20), edgecolor='#fff', column=LAND_USE_NAME, legend=True) #color='#888'
        edges_gdf[edges_gdf[BORDER_LENGTH_NAME]>0].plot(ax=ax, color='#000')
        ax.set_axis_off()
        # pos = {n:(d['x'], d['y']) for n,d in self.adjacency_graph.nodes(data=True)}
        # nx.draw_networkx(self.adjacency_graph, pos=pos, with_labels=False, node_size=4, edge_color='#888', node_color='#000')

    def _get_edges_gdf(self):
        def _edge_to_linestring(edge):
            geom_a = self.blocks_gdf.loc[edge[0],'geometry']
            geom_b = self.blocks_gdf.loc[edge[1],'geometry']
            return shapely.LineString([geom.representative_point() for geom in [geom_a, geom_b]])

        edges_gdf = gpd.GeoDataFrame([{'from': edge[0], 'to': edge[1], **edge[2], 'geometry': _edge_to_linestring(edge)}for edge in nx.to_edgelist(self.adjacency_graph)])
        return edges_gdf.set_crs(self.blocks_gdf.crs)

    @staticmethod
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

    @classmethod
    def _preprocess_blocks(cls, blocks_gdf):
        blocks_gdf = blocks_gdf[['geometry', LAND_USE_NAME]].copy()
        blocks_gdf[AREA_NAME] = blocks_gdf.area
        blocks_gdf[ASPECT_RATIO_NAME] = blocks_gdf.geometry.apply(cls._get_aspect_ratio)
        return blocks_gdf
        
    @staticmethod
    def _get_adjacency_graph(blocks_gdf):
        # sjoin blocks with blocks
        sjoin = blocks_gdf.sjoin(blocks_gdf, predicate='intersects')
        sjoin = sjoin[sjoin.index != sjoin['index_right']]
        sjoin[BORDER_LENGTH_NAME] = sjoin.apply(lambda s : s.geometry.intersection(blocks_gdf.loc[s.index_right].geometry).length, axis=1)
        # create graph
        adj_graph = nx.Graph()
        adj_graph.add_nodes_from([(i, {'x':row['geometry'].representative_point().x, 'y':row['geometry'].representative_point().y}) for i,row in blocks_gdf.iterrows()])
        adj_graph.add_edges_from([(i, row['index_right'], {BORDER_LENGTH_NAME: row[BORDER_LENGTH_NAME]}) for i,row in sjoin.iterrows()])
        return adj_graph