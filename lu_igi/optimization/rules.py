import networkx as nx
from ..models.land_use import LandUse

AREA_RANGES = {
    LandUse.RESIDENTIAL: (2_000, 100_000),
    LandUse.BUSINESS: (50_000, 150_000),
    LandUse.RECREATION: (10_000, 1_000_000),
    LandUse.SPECIAL: (50_000, 500_000),
    LandUse.INDUSTRIAL: (10_000, 800_000),
    LandUse.AGRICULTURE: (300_000, 1_000_000),
    LandUse.TRANSPORT: (50_000, 500_000),
}

ASPECT_RATIO_RANGES = {
    LandUse.RESIDENTIAL: (1, 3),
    LandUse.BUSINESS: (1, 4),
    LandUse.RECREATION: (1, 7),
    LandUse.SPECIAL: (1, 6),
    LandUse.INDUSTRIAL: (1, 5),
    LandUse.AGRICULTURE: (1, 4),
    LandUse.TRANSPORT: (1, 7),
}

ADJACENCY_RULES_LIST = [
    # self adjacency
    (LandUse.RESIDENTIAL, LandUse.RESIDENTIAL),
    (LandUse.BUSINESS, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.RECREATION),
    (LandUse.INDUSTRIAL, LandUse.INDUSTRIAL),
    (LandUse.TRANSPORT, LandUse.TRANSPORT),
    (LandUse.SPECIAL, LandUse.SPECIAL),
    (LandUse.AGRICULTURE, LandUse.AGRICULTURE),
    # recreation can be adjacent to anything
    (LandUse.RECREATION, LandUse.SPECIAL),
    (LandUse.RECREATION, LandUse.INDUSTRIAL),
    (LandUse.RECREATION, LandUse.BUSINESS),
    (LandUse.RECREATION, LandUse.AGRICULTURE),
    (LandUse.RECREATION, LandUse.TRANSPORT),
    (LandUse.RECREATION, LandUse.RESIDENTIAL),
    # residential
    (LandUse.RESIDENTIAL, LandUse.BUSINESS),
    # business
    (LandUse.BUSINESS, LandUse.INDUSTRIAL),
    (LandUse.BUSINESS, LandUse.TRANSPORT),
    # industrial
    (LandUse.INDUSTRIAL, LandUse.SPECIAL),
    (LandUse.INDUSTRIAL, LandUse.AGRICULTURE),
    (LandUse.INDUSTRIAL, LandUse.TRANSPORT),
    # transport
    (LandUse.TRANSPORT, LandUse.SPECIAL),
    (LandUse.TRANSPORT, LandUse.AGRICULTURE),
    # special
    (LandUse.SPECIAL, LandUse.AGRICULTURE),
]

ADJACENCY_RULES_GRAPH = nx.from_edgelist(ADJACENCY_RULES_LIST)