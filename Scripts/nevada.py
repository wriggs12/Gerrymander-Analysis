from gerrychain import Graph, Partition, Election
from gerrychain.updaters import Tally, cut_edges
import geopandas
import numpy as np
from libpysal import weights

nv_sa = geopandas.read_file('./../Nevada/precincts/nv_shapefile.zip')
nv_sa = nv_sa.set_crs('epsg:3857')
# print(nv_sa.crs)

graph = Graph.from_geodataframe(nv_sa)
election = Election("SEN12", {"Dem": "NDV", "Rep": "NRV"})
print(graph)
print(election)

initial_partition = Partition(
    graph,
    assignment="GEOID10",
    updaters={
        "cut_edges": cut_edges,
        "population": Tally("VAP", alias="population"),
        "SEN12": election
    }
)

for district, pop in initial_partition["population"].items():
    print("District {}: {}".format(district, pop))