import geopandas
import utils
import networkx as nx
import numpy as np
from libpysal import weights
import matplotlib.pyplot as plt
import json

# NEVADA Graph
nv_sa = geopandas.read_file('./../Nevada/precincts/nv_voting_precincts.zip')
centroids = np.column_stack((nv_sa.centroid.x, nv_sa.centroid.y))
queen = weights.Queen.from_dataframe(nv_sa)
graph = queen.to_networkx()
positions = dict(zip(graph.nodes, centroids))

# GEORGIA Graph
# ga_cd = geopandas.read_file(utils.GEORGIA_PATH + 'precincts/ga-precincts2022-shape.zip')
# centroids = np.column_stack((ga_cd.centroid.x, ga_cd.centroid.y))
# queen = weights.Queen.from_dataframe(ga_cd)
# graph = queen.to_networkx()
# positions = dict(zip(graph.nodes, centroids))

# ga_cd = nx.read_shp(utils.GEORGIA_PATH + 'precincts/ga-precincts2022-shape.zip')

# print(ga_cd)

# MICHIGAN Graph
# mi_cd = geopandas.read_file(utils.MICHIGAN_PATH + 'precincts/mi_2016.geojson')
# mi_cd = mi_cd.to_json()
# mi_cd = json.loads(mi_cd)

# print(mi_cd)

# # centroids = np.column_stack((mi_cd.centroid.x, mi_cd.centroid.y))
# queen = weights.Queen.from_dataframe(mi_cd)
# graph = queen.to_networkx()
# print(nx.is_connected(graph))

# components = list(nx.connected_components(graph))
# lst = [len(c) for c in components]
# biggest_component_size = max(len(c) for c in components)
# problem_components = [c for c in components if len(c) != biggest_component_size]
# problem_nodes = [node for component in problem_components for node in component]
# problem_geoids = [graph.nodes[node]["GEOID10"] for node in problem_nodes]
# is_a_problem = mi_cd["GEOID10"].isin(problem_geoids)

# positions = dict(zip(graph.nodes, centroids))

# ax = mi_cd.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
# ax.axis("off")
# nx.draw(graph, positions, ax=ax, node_size=5, node_color="r")
# plt.show()


# ga_cd = geopandas.read_file(utils.GEORGIA_PATH + 'ga_congressional_districts.zip')
# for col in ga_cd.columns:
#     print(col)
# centroids = np.column_stack((ga_cd.centroid.x, ga_cd.centroid.y))
# queen = weights.Queen.from_dataframe(ga_cd)
# graph = queen.to_networkx()
# positions = dict(zip(graph.nodes, centroids))

ax = nv_sa.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
ax.axis("off")
nx.draw(graph, positions, ax=ax, node_size=5, node_color="r")
plt.show()