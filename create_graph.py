import geopandas
import utils
import networkx as nx
import numpy as np
from libpysal import weights
import matplotlib.pyplot as plt

# NEVADA Graph
# print(utils.NEVADA_PATH + 'NV_SA_Districts.zip')
# nv_sa = geopandas.read_file(utils.NEVADA_PATH + 'NV_SA_Districts.zip')
# centroids = np.column_stack((nv_sa.centroid.x, nv_sa.centroid.y))
# queen = weights.Queen.from_dataframe(nv_sa)
# graph = queen.to_networkx()
# positions = dict(zip(graph.nodes, centroids))

#GEORGIA Graph
# ga_cd = geopandas.read_file(utils.GEORGIA_PATH + 'ga_voting_precincts.zip')
# centroids = np.column_stack((ga_cd.centroid.x, ga_cd.centroid.y))
# queen = weights.Queen.from_dataframe(ga_cd)
# graph = queen.to_networkx()
# positions = dict(zip(graph.nodes, centroids))

#MICHIGAN Graph
mi_cd = geopandas.read_file(utils.MICHIGAN_PATH + 'mi_voting_precincts.zip')
centroids = np.column_stack((mi_cd.centroid.x, mi_cd.centroid.y))
queen = weights.Queen.from_dataframe(mi_cd)
graph = queen.to_networkx()
positions = dict(zip(graph.nodes, centroids))


ax = mi_cd.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
ax.axis("off")
nx.draw(graph, positions, ax=ax, node_size=5, node_color="r")
plt.show()