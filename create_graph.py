import geopandas
import utils
import networkx as nx
import numpy as np
from libpysal import weights
import matplotlib.pyplot as plt

# print(utils.NEVADA_PATH + 'NV_SA_Districts.zip')
nv_sa = geopandas.read_file(utils.NEVADA_PATH + 'NV_SA_Districts.zip')
centroids = np.column_stack((nv_sa.centroid.x, nv_sa.centroid.y))
queen = weights.Queen.from_dataframe(nv_sa)
graph = queen.to_networkx()
positions = dict(zip(graph.nodes, centroids))

# ax = nv_sa.plot(linewidth=1, edgecolor="grey", facecolor="lightblue")
# ax.axis("off")
# nx.draw(graph, positions, ax=ax, node_size=5, node_color="r")
# plt.show()
# nx.from_pandas_edgelist(nv_sa)
# osmnx.utils_graph.graph_from_gdfs