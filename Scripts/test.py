import geopandas
import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import json
from gerrychain import (GeographicPartition)

data = json.load(open('output.json'))
geopandas.read_file('output.json')
# test = nx.adjacency_graph(data)
# initial_partition = GeographicPartition(test)

# initial_partition.plot()
# plt.show()