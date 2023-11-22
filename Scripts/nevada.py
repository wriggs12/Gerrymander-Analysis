import matplotlib.pyplot as plt
import networkx as nx
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
import geopandas
from gerrychain.random import random

print('Reading Data...')
nv = geopandas.read_file('nevada_data_processed.zip')
print('Done')
graph = Graph.from_geodataframe(nv)

elections = [
    Election("PRES", {"Dem": "G20PREDBID", "Rep": "G20PRERTRU"})
]

my_updaters = {"population": updaters.Tally("ADJPOP", alias="population")}
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = GeographicPartition(graph, assignment='DISTRICT', updaters=my_updaters)

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

proposal = partial(recom,
                   pop_col="ADJPOP",
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2)

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.12)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=50
)

ensemble = []
for itr in range(10):
    seed = itr
    random.seed(seed)
    generated_partition = initial_partition
    for partition in chain.with_progress_bar():
        generated_partition = partition
    ensemble.append(generated_partition)


print(ensemble[0].graph.nodes[0])
# plt.show()
# for node in ensemble[0].graph.nodes:
#     print(ensemble[0].assignment[node])
