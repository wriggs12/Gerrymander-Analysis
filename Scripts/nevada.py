import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
import pandas as pd
import preprocess as data
import json
from networkx.readwrite import json_graph


nv = data.get_nevada_data()
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

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.08)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=1000
)

generated_partitions = []
for partition in chain.with_progress_bar():
    generated_partitions.append(partition)

generated_partitions[-1].plot()
plt.show()

# fig, ax = plt.subplots(figsize=(8, 6))

# ax.axhline(0.5, color="#cccccc")

# calc_data.boxplot(ax=ax, positions=range(len(calc_data.columns)))
# plt.plot(calc_data.iloc[0], "ro")

# ax.set_title("Comparing the 2020 plan to an ensemble")
# ax.set_ylabel("Democratic vote % (State Assembly 2020)")
# ax.set_xlabel("Sorted districts")
# ax.set_ylim(0, 1)
# ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

# plt.show()