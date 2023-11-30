from gerrychain import (GeographicPartition, Graph, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from gerrychain.random import random
import matplotlib.pyplot as plt

def generate_ensemble(size, chain):
    ensemble = []

    for i in range(size):
        random.seed(i)
    
        for partition in chain.with_progress_bar():
            pass

        ensemble.append(partition)
    
    return ensemble

def run(data):
    graph = Graph.from_geodataframe(data)

    elections = [
        Election("PRES", {"Dem": "G20PREDBID", "Rep": "G20PRERTRU"})
    ]

    my_updaters = {"population": updaters.Tally("ADJPOP", alias="population")}
    election_updaters = {election.name: election for election in elections}
    my_updaters.update(election_updaters)

    initial_partition = GeographicPartition(graph, assignment='DISTRICT', updaters=my_updaters)
    initial_partition.plot()
    plt.show()

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

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.15)

    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            # pop_constraint,
            compactness_bound
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=10000
    )

    return generate_ensemble(1000, chain)

def district_plan(partition):
    pass

def calc_stats(ensemble):
    pass

def cluster_analysis(ensemble):
    pass

def optimal_transport():
    pass
