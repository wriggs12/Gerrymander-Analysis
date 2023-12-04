from gerrychain import (GeographicPartition, Graph, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from gerrychain.random import random
from gerrychain.constraints import no_vanishing_districts
import multiprocessing as mp

ENSEMBLE_SIZE=1

def generate_plan(chain, seed):
    random.seed(seed)
    for partition in chain.with_progress_bar():
        pass
    
    generated_plan = partition
    calc_party_split(generated_plan)
    calc_race_demographics(generated_plan)

    return partition

def init_chain(graph):
    elections = [
        Election("PRES", {"Dem": "G20PREDBID", "Rep": "G20PRERTRU"})
    ]

    my_updaters = {"population": updaters.Tally("ADJPOP", alias="population")}
    election_updaters = {election.name: election for election in elections}
    my_updaters.update(election_updaters)

    initial_partition = GeographicPartition(graph, assignment='DISTRICT', updaters=my_updaters)
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.10)

    proposal = partial(recom,
                    pop_col="ADJPOP",
                    pop_target=ideal_population,
                    epsilon=0.10,
                    node_repeats=1)

    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]),
        2*len(initial_partition["cut_edges"])
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            no_vanishing_districts,
            pop_constraint,
            compactness_bound
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=100
    )

    return chain

def run(data):
    graph = Graph.from_geodataframe(data)
    chain = init_chain(graph)

    ensemble = [generate_plan(chain, seed) for seed in range(ENSEMBLE_SIZE)]
    for plan in ensemble:
        print(plan.graph.nodes[0])
    

def calc_party_split(plan):
    rep = 0
    dem = 0
    for node in plan.graph:
        dem = dem + plan.graph.nodes[node]['G20PREDBID']
        rep = rep + plan.graph.nodes[node]['G20PRERTRU']

    if (rep > dem):
        return 'R'
    else:
        return 'D'
    

def calc_race_demographics(plan):
    pass
