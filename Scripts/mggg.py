from gerrychain import (GeographicPartition, Graph, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from gerrychain.random import random
from gerrychain.constraints import no_vanishing_districts
import json
import os

ENSEMBLE_SIZE=1000
stats = {}

def run(data, dist_measure, ensemble_number, size=1000):
    ENSEMBLE_SIZE = size

    graph = Graph.from_geodataframe(data)
    chain = init_chain(graph)

    ensemble = [generate_plan(chain, seed) for seed in range(ENSEMBLE_SIZE)]
    cluster_analysis(ensemble, dist_measure)

    os.mkdir('./ensemble_' + str(ensemble_number))
    for index, plan in enumerate(ensemble):
        plan.graph.to_json('./ensemble_' + str(ensemble_number) + '/partition_' + str(index))

    with open('./ensemble_' + str(ensemble_number) + '.json', 'w') as file:
        json.dump(stats, file)
    

def generate_plan(chain, seed):
    random.seed(seed)
    for partition in chain.with_progress_bar():
        pass
    
    stats[seed] = calc_stats(partition)

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

def calc_stats(plan):
    stats = {
        'wh_dist': [],
        'aa_dist': [],
        'ai_dist': [],
        'hi_dist': [],
        'rep_dist': [],
        'dem_dist': []
    }

    for district in plan.assignment.parts.keys():
        dist = plan.assignment.parts[district]

        wh_pop = 0
        aa_pop = 0
        ai_pop = 0
        hi_pop = 0

        rep_sum = 0
        dem_sum = 0

        for node in dist:
            wh_pop = wh_pop + plan.graph.nodes[node]['TAWHITEALN']
            aa_pop = aa_pop + plan.graph.nodes[node]['TABLACKCMB']
            ai_pop = ai_pop + plan.graph.nodes[node]['TAASIANCMB']
            hi_pop = hi_pop + plan.graph.nodes[node]['TAHISPANIC']

            rep_sum = rep_sum + plan.graph.nodes[node]['G20PRERTRU']
            dem_sum = dem_sum + plan.graph.nodes[node]['G20PREDBID']

        if (max(wh_pop, aa_pop, ai_pop, hi_pop) == wh_pop):
            stats['wh_dist'].append(district)
        elif (max(aa_pop, ai_pop, hi_pop) == aa_pop):
            stats['aa_dist'].append(district)
        elif (max(ai_pop, hi_pop) == ai_pop):
            stats['ai_dist'].append(district)
        else:
            stats['hi_dist'].append(district)

        if rep_sum > dem_sum:
            stats['rep_dist'].append(district)
        else:
            stats['dem_dist'].append(district)

    return stats

def cluster_analysis(ensemble, dist_measure):
    # Find number of clusters and centers
    # calculate variation of clusters
    # calculate number of plans within cluster and avg stats

    pass
