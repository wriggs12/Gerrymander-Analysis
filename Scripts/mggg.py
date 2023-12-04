from gerrychain import (GeographicPartition, Graph, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from gerrychain.random import random
from gerrychain.constraints import no_vanishing_districts

ENSEMBLE_SIZE=1
stats = {'DEM_DIST': [], 'REP_DIST': [], 'AA_POP': [], 'AI_POP': [], 'WH_POP': [], 'HI_POP': []}

def generate_plan(chain, seed):
    random.seed(seed)
    for partition in chain.with_progress_bar():
        pass
    
    calc_party_split(partition)
    calc_race_demographics(partition)

    return (partition)

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

def run(data, dist_measure):
    graph = Graph.from_geodataframe(data)
    chain = init_chain(graph)

    ensemble = [generate_plan(chain, seed) for seed in range(ENSEMBLE_SIZE)]
    find_clusters(ensemble, dist_measure)
    

def calc_party_split(plan, stats):
    rep = 0
    dem = 0
    
    for node in plan.graph:
        if plan.graph.nodes[node]['G20PREDBID'] > plan.graph.nodes[node]['G20PRERTRU']:
            dem = dem + 1
        else:
            rep = rep + 1

    stats['DEM_DIST'].append(dem)
    stats['REP_DIST'].append(rep)

def calc_race_demographics(plan, stats):
    hi_pop = 0
    aa_pop = 0
    ai_pop = 0
    wh_pop = 0

    for node in plan.graph:
        hi_pop = hi_pop + plan.graph.nodes[node]['TAHISPANIC']
        aa_pop = aa_pop + plan.graph.nodes[node]['TABLACKCMB']
        ai_pop = ai_pop + plan.graph.nodes[node]['TAASIANCMB']
        wh_pop = wh_pop + plan.graph.nodes[node]['TAWHITEALN']

    stats['HI_POP'].append(hi_pop)
    stats['AA_POP'].append(aa_pop)
    stats['AI_POP'].append(ai_pop)
    stats['WH_POP'].append(wh_pop)

def save_stats():
    pass

def find_clusters(ensemble, dist_measure):
    pass