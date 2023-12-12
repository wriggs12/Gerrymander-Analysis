from gerrychain import (GeographicPartition, Graph, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from gerrychain.random import random
from gerrychain.constraints import no_vanishing_districts
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import opt_trans as ot
import json
import os
import utils

ENSEMBLE_SIZE=10
stats = {}

def run(data, dist_measure, ensemble_number, size=1000):
    ENSEMBLE_SIZE = size

    print("Setting Up...")
    graph = Graph.from_geodataframe(data)
    chain = init_chain(graph)

    print("Generating Ensemble...")
    ensemble = [generate_plan(chain, seed) for seed in range(ENSEMBLE_SIZE)]

    print("Performing Cluster Analysis...")
    match dist_measure:
        case utils.OPTIMAL_TRANSPORT:
            cluster_analysis_opt_trans(ensemble)
        case utils.HAMMING_DISTANCE:
            cluster_analysis_hamm_dist(ensemble)
        case utils.ENTROPY_DISTANCE:
            cluster_analysis_ent_dist(ensemble)
        case _:
            cluster_analysis_hamm_dist(ensemble)
        
    print("Saving Plan Data...")
    os.mkdir('./ensemble_' + str(ensemble_number))
    for index, plan in enumerate(ensemble):
        plan.graph.to_json('./ensemble_' + str(ensemble_number) + '/partition_' + str(index))

    print("Saving Ensemble Stats...")
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
        total_steps=10000
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

def cluster_analysis_opt_trans(ensemble):
    # Find number of clusters and centers
    distances = [[0]*ENSEMBLE_SIZE]*ENSEMBLE_SIZE

    for outer_idx, outer_plan in enumerate(ensemble):
        for inner_idx, inner_plan in range(outer_idx + 1, ENSEMBLE_SIZE):
            inner_plan = ensemble[inner_idx]
            distances[outer_idx, inner_idx] = ot.Pair(outer_plan, inner_plan).distance
            distances[inner_idx, outer_idx] = distances[outer_idx, inner_idx]

    cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_dist = compute_clusters(distances, ensemble)

def cluster_analysis_hamm_dist(ensemble):
    pass

def cluster_analysis_ent_dist(ensemble):
    pass

def compute_clusters(dist_matrix, partitions):
    mds = MDS(n_components=2, random_state=0, dissimilarity='precomputed')
    pos = mds.fit(dist_matrix).embedding_

    total_distance_all_points = 0
    num_points = len(pos)
    count_all_pairs = num_points * (num_points - 1) / 2

    for i in range(num_points):
        for j in range(i + 1, num_points):
            total_distance_all_points += distance.euclidean(pos[i], pos[j])

    avg_ensemble_dist = total_distance_all_points / count_all_pairs

    sse = []
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pos)
        sse.append(kmeans.inertia_)

        score = silhouette_score(pos, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = range(2, 10)[silhouette_scores.index(max(silhouette_scores))]

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(pos)
    centers = kmeans.cluster_centers_

    cluster_partition_mapping = {i: [] for i in range(len(centers))}
    for idx, (partition, cluster_id) in enumerate(zip(partitions, clusters)):
        cluster_partition_mapping[cluster_id].append(partition)
        mds_coords = pos[idx]
        partition.plan.mds_centroid = [mds_coords[0], mds_coords[1]]

    centroids_mapping = {i: centroid for i, centroid in enumerate(centers)}

    average_plans = {}
    for cluster_id, centroid in enumerate(centers):
        min_dist = float('inf')
        closest_partition = None

        for partition in cluster_partition_mapping[cluster_id]:
            partition_index = partitions.index(partition)
            partition_pos = pos[partition_index]

            dist = distance.euclidean(partition_pos, centroid)

            if dist < min_dist:
                min_dist = dist
                closest_partition = partition

        average_plans[cluster_id] = closest_partition

    avg_distances = {}
    for cluster_id in range(len(centers)):
        cluster_points = [partition.plan.mds_centroid for partition in partitions if
                            clusters[partitions.index(partition)] == cluster_id]
        total_distance = 0
        count = 0

        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                total_distance += distance.euclidean(cluster_points[i], cluster_points[j])
                count += 1

        if count > 0:
            avg_distances[cluster_id] = total_distance / count
        else:
            avg_distances[cluster_id] = 0

    return cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_dist