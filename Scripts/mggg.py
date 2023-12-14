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

ENSEMBLE_SIZE=1000
ENSEMBLE_ID=0

class Plan():
    def __init__(self, id, dem_pct, rep_pct, rep_dists, dem_dists, opp_dists, pop_data, area_data):
        self.id = id
        self.centroid = 0
        self.dem_pct = dem_pct
        self.rep_pct = rep_pct
        self.rep_dists = rep_dists
        self.dem_dists = dem_dists
        self.opportunity_districts = opp_dists
        self.population_data = pop_data
        self.area_data = area_data

    def format(self):
        return {
            'id': self.id,
            'mds_centroid': self.centroid,
            'dem_pct': self.dem_pct,
            'rep_pct': self.rep_pct,
            'rep_dists': self.rep_dists,
            'dem_dists': self.dem_dists,
            'opportunity_districts': self.opportunity_districts,
            'population_data': self.population_data,
            'area_data': self.area_data
        }

def run(data, dist_measure, ensemble_id, size=1000):
    global ENSEMBLE_SIZE
    global ENSEMBLE_ID
    ENSEMBLE_SIZE = size
    ENSEMBLE_ID = ensemble_id

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
        
    # print("Saving Plan Data...")
    # os.mkdir('./ensemble_' + str(ensemble_number))
    # for index, plan in enumerate(ensemble):
    #     print(type(plan.graph))
    #     plan.graph.to_json('./ensemble_' + str(ensemble_number) + '/partition_' + str(index), include_geometries_as_geojson=True)

    # print("Saving Cluster Data...")

    # print("Saving Ensemble Stats...")
    # with open('./ensemble_' + str(ensemble_number) + '.json', 'w') as file:
    #     json.dump(stats, file)
    
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

def generate_plan(chain, seed):
    random.seed(seed)
    for partition in chain.with_progress_bar():
        pass
    
    partition.stats = calc_stats(partition, seed)
    return partition

def calc_stats(plan, seed):
    dem_pct = 0
    rep_pct = 0
    rep_dists = []
    dem_dists = []
    opportunity_districts = []
    population_data = {
        'white_pct': 0,
        'asian_pct': 0,
        'black_pct': 0,
        'hisp_pct': 0
    }
    area_data = plan.area

    white_pcts = []
    black_pcts = []
    asian_pcts = []
    hisp_pcts = []

    for district in plan.assignment.parts.keys():
        dist = plan.assignment.parts[district]

        white_sum = 0
        black_sum = 0
        asian_sum = 0
        hisp_sum = 0

        rep_sum = 0
        dem_sum = 0

        for node in dist:
            white_sum = white_sum + plan.graph.nodes[node]['TAWHITEALN']
            black_sum = black_sum + plan.graph.nodes[node]['TABLACKCMB']
            asian_sum = asian_sum + plan.graph.nodes[node]['TAASIANCMB']
            hisp_sum = hisp_sum + plan.graph.nodes[node]['TAHISPANIC']

            rep_sum = rep_sum + plan.graph.nodes[node]['G20PRERTRU']
            dem_sum = dem_sum + plan.graph.nodes[node]['G20PREDBID']

        pop_sum = white_sum + black_sum + asian_sum + hisp_sum
        white_pcts.append(white_sum / pop_sum)
        black_pcts.append(black_sum / pop_sum)
        asian_pcts.append(asian_sum / pop_sum)
        hisp_pcts.append(hisp_sum / pop_sum)

        if (black_pcts[-1] > 0.2 or asian_pcts[-1] > 0.2 or hisp_pcts[-1] > 0.2):
            opportunity_districts.append(district)

        if rep_sum > dem_sum:
            rep_dists.append(district)
        else:
            dem_dists.append(district)

    dem_pct = len(dem_dists) / (len(dem_dists) + len(rep_dists))
    rep_pct = len(rep_dists) / (len(dem_dists) + len(rep_dists))

    avg_white_pct = sum(white_pcts) / len(white_pcts)
    avg_black_pct = sum(black_pcts) / len(black_pcts)
    avg_asian_pct = sum(asian_pcts) / len(asian_pcts)
    avg_hisp_pct = sum(hisp_pcts) / len(hisp_pcts)

    population_data['white_pct'] = avg_white_pct
    population_data['black_pct'] = avg_black_pct
    population_data['asian_pct'] = avg_asian_pct
    population_data['hisp_pct'] = avg_hisp_pct

    plan_data = Plan(seed, dem_pct, rep_pct, rep_dists, dem_dists, opportunity_districts, population_data, area_data)

    return plan_data

def cluster_analysis_opt_trans(ensemble):
    print('Calculating Distances...')
    distances = [[0 for i in range(ENSEMBLE_SIZE)] for j in range(ENSEMBLE_SIZE)]

    for outer_idx, outer_plan in enumerate(ensemble):
        for inner_idx in range(outer_idx + 1, ENSEMBLE_SIZE):
            inner_plan = ensemble[inner_idx]
            distances[outer_idx][inner_idx] = ot.Pair(outer_plan, inner_plan).distance
            distances[inner_idx][outer_idx] = distances[outer_idx][inner_idx]

    print(distances)

    cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_dist = compute_clusters(distances, ensemble)
    save_cluster_data(cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_dist)

def cluster_analysis_hamm_dist(ensemble):
    print('Calculating Distances...')
    distances = [[0 for i in range(ENSEMBLE_SIZE)] for j in range(ENSEMBLE_SIZE)]

    for outer_idx, outer_plan in enumerate(ensemble):
        for inner_idx in range(outer_idx + 1, ENSEMBLE_SIZE):
            inner_plan = ensemble[inner_idx]
            
            hamm_dist = 0
            for precinct in range(len(outer_plan.assignment)):
                if (inner_plan.assignment[precinct] != outer_plan.assignment[precinct]):
                    hamm_dist += 1
            
            hamm_dist = hamm_dist / len(outer_plan.assignment)

            distances[inner_idx][outer_idx] = hamm_dist
            distances[outer_idx][inner_idx] = hamm_dist

    print(distances)

    cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_dist = compute_clusters(distances, ensemble)
    save_cluster_data(cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_dist)

def cluster_analysis_ent_dist(ensemble):
    pass

def compute_clusters(dist_matrix, partitions):
    print('Computing Clusters...')
    mds = MDS(n_components=2, random_state=0, dissimilarity='precomputed', normalized_stress='auto')
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
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(pos)
        sse.append(kmeans.inertia_)

        score = silhouette_score(pos, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_k = range(2, 10)[silhouette_scores.index(max(silhouette_scores))]

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(pos)
    centers = kmeans.cluster_centers_

    cluster_partition_mapping = {i: [] for i in range(len(centers))}
    for idx, (partition, cluster_id) in enumerate(zip(partitions, clusters)):
        cluster_partition_mapping[cluster_id].append(partition)
        mds_coords = pos[idx]
        partition.stats.centroid = [mds_coords[0], mds_coords[1]]

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
        cluster_points = [partition.stats.centroid for partition in partitions if
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

def save_cluster_data(cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_dist):
    for cluster in cluster_partition_mapping:
        for i in range(len(cluster_partition_mapping[cluster])):
            cluster_partition_mapping[cluster][i] = cluster_partition_mapping[cluster][i].stats.id

    for plan in average_plans:
        average_plans[plan] = average_plans[plan].stats.id

    for centroid in centroids_mapping:
        centroids_mapping[centroid] = centroids_mapping[centroid].tolist()

    ensemble_stats = {
        'cluster_mapping': cluster_partition_mapping,
        'centroid_mapping': centroids_mapping,
        'avg_plans': average_plans,
        'avg_distances': avg_distances,
        'avg_enesmble_distance': avg_ensemble_dist
    }

    with open('./ensemble_' + str(ENSEMBLE_ID) + '.json', 'w') as file:
        json.dump(ensemble_stats, file)