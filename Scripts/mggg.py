from gerrychain import (GeographicPartition, Graph, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from gerrychain.random import random
from gerrychain.constraints import no_vanishing_districts
from gerrychain.constraints import within_percent_of_ideal_population
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
import concurrent.futures
import multiprocessing as mp
from datatypes import Plan
from datatypes import Cluster
from datatypes import Ensemble
import opt_trans as ot
import json
import os
import utils

ENSEMBLE_SIZE = 1000
ENSEMBLE_ID = 0
OUTPUT_PATH = ''

PLANS = []
CLUSTERS = []
ENSEMBLE = Ensemble()

def run(data, dist_measure, ensemble_number, size=1000):
    global OUTPUT_PATH
    global ENSEMBLE_SIZE
    global ENSEMBLE_ID
    global ENSEMBLE

    ENSEMBLE_SIZE = size
    ENSEMBLE_ID = f'ensemble_{ensemble_number}'
    ENSEMBLE.ensemble_id = ENSEMBLE_ID
    OUTPUT_PATH = f'{utils.OUTPUT_PATH}{ENSEMBLE_ID}'

    print("Setting Up...")
    graph = Graph.from_geodataframe(data)
    ensemble = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []

        for seed in range(ENSEMBLE_SIZE):
            future = executor.submit(generate_plan, init_chain(graph), seed * (ensemble_number + 1))
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result(timeout=300)
                ensemble.append(result)
            except concurrent.futures.TimeoutError:
                print("Timeout occurred for generate_plan call.")
            except Exception as e:
                print(f"Exception occurred: {e}")

    print("Performing Cluster Analysis...")
    match dist_measure:
        case utils.OPTIMAL_TRANSPORT:
            cluster_analysis_opt_trans(ensemble)
        case utils.HAMMING_DISTANCE:
            cluster_analysis_hamm_dist(ensemble)
        case _:
            cluster_analysis_hamm_dist(ensemble)

    ENSEMBLE.num_of_clusters = len(CLUSTERS)
    ENSEMBLE.num_of_plans = len(PLANS)
    
    print("Saving Data...")
    save_data(ensemble, data)
    
def init_chain(graph):
    elections = [
        Election("PRES", {"Dem": "G20PREDBID", "Rep": "G20PRERTRU"})
    ]

    my_updaters = {"population": updaters.Tally("ADJPOP", alias="population")}
    election_updaters = {election.name: election for election in elections}
    my_updaters.update(election_updaters)

    initial_partition = GeographicPartition(graph, assignment='DISTRICT', updaters=my_updaters)
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

    pop_constraint = within_percent_of_ideal_population(initial_partition, 0.10)

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
        total_steps=2500
    )

    return chain

def generate_plan(chain, seed):
    random.seed(seed)
    for partition in chain.with_progress_bar():
        pass
    
    global PLANS
    PLANS.append(calc_plan_stats(partition, f'{ENSEMBLE_ID}.plan_{seed}'))

    return partition

def calc_plan_stats(plan, plan_id):
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

        if (black_pcts[-1] > 0.35 or asian_pcts[-1] > 0.35 or hisp_pcts[-1] > 0.35):
            opportunity_districts.append(district)

        if rep_sum > dem_sum:
            rep_dists.append(district)
        else:
            dem_dists.append(district)

    dem_pct = len(dem_dists) / (len(dem_dists) + len(rep_dists))
    rep_pct = len(rep_dists) / (len(dem_dists) + len(rep_dists))
    opp_pct = len(opportunity_districts) / (len(dem_dists) + len(rep_dists))

    avg_white_pct = sum(white_pcts) / len(white_pcts)
    avg_black_pct = sum(black_pcts) / len(black_pcts)
    avg_asian_pct = sum(asian_pcts) / len(asian_pcts)
    avg_hisp_pct = sum(hisp_pcts) / len(hisp_pcts)

    population_data['white_pct'] = avg_white_pct
    population_data['black_pct'] = avg_black_pct
    population_data['asian_pct'] = avg_asian_pct
    population_data['hisp_pct'] = avg_hisp_pct

    plan_data = Plan()

    plan_data.plan_id = plan_id
    plan_data.dem_pct = dem_pct
    plan_data.rep_pct = rep_pct
    plan_data.dem_dists = dem_dists
    plan_data.rep_dists = rep_dists
    plan_data.opp_pct = opp_pct
    plan_data.opportunity_districts = opportunity_districts
    plan_data.population_data = population_data
    plan_data.area_data = area_data

    return plan_data

def cluster_analysis_opt_trans(ensemble):
    print('Calculating Distances...')
    distances = [[0 for _ in range(ENSEMBLE_SIZE)] for _ in range(ENSEMBLE_SIZE)]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for outer_idx, outer_plan in enumerate(ensemble):
            for inner_idx in range(outer_idx + 1, ENSEMBLE_SIZE):
                futures.append(executor.submit(opt_trans, outer_idx, inner_idx, ensemble))

        concurrent.futures.wait(futures)

        results = [future.result() for future in futures]

    idx = 0
    for outer_idx in range(ENSEMBLE_SIZE - 1):
        for inner_idx in range(outer_idx + 1, ENSEMBLE_SIZE):
            distances[outer_idx][inner_idx] = results[idx]
            distances[inner_idx][outer_idx] = results[idx]
            idx += 1

    compute_clusters(distances, ensemble)

def opt_trans(idx_1, idx_2, ensemble):
    return ot.Pair(ensemble[idx_1], ensemble[idx_2]).distance

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

    compute_clusters(distances, ensemble)

def compute_clusters(dist_matrix, partitions):
    global PLANS
    global ENSEMBLE
    global CLUSTERS
    global ENSEMBLE_ID

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
    ENSEMBLE.avg_distance = avg_ensemble_dist

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

    cluster_partition_mapping = {i: Cluster() for i in range(len(centers))}
    for idx, (partition, cluster_id) in enumerate(zip(partitions, clusters)):
        cur_plan = PLANS[idx]
        cluster_partition_mapping[cluster_id].plan_ids.append(cur_plan.plan_id)

        mds_coord = pos[idx]
        cur_plan.mds_coord = list([mds_coord[0], mds_coord[1]])

    for idx, mds_coord in enumerate(centers):
        cluster_partition_mapping[idx].mds_coord = list(mds_coord)
        cluster_partition_mapping[idx].cluster_id = f'{ENSEMBLE_ID}.cluster_{idx}'
        cluster_partition_mapping[idx].num_of_plans = len(cluster_partition_mapping[idx].plan_ids)
        ENSEMBLE.cluster_ids.append(cluster_partition_mapping[idx].cluster_id)

    for cluster_id, centroid in enumerate(centers):
        min_dist = float('inf')
        closest_partition = None

        for plan_id in cluster_partition_mapping[cluster_id].plan_ids:
            partition_pos = 0
            for plan in PLANS:
                if plan.plan_id == plan_id:
                    partition_pos = plan.mds_coord
                    break
        
            dist = distance.euclidean(partition_pos, centroid)

            if dist < min_dist:
                min_dist = dist
                closest_partition = plan_id

        cluster_partition_mapping[cluster_id].avg_plan = closest_partition

        for plan in PLANS:
            if plan.plan_id == closest_partition:
                plan.geo_id = f'{plan.plan_id}_GEO'
                break

    for cluster_id in range(len(centers)):
        cluster_points = []
        for plan_id in cluster_partition_mapping[cluster_id].plan_ids:
            for plan in PLANS:
                if plan_id == plan.plan_id:
                    cluster_points.append(plan.mds_coord)
                    break

        total_distance = 0
        count = 0

        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                total_distance += distance.euclidean(cluster_points[i], cluster_points[j])
                count += 1

        if count > 0:
            cluster_partition_mapping[cluster_id].variation = total_distance / count
        else:
            cluster_partition_mapping[cluster_id].variation = 0

    max_opp_pct = 0
    min_opp_pct = PLANS[0].opp_pct
    max_dem_pct = 0
    max_rep_pct = 0

    max_opp_idx = 0
    min_opp_idx = 0
    max_dem_idx = 0
    max_rep_idx = 0
    for idx, plan in enumerate(PLANS):
        if plan.opp_pct > max_opp_pct:
            max_opp_pct = plan.opp_pct
            max_opp_idx = idx

        if plan.opp_pct < min_opp_pct:
            min_opp_pct = plan.opp_pct
            min_opp_idx = idx
        
        if plan.dem_pct > max_dem_pct:
            max_dem_pct = plan.dem_pct
            max_dem_idx = idx

        if plan.rep_pct > max_rep_pct:
            max_rep_pct = plan.rep_pct
            max_rep_idx = idx

    PLANS[max_opp_idx].geo_id = f'{PLANS[max_opp_idx].plan_id}_GEO'
    PLANS[min_opp_idx].geo_id = f'{PLANS[min_opp_idx].plan_id}_GEO'
    PLANS[max_dem_idx].geo_id = f'{PLANS[max_dem_idx].plan_id}_GEO'
    PLANS[max_rep_idx].geo_id = f'{PLANS[max_rep_idx].plan_id}_GEO'

    ENSEMBLE.max_opp_pct = max_opp_pct
    ENSEMBLE.min_opp_pct = min_opp_pct
    ENSEMBLE.max_rep_pct = max_rep_pct
    ENSEMBLE.max_dem_pct = max_dem_pct

    for idx in cluster_partition_mapping:
        CLUSTERS.append(cluster_partition_mapping[idx])

def save_data(ensemble, data):
    global PLANS
    global OUTPUT_PATH    
    
    os.mkdir(f'{OUTPUT_PATH}')

    save_plan_data()
    save_cluster_data()
    save_ensemble_data()
    save_plan_geo_data(ensemble, data)

def save_plan_data():
    global PLANS
    global OUTPUT_PATH

    with open(f'{OUTPUT_PATH}/plans.json', 'w') as file:
        plan_data = {}
        for plan in PLANS:
            plan_data[plan.plan_id] = plan.format()

        json.dump(plan_data, file)

def save_cluster_data():
    global CLUSTERS
    global OUTPUT_PATH
    
    with open(f'{OUTPUT_PATH}/clusters.json', 'w') as file:
        cluster_data = {}
        for cluster in CLUSTERS:
            cluster_data[cluster.cluster_id] = cluster.format()

        json.dump(cluster_data, file)

def save_ensemble_data():
    global ENSEMBLE
    global OUTPUT_PATH

    with open(f'{OUTPUT_PATH}/ensemble.json', 'w') as file:
        ensemble_data = {
            f'{ENSEMBLE_ID}': ENSEMBLE.format()
        }

        json.dump(ensemble_data, file)

def save_plan_geo_data(ensemble, data):
    plans_geo_data = {
        'geo_data': []
    }

    for idx, plan in enumerate(PLANS):
        if plan.geo_id == 'N/A':
            continue
        
        print("Saving Geo Data...")

        data_copy = data.copy()
        data_copy['district'] = data_copy.index.map(ensemble[idx].assignment)
        districts = data_copy.dissolve(by='district')

        geojson_str = districts.to_json()
        geojson_dict = json.loads(geojson_str)
        geojson_dict = {
            f'{plan.geo_id}': geojson_dict
        }

        plans_geo_data['geo_data'].append(geojson_dict)

    with open(f'{OUTPUT_PATH}/plan_geo.json', 'w') as file:
        json.dump(plans_geo_data, file)