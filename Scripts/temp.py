# Standard library imports
import json
import multiprocessing
import uuid
import sys
from functools import partial
from multiprocessing import Pool

# Third-party imports
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        accept, constraints, proposals, updaters, Election)
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part
from networkx.readwrite import json_graph
from optimaltransport import Pair
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
import cProfile

# Local imports
from database_classes import district_plan_geojson
from database_classes.cluster import build_cluster
from database_classes.district_plan import build_district_plan
from database_classes.ensemble import build_ensemble

"""
# ======================================================================================================================
# SECTION 1: MULTIPROCESSING METHODS
# ======================================================================================================================
# This is the code that handles the comparison of partitions. The serialize functions were inspired from the json
# functions already found in the graph folders of gerrychain, allows for the partition objects to be serialized.
# ======================================================================================================================
"""


def json_serialize(input_object):
    if pd.api.types.is_integer_dtype(input_object):
        return int(input_object)


def serialize_graph_to_json_string(graph):
    # Convert the graph to a JSON-compatible dictionary
    data = json_graph.adjacency_data(graph)
    # Convert the dictionary to a JSON string
    json_string = json.dumps(data, default=json_serialize)
    return json_string


def deserialize_graph_from_json_string(json_string):
    # Convert the JSON string back to a dictionary
    data = json.loads(json_string)
    # Convert the dictionary back to a graph
    graph = json_graph.adjacency_graph(data)
    return graph


def distance_worker(i, j, json_string_i, assignment_data_i, json_string_j, assignment_data_j):
    # Deserialize the graphs from JSON strings
    graph_i = deserialize_graph_from_json_string(json_string_i)
    graph_j = deserialize_graph_from_json_string(json_string_j)

    # Reconstruct the Partition objects
    partition_i = Partition(graph_i, assignment_data_i)
    partition_j = Partition(graph_j, assignment_data_j)

    # Calculate the distance
    distance = Pair(partition_i, partition_j).distance
    return i, j, distance


def compare_partitions(partitions, num_simulations):
    with Pool() as pool:
        tasks = []
        for i in range(num_simulations):
            for j in range(i, num_simulations):
                # Serialize the graph data to JSON strings
                json_string_i = serialize_graph_to_json_string(partitions[i].graph.graph)
                json_string_j = serialize_graph_to_json_string(partitions[j].graph.graph)

                # Get the assignment data directly
                assignment_data_i = partitions[i].assignment
                assignment_data_j = partitions[j].assignment

                # Append the task with graph and assignment data
                tasks.append((i, j, json_string_i, assignment_data_i, json_string_j, assignment_data_j))

        # Execute tasks in parallel
        results = pool.starmap(distance_worker, tasks)

    # Construct the distance matrix
    distance_matrix = np.zeros((num_simulations, num_simulations))
    for i, j, distance in results:
        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance

    # Normalize the distance matrix
    min_value = np.min(distance_matrix[distance_matrix != 0])
    max_value = np.max(distance_matrix)
    distance_matrix = (distance_matrix - min_value) / (max_value - min_value)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def main():
    """
    # ======================================================================================================================
    # SECTION 2: RECOM
    # ======================================================================================================================
    # The standard recom code. Change the global variables NUM_DISTRICTS, NUM_PARTITIONS, NUM_STEPS, and STATE to control
    # the specific things. Should end with a list of partitions at the end.
    # ======================================================================================================================
    """

    # NORMAL ARGS
    # NUM_DISTRICTS = 10  # EX. 30 for Arizona
    # NUM_PARTITIONS = 10  # How many plans made
    # NUM_STEPS = 100  # Steps for every plan made
    # STATE = "az"  # The state to redistrict

    # SLURM SCRIPT ARGS
    NUM_DISTRICTS = int(sys.argv[1])
    NUM_PARTITIONS = int(sys.argv[2])
    NUM_STEPS = int(sys.argv[3])
    STATE = str(sys.argv[4])

    # PART 2.1: The preprocessed file we have isn't in a graph format that gerrychain uses, so import geopandas and create a graph from the json as seen below

    gdf = gpd.read_file(f'data/{STATE}precincts.json')
    graph = Graph.from_geodataframe(gdf)

    # PART 2.2: Create the election array / the election objects

    elections = [
        Election("SEN20", {"Democratic": "adv_20", "Republican": "arv_20"})
        # Use "SEN20" to access election results in a partition
    ]

    # PART 2.3: Configure the updaters, these are the fields one can access for all the partitions

    my_updaters = {
        "population": updaters.Tally("vap", alias="population"),
        "pop_white": updaters.Tally("vap_white", alias="pop_white"),
        "pop_hisp": updaters.Tally("vap_hisp", alias="pop_hisp"),
        "pop_black": updaters.Tally("vap_black", alias="pop_black"),
        "pop_asian": updaters.Tally("vap_asian", alias="pop_asian"),
        "pop_two": updaters.Tally("vap_two", alias="pop_two")
    }

    election_updaters = {election.name: election for election in elections}
    my_updaters.update(election_updaters)

    # PART 2.4: Configure the assignments, this has stuff like the number of districts we want to start with

    total_pop = sum([graph.nodes[n]["vap"] for n in graph.nodes])

    assignment = recursive_tree_part(
        graph,
        range(NUM_DISTRICTS),  # district names, in this case {0,1,2,3}
        total_pop / NUM_DISTRICTS,  # ideal population for a district (we want them to all be even)
        "vap",
        0.10  # maximum allowed population deviation, 5% for now
    )

    # PART 2.5: Configure the initial partition

    initial_partition = GeographicPartition(graph, assignment=assignment, updaters=my_updaters)
    initial_partition.plot(gdf)  # If you want to see how the map first looks like

    # PART 2.6: Set up the ReCom proposal

    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

    proposal = partial(recom,
                       pop_col="vap",
                       pop_target=ideal_population,
                       epsilon=0.10,
                       node_repeats=2
                       )

    # PART 2.7: Set up the constraints

    compactness_bound = constraints.UpperBound(lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"]))

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.10)

    # PART 2.8: Configure the markov chain and run it, making 10 at 500 steps for testing

    partitions = []

    for simulation in range(NUM_PARTITIONS):
        chain = MarkovChain(
            proposal=proposal,
            constraints=[
                pop_constraint,
                compactness_bound
            ],
            accept=accept.always_accept,
            initial_state=initial_partition,
            total_steps=NUM_STEPS
        )

        for partition in chain.with_progress_bar():
            pass

        partitions.append(partition)  # Save the final state of the chain

    """
    # ======================================================================================================================
    # SECTION 3: DISTRICT PLAN CREATION
    # ======================================================================================================================
    # Create a district plan object for every partition, has all of the calculations and used for the cluster average
    # calculations. Given as an instance variable to every partition object to preserve ordering.
    # ======================================================================================================================
    """

    for partition in partitions:
        partition.plan = build_district_plan(STATE, partition)

    """
    # ======================================================================================================================
    # SECTION 4: PARTITION COMPARISON
    # ======================================================================================================================
    # Compare all of the district plans using the optimal transport code
    # ======================================================================================================================
    """

    distance_matrix = compare_partitions(partitions, NUM_PARTITIONS)

    """
    # ======================================================================================================================
    # SECTION 5: CLUSTERING
    # ======================================================================================================================
    # Given the normalized distance matrix clusters are found using k-means, at the end the cluster partition mapping,
    # centroid mapping, average plan for cluster, average distance for the cluster, and the overall average ensemble distance
    # ======================================================================================================================
    """

    def find_clusters(distance_matrix, partitions):
        # PART 5.1: Perform MDS on the distance matrix

        mds = MDS(n_components=2, random_state=0, dissimilarity='precomputed')
        pos = mds.fit(distance_matrix).embedding_

        # PART 5.2: Get the average distance for all points, this is for the ensemble
        total_distance_all_points = 0
        num_points = len(pos)
        count_all_pairs = num_points * (num_points - 1) / 2  # Number of unique pairs

        for i in range(num_points):
            for j in range(i + 1, num_points):
                total_distance_all_points += distance.euclidean(pos[i], pos[j])

        # Calculate the average distance for all points
        avg_ensemble_distance = total_distance_all_points / count_all_pairs

        # PART 5.3: Find the ideal number of clusters by finding optimal k

        sse = []  # Sum of squared distances (For plotting)
        silhouette_scores = []  # List to store silhouette scores for each k
        for k in range(2, 10):  # Silhouette score is only valid for k >= 2
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(pos)
            sse.append(kmeans.inertia_)  # (For plotting)

            # Calculate silhouette score and append to the list
            score = silhouette_score(pos, kmeans.labels_)
            silhouette_scores.append(score)

        optimal_k = range(2, 10)[silhouette_scores.index(max(silhouette_scores))]

        # PART 5.4: Perform  k means clustering

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(pos)

        # OPTIONAL: Uncomment if you want to see the plotted clustered data
        # plt.scatter(pos[:, 0], pos[:, 1], c=clusters, cmap='viridis', marker='o')
        centers = kmeans.cluster_centers_
        # plt.scatter(centers[:, 0], centers[:, 1], s=300, c='red', marker='x')  # Plot the centroids
        # plt.title('MDS with K-Means Clustering')
        # plt.xlabel('Component 1')
        # plt.ylabel('Component 2')
        # plt.show()

        # PART 5.5: Associate each of the partition objects with a cluster
        # PART 5.5.1: Map the partition objects with the clusters, also get the coordinates for each plan
        cluster_partition_mapping = {i: [] for i in range(len(kmeans.cluster_centers_))}
        for idx, (partition, cluster_id) in enumerate(zip(partitions, clusters)):
            # Make dictionary mapping partitions in an array to cluster
            cluster_partition_mapping[cluster_id].append(partition)

            # Associate the mds coords with each district plan
            mds_coords = pos[idx]
            partition.plan.mds_centroid = [mds_coords[0], mds_coords[1]]

        # PART 5.5.2: Map the centroid of each cluster with the cluster, needed for cluster placement in UI
        centroids_mapping = {i: centroid for i, centroid in enumerate(kmeans.cluster_centers_)}

        # PART 5.6: Find the partition closest to each cluster's centroid, this is the average plan (SeaWulf-14)
        average_plans = {}
        for cluster_id, centroid in enumerate(centers):
            min_distance = float('inf')
            closest_partition = None

            # Iterate over each object in the cluster
            for partition in cluster_partition_mapping[cluster_id]:
                partition_index = partitions.index(partition)
                partition_pos = pos[partition_index]

                # Calculate the distance from the object to the centroid
                dist = distance.euclidean(partition_pos, centroid)

                # Check if this object is closer than the current closest
                if dist < min_distance:
                    min_distance = dist
                    closest_partition = partition

            average_plans[cluster_id] = closest_partition

        # PART 5.7: Find the average distance between district plans in a cluster

        avg_distances = {}
        for cluster_id in range(len(kmeans.cluster_centers_)):
            cluster_points = [partition.plan.mds_centroid for partition in partitions if
                              clusters[partitions.index(partition)] == cluster_id]
            total_distance = 0
            count = 0

            # Calculate pairwise distances within the cluster
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    total_distance += distance.euclidean(cluster_points[i], cluster_points[j])
                    count += 1

            # Avoid division by zero for clusters with a single point
            if count > 0:
                avg_distances[cluster_id] = total_distance / count
            else:
                avg_distances[cluster_id] = 0

        return cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_distance

    cluster_partition_mapping, centroids_mapping, average_plans, avg_distances, avg_ensemble_distance = find_clusters(
        distance_matrix, partitions)

    """
    # ======================================================================================================================
    # SECTION 6: AVERAGE PLAN GEOJSONS
    # ======================================================================================================================
    # Generate the geojson for the average plan. Append this to the district_plan_geojsons array that will then converted
    # to the json file later
    # ======================================================================================================================
    """

    # Initialize a list that will hold all the geojson made during this run, will export these to a json file at the end
    district_plan_geojsons = []

    for cluster_id in cluster_partition_mapping:
        average_plan = average_plans[cluster_id]  # What is the average plan (closest object to the centroid)

        gdf_copy = gdf.copy()
        gdf_copy['district'] = gdf_copy.index.map(average_plan.assignment)
        districts = gdf_copy.dissolve(by='district')

        geojson_str = districts.to_json()  # Convert to string format
        geojson_dict = json.loads(geojson_str)  # Convert to dictionary format

        unique_geojson_id = uuid.uuid4()

        average_plan.plan.geojson_id = unique_geojson_id  # The district plan object associated with geojson
        district_plan = district_plan_geojson.DistrictPlanGeoJSON(unique_geojson_id, geojson_dict)

        district_plan_geojsons.append(district_plan)

    """
    # ======================================================================================================================
    # SECTION 7: BUILD THE CLUSTER OBJECTS
    # ======================================================================================================================
    # Create a cluster object for every partition clustering. Also find interesting plans in the clusters (if any)
    # ======================================================================================================================
    """

    clusters = []
    for cluster_id in cluster_partition_mapping:
        partitions_list = cluster_partition_mapping[cluster_id]  # List of partitions for this cluster
        centroid = centroids_mapping[cluster_id]  # Centroid of the cluster
        average_plan = average_plans[cluster_id]  # What is the average plan (closest object to the centroid)
        avg_distance = avg_distances[cluster_id]  # The average pairwise distance amoung points in the cluster

        # PART 7.1: Build the custom cluster object

        cluster_thing = build_cluster(partitions_list, centroid, average_plan, avg_distance)

        # PART 7.2: Find the interesting plans for this cluster, if any

        interesting_partitions = []  # List to hold the partitions that we will build DistrictPlanGeoJSON objects
        interesting_plans = {}  # Dictionary we will give the cluster object

        if len(partitions_list) < 3:  # If the cluster is too small there's no real point
            pass
        elif len(partitions_list) < 5:
            min_opportunity_district_plan, max_opportunity_district_plan = district_plan_geojson.find_most_and_least_opportunity_districts(partitions_list)

            interesting_partitions.append(min_opportunity_district_plan)
            interesting_partitions.append(max_opportunity_district_plan)

            interesting_plans = {
                "min_opportunity_district_plan": str(min_opportunity_district_plan.plan.plan_id),
                "max_opportunity_district_plan": str(max_opportunity_district_plan.plan.plan_id),
            }
        else:  # Originally had more but saving 6 takes up way too much space :(
            min_opportunity_district_plan, max_opportunity_district_plan = district_plan_geojson.find_most_and_least_opportunity_districts(partitions_list)
            lowest_population_margin_partition, highest_population_margin_partition = district_plan_geojson.find_most_and_least_population_margins(partitions_list)

            interesting_partitions.append(min_opportunity_district_plan)
            interesting_partitions.append(max_opportunity_district_plan)
            interesting_partitions.append(lowest_population_margin_partition)
            interesting_partitions.append(highest_population_margin_partition)

            interesting_plans = {
                "min_opportunity_district_plan": str(min_opportunity_district_plan.plan.plan_id),
                "max_opportunity_district_plan": str(max_opportunity_district_plan.plan.plan_id),
                "lowest_population_margin_partition": str(lowest_population_margin_partition.plan.plan_id),
                "highest_population_margin_partition": str(highest_population_margin_partition.plan.plan_id),
            }

        cluster_thing.interesting_plans = interesting_plans

        # PART 7.3: Generate the geojson object for all the interesting plans and append it to the district_plan_geojsons array | NO DUPLICATES!!!!

        for partition in interesting_partitions:
            if partition.plan.geojson_id == 0:  # Dont want to make repeat geojson objects
                gdf_copy = gdf.copy()
                gdf_copy['district'] = gdf_copy.index.map(partition.assignment)
                districts = gdf_copy.dissolve(by='district')

                geojson_str = districts.to_json()  # Convert to string format
                geojson_dict = json.loads(geojson_str)  # Convert to dictionary format

                unique_geojson_id = uuid.uuid4()

                partition.plan.geojson_id = unique_geojson_id  # The district plan object associated with geojson
                district_plan = district_plan_geojson.DistrictPlanGeoJSON(unique_geojson_id, geojson_dict)

                district_plan_geojsons.append(district_plan)

            # PART 7.4: Append it to the clusters list

        clusters.append(cluster_thing)

    """
    # ======================================================================================================================
    # SECTION 8: ENSEMBLE CREATION
    # ======================================================================================================================
    # Create the ensemble object
    # ======================================================================================================================
    """

    ensemble = build_ensemble(NUM_PARTITIONS, clusters, distance_matrix.tolist(), avg_ensemble_distance)

    """
    # ======================================================================================================================
    # SECTION 9: EXPORTING
    # ======================================================================================================================
    # Export the district plans, clusters, ensemble, and geojsons to a json file to be imported into our mongo database
    # ======================================================================================================================
    """

    # PARTITIONS

    plan_array = []
    for partition in partitions:
        plan_array.append(partition.plan.to_dict())

    with open(f'{STATE}_district_plan_{NUM_PARTITIONS}.json', 'w+') as file:
        for plan in plan_array:
            file.write(json.dumps(plan) + '\n')

    # CLUSTERS

    cluster_array = []
    for cluster in clusters:
        cluster_array.append(cluster.to_dict())

    with open(f'{STATE}_cluster_{NUM_PARTITIONS}.json', 'w+') as file:
        for cluster in cluster_array:
            file.write(json.dumps(cluster) + '\n')

    # PARTITION GEOJSONS

    plan_geojson_array = [plan_geojson.to_dict() for plan_geojson in district_plan_geojsons]

    with open(f'{STATE}_geojson_{NUM_PARTITIONS}.json', 'w+') as file:
        for plan_geojson in plan_geojson_array:
            file.write(json.dumps(plan_geojson) + '\n')

    # ENSEMBLES

    with open(f'{STATE}_ensemble_{NUM_PARTITIONS}.json', 'w+') as file:
        file.write(json.dumps(ensemble.to_dict()))


if __name__ == "__main__":
    cProfile.run('main()', f'{str(sys.argv[4])}_profiling_output_{int(sys.argv[2])}.prof')