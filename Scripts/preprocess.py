import geopandas
import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np
import maup

def preprocess(precincts, populations, districts, election_results):
    precincts = precincts.to_crs('EPSG:3421')
    districts = districts.to_crs('EPSG:3421')

    precincts['NEIGHBORS'] = None

    print("Generating Precinct Neighbors...")
    for index, precinct in precincts.iterrows():
        neighbors = []
        for i, possible_neighbor in precincts.iterrows():
            if i == index:
                continue

            border_length = precinct['geometry'].intersection(possible_neighbor['geometry']).length
            if border_length > 66:
                neighbors.append(possible_neighbor['GEOID20'])

        precincts.at[index, 'NEIGHBORS'] = ', '.join(neighbors)
    
    print("Done\n")

    precincts = precincts.join(populations.set_index('GEOID20'), on='GEOID20')
    precincts = precincts.join(election_results.set_index('GEOID20'), on='GEOID20')

    print("Assigning Districts...")
    precinct_to_district_assignment = maup.assign(precincts, districts)
    precincts['DISTRICT'] = precinct_to_district_assignment
    print("Done\n")

    return precincts

def get_nevada_data():
    print("Reading Precinct Boundry Data...")
    precincts = geopandas.read_file(utils.NEVADA_PATH + 'Hope/nv_vtd_2020_bound.zip')
    print("Done\n")

    print("Reading Precinct Demographic Data...")
    populations = pd.read_csv(utils.NEVADA_PATH + 'Hope/2020PL94-171_ADJPOP11-13-2021_Precincts.csv')
    print("Done\n")

    print("Reading Precinct Election Data...")
    election_results = pd.read_csv(utils.NEVADA_PATH + 'Hope/nv_2020_election.csv')
    print("Done\n")

    print("Reading District Boundry Data...")
    districts = geopandas.read_file(utils.NEVADA_PATH + 'Hope/nv_sldl_2021.zip')
    print("Done\n")

    populations = populations.drop(['STATEFP20', 'COUNTYFP20', 'VTDST20', 'NAME20'], axis='columns')

    precincts = preprocess(precincts, populations, districts, election_results)

    precincts = precincts.drop(['STATEFP20', 'STATEFP', 'COUNTYFP20', 'COUNTYFP', 'G20PRELJOR', 'G20PREIBLA', 'G20PREONON', 'GEOID', 'VTDST20', 'VTDI20', 'NAMELSAD20', 'LSAD20', 'MTFCC20', 'FUNCSTAT20', 'ALAND20', 'AWATER20'], axis='columns')

    return precincts



if __name__ == '__main__':
    nevada_data = get_nevada_data()
    nevada_data.to_file('nevada_data.shp')