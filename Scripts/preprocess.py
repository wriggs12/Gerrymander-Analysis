import geopandas
import pandas as pd
import json
import utils
import matplotlib.pyplot as plt
import csv
import numpy as np
import random
import maup


def preprocess(precincts, populations, districts, election_results):
    precincts = precincts.to_crs('EPSG:3421')
    districts = districts.to_crs('EPSG:3421')

    precincts['NEIGHBORS'] = None

    for index, precinct in precincts.iterrows():
        neighbors = np.array(precincts[precincts.geometry.touches(precinct['geometry'])].GEOID20)
        precincts.at[index, 'NEIGHBORS'] = ', '.join(neighbors)

    precincts = precincts.join(populations.set_index('GEOID20'), on='GEOID20')
    precincts = precincts.join(election_results.set_index('GEOID20'), on='GEOID20')

    precinct_to_district_assignment = maup.assign(precincts, districts)
    precincts['DISTRICT'] = precinct_to_district_assignment

    return precincts

def get_nevada_data():
    precincts = geopandas.read_file(utils.NEVADA_PATH + 'Hope/nv_vtd_2020_bound.zip')
    populations = pd.read_csv(utils.NEVADA_PATH + 'Hope/2020PL94-171_ADJPOP11-13-2021_Precincts.csv')
    districts = geopandas.read_file(utils.NEVADA_PATH + 'Hope/nv_sldl_2021.zip')
    election_results = pd.read_csv(utils.NEVADA_PATH + 'Hope/nv_2020_election.csv')

    populations = populations.drop(['STATEFP20', 'COUNTYFP20', 'VTDST20', 'NAME20'], axis='columns')

    precincts = preprocess(precincts, populations, districts, election_results)

    return precincts



if __name__ == '__main__':
    get_nevada_data()