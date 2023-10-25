import geopandas
import pandas as pd
import json
import utils
import matplotlib.pyplot as plt
import csv
import numpy as np

def preprocess(precincts, populations):
    precincts['NEIGHBORS'] = None

    for index, precinct in precincts.iterrows():
        neighbors = np.array(precincts[precincts.geometry.touches(precinct['geometry'])].GEOID)
        precincts.at[index, 'NEIGHBORS'] = ', '.join(neighbors)

    print(precincts.geometry)
    print(precincts)

    precincts = precincts.set_index('GEOID').join(populations.set_index('GEOID'))

    precincts.plot(column='VAP')
    plt.show()

def get_nevada_data():
    precincts = geopandas.read_file(utils.NEVADA_PATH + 'tl_rd22_32_bg.zip')
    populations = pd.read_csv(utils.NEVADA_PATH + 'voting_age_population.csv')

    populations = populations.rename(columns={'GEO_ID': 'GEOID'})
    populations = populations.drop(populations.index[0])
    populations = populations.dropna(how='all', axis='columns')
    populations = populations.drop('NAME', axis='columns')

    populations['VAP'] = populations.drop('GEOID', axis='columns').astype(float).sum(axis='columns')
    populations['GEOID'] = populations['GEOID'].apply(lambda id : id[9:])
    
    populations = populations.loc[:, populations.columns.intersection(['GEOID', 'VAP'])]

    preprocess(precincts, populations)



if __name__ == '__main__':
    get_nevada_data()

# precincts.plot()

# precincts.merge(districts, how='inner')

# print(precincts)