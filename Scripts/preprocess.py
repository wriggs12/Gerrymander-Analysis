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

    print(precincts.columns)
    print(precincts['GEOID10'])

def get_nevada_data():
    precincts = geopandas.read_file(utils.NEVADA_PATH + 'tl_rd22_32_bg.zip')
    populations = pd.read_csv(utils.NEVADA_PATH + 'NV_POPULATION_DATA.csv')

    preprocess(precincts, populations)
    
    # print(districts.columns)
    # print(precincts)

    # precincts.plot()
    # plt.show()

    # districts.plot()
    # plt.show()



if __name__ == '__main__':
    get_nevada_data()

# precincts.plot()

# precincts.merge(districts, how='inner')

# print(precincts)