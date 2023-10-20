import geopandas
import pandas as pd
import json
import utils
import matplotlib.pyplot as plt
import csv

# from numba import njit

def is_in_state(precincts, state):
    for i in range(len(precincts)):
        if (not state.contains(precincts[i], align=True).bool()):
            precincts.drop(i)
    return precincts

def get_nevada_districts():
    state = geopandas.read_file(utils.NEVADA_PATH + 'nv_state.zip')
    precincts = geopandas.read_file(utils.NEVADA_PATH + 'precincts-with-results.zip')

    print(state.geometry)
    print(precincts.geometry)

    state = state.to_crs(precincts.crs)

    print(state.crs)
    print(precincts.crs)

    precincts = is_in_state(precincts.geometry, state.geometry)

    print(precincts)

    precincts.to_file('output.shp')


# get_nevada_districts()

if __name__ == '__main__':
    get_nevada_districts()
# precincts.plot()

# precincts.merge(districts, how='inner')

# print(precincts)