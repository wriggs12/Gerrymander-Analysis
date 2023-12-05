import geopandas
import pandas as pd
import utils
import maup

def preprocess(precincts, populations, districts, election_results):
    precincts = precincts.join(populations.set_index('GEOID20'), on='GEOID20')
    precincts = precincts.join(election_results.set_index('GEOID20'), on='GEOID20')
    
    precincts = precincts.to_crs(utils.PROJECTED_CRS)
    districts = districts.to_crs(utils.PROJECTED_CRS)

    precincts['NEIGHBORS'] = None

    print("Generating Precinct Neighbors...")
    for index, precinct in precincts.iterrows():
        neighbors = []
        for i, possible_neighbor in precincts.iterrows():
            if i == index:
                continue

            if (not precinct['geometry'].intersects(possible_neighbor['geometry'])):
                continue

            border_length = precinct['geometry'].intersection(possible_neighbor['geometry']).length
            if border_length > 80:
                neighbors.append(possible_neighbor['GEOID20'])

        precincts.at[index, 'NEIGHBORS'] = ', '.join(neighbors)

    print("Assigning Districts...")
    precinct_to_district_assignment = maup.assign(precincts, districts)
    precincts['DISTRICT'] = precinct_to_district_assignment

    for index, precinct in precincts.iterrows():
        neighbors = precinct.NEIGHBORS.split(', ')
        if len(neighbors) < 1:
            continue
        
        dists = {}

        for neighbor in neighbors:
            try:
                dist = precincts.iloc[precincts.index[precincts.GEOID20 == neighbor][0]].DISTRICT
                if dist not in dists.keys():
                    dists[dist] = 1
                else:
                    dists[dist] = dists[dist] + 1
            except:
                print(neighbor, ' Not Found')

        if precinct.DISTRICT not in dists.keys():
            print(precincts.iloc[index].DISTRICT)
            precincts.at[index, 'DISTRICT'] = precincts.iloc[precincts.index[precincts.GEOID20 == neighbor][0]].DISTRICT
            print(precincts.iloc[index].DISTRICT)
            print('')

    return precincts

def get_nevada_data():
    print("Reading Precinct Boundry Data...")
    precincts = geopandas.read_file(utils.NEVADA_PATH + 'precinct_geometry.zip')

    print("Reading Precinct Demographic Data...")
    demogrphics = pd.read_csv(utils.NEVADA_PATH + 'precinct_demographics.csv')

    print("Reading Precinct Election Data...")
    election_results = pd.read_csv(utils.NEVADA_PATH + 'election_results.csv')

    print("Reading District Boundry Data...")
    districts = geopandas.read_file(utils.NEVADA_PATH + 'district_geometry.zip')

    demogrphics = demogrphics.drop(['STATEFP20', 'COUNTYFP20', 'VTDST20', 'NAME20'], axis='columns')

    precincts = preprocess(precincts, demogrphics, districts, election_results)
    precincts = precincts.drop(['STATEFP20', 'STATEFP', 'COUNTYFP20', 'COUNTYFP', 'G20PRELJOR', 'G20PREIBLA', 'G20PREONON', 'GEOID', 'VTDST20', 'VTDI20', 'NAMELSAD20', 'LSAD20', 'MTFCC20', 'FUNCSTAT20', 'ALAND20', 'AWATER20', 'INTPTLAT20', 'INTPTLON20', 'TA2RACE', 'TANHOPICMB', 'TAOTHERALN', 'NAME20'], axis='columns')

    return precincts

def get_michigan_data():
    print("Reading Precinct Boundry Data...")
    precincts = geopandas.read_file(utils.MICHIGAN_PATH + 'precinct_geometry.zip')

    print("Reading Precinct Demographic Data...")
    demogrphics = pd.read_csv(utils.MICHIGAN_PATH + 'precinct_demographics.csv')

    print("Reading Precinct Election Data...")
    election_results = pd.read_csv(utils.MICHIGAN_PATH + 'election_results.csv')

    print("Reading District Boundry Data...")
    districts = geopandas.read_file(utils.MICHIGAN_PATH + 'district_geometry.zip')

    precincts = preprocess(precincts, demogrphics, districts, election_results)
    precincts = precincts.drop(['GEOID', 'VTDI20', 'NAMELSAD20', 'LSAD20', 'MTFCC20', 'ALAND20', 'AWATER20', 'INTPTLAT20', 'INTPTLON20', 'NAME20'], axis='columns')

    return precincts

def get_georiga_data():
    print("Reading Precinct Boundry Data...")
    precincts = geopandas.read_file(utils.GEORGIA_PATH + 'precinct_geometry.zip')

    print("Reading Precinct Demographic Data...")
    demogrphics = pd.read_csv(utils.GEORGIA_PATH + 'precinct_demographics.csv')

    print("Reading Precinct Election Data...")
    election_results = pd.read_csv(utils.GEORGIA_PATH + 'election_results.csv')

    print("Reading District Boundry Data...")
    districts = geopandas.read_file(utils.GEORGIA_PATH + 'district_geometry.zip')

    precincts = preprocess(precincts, demogrphics, districts, election_results)
    precincts = precincts.drop(['COUNTYFP', 'NAMELSAD20', 'LSAD20', 'MTFCC20', 'FUNCSTAT20', 'NAME20'], axis='columns')

    return precincts

if __name__ == '__main__':
    nevada_data = get_nevada_data()
    nevada_data.to_file('nevada_data_processed.shp')