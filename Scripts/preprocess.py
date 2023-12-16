import geopandas
import pandas as pd
import utils
import maup

def preprocess(precincts): #, populations, districts, election_results):
    # precincts = precincts.join(populations.set_index('GEOID20'), on='GEOID20')
    # precincts = precincts.join(election_results.set_index('GEOID20'), on='GEOID20')
    
    precincts = precincts.to_crs(utils.PROJECTED_CRS)
    precincts['NEIGHBORS'] = None

    print("Generating Precinct Neighbors...")
    for index, precinct in precincts.iterrows():
        neighbors = precincts[precincts.geometry.touches(precinct['geometry'])].GEOID20.tolist()
        
        # neighbors = []
        # for i, possible_neighbor in precincts.iterrows():
        #     if i == index:
        #         continue

        #     if (not precinct['geometry'].intersects(possible_neighbor['geometry'])):
        #         continue

        #     border_length = precinct['geometry'].intersection(possible_neighbor['geometry']).length
        #     if border_length > 80:
        #         neighbors.append(possible_neighbor['GEOID20'])

        precincts.at[index, 'NEIGHBORS'] = ', '.join(neighbors)

    # print("Assigning Districts...")
    # precinct_to_district_assignment = maup.assign(precincts, districts)
    # precincts['DISTRICT'] = precinct_to_district_assignment

    # for index, precinct in precincts.iterrows():
    #     neighbors = precinct.NEIGHBORS.split(', ')
    #     if len(neighbors) < 1:
    #         continue
        
    #     dists = {}

    #     for neighbor in neighbors:
    #         try:
    #             dist = precincts.iloc[precincts.index[precincts.GEOID20 == neighbor][0]].DISTRICT
    #             if dist not in dists.keys():
    #                 dists[dist] = 1
    #             else:
    #                 dists[dist] = dists[dist] + 1
    #         except:
    #             print(neighbor, ' Not Found')

    #     if precinct.DISTRICT not in dists.keys():
    #         print(precincts.iloc[index].DISTRICT)
    #         precincts.at[index, 'DISTRICT'] = precincts.iloc[precincts.index[precincts.GEOID20 == neighbor][0]].DISTRICT
    #         print(precincts.iloc[index].DISTRICT)
    #         print('')

    return precincts

def get_georgia_data():
    print("Reading Precinct Boundry Data...")
    data = geopandas.read_file(utils.GEORGIA_PATH + 'GA_precincts.zip')

    data = data.drop(['PRECINCT_N', 'ID', 'CTYNAME', 'FIPS1', 'FIPS2',
                               'PRES16L', 'SEN16D', 'SEN16R', 'SEN16L', 'NH_AMIN', 'NH_NHPI',
                               'NH_OTHER', 'NH_2MORE', 'HVAP', 'WVAP', 'BVAP', 'AMINVAP', 'ASIANVAP', 'NHPIVAP', 'OTHERVAP', '2MOREVAP',
                               'HDIST', 'SEND', 'H_OTHER', 'H_2MORE', 'DISTRICT'], axis='columns')
    
    data.HISP = data.HISP + data.H_WHITE + data.H_BLACK + data.H_AMIN + data.H_ASIAN + data.H_NHPI
    data = data.drop(['H_BLACK', 'H_AMIN', 'H_ASIAN', 'H_NHPI', 'H_WHITE'], axis='columns')
    data = data.rename(columns={'PRECINCT_I': 'GEOID20', 'CD': 'DISTRICT', 'PRES16D': 'G20PREDBID', 'PRES16R': 'G20PRERTRU', 'TOTPOP': 'ADJPOP',
                                'NH_WHITE': 'TAWHITEALN', 'NH_BLACK': 'TABLACKCMB', 'NH_ASIAN': 'TAASIANCMB', 'HISP': 'TAHISPANIC'})
    
    data = preprocess(data)

    data['G20PREDBID'] = data['G20PREDBID'].astype(int)
    data['G20PRERTRU'] = data['G20PRERTRU'].astype(int)
    data['TAWHITEALN'] = data['TAWHITEALN'].astype(int)
    data['TABLACKCMB'] = data['TABLACKCMB'].astype(int)
    data['TAASIANCMB'] = data['TAASIANCMB'].astype(int)
    data['TAHISPANIC'] = data['TAHISPANIC'].astype(int)

    print(data.columns)
    print(data)

    return data

if __name__ == '__main__':
    georgia_data = get_georgia_data()
    georgia_data.to_file('georgia_data_processed.shp')


# def get_nevada_data():
#     print("Reading Precinct Boundry Data...")
#     precincts = geopandas.read_file(utils.NEVADA_PATH + 'precinct_geometry.zip')

#     print("Reading Precinct Demographic Data...")
#     demogrphics = pd.read_csv(utils.NEVADA_PATH + 'precinct_demographics.csv')

#     print("Reading Precinct Election Data...")
#     election_results = pd.read_csv(utils.NEVADA_PATH + 'election_results.csv')

#     print("Reading District Boundry Data...")
#     districts = geopandas.read_file(utils.NEVADA_PATH + 'district_geometry.zip')

#     demogrphics = demogrphics.drop(['STATEFP20', 'COUNTYFP20', 'VTDST20', 'NAME20'], axis='columns')

#     precincts = preprocess(precincts, demogrphics, districts, election_results)
#     precincts = precincts.drop(['STATEFP20', 'STATEFP', 'COUNTYFP20', 'COUNTYFP', 'G20PRELJOR', 'G20PREIBLA', 'G20PREONON', 'GEOID', 'VTDST20', 'VTDI20', 'NAMELSAD20', 'LSAD20', 'MTFCC20', 'FUNCSTAT20', 'ALAND20', 'AWATER20', 'INTPTLAT20', 'INTPTLON20', 'TA2RACE', 'TANHOPICMB', 'TAOTHERALN', 'NAME20'], axis='columns')

#     return precincts
    
# def get_michigan_data():
#     print("Reading Precinct Data...")
#     data = geopandas.read_file(utils.MICHIGAN_PATH + 'precinct_data.zip')

#     data = data.drop(['VTD2016_x', 'ShapeSTLen', 'CountyFips', 'Jurisdicti', 'ElectionYe', 'Label', 'county_nam', 'county_fip',
#                'county_lat', 'county_lon', 'jurisdic_1', 'PRES16L', 'PRES16G', 'GOV18D', 'GOV18R', 'SOS18D', 'SOS18R', 'AG18D', 'AG18R', 'SEN18D',
#                'SEN18R', 'SENDIST', 'NH_AMIN', 'NH_OTHER', 'H_OTHER', 'HVAP', 'WVAP', 'BVAP', 'AMINVAP', 'ASIANVAP', 'NHPIVAP', 'OTHERVAP',
#                '2MOREVAP', 'NH_2MORE', 'H_2MORE','NH_NHPI', 'VAP'], axis='columns')

#     # print("Reading Precinct Demographic Data...")
#     # demogrphics = pd.read_csv(utils.MICHIGAN_PATH + 'precinct_demographics.csv')

#     data.HISP = data.HISP + data.H_WHITE + data.H_BLACK + data.H_AMIN + data.H_ASIAN + data.H_NHPI
#     data = data.drop(['H_BLACK', 'H_AMIN', 'H_ASIAN', 'H_NHPI', 'H_WHITE'], axis='columns')

#     data = data.rename(columns={'VTD': 'GEOID20', 'CD': 'DISTRICT', 'PRES16D': 'G20PREDBID', 'PRES16R': 'G20PRERTRU', 'TOTPOP': 'ADJPOP',
#                                 'NH_WHITE': 'TAWHITEALN', 'NH_BLACK': 'TABLACKCMB', 'NH_ASIAN': 'TAASIANCMB', 'HISP': 'TAHISPANIC'})

#     # print("Reading Precinct Election Data...")
#     # election_results = pd.read_csv(utils.MICHIGAN_PATH + 'election_results.csv')

#     # print("Reading District Boundry Data...")
#     # districts = geopandas.read_file(utils.MICHIGAN_PATH + 'district_geometry.zip')

#     data = preprocess(data)
#     # precincts = precincts.drop(['GEOID', 'VTDI20', 'NAMELSAD20', 'LSAD20', 'MTFCC20', 'ALAND20', 'AWATER20', 'INTPTLAT20', 'INTPTLON20', 'NAME20'], axis='columns')

#     return data