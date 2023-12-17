import pymongo
import geopandas
import json
import utils

cluster = pymongo.MongoClient('mongodb+srv://winston:Ybq0Pn7jVIK1jf69@cluster0.famlsic.mongodb.net/')
db = cluster['michigan']
collection = db['geometries']

# mi_state = geopandas.read_file(f'{utils.MICHIGAN_PATH}michigan_state.zip')
# mi_state = mi_state.to_json()
# mi_state = json.loads(mi_state)

# collection.insert_one({'state': mi_state})

collection = db['ensembles']

collection = db['clusters']

collection = db['plans']


db = cluster['nevada']
collection = db['geometries']

# nv_state = geopandas.read_file(f'{utils.NEVADA_PATH}nevada_state.zip')
# nv_state = nv_state.to_json()
# nv_state = json.loads(nv_state)

# collection.insert_one({'state': nv_state})

nv_plan_geos = json.load(open('output/ensemble_0/plan_geo.json'))['geo_data']
for plan in nv_plan_geos:
    collection.insert_one(plan)

collection = db['ensembles']

# nv_ensemble = json.load(open('output/ensemble_0/ensemble.json'))
# collection.insert_one({'ensemble_0': nv_ensemble})

collection = db['clusters']

# nv_clusters = json.load(open('output/ensemble_0/clusters.json'))
# for nv_cluster in nv_clusters:
    # collection.insert_one({nv_cluster: nv_clusters[nv_cluster]})

collection = db['plans']

# nv_plans = json.load(open('output/ensemble_0/plans.json'))
# for nv_plan in nv_plans:
    # collection.insert_one({nv_plan: nv_plans[nv_plan]})

db = cluster['georgia']
collection = db['geometries']

# ga_state = geopandas.read_file(f'{utils.GEORGIA_PATH}georgia_state.zip')
# ga_state = ga_state.to_json()
# ga_state = json.loads(ga_state)

# collection.insert_one({'state': ga_state})

collection = db['ensembles']

collection = db['clusters']

collection = db['plans']
