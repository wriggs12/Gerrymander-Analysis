import pymongo
import geopandas
import json
import utils
import matplotlib.pyplot as plt

cluster = pymongo.MongoClient('mongodb+srv://winston:Ybq0Pn7jVIK1jf69@cluster0.famlsic.mongodb.net/')

## MICHIGAN ##
# db = cluster['michigan']
# collection = db['geometries']

# mi_plan_geos = json.load(open('output/ensemble_70/plan_geo.json'))['geo_data']
# for plan in mi_plan_geos:
#     key = list(plan.keys())[0]
    
#     gdf = geopandas.GeoDataFrame.from_features(plan[key])
#     gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)

#     gdf.plot()
#     plt.show()
#     data = gdf.to_json()
#     data = json.loads(data)

#     collection.insert_one({'properties': {'id': key, 'geometry': data}})

# collection = db['ensembles']

# mi_ensemble = json.load(open('output/ensemble_70/ensemble.json'))
# collection.insert_one(mi_ensemble)

# collection = db['clusters']

# mi_clusters = json.load(open('output/ensemble_70/clusters.json'))
# for mi_cluster in mi_clusters:
#     collection.insert_one({'properties': mi_clusters[mi_cluster]})

# collection = db['plans']

# mi_plans = json.load(open('output/ensemble_70/plans.json'))
# for mi_plan in mi_plans:
#     collection.insert_one({'properties': mi_plans[mi_plan]})


## NEVADA ##
# db = cluster['nevada']
# collection = db['geometries']

# nv_plan_geos = json.load(open('output/ensemble_69/plan_geo.json'))['geo_data']
# for plan in nv_plan_geos:
#     key = list(plan.keys())[0]
    
#     gdf = geopandas.GeoDataFrame.from_features(plan[key])
#     gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)

#     data = gdf.to_json()
#     data = json.loads(data)

#     collection.insert_one({'properties': {'id': key, 'geometry': data}})

# collection = db['ensembles']

# nv_ensemble = json.load(open('output/ensemble_65/ensemble.json'))
# collection.insert_one(nv_ensemble)

# collection = db['clusters']

# nv_clusters = json.load(open('output/ensemble_69/clusters.json'))
# for nv_cluster in nv_clusters:
#     collection.insert_one({'properties': nv_clusters[nv_cluster]})

# collection = db['plans']

# nv_plans = json.load(open('output/ensemble_69/plans.json'))
# for nv_plan in nv_plans:
#     collection.insert_one({'properties': nv_plans[nv_plan]})


## GEORGIA ##
# db = cluster['georgia']
# collection = db['geometries']

# ga_plan_geos = json.load(open('output/ensemble_51/plan_geo.json'))['geo_data']
# for plan in ga_plan_geos:
#     key = list(plan.keys())[0]
    
#     gdf = geopandas.GeoDataFrame.from_features(plan[key])
#     gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)

#     data = gdf.to_json()
#     data = json.loads(data)

#     collection.insert_one({'properties': {'id': key, 'geometry': data}})

# collection = db['ensembles']

# ga_ensemble = json.load(open('output/ensemble_51/ensemble.json'))
# collection.insert_one(ga_ensemble)

# collection = db['clusters']

# ga_clusters = json.load(open('output/ensemble_19/clusters.json'))
# for ga_cluster in ga_clusters:
#     collection.insert_one({'properties': ga_clusters[ga_cluster]})

# collection = db['plans']

# ga_plans = json.load(open('output/ensemble_19/plans.json'))
# for ga_plan in ga_plans:
#     collection.insert_one({'properties': ga_plans[ga_plan]})
