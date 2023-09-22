import pymongo
import geopandas

cluster = pymongo.MongoClient('mongodb+srv://winston:Ybq0Pn7jVIK1jf69@cluster0.famlsic.mongodb.net/')

ga_cd = geopandas.read_file('./Georgia/ga_congressional_districts.zip')
ga_cd = ga_cd.to_json()

db = cluster['georgia']
collection = db['districts']
collection.insert_one({'2022_districts': ga_cd})

mi_cd = geopandas.read_file('./Michigan/mi_congressional_districts.zip')
mi_cd = mi_cd.to_json()

db = cluster['michigan']
collection = db['districts']
collection.insert_one({'2022_districts': mi_cd})

nv_sa = geopandas.read_file('./Nevada/NV_SA_Districts.zip')
nv_sa = nv_sa.to_json()

db = cluster['nevada']
collection = db['districts']
collection.insert_one({'2022_districts': nv_sa})
