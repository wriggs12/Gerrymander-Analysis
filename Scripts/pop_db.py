import pymongo
import geopandas
import json

cluster = pymongo.MongoClient('mongodb+srv://winston:Ybq0Pn7jVIK1jf69@cluster0.famlsic.mongodb.net/')
db = cluster['georgia']
collection = db['districts']

ga_cd = geopandas.read_file('./Georgia/districts/ga_congressional_districts.zip')
ga_cd = ga_cd.to_json()
ga_cd = json.loads(ga_cd)

# collection.insert_one({'2022_districts': ga_cd})


# db = cluster['georgia']
# collection = db['districts']
# collection.insert_one({'2022_districts': ga_cd})

# mi_cd = geopandas.read_file('./Michigan/mi_congressional_districts.zip')
# mi_cd = mi_cd.to_json()
# mi_cd = json.loads(mi_cd)

# db = cluster['michigan']
# collection = db['districts']
# collection.insert_one({'2022_districts': mi_cd})

# nv_sa = geopandas.read_file('./Nevada/districts/NV_SA_Districts.zip')
# nv_sa = nv_sa.to_json()
# nv_sa = json.loads(nv_sa)

# db = cluster['nevada']
# collection = db['districts']
# collection.insert_one({'2022_districts': nv_sa})
