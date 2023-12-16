class Plan():
    def __init__(self):
        self.plan_id = ''
        self.geo_id = 'N/A'
        self.mds_coord = []
        self.dem_pct = 0.0
        self.rep_pct = 0.0
        self.rep_dists = []
        self.dem_dists = []
        self.opp_pct = 0.0
        self.opportunity_districts = []
        self.population_data = {}
        self.area_data = {}

    def format(self):
        return {
            'id': self.plan_id,
            'geo_id': self.geo_id,
            'mds_coord': self.mds_coord,
            'dem_pct': self.dem_pct,
            'rep_pct': self.rep_pct,
            'rep_dists': self.rep_dists,
            'dem_dists': self.dem_dists,
            'opp_pct': self.opp_pct,
            'opportunity_districts': self.opportunity_districts,
            'population_data': self.population_data,
            'area_data': self.area_data
        }

class Cluster():
    def __init__(self):
        self.cluster_id = ''
        self.plan_ids = []
        self.avg_plan = ''
        self.variation = 0.0
        self.num_of_plans = 0
        self.mds_coord = []

    def format(self):
        return {
            'cluster_id': self.cluster_id,
            'num_of_plans': self.num_of_plans,
            'variation': self.variation,
            'mds_coord': self.mds_coord,
            'avg_plan': self.avg_plan,
            'plan_ids': self.plan_ids
        }
    
class Ensemble():
    def __init__(self):
        self.ensemble_id = ''
        self.avg_distance = 0.0
        self.max_opp_pct = 0.0
        self.max_rep_pct = 0.0
        self.max_dem_pct = 0.0
        self.num_of_clusters = 0
        self.num_of_plans = 0
        self.cluster_ids = []

    def format(self):
        return {
            'ensemble_id': self.ensemble_id,
            'num_of_clusters': self.num_of_clusters,
            'num_of_plans': self.num_of_plans,
            'avg_distance': self.avg_distance,
            'max_opp_pct': self.max_opp_pct,
            'max_dem_pct': self.max_dem_pct,
            'max_rep_pct': self.max_rep_pct,
            'cluster_ids': self.cluster_ids
        }