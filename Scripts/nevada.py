# from mpi4py import MPI

import geopandas
import mggg

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# Open Data
# print("Rank ", rank)
nv = geopandas.read_file('nevada_data_processed.zip')
mggg.run(nv)

# Perform Analysis?