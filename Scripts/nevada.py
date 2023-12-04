# from mpi4py import MPI

import geopandas
import mggg

# Run with `mpiexec -n # python ./nevada.py`
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

nv = geopandas.read_file('nevada_data_processed.zip')
mggg.run(nv)
