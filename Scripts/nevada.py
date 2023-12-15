from mpi4py import MPI

import geopandas
import mggg
import utils
import os

def main():
    # Run with `mpiexec -n # python ./nevada.py`
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nv = geopandas.read_file('nevada_data_processed.zip')
    # ga = geopandas.read_file('georgie_data_processed.zip')
    # mi = geopandas.read_file('michigan_data_processed.zip')

    match rank:
        case 0:
            mggg.run(nv, utils.HAMMING_DISTANCE, rank, 10)
        case 1:
            mggg.run(nv, utils.HAMMING_DISTANCE, rank, 20)

    # mggg.run(nv, utils.OPTIMAL_TRANSPORT, rank, 250)
    # mggg.run(nv, utils.HAMMING_DISTANCE, rank, 5000)
    # mggg.run(nv, utils.HAMMING_DISTANCE, rank, 1000)
    # mggg.run(nv, utils.HAMMING_DISTANCE, rank, 500)
    # mggg.run(nv, utils.ENTROPY_DISTANCE, rank, 5000)
    # mggg.run(nv, utils.ENTROPY_DISTANCE, rank, 1000)
    # mggg.run(nv, utils.ENTROPY_DISTANCE, rank, 500)

    # mggg.run(ga, utils.OPTIMAL_TRANSPORT, rank, 250)
    # mggg.run(ga, utils.HAMMING_DISTANCE, rank, 5000)
    # mggg.run(ga, utils.HAMMING_DISTANCE, rank, 1000)
    # mggg.run(ga, utils.HAMMING_DISTANCE, rank, 500)
    # mggg.run(ga, utils.ENTROPY_DISTANCE, rank, 5000)
    # mggg.run(ga, utils.ENTROPY_DISTANCE, rank, 1000)
    # mggg.run(ga, utils.ENTROPY_DISTANCE, rank, 500)

    # mggg.run(mi, utils.OPTIMAL_TRANSPORT, rank, 250)
    # mggg.run(mi, utils.HAMMING_DISTANCE, rank, 5000)
    # mggg.run(mi, utils.HAMMING_DISTANCE, rank, 1000)
    # mggg.run(mi, utils.HAMMING_DISTANCE, rank, 500)
    # mggg.run(mi, utils.ENTROPY_DISTANCE, rank, 5000)
    # mggg.run(mi, utils.ENTROPY_DISTANCE, rank, 1000)
    # mggg.run(mi, utils.ENTROPY_DISTANCE, rank, 500)

if __name__ == '__main__':
    if not os.path.exists(f'{utils.OUTPUT_PATH}'):
        os.mkdir(f'{utils.OUTPUT_PATH}')
    main()