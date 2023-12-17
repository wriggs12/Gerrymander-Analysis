import geopandas
import mggg
import utils
import os
import sys

def main():
    print("Reading Data...")
    nv = geopandas.read_file('nevada_data_processed.zip')
    ga = geopandas.read_file('georgia_data_processed.zip')
    mi = geopandas.read_file('michigan_data_processed.zip')

    rank = int(sys.argv[1])
    if rank == 50:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 500)
    elif rank == 51:
        mggg.run(ga, utils.HAMMING_DISTANCE, rank, 1000)
    elif rank == 52:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 1000)
    elif rank == 60:
        mggg.run(nv, utils.OPTIMAL_TRANSPORT, rank, 250)
    elif rank == 61:
        mggg.run(ga, utils.OPTIMAL_TRANSPORT, rank, 250)
    elif rank == 62:
        mggg.run(mi, utils.OPTIMAL_TRANSPORT, rank, 250)
    elif rank == 65:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 500)
    elif rank == 66:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 1000)

if __name__ == '__main__':
    if not os.path.exists(f'{utils.OUTPUT_PATH}'):
        os.mkdir(f'{utils.OUTPUT_PATH}')
    main()