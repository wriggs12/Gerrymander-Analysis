import geopandas
import mggg
import utils
import os
import sys

def main():
    print("Reading Data...")
    nv = geopandas.read_file('nevada_data_processed.zip')
    # ga = geopandas.read_file('georgia_data_processed.zip')
    mi = geopandas.read_file('michigan_data_processed.zip')

    rank = int(sys.argv[1])
    if rank == 0:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 10000)
    elif rank == 1:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 7500)
    elif rank == 2:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 5000)
    elif rank == 3:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 2000)
    elif rank == 4:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 1000)
    elif rank == 5:
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 500)
    elif rank == 6:
        mggg.run(nv, utils.OPTIMAL_TRANSPORT, rank, 250)
    # case 7:
    #     mggg.run(ga, utils.HAMMING_DISTANCE, rank, 10000)
    # case 8:
    #     mggg.run(ga, utils.HAMMING_DISTANCE, rank, 7500)
    # case 9:
    #     mggg.run(ga, utils.HAMMING_DISTANCE, rank, 5000)
    # case 10:
    #     mggg.run(ga, utils.HAMMING_DISTANCE, rank, 2000)
    # case 11:
    #     mggg.run(ga, utils.HAMMING_DISTANCE, rank, 1000)
    # case 12:
    #     mggg.run(ga, utils.HAMMING_DISTANCE, rank, 500)
    # case 13:
    #     mggg.run(ga, utils.OPTIMAL_TRANSPORT, rank, 250)
    elif rank == 7:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 10)
    elif rank == 8:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 7500)
    elif rank == 9:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 5000)
    elif rank == 10:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 2000)
    elif rank == 11:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 1000)
    elif rank == 12:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 500)
    elif rank == 13:
        mggg.run(mi, utils.OPTIMAL_TRANSPORT, rank, 250)
        
if __name__ == '__main__':
    if not os.path.exists(f'{utils.OUTPUT_PATH}'):
        os.mkdir(f'{utils.OUTPUT_PATH}')
    main()