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
        mggg.run(nv, utils.HAMMING_DISTANCE, rank, 1000)
    elif rank == 51:
        mggg.run(ga, utils.HAMMING_DISTANCE, rank, 1000)
    elif rank == 52:
        mggg.run(mi, utils.HAMMING_DISTANCE, rank, 1000)
        
if __name__ == '__main__':
    if not os.path.exists(f'{utils.OUTPUT_PATH}'):
        os.mkdir(f'{utils.OUTPUT_PATH}')
    main()