import geopandas
import mggg
import utils
import os
import sys

def main():
    state = sys.argv[1]
    rand_key = int(sys.argv[2])
    ensemble_size = int(sys.argv[3])

    print("Reading Data...")
    match state:
        case 'nv':
            nv = geopandas.read_file('Data/nevada_data_processed.zip')
            mggg.run(nv, utils.HAMMING_DISTANCE, rand_key, ensemble_size)
        case 'mi':
            mi = geopandas.read_file('Data/michigan_data_processed.zip')
            mggg.run(mi, utils.HAMMING_DISTANCE, rand_key, ensemble_size)
        case 'ga':
            ga = geopandas.read_file('Data/georgia_data_processed.zip')
            mggg.run(ga, utils.HAMMING_DISTANCE, rand_key, ensemble_size)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("USAGE: main.py <nv | mi | ga> <random key> <ensemble size>")
        os._exit(1)

    if not os.path.exists(f'{utils.OUTPUT_PATH}'):
        os.mkdir(f'{utils.OUTPUT_PATH}')
    main()
