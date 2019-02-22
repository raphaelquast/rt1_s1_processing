'''
generate tiff-images based on csv-files
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd
import rasterio


# ----------------
equi7_eu_crs = '+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs'
# ----------------


def splitx(s):
    c, r = np.array(s.split('_'), dtype=int)
    return c, r

def get_values_index(filepath, param, encoding):
    csv_data = pd.read_csv(filepath)
    csv_data['id'] = csv_data['id'].apply(splitx)
    indexes = csv_data['id'].values
    values = csv_data[param].values*encoding
    return values, indexes

def generatetiff_TS(csvfilename, csv_path, param, ref_raster,
                    block_size, out_path, encoding=200):
    print(f"processing {param}: {csvfilename.replace('.csv', '')}")
    filepath = os.path.join(csv_path, param, csvfilename)
    values, indexes = get_values_index(filepath, param, encoding)

    filename = csvfilename.replace('.csv', '') + '.tif'
    out_raster = os.path.join(out_path, param, filename)
    try:
        # open raster
        dataset = rasterio.open(out_raster)
        first_time_write = False
    except:
        dataset = rasterio.open(ref_raster)
        first_time_write = True

    transform = dataset.transform
    # read band 1
    in_array = dataset.read(1)

    if first_time_write:
        in_array.fill(255)

    for val, [j, i] in zip(values, indexes):
        in_array[block_size * i:block_size * (i + 1),
                 block_size * j:block_size * (j + 1)] = val

    # write raster
    with rasterio.open(out_raster, 'w', driver='GTiff', height=in_array.shape[0],
                       width=in_array.shape[1], count=1, dtype=str(in_array.dtype),
                       compress='lzw', crs=equi7_eu_crs, transform=transform,
                       nodata=255) as dst:
        dst.write(in_array, 1)

def generate_image_from_csv(csv_path, param, ref_raster, block_size, out_path,
                            encoding=200, mp_threads=1):
    '''
    generate a tiff based on the input from a csv-file
    '''
    if param == 'all':
        params = []
        for i in os.listdir(csv_path):
            if i.endswith('.csv'):
                params += [i.replace('.csv', '')]
            else:
                params += [i]
    elif isinstance(param, list):
        params = param
    else:
        params = [param]

    for param in params:
        # check if a file named "param.csv" exists
        if os.path.isfile(os.path.join(csv_path, param + '.csv')):
            print(f'generating tif for constant parameter {param} ')
            if not os.path.exists(out_path):
                os.mkdir(out_path)

            filepath = os.path.join(csv_path, param + '.csv')
            values, indexes = get_values_index(filepath, param, encoding)
            filename = param + '.tif'
            out_raster = os.path.join(out_path, filename)
            try:
                # open raster
                dataset = rasterio.open(out_raster)
                first_time_write = False
            except:
                dataset = rasterio.open(ref_raster)
                first_time_write = True

            transform = dataset.transform
            # read band 1
            in_array = dataset.read(1)

            if first_time_write:
                in_array.fill(255)

            for val, [j, i] in zip(values, indexes):
                in_array[block_size * i:block_size * (i + 1),
                         block_size * j:block_size * (j + 1)] = val

            # write raster
            with rasterio.open(out_raster, 'w', driver='GTiff', height=in_array.shape[0],
                               width=in_array.shape[1], count=1, dtype=str(in_array.dtype),
                               compress='lzw', crs=equi7_eu_crs, transform=transform,
                               nodata=255) as dst:
                dst.write(in_array, 1)
        # check if a folder named "param" exists containing csv-files
        elif os.path.exists(os.path.join(csv_path, param)) and os.listdir(os.path.join(csv_path, param))[0].endswith('.csv'):
            print(f'generating tif-timeseries for parameter {param} on {mp_threads} cores')

            if not os.path.exists(out_path):
                os.mkdir(out_path)
            if not os.path.exists(os.path.join(out_path, param)):
                os.mkdir(os.path.join(out_path, param))


            csvfilenames = os.listdir(os.path.join(csv_path, param))

            parallelfunc = partial(generatetiff_TS, csv_path=csv_path, param=param,
                             ref_raster=ref_raster, block_size=block_size,
                             out_path=out_path, encoding=encoding)

            pool = mp.Pool(mp_threads)
            pool.map(parallelfunc, csvfilenames)
            pool.close()


# %%
if __name__ == '__main__':

    # reference raster-image (used for geolocation)
    ref_raster=r"E:\RADAR\E051N016T1\sig0\D20160101_050131--_SIG0-----_S1AIWGRDH1VHD_124_A0105_EU010M_E051N016T1.tif"
    # the pixel-block size
    block_size=1

    # number of threads used for generation of timeseries-tiffs
    mp_threads=10
    # the parent-path where the csv-files are stored
    # (timeseries are expected to be stored in a subfolder named after the parameter)
    csv_path = r"D:\USERS\rq\delete_me\csv_output"
    param = ['v2', 'omega', 'frac']

    # the path where the generated tif-files will be stored
    out_path = r"D:\USERS\rq\delete_me\tif"
    out_path = r"H:\Projects_Visits_Etc\S1_processing\tif"
    # the multiplier for converting floats to integers
    encoding=200

    # ----------------------------------------------------------------------
    # generate timeseries images
    generate_image_from_csv(csv_path, param, ref_raster, block_size, out_path,
                            encoding, mp_threads)
