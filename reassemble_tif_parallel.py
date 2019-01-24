'''
generate tiff-images based on a folder of dump-files
'''

import os
import cloudpickle
import multiprocessing as mp

import numpy as np
import pandas as pd
import rasterio

from functools import partial

# ----------------
equi7_eu_crs = '+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs'
# ----------------

# read dump to list
def get_dictlist(filepath):
    '''
    get a dictionary that contains all parameter-results
        - constant parameters will be stored as single float/pixel as:
            returndict[parameter] = ['row_column', parameter-value]

        - dynamic parameters will be stored as pandas.DataFrame as:
            returndict[parameter] = pd.DataFrame(parameter-timeseries)

    dump-filenames are expected to be in the format:    'row_column.dump'

    '''
    # the parameter-values will be multiplied by this number for integer-conversion
    # ( DN = parameter * encoding )
    encoding = 200

    dump_folder, file = filepath

    if file.endswith(".dump"):
        #print('processing', file)
        dump_file = os.path.join(dump_folder, file)
        try:
            with open(dump_file, 'rb') as file_file:
                res = cloudpickle.load(file_file)

            returndict = {}
            for param, val in res.result[6].items():
                if len(np.unique(val)) == 1:
                    # if the parameter is constant, return only a single value
                    # (much faster than generating a pandas DataFrame)
                    returndict[param] = [file[:-5], res.result[6][param][0] * encoding]
                else:
                    returndict[param] = pd.DataFrame(val * encoding, res.index, columns=[file[:-5]])
            return returndict

        except Exception as e:
            print(dump_file, e)
            pass


def generate_image(result, param, ref_raster, block_size, out_path):
    '''
    generate tiffs for the whole timeseries (for dynamic parameters)
    '''

    try:
        os.mkdir(os.path.join(out_path))
    except FileExistsError:
        pass

    try:
        os.mkdir(os.path.join(out_path, param))
    except FileExistsError:
        pass

    filename = result.name.strftime("%Y-%m-%d_") + param + '.tif'
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

    indexes = np.array([cr.split('_') for cr in result.keys()], dtype=int)
    values = np.array(result.values, dtype=int)

    for val, [j, i] in zip(values, indexes):
        in_array[block_size * i:block_size * (i + 1),
                 block_size * j:block_size * (j + 1)] = val

    # write raster
    with rasterio.open(out_raster, 'w', driver='GTiff', height=in_array.shape[0],
                       width=in_array.shape[1], count=1, dtype=str(in_array.dtype),
                       compress='lzw', crs=equi7_eu_crs, transform=transform,
                       nodata=255) as dst:
        dst.write(in_array, 1)




def generate_image_const(result, param, ref_raster):
    '''
    generate a single tiff (for constant parameters)
    '''
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    indexes = np.array([res[0].split('_') for res in result], dtype=int)
    values = np.array([res[1] for res in result], dtype=int)

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



if __name__ == '__main__':

    # reference raster-image (used for geolocation)
    ref_raster=r"E:\RADAR\E051N016T1\sig0\D20160101_050131--_SIG0-----_S1AIWGRDH1VHD_124_A0105_EU010M_E051N016T1.tif"
    # the pixel-block size
    block_size=10

    # number of threads used for reading and image-generation
    mp_threads=30

    # the parent-dictionary where the dump-files are stored (all subfolders will be considered!)
    # dump_folder = r"E:\RADAR\20190116_sbdsc"
    # dump_folder = r'E:\USERS\tle\20190122_sbdsc_test'
    # dump_folder = r'D:\USERS\rq\testresults_new'
    dump_folder = r'E:\USERS\tle\20190122_sbdsc_test'

    # the path where the generated tif-files will be stored
    out_path = r'R:\Projects_work\SBDSC\data\codes\Qgis\testdata\temp\testresults_new\VSC'
    #out_path = os.path.join(dump_folder, 'tiffs')

    # ----------------------------------------------------------------------


    # generate timeseries images
    print('start of multiprocessing')
    pool = mp.Pool(mp_threads)
    print('reading files')

    # get list of files
    # this will use all files in all subfolders that end with .dump !!!)
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dump_folder):
        listOfFiles += [[dirpath, file] for file in filenames if file.endswith('.dump')]

    dictlist = pool.map_async(get_dictlist, listOfFiles)
    dictlist.wait()
    results = dictlist.get()
    print('generating images')
    for param in results[0].keys():
        if len(results[0][param]) == 2:
            # generate images of constant parameters
            generate_image_const([i[param] for i in results], param=param,
                                 ref_raster=ref_raster, out_path=out_path,
                                 block_size=block_size)
        else:
            # generate images of dynamic parameters
            results_df = pd.concat([i[param] for i in results], axis=1)
            results_df = results_df.fillna(255)
            pool.map_async(partial(generate_image, param=param,
                                 ref_raster=ref_raster, out_path=out_path,
                                 block_size=block_size),
                           [results_df.loc[index] for index in results_df.index])
    pool.close()

