import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from ImageStack import create_imagestack_dataset
from datetime import datetime
import pandas as pd
import numpy as np
from rt1_processing_funcs_juhuu import inpdata_inc_average
import random, string
import multiprocessing as mp
from common import get_worker_id,chunkIt, parse_args, read_cfg, parallelfunc
from common import get_processed_list, get_str_occ
import rasterio
from rasterio.mask import mask
import fiona
from common import prepare_index_array

def read_data(sig0_dir, plia_dir, block_size, feature_list, output_dir, ndvi_dir=None, orbit_direction='',
              tif_size=10000, ref_image='/home/tle/temp/E044N021T1_ref.tif', shp_file_base=''):
    '''
    Read sig0 and plia rasters stack into a virtual raster stack, then read the time-series block-by block
    Parameters
    ----------
    sig0_dir: str
        directory of sig0 tif images
    plia_dir: str
        directory of plia tif images
    block_size: int
        block size: the size of the block. e.g if you would like to take 100x100m block, blocksize is 10
    Returns
    -------
    '''

    random_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

    # read ndvi stack if ndvi_dir exist
    if ndvi_dir:
        files_ndvi = os.listdir(ndvi_dir)
        filelist_ndvi = []
        times_ndvi = []
        for fil in files_ndvi:
            if 'NDVI300' in fil and fil.endswith('.tif'):
                filelist_ndvi.append(os.path.join(ndvi_dir, fil))
                times_ndvi.append(datetime.strptime(fil[19:27], "%Y%m%d"))
    else:
        ndvi_stack = None

    # prepare timelist and filelist for sig0 reader
    files_sig0 = os.listdir(sig0_dir)
    filelist_sig0 = []
    times_sig0 = []
    sig0_pattern = 'IWGRDH1VV%s' % orbit_direction

    for fil in files_sig0:
        if 'SIG' in fil and sig0_pattern in fil and fil.endswith('.tif'):
            filelist_sig0.append(os.path.join(sig0_dir, fil))
            times_sig0.append(datetime.strptime(fil[1:16], "%Y%m%d_%H%M%S"))

    # prepare timelist and filelist for plia reader
    files_plia = os.listdir(plia_dir)
    filelist_plia = []
    times_plia = []
    plia_pattern = 'IWGRDH1--%s' % orbit_direction

    for fil in files_plia:
        if 'PLIA' in fil and plia_pattern in fil and fil.endswith('.tif'):
            filelist_plia.append(os.path.join(plia_dir, fil))
            times_plia.append(datetime.strptime(fil[1:16], "%Y%m%d_%H%M%S"))

    # raise a warning if number of sig0 and plia is not equal
    if len(times_sig0) != len(times_plia):
        print("Warning! The number of sig0 and plia images is not equal")

    # check if there is any files in sig0 which the correspond plia doesn't exist
    for time in times_sig0:
        if time not in times_plia:
            idx_sig0_no_plia = times_sig0.index(time)
            del times_sig0[idx_sig0_no_plia]
            del filelist_sig0[idx_sig0_no_plia]
            print("Warning! The scene in this date ", time, " doesn't have the correspond plia. Removing...")

    # sort those lists. After sorting, times_plia and times_sig0 must be equal!
    filelist_plia.sort()
    times_plia.sort()
    filelist_sig0.sort()
    times_sig0.sort()

    # check index
    for time in times_sig0:
        idx = times_sig0.index(time)
        if times_sig0[idx] != times_plia[idx] or os.path.basename(filelist_sig0[idx])[1:16] != os.path.basename(
                filelist_plia[idx])[1:16]:
            raise ValueError("index is wrong!")


    # read ndvi to virtual stack
    if ndvi_dir:
        ndvi_stack = create_imagestack_dataset(name=random_name + 'ndvi', filelist=filelist_ndvi, times=times_ndvi,
                                               nodata=-9999)

    # read sig0 to virtual stack
    sig0_stack = create_imagestack_dataset(name=random_name + 'SIG', filelist=filelist_sig0, times=times_sig0,
                                          nodata=-9999)
    # read plia to virtual stack
    plia_stack = create_imagestack_dataset(name=random_name + 'LIA', filelist=filelist_plia, times=times_plia,
                                           nodata=-9999)
    # read sig and plia blocks

    for count, feature in enumerate(feature_list):
        feature_geometry = [feature["geometry"]]
        feature_id = feature["id"]

        # etablish mask array
        with rasterio.open(ref_image) as raster:
            out_image, out_transform = mask(raster, feature_geometry, crop=False, all_touched=False)

        # this is the mask array 10000x10000, value = 1 at feature coverage
        mask_array = out_image[0]
        from numpy import argwhere

        # find the bounding box (for read_ts)
        try:
            loc_feature = argwhere(mask_array == 1)
            (cstart, rstart), (cstop, rstop) = loc_feature.min(0), loc_feature.max(0)
        except Exception as e:
            print(e, feature_id)
            continue
        # incease by one to cover the whole feature
        cstop += 1
        rstop += 1

        # in_array[cstart:cstop, rstart:rstop] = 1
        bbox_array = mask_array[cstart:cstop, rstart:rstop]


        # read 3d array based on bounding box location above
        time_sig0_list, data_sig0_list = sig0_stack.read_ts(int(rstart), int(cstart), int(rstop - rstart),
                                        int(cstop - cstart))
        time_plia_list, data_plia_list = plia_stack.read_ts(int(rstart), int(cstart), int(rstop - rstart),
                                        int(cstop - cstart))


        # create 3d mask for new 3d array
        arr3d_mask = np.zeros(data_sig0_list.shape, dtype=bool)
        arr3d_mask[:, :, :] = bbox_array[np.newaxis, :, :] == 255
        sig0_value_masked = np.ma.array(data_sig0_list, mask=arr3d_mask)
        plia_value_masked = np.ma.array(data_plia_list, mask=arr3d_mask)

        # change -9999 to masked, loop through time slice
        for i in range(sig0_value_masked.shape[0]):
            sig0_value_masked[i].mask[sig0_value_masked[i].data == -9999] = True
            plia_value_masked[i].mask[plia_value_masked[i].data == -9999] = True


        px_sig0 = sig0_value_masked.filled(-9999) # todo remove hardcode here
        px_plia = plia_value_masked.filled(-9999) # todo remove hardcode here
        # prepare index array
        px_time = prepare_index_array(time_sig0_list, px_sig0)

        sig0_plia_stack = np.vstack((px_sig0.flatten(), px_plia.flatten()))
        df = pd.DataFrame(data=sig0_plia_stack.transpose(),
                          index=px_time.flatten(),
                          columns=['sig', 'inc'])

        # nan handling
        df = df.replace(-9999, np.nan)

        df = df.dropna()

        # true value
        df['sig'] = df['sig'].div(100)
        df['inc'] = df['inc'].div(100)

        # convert to radian
        df['inc'] = np.deg2rad(df['inc'])

        try:
            df = inpdata_inc_average(df)
        except Exception as e:
            print(get_worker_id(), "inpdata_inc_average failed!", e)
            return

        #TODO: NDVI
        if ndvi_stack:
            print('read ndvi stack...:', datetime.now())
            time_ndvi_list, data_ndvi_list = ndvi_stack.read_ts(int(rstart), int(cstart), int(rstop - rstart),
                                                                int(cstop - cstart))

            # create 3d mask for new 3d array
            arr3d_mask_ndvi = np.zeros(data_ndvi_list.shape, dtype=bool)
            arr3d_mask_ndvi[:, :, :] = bbox_array[np.newaxis, :, :] == 255
            ndvi_value_masked = np.ma.array(data_ndvi_list, mask=arr3d_mask_ndvi)

            # change -9999 to masked, loop through time slice
            for i in range(ndvi_value_masked.shape[0]):
                ndvi_value_masked[i].mask[ndvi_value_masked[i].data == -9999] = True


            px_ndvi = ndvi_value_masked.filled(-9999)  # todo remove hardcode here

            # prepare index array
            px_time_ndvi = prepare_index_array(time_ndvi_list, px_ndvi)

            df_ndvi = pd.DataFrame(data=px_ndvi.flatten(),
                                   index=px_time_ndvi.flatten(),
                                   columns=['ndvi'])

            # nan handling
            df_ndvi = df_ndvi.replace(list(range(251, 256)), np.nan)

            df_ndvi = df_ndvi.dropna()

            # convert to physical value
            df_ndvi['ndvi'] = df_ndvi['ndvi'] / 250 - 0.08

            # average daily
            df_ndvi = df_ndvi.groupby(df_ndvi.index).mean()
        else:
            df_ndvi = None


        out_dict = {'dataset': df, 'df_ndvi': df_ndvi, '_fnevals_input': None,
                    'c':shp_file_base, 'r': str(feature_id), 'outdir': output_dir}

        parallelfunc(out_dict)

        # equi7_eu_crs = '+proj=aeqd +lat_0=53 +lon_0=24 +x_0=5837287.81977 +y_0=2121415.69617 +datum=WGS84 +units=m +no_defs'
        # out_raster='/home/tle/temp/testxxx.tif'
        # in_array=out_image[0]
        # with rasterio.open(out_raster, 'w', driver='GTiff', height=in_array.shape[0],
        #                    width=in_array.shape[1], count=1, dtype=str(in_array.dtype),
        #                    compress='lzw',
        #                    crs=equi7_eu_crs,
        #                    transform=out_transform, nodata=255) as dst:
        #     dst.write(in_array, 1)

        print('pass')



def main(args, test_vsc_param=False):
    '''
    Main entry point to start the processor
    Parameters
    ----------
    args: list
        command line arguments
    test_vsc_param: bool
        if set, the processing will return the settings
    Returns
    -------
    '''

    args = parse_args(args)
    cfg = read_cfg(args.cfg_file)
    sig0_dir = cfg['PATH']['sig0_dir']
    plia_dir = cfg['PATH']['plia_dir']
    out_dir = cfg['PATH']['out_dir']
    ndvi_dir = cfg['PATH']['ndvi_dir']
    if ndvi_dir == 'None':
        ndvi_dir = None

    shp_file = cfg['PARAMETER']['shp_file']
    shp_file_base = os.path.basename(shp_file).split('.')[0]
    ref_image = cfg['PARAMETER']['ref_image']
    tif_size = int(cfg['PARAMETER']['tif_size'])
    block_size = int(cfg['PARAMETER']['block_size'])
    mp_threads = int(cfg['PARAMETER']['mp_threads'])
    orbit_direction = cfg['PARAMETER']['orbit_direction']
    if orbit_direction == 'None':
        orbit_direction = ''

    # take total array number and array number from arg
    if args.arraynumber:
        arr_number = int(args.arraynumber)
    else:
        raise Warning('-arraynumber missing')

    if args.totalarraynumber:
        total_arr_number = int(args.totalarraynumber)
    else:
        raise Warning('-totalarraynumber missing')

    pixels_per_side = int(tif_size / block_size)



    # TODO modify from here:
    # loop through features
    feature_list = []
    with fiona.open(shp_file, "r") as shapefile:
        no_features = len(shapefile)
        print('processing', no_features, 'features of ', shp_file_base)
        for feature in shapefile:
            # only select the feature which have number of fields more than min_num_field
            feature_list.append(feature)

    # todo: skip processed feature
    # prepare processing dicts with list of features

    # distributing to VSC computing nodes, if run on single machine, total_arr_number should be 1
    list_to_process_all = chunkIt(feature_list, total_arr_number)  # list of 5 lists, each list 200 line (1000/5)
    list_to_process_this_node = list_to_process_all[arr_number - 1]  # e.g array number 1 take list_to_process_all[0]

    # divide list_procss_this_node into a list of lists
    list_to_process_this_node = chunkIt(list_to_process_this_node, 50) # TODO remove hardcode here

    if test_vsc_param:
        # # print out test parameters
        # print('sig0_dir', sig0_dir)
        # print('plia_dir', plia_dir)
        # print('block_size', block_size)
        # print('out_dir', out_dir)
        # print('mp_threads', mp_threads)
        # # print('cr_list', list_to_process_node)
        # print('len_all (total px)', len(list_all_lines))
        # print('number of cr_list (must equal arraytotalnumber)', len(list_to_process_all))
        # print('totalarraynumber', total_arr_number)
        # print('arraynumber', arr_number)
        # print('len_cr_list (px for this node)', len(list_to_process_this_node))
        print('function is deprecated!')
        pass

    else:
        # single thread processing
        if mp_threads in [0, 1]:
            print("single thread processing")
            read_data(sig0_dir=sig0_dir,
                      plia_dir=plia_dir,
                      block_size=block_size,
                      feature_list=feature_list,
                      output_dir=out_dir,
                      ndvi_dir=ndvi_dir,
                      orbit_direction=orbit_direction,
                      tif_size=tif_size,
                      ref_image=ref_image,
                      shp_file_base=shp_file_base)
        else:
            # multiprocessing
            process_list = []
            # chunk the line list into smaller lists for processing
            # list_to_process_node_chunked = chunkIt(list_to_process_this_node, mp_threads * 2)
            for feature_list_node in list_to_process_this_node:
                process_dict = {}
                process_dict['sig0_dir'] = sig0_dir
                process_dict['plia_dir'] = plia_dir
                process_dict['block_size'] = block_size
                process_dict['feature_list'] = feature_list_node
                process_dict['output_dir'] = out_dir
                process_dict['ndvi_dir'] = ndvi_dir
                process_dict['orbit_direction'] = orbit_direction
                process_dict['tif_size'] = tif_size
                process_dict['ref_image'] = ref_image
                process_dict['shp_file_base'] = shp_file_base
                process_list.append(process_dict)

            print("Node:", arr_number, "/", total_arr_number, "start the MP...:", datetime.now())
            print('Target: process ', len(process_list), 'lines...')
            pool = mp.Pool(mp_threads)
            pool.map(read_data_mp, process_list)
            pool.close()


def read_data_mp(process_dict):
    read_data(sig0_dir=process_dict['sig0_dir'],
              plia_dir=process_dict['plia_dir'],
              block_size=process_dict['block_size'],
              feature_list=process_dict['line'],
              output_dir=process_dict['output_dir'],
              ndvi_dir=process_dict['ndvi_dir'],
              orbit_direction=process_dict['orbit_direction'],
              tif_size=process_dict['tif_size'])


if __name__ == '__main__':
    import sys

    # comment those lines if you're working on the VSC
    sys.argv.append("config/config_4dformat_tle.ini")
    sys.argv.append("-totalarraynumber")
    sys.argv.append("1")
    sys.argv.append("-arraynumber")
    sys.argv.append("1")

    print("-------------START------------", datetime.now())
    main(sys.argv[1:], test_vsc_param=False)

    pass