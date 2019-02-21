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
from common import get_worker_id, move_dir, make_tmp_dir, chunkIt, parse_args, read_cfg, parallelfunc
from common import get_processed_list, get_str_occ, prepare_index_array


def prepare_data_mp(input_dict):
    prepare_data(time_sig0_list=input_dict['time_sig0_list'],
                 data_sig0_list=input_dict['data_sig0_list'],
                 data_plia_list=input_dict['data_plia_list'],
                 time_ndvi_list=input_dict['time_ndvi_list'],
                 data_ndvi_list=input_dict['data_ndvi_list'],
                 col=input_dict['col'],
                 row=input_dict['row'],
                 block_size=input_dict['block_size'],
                 out_dir=input_dict['out_dir'])


def prepare_data(time_sig0_list, data_sig0_list, data_plia_list, time_ndvi_list, data_ndvi_list,
                 col, row, block_size, out_dir):
    print(get_worker_id(), 'preparing the data for col', col, 'row', row, datetime.now())

    # convert plia, sig0 into df
    px_sig0 = data_sig0_list[:, :, col * block_size:(col + 1) * block_size]
    px_plia = data_plia_list[:, :, col * block_size:(col + 1) * block_size]

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

    # -----------------------------------------

    if time_ndvi_list:
        px_ndvi = data_ndvi_list[:, :, col * block_size:(col + 1) * block_size]

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
                'c': col, 'r': row, 'outdir': out_dir}

    parallelfunc(out_dict)


def read_stack_line(sig0_dir, plia_dir, block_size, line_list, output_dir, ndvi_dir=None, orbit_direction='',
                    processed=[], tif_size=10000, mp_threads=10):
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
    times_sig0_new = []
    times_plia_new = []
    filelist_sig0_new = []
    filelist_plia_new = []
    for time_sig0 in times_sig0:
        for time_plia in times_plia:
            if time_sig0 == time_plia:
                # add to new lists
                idx_sig0 = times_sig0.index(time_sig0)
                idx_plia = times_plia.index(time_plia)
                times_sig0_new.append(times_sig0[idx_sig0])
                filelist_sig0_new.append(filelist_sig0[idx_sig0])
                times_plia_new.append(times_sig0[idx_plia])
                filelist_plia_new.append(filelist_sig0[idx_plia])

    times_sig0 = times_sig0_new
    filelist_sig0 = filelist_sig0_new
    times_sig0_new = None
    filelist_sig0_new = None

    times_plia = times_plia_new
    filelist_plia = filelist_plia_new
    times_plia_new = None
    filelist_plia_new = None

    # sort those lists. After sorting, times_plia and times_sig0 must be equal!
    filelist_plia.sort()
    times_plia.sort()
    filelist_sig0.sort()
    times_sig0.sort()

    if times_sig0 != times_plia:
        raise ValueError("times_sig0 must be equal to times_plia")

    # check index
    for time in times_sig0:
        idx = times_sig0.index(time)
        if times_sig0[idx] != times_plia[idx] or os.path.basename(filelist_sig0[idx])[1:16] != os.path.basename(
                filelist_plia[idx])[1:16]:
            raise ValueError("index is wrong!")

    # read ndvi to virtual stack, write the vrt to VSC ram disk
    if ndvi_dir:
        ndvi_stack = create_imagestack_dataset(name=random_name + 'ndvi', filelist=filelist_ndvi, times=times_ndvi,
                                               nodata=-9999)

    # read sig0 to virtual stack, write the vrt to VSC ram disk
    sig_stack = create_imagestack_dataset(name=random_name + 'SIG', filelist=filelist_sig0, times=times_sig0,
                                          nodata=-9999)
    # read plia to virtual stack, write the vrt to VSC ram disk
    plia_stack = create_imagestack_dataset(name=random_name + 'LIA', filelist=filelist_plia, times=times_plia,
                                           nodata=-9999)
    # read sig and plia blocks

    for row in line_list:
        # make temp dir in VSC's ram disk
        tmp_dir = make_tmp_dir(str(row))
        print('read sig0 and plia stack... line:', row, datetime.now())
        # read sig0 and plia line (col: 0 - tifsize, row: row*blocksize - (row+1)*blocksize)
        time_sig0_list, data_sig0_list = sig_stack.read_ts(0, row * block_size, tif_size, block_size)
        time_plia_list, data_plia_list = plia_stack.read_ts(0, row * block_size, tif_size, block_size)

        if ndvi_stack:
            print('read ndvi stack..., line:', row, datetime.now())
            # read ndvi line (col: 0 - tifsize, row: row*blocksize - (row+1)*blocksize)
            time_ndvi_list, data_ndvi_list = ndvi_stack.read_ts(0, row * block_size, tif_size, block_size)
        else:
            time_ndvi_list = None
            data_ndvi_list = None

        # single thread processing
        if mp_threads in [0, 1]:
            for col in range(int(tif_size / block_size)):  # number of pixel per line
                # if the [col_row] is processed already, continue
                if str(col) + '_' + str(row) in processed:
                    # print(str(col) + '_' + str(row) + 'is processed')
                    continue
                else:
                    prepare_data(time_sig0_list=time_sig0_list,
                                 data_sig0_list=data_sig0_list,
                                 data_plia_list=data_plia_list,
                                 time_ndvi_list=time_ndvi_list,
                                 data_ndvi_list=data_ndvi_list,
                                 col=col,
                                 row=row,
                                 block_size=block_size,
                                 out_dir=output_dir)  # write output file directly to output_dir


        else:
            # start multiprocessing
            process_list = []
            for col in range(int(tif_size / block_size)):  # number of pixel per line
                # if the [col_row] is processed already, continue
                if str(col) + '_' + str(row) in processed:
                    # print(str(col) + '_' + str(row) + 'is processed')
                    continue
                else:
                    # prepare mp input dictionaries
                    process_dict = {}
                    process_dict['time_sig0_list'] = time_sig0_list
                    process_dict['data_sig0_list'] = data_sig0_list
                    process_dict['data_plia_list'] = data_plia_list
                    process_dict['time_ndvi_list'] = time_ndvi_list
                    process_dict['data_ndvi_list'] = data_ndvi_list
                    process_dict['col'] = col
                    process_dict['row'] = row
                    process_dict['block_size'] = block_size
                    process_dict['out_dir'] = tmp_dir  # write output file to tmp_dir
                    process_list.append(process_dict)

            # start the pool
            pool = mp.Pool(mp_threads)
            pool.map(prepare_data_mp, process_list)
            pool.close()
            pool = None
            # move whole processed line from VSC's ram disk to output dir
            move_dir(tmp_dir, output_dir)


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

    tif_size = int(cfg['PARAMETER']['tif_size'])
    block_size = int(cfg['PARAMETER']['block_size'])
    mp_threads = int(cfg['PARAMETER']['mp_threads'])
    orbit_direction = cfg['PARAMETER']['orbit_direction']
    if orbit_direction == 'None':
        orbit_direction = ''

    test_corner = cfg['PARAMETER']['test_corner']
    if test_corner == 'None':
        test_corner = False
        upper_left_index = None
    else:
        test_corner = int(test_corner)
        upper_left_index = cfg['PARAMETER']['upper_left_index'].split(',')
        col_test, row_test = list(map(int, upper_left_index))

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

    # get processed lines:
    processed = get_processed_list(out_dir)

    # prepare lines list
    list_all_lines = []
    if test_corner:
        row_to_proc = range(row_test, row_test + test_corner)
        col_to_proc = range(col_test, col_test + test_corner)
        for line in row_to_proc:
            # line filtering
            list_all_lines.append(line)
            # column filtering
            for col_test in range(pixels_per_side):
                if col_test not in list(col_to_proc):
                    processed.append(str(col_test) + '_' + str(line))
    # process full tile (all the lines in tile)
    else:
        for line in range(pixels_per_side):
            list_all_lines.append(line)

    # distributing to VSC computing nodes, if run on single machine, total_arr_number should be 1
    list_to_process_all = chunkIt(list_all_lines, total_arr_number)  # list of 5 lists, each list 200 line (1000/5)
    list_to_process_this_node = list_to_process_all[arr_number - 1]  # e.g array number 1 take list_to_process_all[0]

    if not test_corner:
        # remove fully processed line only test_corner is not active
        # condition: only add line to processing list if "_line_" occurences less than 1000
        list_to_process_this_node_filt = []
        for line in list_to_process_this_node:
            if get_str_occ(processed, '_' + str(line)) >= pixels_per_side:
                print(line, "was fully processed, removing..")
            else:
                list_to_process_this_node_filt.append(line)
        list_to_process_this_node = list_to_process_this_node_filt
        list_to_process_this_node_filt = None

    if test_vsc_param:
        # print out test parameters
        print('sig0_dir', sig0_dir)
        print('plia_dir', plia_dir)
        print('block_size', block_size)
        print('out_dir', out_dir)
        print('mp_threads', mp_threads)
        # print('cr_list', list_to_process_node)
        print('len_all (total px)', len(list_all_lines))
        print('number of cr_list (must equal arraytotalnumber)', len(list_to_process_all))
        print('totalarraynumber', total_arr_number)
        print('arraynumber', arr_number)
        print('len_cr_list (px for this node)', len(list_to_process_this_node))

    else:
        print("Node:", arr_number, "/", total_arr_number, datetime.now())
        print('Target: process ', len(list_to_process_this_node), 'lines...')
        read_stack_line(sig0_dir=sig0_dir,
                        plia_dir=plia_dir,
                        block_size=block_size,
                        line_list=list_to_process_this_node,
                        output_dir=out_dir,
                        ndvi_dir=ndvi_dir,
                        orbit_direction=orbit_direction,
                        processed=processed,
                        tif_size=tif_size,
                        mp_threads=mp_threads)


if __name__ == '__main__':
    import sys

    # comment those lines if you're working on the VSC
    sys.argv.append("config/config_tle.ini")
    sys.argv.append("-totalarraynumber")
    sys.argv.append("1")
    sys.argv.append("-arraynumber")
    sys.argv.append("1")

    print("-------------START------------", datetime.now())
    main(sys.argv[1:], test_vsc_param=False)

    pass
