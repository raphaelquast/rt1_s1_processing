import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import cloudpickle

from ImageStack import create_imagestack_dataset
from datetime import datetime
import pandas as pd
import numpy as np
from configparser import ConfigParser
import argparse
import copy
from rt1_processing_funcs_juhuu import inpdata_inc_average
import random, string
from scipy.signal import savgol_filter
import multiprocessing as mp
import shutil


def get_worker_id():
    try:
        return str(mp.current_process())[21:30]
    except Exception:
        return ("Non MP")


def read_stack_line(sig0_dir, plia_dir, block_size, line_list, output_dir, ndvi_dir=None, orbit_direction=''):
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

    # read ndvi to virtual stack
    if ndvi_dir:
        ndvi_stack = create_imagestack_dataset(name=random_name + 'ndvi', filelist=filelist_ndvi, times=times_ndvi,
                                               nodata=-9999)

    # read sig0 to virtual stack
    sig_stack = create_imagestack_dataset(name=random_name + 'SIG', filelist=filelist_sig0, times=times_sig0,
                                          nodata=-9999)
    # read plia to virtual stack
    plia_stack = create_imagestack_dataset(name=random_name + 'LIA', filelist=filelist_plia, times=times_plia,
                                           nodata=-9999)
    # read sig and plia blocks
    try:
        print(get_worker_id(), line_list, datetime.now())
    except Exception:
        pass

    for line in line_list:
        # make temp dir:
        tmp_dir = make_tmp_dir(str(line))
        # don't re-process files that already exist
        # TODO: add here check output function (probably a blacklist), remove hardcode
        print('read sig0 and plia stack... line:', line, datetime.now())
        sig0_block = sig_stack.read_ts(0, line * block_size, 10000, block_size)
        plia_block = plia_stack.read_ts(0, line * block_size, 10000, block_size)
        time_sig0_list = sig0_block[0]
        data_sig0_list = sig0_block[1]
        time_plia_list = plia_block[0]
        data_plia_list = plia_block[1]

        if ndvi_stack:
            print('read ndvi stack..., line:', line, datetime.now())
            ndvi_block = ndvi_stack.read_ts(0, line * block_size, 10000, block_size)
            time_ndvi_list = ndvi_block[0]
            data_ndvi_list = ndvi_block[1]

        # TODO: remove hard code in here
        for px in range(1000):  # number of pixel per line
            df_px_list = []
            print(get_worker_id(), 'preparing the data for line', line, "px", px, datetime.now())
            for time in time_sig0_list:
                idx_sig0 = time_sig0_list.index(time)
                idx_plia = time_plia_list.index(time)
                px_sig0 = data_sig0_list[idx_sig0][:, px * block_size:(px + 1) * block_size]
                px_plia = data_plia_list[idx_plia][:, px * block_size:(px + 1) * block_size]

                sig0_plia_stack = np.vstack((px_sig0.flatten(), px_plia.flatten()))
                df_px = pd.DataFrame(data=sig0_plia_stack.transpose(),
                                     index=np.full((sig0_plia_stack.shape[1],), time),
                                     columns=['sig', 'inc'])
                df_px_list.append(df_px)

                # old stupid way
                # loop over slice
                # for row in range(block_size):
                #     for col in range(px * block_size, (px + 1) * block_size):
                #         sig0_value = data_sig0_list[idx_sig0][row][col]
                #         plia_value = data_plia_list[idx_plia][row][col]
                #         final_list.append([time, sig0_value, plia_value])

            df = pd.concat(df_px_list)

            # nan handling
            df = df.replace(-9999, np.nan)

            df = df.dropna()

            # true value
            df['sig'] = df['sig'].div(100)
            df['inc'] = df['inc'].div(100)

            # convert to radian
            df['inc'] = np.deg2rad(df['inc'])

            # sort by index
            df.sort_index(inplace=True)

            df = inpdata_inc_average(df)

            # -----------------------------------------

            if ndvi_stack:
                ndvi_list = []
                for time in time_ndvi_list:
                    idx_ndvi = time_ndvi_list.index(time)
                    ndvi_list.append([time, np.mean(data_ndvi_list[idx_ndvi])])

                df_ndvi = pd.DataFrame(ndvi_list)
                df_ndvi.columns = ['time', 'ndvi']

                # nan handling
                df_ndvi = df_ndvi.replace(list(range(251, 256)), np.nan)

                df_ndvi = df_ndvi.dropna()

                # convert to physical value
                df_ndvi['ndvi'] = df_ndvi['ndvi'] / 250 - 0.08

                # set index
                df_ndvi = df_ndvi.set_index('time')

                # sort by index
                df_ndvi.sort_index(inplace=True)

            # ------------------------ RQ's manipulation
            # TODO: add a condition to work without NDVI
            VOD_input = df_ndvi.resample('D').interpolate(method='nearest')
            # get a smooth curve
            # VOD_input = VOD_input.rolling(window=60, center=True, min_periods=1).mean()
            VOD_input = VOD_input.clip_lower(0).apply(savgol_filter, window_length=61, polyorder=2).clip_lower(0)
            # reindex to input-dataset
            VOD_input = VOD_input.reindex(df.index.drop_duplicates()).dropna()
            # ensure that there are no ngative-values appearing (possible due to rolling-mean and interpolation)
            VOD_input = VOD_input.clip_lower(0)
            # drop all measurements where no VOD estimates are available
            df = df.loc[VOD_input.dropna().index]

            # manual_dyn_df = pd.DataFrame(df.index.month.values.flatten(), df.index, columns=['VOD'])
            defdict_i = {
                'bsf': [False, 0.01, None, ([0.01], [.25])],
                'v': [False, 0.4, None, ([0.01], [.4])],
                'v2': [True, 1., None, ([0.5], [1.5])],
                'VOD': [False, VOD_input.values.flatten()],
                'SM': [True, 0.25, 'D', ([0.05], [0.5])],
                # 'VOD'   : [True, 0.25,'30D', ([0.01], [1.])],
                'frac': [True, 0.5, None, ([0.01], [1.])],
                'omega': [True, 0.3, None, ([0.05], [0.6])],
            }
            out_dict = {'dataset': df, 'defdict': defdict_i, '_fnevals_input': None, 'c': px, 'r': line,
                        'outdir': tmp_dir}
            # print('processed: ', px, line, datetime.now())
            parallelfunc(out_dict)

        # move temp folder into output dir
        move_tmp_dir(tmp_dir, output_dir)


def chunkIt(seq, num):
    '''
    Chunk a list in to a approximately equal length
    Parameters
    ----------
    seq: list
    num: number of part

    Returns
    -------

    '''
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def parse_args(args):
    """
    Parse command line parameters.

    Parameters
    ----------
    args: list
        command line arguments

    Returns
    -------
    parser: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="RT1 processing")

    parser.add_argument(
        "cfg_file", help="Path to config file for this processing step")

    parser.add_argument("-arraynumber", "--arraynumber", metavar='arraynumber', type=str,
                        help=("Number of VSC3 array (int)"))

    parser.add_argument("-totalarraynumber", "--totalarraynumber", metavar='totalarraynumber', type=int,
                        help=("Total number of VSC3 arrays (int)"))

    return parser.parse_args(args)


def read_cfg(cfg_file, include_default=True):
    """
    Parse a config file.

    Parameters
    ----------
    cfg_file: string
        filename of the config file
    include_default: boolean, optional
        If true then the DEFAULT section settings will
        be included in all other sections.
    """

    config = ConfigParser()
    config.optionxform = str
    config.read(cfg_file)

    ds = {}

    for section in config.sections():

        ds[section] = {}
        for item, value in config.items(section):

            if 'path' in item:
                value = value.replace(' ', '')
                path = value.split(',')
                if path[0][0] == '.':
                    # relative path
                    value = os.path.join(os.path.split(cfg_file)[0], *path[0:])
                elif path[0][0] == '/' or path[0][1] == ':':
                    # absolute path in linux or windows
                    value = os.path.sep.join(path)
                else:
                    print(section, item, ' got a blank value, set to  None')
                    value = None

            if item.startswith('kws'):
                if item[4:] == 'custom_dtype':
                    value = {item[4:]: eval(value)}
                else:
                    value = {item[4:]: value}
                item = 'kws'

            if include_default or item not in config.defaults().keys():
                if item == 'kws':
                    ds[section][item].update(value)
                else:
                    ds[section][item] = value

    return ds


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
        upper_left_index = list(map(int, upper_left_index))

    # take total array number and array number from arg
    if args.arraynumber:
        arr_number = int(args.arraynumber)
    else:
        raise Warning('-arraynumber needed')

    if args.totalarraynumber:
        total_arr_number = int(args.totalarraynumber)
    else:
        raise Warning('-totalarraynumber needed')

    pixels_per_side = int(tif_size / block_size)

    # prepare corner list
    list_all_lines = []
    if test_corner:
        # TODO: make test_corner work again (or probably test line)
        pass
    else:
        for line in range(pixels_per_side):
            list_all_lines.append(line)

    list_to_process_all = chunkIt(list_all_lines, total_arr_number)  # list of 5 lists, each list 200 line (1000/5)
    list_to_process_this_node = list_to_process_all[
        arr_number - 1]  # e.g array number 1 will take list_to_process_all[0]

    if test_vsc_param:
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
        # print("load the stack...:", datetime.now())
        if mp_threads in [0, 1]:
            read_stack_line(sig0_dir=sig0_dir,
                            plia_dir=plia_dir,
                            block_size=block_size,
                            line_list=list_to_process_this_node,
                            output_dir=out_dir,
                            ndvi_dir=ndvi_dir,
                            orbit_direction=orbit_direction)
        else:
            # implement the multiprocessing here

            process_list = []
            # chunk the line list into smaller lists for processing
            list_to_process_node_chunked = chunkIt(list_to_process_this_node, mp_threads * 2)
            for cr_list in list_to_process_node_chunked:
                process_dict = {}
                process_dict['sig0_dir'] = sig0_dir
                process_dict['plia_dir'] = plia_dir
                process_dict['block_size'] = block_size
                process_dict['cr_list'] = cr_list
                process_dict['output_dir'] = out_dir
                process_dict['ndvi_dir'] = ndvi_dir
                process_dict['orbit_direction'] = orbit_direction
                process_list.append(process_dict)

            print("start the mp...:", datetime.now())
            print('each computing node is processing ', len(process_list), 'lists...')
            print('each list has about ', len(list_to_process_node_chunked[0]), 'lines...')
            pool = mp.Pool(mp_threads)
            pool.map(read_stack_line_mp, process_list)
            pool.close()

            # import ipyparallel as ipp
            # client = ipp.Client()
            # dview = client[:]
            # dview.use_cloudpickle()
            # balanced_view = client.load_balanced_view()

            # async_res = balanced_view.map_async(read_stack_mp, process_list)
            # return async_res


def read_stack_line_mp(process_dict):
    read_stack_line(sig0_dir=process_dict['sig0_dir'],
                    plia_dir=process_dict['plia_dir'],
                    block_size=process_dict['block_size'],
                    line_list=process_dict['cr_list'],
                    output_dir=process_dict['output_dir'],
                    ndvi_dir=process_dict['ndvi_dir'],
                    orbit_direction=process_dict['orbit_direction'])


# -------------------------------------------------


def parallelfunc(import_dict):
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    c = import_dict['c']
    r = import_dict['r']
    outdir = import_dict['outdir']
    dataset = copy.deepcopy(import_dict['dataset'])
    dataset['sig'] = 10 ** (dataset['sig'] / 10.)

    defdict = import_dict['defdict']
    _fnevals_input = import_dict['_fnevals_input']

    try:
        print(get_worker_id(), 'processing site C:', c, ' R:', r, 'time:', datetime.now())
    except Exception:
        pass

    from rt1.rtfits import Fits
    '''
    def set_V_SRF(isopart, alpha, v1, v2, v_forward, v_bounceoff, VOD,
                  omega, tt, s1, s2, aa, SM, **kwargs):
        from rt1.volume import LinCombV, HenyeyGreenstein
        from rt1.surface import HG_nadirnorm

        V = LinCombV(Vchoices=
                     [[isopart,
                       HenyeyGreenstein(t=0.001, ncoefs=3)],
                      [(1. - isopart) * (1. - alpha),
                       HenyeyGreenstein(t=v_forward, ncoefs=8)],
                      [(1. - isopart) * alpha,
                       HenyeyGreenstein(t=-v_bounceoff, ncoefs=8, a=[1., 1., 1.])]
                      ],
                     tau=v1 + v2 * VOD,
                     omega=omega)

        SRF = HG_nadirnorm(t=tt, ncoefs=10, a=[aa, 1., 1.],
                           NormBRDF=s1 + s2 * SM)
        return V, SRF
    '''

    def set_V_SRF(frac, omega, SM, VOD, v, v2, **kwargs):
        from rt1.volume import HenyeyGreenstein
        from rt1.surface import LinCombSRF, HG_nadirnorm

        SRFchoices = [
            [frac, HG_nadirnorm(t=0.01, ncoefs=2, a=[-1., 1., 1.])],
            [(1. - frac), HG_nadirnorm(t=0.6, ncoefs=10, a=[1., 1., 1.])]
        ]
        SRF = LinCombSRF(SRFchoices=SRFchoices, NormBRDF=SM)

        V = HenyeyGreenstein(t=v, omega=omega, tau=v2 * VOD, ncoefs=8)

        return V, SRF

    fit = Fits(sig0=True, dB=False, dataset=dataset,
               set_V_SRF=set_V_SRF,
               defdict=defdict)

    fitset = {'int_Q': False,
              '_fnevals_input': _fnevals_input,
              # least_squares kwargs:
              'verbose': 1,  # verbosity of least_squares
              # 'verbosity' : 1, # verbosity of monofit
              'ftol': 1.e-5,
              'gtol': 1.e-5,
              'xtol': 1.e-5,
              'max_nfev': 100,
              'method': 'trf',
              'tr_solver': 'lsmr',
              'x_scale': 'jac'
              }

    fit.performfit(**fitset)
    fit.result[1].fn = 1

    if import_dict['_fnevals_input'] is None:
        import_dict['_fnevals_input'] = fit.result[1]._fnevals

    with open(os.path.join(outdir, str(c) + '_' + str(r) + '.dump'), 'wb') as file:
        cloudpickle.dump(fit, file)
        # return fit
    print(get_worker_id(), "processing done", 'C:', c, ' R:', r, 'time:', datetime.now())


def make_tmp_dir(line):
    if 'TMPDIR' in os.environ:
        tmp_dir = os.path.join(os.environ["TMPDIR"], line)
    else:
        tmp_dir = os.path.join('/tmp', line)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir


def move_tmp_dir(tmp_dir, outdir):
    try:
        shutil.copy(tmp_dir, outdir)
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    import sys

    # sys.argv.append("/home/tle/code/new/rt1_s1_processing/config_tle.ini")
    # sys.argv.append("-totalarraynumber")
    # sys.argv.append("1")
    # sys.argv.append("-arraynumber")
    # sys.argv.append("1")

    print("Start", datetime.now())
    main(sys.argv[1:], test_vsc_param=False)

    pass
