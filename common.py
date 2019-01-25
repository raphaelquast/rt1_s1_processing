import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from configparser import ConfigParser
import argparse
import multiprocessing as mp
import shutil
import cloudpickle
from datetime import datetime
import tempfile
import numpy as np


def get_worker_id():
    '''
    return multiprocessing worker ID
    Returns
    -------

    '''
    try:
        if "Main" in str(mp.current_process()):
            return ("Single thread:")
        return str(mp.current_process())
    except Exception as e:
        print(e)


def prepare_index_array(time_list, data_array):
    px_time = np.empty(data_array.shape, dtype=object)
    for time in time_list:
        idx = time_list.index(time)
        px_time[idx] = time
    return px_time

def make_tmp_dir(sub_folder_name):
    '''
    make a child folder in tempdir of the current machine
    Parameters
    ----------
    sub_folder_name: str

    Returns
    -------

    '''
    if 'TMPDIR' in os.environ:
        tmp_dir = os.path.join(os.environ["TMPDIR"], sub_folder_name)
    else:
        tmp_dir = os.path.join(tempfile.gettempdir(), sub_folder_name)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return tmp_dir


def move_dir(tmp_dir, outdir):
    '''
    Move a target folder to destination folder, overwrite the destination folder
    Parameters
    ----------
    tmp_dir
    outdir

    Returns
    -------

    '''
    out_path = os.path.join(outdir, os.path.basename(tmp_dir))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    copytree(tmp_dir, out_path)
    shutil.rmtree(tmp_dir)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


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


def get_processed_list(out_dir):
    '''
    return the list of existing dump files without the extension (in out_dir)
    Parameters
    ----------
    out_dir

    Returns
    -------

    '''
    # traverse root directory, and list directories as dirs and files as files
    processed_dump = []
    for root, dirs, files in os.walk(out_dir):
        processed_dump += [fil.replace('.dump', '') for fil in files if fil.endswith(".dump")]
    return processed_dump


def get_str_occ(list, str):
    '''
    get string occurences in list of strings
    Parameters
    ----------
    list: list of str
    str: str

    Returns
    -------

    '''
    return sum(str in s for s in list)


def parallelfunc(import_dict):
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    from scipy.signal import savgol_filter

    c = import_dict['c']
    r = import_dict['r']
    outdir = import_dict['outdir']

    # get the sig0 dataset
    dataset = import_dict['dataset']
    # convert it to linear units
    dataset['sig'] = 10 ** (dataset['sig'] / 10.)

    df_ndvi = import_dict['df_ndvi']
    # transform ndvi dataset if available
    if df_ndvi is not None:
        VOD_input = df_ndvi.resample('D').interpolate(method='nearest')
        # get a smooth curve
        # VOD_input = VOD_input.rolling(window=60, center=True, min_periods=1).mean()
        VOD_input = VOD_input.clip_lower(0).apply(savgol_filter, window_length=61, polyorder=2).clip_lower(0)
        # reindex to input-dataset
        VOD_input = VOD_input.reindex(dataset.index.drop_duplicates()).dropna()
        # ensure that there are no ngative-values appearing (possible due to rolling-mean and interpolation)
        VOD_input = VOD_input.clip_lower(0)
        # drop all measurements where no VOD estimates are available
        dataset = dataset.loc[VOD_input.dropna().index]

        # manual_dyn_df = pd.DataFrame(dataset.index.month.values.flatten(),
        #                              dataset.index, columns=['VOD'])
        defdict = {
            'bsf': [False, 0.01, None, ([0.01], [.25])],
            'v': [False, 0.4, None, ([0.01], [.4])],
            # 'v2'    : [True, 1., None, ([0.5], [1.5])],
            'v2': [True, 1., None, ([0.1], [1.5])],
            # 'VOD'   : [False, VOD_input.values.flatten()],
            # 'VOD'   : [True, 0.25,'30D', ([0.01], [1.])],
            'VOD': [False, ((VOD_input - VOD_input.min()) / (VOD_input - VOD_input.min()).max()).values.flatten()],
            # 'SM'    : [True, 0.25,  'D',   ([0.05], [0.5])],
            'SM': [True, 0.1, 'D', ([0.01], [0.2])],
            'frac': [True, 0.5, None, ([0.01], [1.])],
            'omega': [True, 0.3, None, ([0.05], [0.6])],
        }

    _fnevals_input = import_dict['_fnevals_input']

    try:
        print(get_worker_id(), 'processing site C:', c, ' R:', r, 'time:', datetime.now())
    except Exception:
        pass

    from rt1.rtfits import Fits

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


if __name__ == '__main__':
    # print(get_processed_list('/tmp'))
    pass
