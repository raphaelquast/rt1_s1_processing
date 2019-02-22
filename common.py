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
import pandas as pd

def inpdata_inc_average(inpdata, round_digits=0, returnall = False):
    '''
    A function to average the data based on a given number of digits for the
    incidence-angle information ('inc' column of inpdata)

    Parameters:

    inpdata : pandas.DataFrame
        a dataframe with columns 'inc' and 'sig' and a datetime-index
    round_digits : int (default = 0)
        the number of digits to which the incidence-angles will
        be rounded (input of np.round() )
    returnall : bool (default = False)
        indicator if all generated data should be returned or only
        the averaged inpdata


    Returns:

    inpdata_new : pandas.DataFrame
        a dataframe with the 'sig' values averaged based on
        the incidence-angles which have been rounded with respect to the
        'round_digits' parameter
    index : array
        only returned if returnall == True !

        the datetime-index of the unique incidence-angle values
    meanuniquesigs : array
        only returned if returnall == True !

        averaged sig0 values corresponding to the rounded incidence-angles
    uniquesigs : array
        only returned if returnall == True !

        all sig0 values corresponding to the rounded incidence-angles
        (call signature: uniquesigs[day][#average incidence-angle][#measurement])

    uniqueinc : array
        only returned if returnall == True !

        all (rounded) incidence-angle values of the corresponding day
        (call signature: uniqueinc[day][#average incidence-angle])
    '''

    asdfinc = np.rad2deg(inpdata['inc']).resample('D').apply(np.array).reindex(inpdata.resample('D').mean().dropna().index)
    index = asdfinc.index
    asdfinc = asdfinc.values.flatten()
    asdfsig = inpdata['sig'].resample('D').apply(np.array).reindex(inpdata.resample('D').mean().dropna().index).values.flatten()
    # mask nan-values
    nanmask = [np.where(~np.isnan(i)) for i in asdfinc]

    asdfinc = np.array([np.round(val[nanmask[i]], round_digits) for i, val in enumerate(asdfinc)])
    asdfsig = np.array([val[nanmask[i]] for i, val in enumerate(asdfsig)])

    uniques = np.array([np.unique(i, return_inverse=True) for i in asdfinc])

    uniqueinc = uniques[:,0]
    unique_index = uniques[:,1]

    uniquesigs = []
    for i, sigvals in enumerate(asdfsig):
        uniquesigs_day = []
        for unique_id in np.unique(unique_index[i]):
            uniquesigs_day += [sigvals[np.where(unique_index[i] == unique_id)]]
        uniquesigs += [np.array(uniquesigs_day)]
    #uniquesigs = np.array(uniquesigs)

    meanuniquesigs = []
    for i in uniquesigs:
        meanuniquesigs_day = []
        for j in i:
            meanuniquesigs_day += [np.mean(j)]
        meanuniquesigs += [np.array(meanuniquesigs_day)]

    newindex = np.concatenate([np.array([ind]*len(uniqueinc[i])) for i, ind in enumerate(index)])
    inpdata_new = pd.DataFrame({'inc':np.deg2rad(np.concatenate(uniqueinc)), 'sig':np.concatenate(meanuniquesigs)}, index=newindex)

    if returnall is True:
        return inpdata_new, index, meanuniquesigs, uniquesigs, uniqueinc
    else:
        return inpdata_new


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

    from rt1.rtfits import Fits
    import model_definition
    from scipy.signal import savgol_filter

    if import_dict.get('refit', False) is True:
        print('refitting result')
        c = import_dict['c']
        r = import_dict['r']
        outdir = import_dict['outdir']
        dataset = import_dict['dataset']
        if model_definition.defdict['VOD'][1] == 'auxiliary' and model_definition.defdict['VOD'][0] is False:
            model_definition.defdict['VOD'][1] = import_dict['VOD']
    else:
        c = import_dict['c']
        r = import_dict['r']
        outdir = import_dict['outdir']
        # get the sig0 dataset
        dataset = import_dict['dataset']
        # convert it to linear units
        dataset['sig'] = 10 ** (dataset['sig'] / 10.)

        dataset.index = [pd.to_datetime(i.date()) for i in dataset.index]

        #dataset = inpdata_inc_average(dataset)


        df_ndvi = import_dict['df_ndvi']
        # transform ndvi dataset if available
        if 'df_ndvi' in import_dict and model_definition.defdict['VOD'][1] == 'auxiliary' and model_definition.defdict['VOD'][0] is False:
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

            model_definition.defdict['VOD'][1] = VOD_input.values.flatten()

    try:
        print(get_worker_id(), 'processing site C:', c, ' R:', r, 'time:', datetime.now())
    except Exception:
        pass


    fit = Fits(sig0=model_definition.sig0, dB=model_definition.dB,
               set_V_SRF=model_definition.set_V_SRF,
               defdict=model_definition.defdict,
               dataset=dataset)


#    if 'fnevals_input' in import_dict and import_dict['fnevals_input'] is not None:
#        model_definition.fitset['fnevals_input'] = import_dict['fnevals_input']
#    if import_dict['_fnevals_input'] is None:
#        import_dict['_fnevals_input'] = fit.result[1]._fnevals

    fit.performfit(**model_definition.fitset)
    fit.result[1].fn = 1


    with open(os.path.join(outdir, str(c) + '_' + str(r) + '.dump'), 'wb') as file:
        cloudpickle.dump(fit, file)
        # return fit


    # get the ids
    # TODO replace this with the id from import_dict!
    site_id =  f"{import_dict['c']}_{import_dict['r']}"

    # extract parameters for csv-output
    # get the keys of the constant parameters
    paramkeys = [key for key, val in model_definition.defdict.items() if len(val) > 2 and val[0] is True and val[2] is None]
    # get the keys of the temporally varying parameters
    tskeys = [key for key, val in model_definition.defdict.items() if len(val) > 2 and val[0] is True and val[2] is not None]
    # extract results
    csv_parameters = pd.DataFrame({key : val[0] for key, val in fit.result[6].items() if key in paramkeys}, index=[site_id])
    csv_timeseries = pd.DataFrame({key : val for key, val in fit.result[6].items() if key in tskeys}, index=fit.index)

    # generate csv-files
    csv_folder_path = os.path.join(outdir, 'csv_output')
    # generate csv_output folder if it does not exist
    if not os.path.exists(csv_folder_path):
        os.mkdir(csv_folder_path)

    # generate parameter-csv_files:
    for key, val in csv_parameters.items():
        vsv_filepath = os.path.join(csv_folder_path, key + '.csv')
        if not os.path.exists(vsv_filepath):
            with open(vsv_filepath, 'w') as file:
                val.to_csv(file, header=True, index_label='id')
        else:
            with open(vsv_filepath, 'a') as file:
                val.to_csv(file, header=False)

    # generate timeseries-csv_files:
    for key, val in csv_timeseries.items():
        ts_folderpath = os.path.join(csv_folder_path, key)
        if not os.path.exists(ts_folderpath):
            os.mkdir(ts_folderpath)

        for time_key, time_val in val.items():

            ts_filepath = os.path.join(ts_folderpath,
                                       pd.datetime.strftime(time_key, format='%Y-%m-%d') + '.csv')

            if not os.path.exists(ts_filepath):
                with open(ts_filepath, 'w') as file:
                    pd.Series([time_val], [site_id], name=key).to_csv(file, header=True, index_label='id')
            else:
                with open(ts_filepath, 'a') as file:
                    pd.Series([time_val], [site_id], name=key).to_csv(file)

if __name__ == '__main__':
    # print(get_processed_list('/tmp'))
    pass
