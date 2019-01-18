# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:13:57 2018

@author: rquast
"""
import os
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



def monitorresult(result):
    '''
    a function to monitor the standard-console
    output (stdout) of ipyparallel jobs
    '''
    print('...monitoring...')
    oldout = result.stdout
    nproc = 0
    while result.done() is False:
        for i, out in enumerate(result.stdout):
            if out == oldout[i]:
                continue
            print('kernel ', i, ': ', out[len(oldout[i]):-1])
            nproc += 1
            print(nproc)
            oldout = result.stdout





# %% data-Reader


def read_s1_data(read_sites = None,
                 mainpath = 'R:\\Projects_work\\SBDSC\\data\\Sentinel_1\\sig0_plia_csv_highresdem'):
    '''
    read in all the in-situ soil-moisture datasets provided as csv-files
    and return a dictionary of pandas DataFrames where the keys are
    the site_id's  (inferred from the first 3 digit of the filenames).

    Parameters:
    ---------------
    read_sites : list
                 a list of the site-numbers (if None all will be read)
    mainpath : str (default = 'R:\Projects_work\SBDSC\data\Sentinel_1\20181121_extract_ts')
               path to the main folder where the data is located

    '''
    print('...reading sentinel-1 data...')


    # additional properties of the csv-files
    csv_props = {'sep':',',
                 'index_col':0,
                 'date_parser':pd.to_datetime,
                 'parse_dates':['time'],
                 'na_values' : -9999}

    # get the paths to all csv_files
    csv_filenames = {}
    for file in os.listdir(mainpath):
        if file.endswith(".txt"):
            csv_filenames[file[:-4]] = os.path.join(mainpath, file)


    if read_sites is None:
        read_sites = list(csv_filenames.keys())
    else:
        for i in read_sites:
            assert i in csv_filenames.keys(); 'file not found'



    s1_timeseries = {}
    # read only the csv-files that start with site_ids
    for site in read_sites:
        ts = pd.DataFrame()
        # read in pandas timeseries
        ts = pd.read_csv(csv_filenames[site], **csv_props)/100

        names = [key[:-3] for key in ts.keys() if 'vv' in key]

        sigdf = pd.concat([ts[name + '_vv'] for name in names])
        incdf = pd.concat([ts[name + '_lia'] for name in names])

        ts = pd.concat([incdf, sigdf], keys=['inc', 'sig'], axis=1)
        ts = ts.dropna().sort_index()
        s1_timeseries[site] = ts

    return s1_timeseries




def read_ndvi_lai_data(read_sites = None,
                       dataset = 'lai',
                       mainpath = 'R:\\Projects_work\\SBDSC\\data\\Copernicus\\resampled',
                       remove_outliers=False,
                       outlier_window = 90):
    '''
    read in all the in-situ soil-moisture datasets provided as csv-files
    and return a dictionary of pandas DataFrames where the keys are
    the site_id's  (inferred from the first 3 digit of the filenames).

    Parameters:
    ---------------
    read_sites : list
                 a list of the site-numbers (if None all will be read)
    dataset : str (default = 'lai')
              the dataset to be read (either 'lai' or 'ndvi')
    mainpath : str (default = 'R:\Projects_work\SBDSC\data\Sentinel_1\20181121_extract_ts')
               path to the main folder where the data is located
    remove_outliers : bool (default = False)
                      indicator if outliers (value > rolling_mean + 3*std) are removed
    outlier_window : int (default = 90)
                     the number of days used in the pd.rolling() instance to
                     calculate the rolling mean of the datset
    '''
    print('...reading', dataset,' data...')

    mainpath += '\\' + dataset + '_csv'

    # additional properties of the csv-files
    csv_props = {'sep':',',
                 'index_col':0,
                 'date_parser':pd.to_datetime,
                 'parse_dates':['datetime'],
                 'skiprows' : 1,
                 'names' : ['datetime', dataset]}


    # get the paths to all csv_files
    csv_filenames = {}
    for file in os.listdir(mainpath):
        if file.endswith(".txt"):
            csv_filenames[file[:-4]] = os.path.join(mainpath, file)


    if read_sites is None:
        read_sites = list(csv_filenames.keys())
    else:
        for i in read_sites:
            assert i in csv_filenames.keys(); 'file not found'



    dataset_timeseries = {}
    # read only the csv-files that start with site_ids
    for site in read_sites:
        ts = pd.DataFrame()
        # read in pandas timeseries
        ts = pd.read_csv(csv_filenames[site], **csv_props)

        if dataset == 'ndvi':
            ts = ts[ts<250]
            ts = ts / 250. - 0.08

        if dataset == 'lai':
            ts = ts[ts<210]
            ts = ts / 30.

        if remove_outliers is True:
            mask = (ts.resample('D').mean().interpolate() > ts.resample('D').mean().interpolate().rolling(window=outlier_window, min_periods=2, center=True).mean() + ts.std()).reindex(ts.index)

            if np.any(mask):
                print('site: ', site,
                      '... some outliers are detected and removed!')
                ts = ts[~mask]

        dataset_timeseries[site] = ts
    return dataset_timeseries





















