#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:34:58 2024

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sn_tools.sn_utils import multiproc
from optparse import OptionParser
import time
from sn_tools.sn_io import checkDir

url = "http://stilism.obspm.fr/reddening?frame=galactic&vlong={}&ulong=deg&vlat={}&ulat=deg&distance={}"


def complete_with_stilism(df):
    """
    Function to get E(B-V) from stilism

    Parameters
    ----------
    df : pandas df
        Data to process.

    Returns
    -------
    df : pandas df
        Data+reddening.

    """

    df.loc[:, "distance[pc][stilism]"] = np.nan
    df.loc[:, "reddening[mag][stilism]"] = np.nan
    df.loc[:, "distance_uncertainty[pc][stilism]"] = np.nan
    df.loc[:, "reddening_uncertainty_min[mag][stilism]"] = np.nan
    df.loc[:, "reddening_uncertainty_max[mag][stilism]"] = np.nan

    for index, row in df.iterrows():
        # print("l:", row["l"], "deg, b:", row["b"],
        #      "deg, distance:", row["distance"], "pc")
        res = requests.get(url.format(
            row["l"], row["b"], row["distance"]), allow_redirects=True)
        if res.ok:
            file = StringIO(res.content.decode("utf-8"))
            dfstilism = pd.read_csv(file)
            # print(dfstilism)
            df.loc[index, "distance[pc][stilism]"] = dfstilism["distance[pc]"][0]
            df.loc[index, "reddening[mag][stilism]"] = dfstilism["reddening[mag]"][0]
            df.loc[index, "distance_uncertainty[pc][stilism]"] = dfstilism["distance_uncertainty[pc]"][0]
            df.loc[index, "reddening_uncertainty_min[mag][stilism]"] = dfstilism["reddening_uncertainty_min[mag]"][0]
            df.loc[index, "reddening_uncertainty_max[mag][stilism]"] = dfstilism["reddening_uncertainty_max[mag]"][0]

    return df


def stilism_multiproc(ll, params, j=0, output_q=None):
    """
    Function to get reddening from stilism using multiprocessing

    Parameters
    ----------
    ll : list(str)
        List to process.
    params : dict
        Parameters.
    j : int, optional
        Tag for multiprocessing. The default is 0.
    output_q : multiprocessing queue, optional
        Where to put the data. The default is None.

    Returns
    -------
    pandas df/dict in queue
        Processed data.

    """

    data = params['data']
    thecol = params['thecol']

    idx = data[thecol].isin(ll)
    sel = pd.DataFrame(df[idx])
    print('processing', j, len(sel))
    dfb = complete_with_stilism(sel)

    if output_q is not None:
        return output_q.put({j: dfb})
    else:
        return dfb


parser = OptionParser(description='Script to get reddening from stilism')

parser.add_option('--input_dir', type=str,
                  default='../gaia_files',
                  help='data dir [%default]')
parser.add_option('--input_file', type=str,
                  default='gaia_dr2_0.hdf5',
                  help='file to process [%default]')
parser.add_option('--output_dir', type=str,
                  default='../gaia_files_reddening',
                  help='output data dir [%default]')
parser.add_option('--output_file', type=str,
                  default='gaia_dr2_0_reddening.hdf5',
                  help='output file name [%default]')
parser.add_option('--nproc', type=int,
                  default=8,
                  help='number of procs [%default]')

opts, args = parser.parse_args()

input_dir = opts.input_dir
input_file = opts.input_file
output_dir = opts.output_dir
output_file = opts.output_file
nproc = opts.nproc

checkDir(output_dir)
time_ref = time.time()

full_input_path = '{}/{}'.format(input_dir, input_file)
df = pd.read_hdf(full_input_path)

thecol = 'SOURCE_ID'
params = {}
params['thecol'] = thecol
params['data'] = df
res = multiproc(df[thecol].to_list(), params, stilism_multiproc, nproc)
print('finished', len(res), time.time()-time_ref)

full_output_path = '{}/{}'.format(output_dir, output_file)
res.to_hdf(full_output_path, key='star')
