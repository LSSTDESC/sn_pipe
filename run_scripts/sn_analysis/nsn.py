#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:48:17 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_tools.sn_utils import get_val
import pandas as pd
from sn_tools.sn_io import load_DataFrame
from sn_analysis.sn_selection import selection_criteria
from sn_tools.sn_utils import multiproc
from sn_analysis.sn_nsn_effi import get_nsn
from sn_tools.sn_io import checkDir


def save_data(outDir, dbName, data, outName):
    """
    Function to save the data

    Parameters
    ----------
    outDir : str
        output dir.
    dbName : str
        OS name.
    data : pandas df
        Data to save.
    outName : str
        output file name.

    Returns
    -------
    None.

    """

    outDir_ = '{}/{}'.format(outDir, dbName)
    checkDir(outDir_)
    outName_t = '{}/{}'.format(outDir_, outName)
    data.to_hdf(outName_t, key='nsn')


def process_dbName(dataType, dataDir, dbName, runType,
                   timescale, timeslots, fieldType, listfields, sellist):
    """
    Function to process data

    Parameters
    ----------
    dataType : pandas df
        Data to process.
    dataDir : str
        Data dir.
    dbName : str
        OS name.
    runType : str
        run type.
    timescale : str
        timescale to consider.
    timeslots : list(int)
        time slots.
    fieldType : str
        Field type.
    listfields : list(str)
        List of fields.
    sellist : str
        selection criteria.

    Returns
    -------
    pandas df
     output data

    """

    # load the data
    tt = 'load_{}(\'{}\',\'{}\',\'{}\',\'{}\',{},\'{}\')'.format(
        dataType, dataDir, dbName, runType,
        timescale, timeslots, fieldType)
    df = eval(tt)

    res = pd.DataFrame()
    for field in listfields:
        idx = df['field'] == field
        sel = df[idx]
        if len(sel) == 0:
            continue
        params = {}
        params['data'] = pd.DataFrame(sel)
        params['sellist'] = sellist
        hpixes = sel['healpixID'].unique().tolist()
        # print(field, len(hpixes))
        nsn = multiproc(hpixes, params, process_pixels, nproc=8)
        nsn['field'] = field
        res = pd.concat((res, nsn))
        del sel
        del nsn

    return res


def process_pixels(healpixIDs, params, j=0, output_q=None):
    """
    Function to process data using multiprocessing

    Parameters
    ----------
    healpixIDs : list(int)
        List of healpixIDs.
    params : dict
        parameters.
    j : int, optional
        internal int for multiproc. The default is 0.
    output_q : multiprocessing queue, optional
        where to put the data. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    data = params['data']
    selllist = params['sellist']
    vv = ['healpixID', 'season']
    nsn = pd.DataFrame()

    for hpix in healpixIDs:
        idx = data['healpixID'] == hpix
        sel = data[idx]
        nsn_ = sel.groupby(['healpixID', 'season']).apply(
            lambda x: get_nsn(x, sellist)).reset_index()
        for v in vv:
            nsn_[v] = nsn_[v].astype(int)
        nsn = pd.concat((nsn, nsn_))

    if output_q is not None:
        return output_q.put({j: nsn})
    else:
        return nsn


parser = OptionParser(
    'Script to estimate the number of supernovae estimated fromm observing efficiency')

parser.add_option("--dataDir", type=str,
                  default='../Output_SN_sigmaInt_0.0_Hounsell',
                  help="data dir[%default]")
parser.add_option("--zType", type=str,
                  default='spectroz', help="z type (spectroz/photz) [%default]")
parser.add_option("--listFields", type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help=" [%default]")
parser.add_option("--fieldType", type=str,
                  default='DDF',
                  help="field type [%default]")
parser.add_option('--runType', type=str,
                  default='spectroz',
                  help='run type  [%default]')
parser.add_option('--timescale', type=str,
                  default='year',
                  help='timescale of the files to process [%default]')
parser.add_option('--timeslots', type=str,
                  default='1-10',
                  help='time slot (season or year) to process [%default]')
parser.add_option('--dataType', type=str,
                  default='DataFrame',
                  help='data type [%default]')
parser.add_option('--dbList', type=str,
                  default='list_OS.csv',
                  help='list of OS to process [%default]')
parser.add_option('--selconfig', type=str,
                  default='G10_JLA',
                  help='selection [%default]')
parser.add_option("--outDir", type=str,
                  default='../sn_ddf',
                  help="output dir[%default]")

opts, args = parser.parse_args()

dataDir = opts.dataDir
fieldType = opts.fieldType
listFields = opts.listFields.split(',')
runType = opts.runType
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
dataType = opts.dataType
dbList = opts.dbList
selconfig = opts.selconfig
outDir = opts.outDir


checkDir(outDir)

# OS to process
dbNames = pd.read_csv(dbList, comment='#')

# load selection
sellist = selection_criteria()[selconfig]

# load data
res = pd.DataFrame()

outName = 'nsn_{}_{}.hdf5'.format(fieldType.lower(), selconfig)
for i, row in dbNames.iterrows():
    for tt in timeslots:
        rr = process_dbName(dataType, dataDir, row['dbName'], runType,
                            timescale, [tt], fieldType, listFields, sellist)
        res = pd.concat((res, rr))
    save_data(outDir, row['dbName'], res, outName)


print(res)
