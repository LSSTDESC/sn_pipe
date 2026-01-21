#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:58:22 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import glob
import pandas as pd
from sn_analysis.sn_selection import selection_criteria
from sn_analysis.sn_tools import complete_df, get_pulls
import numpy as np
import re
import operator
from sn_tools.sn_io import checkDir


def load_data(dbDir, dbName, runType, field):
    """
    Function to load the data

    Parameters
    ----------
    dbDir : str
        data dir.
    dbName : str
        OS to process.
    runType : str
        runtype.
    field : str
        field.

    Returns
    -------
    df : pandas df
        loaded data.

    """

    theDir = '{}/{}/{}'.format(dbDir, dbName, runType)

    ('scanning', theDir)
    fis = glob.glob('{}/*{}*.hdf5'.format(theDir, field))

    df = pd.DataFrame()

    for fi in fis:

        df_ = pd.read_hdf(fi)

        df = pd.concat((df, df_))

    return df


def get_nsn(vala, valb, norm_factor):
    """
    Function to grab infos

    Parameters
    ----------
    vala : int
        number of sn after sel.
    valb : int
        number of sn before sel.
    norm_factor : float
        normalization factor.

    Returns
    -------
    list
        DESCRIPTION.

    """

    effi = vala/valb

    err_effi = np.sqrt(effi*(1.-effi)/valb)

    nsn = int(effi*valb/norm_factor)

    err_nsn = int(err_effi*valb/norm_factor)

    effi *= 100.
    err_effi *= 100.

    res = [(nsn, err_nsn, np.round(effi, 1), np.round(err_effi, 1))]
    cols = ['nsn', 'err_nsn', 'effi', 'err_effi']

    return pd.DataFrame(res, columns=cols)


def select_str(res, list_sel):
    """
    Function to select a pandas df

    Parameters
    ----------
    res : pandas df
        data to select.

    Returns
    -------
    pandas df
        selected df.

    """
    idx = True
    for vals in list_sel:
        idx &= vals[1](res[vals[0]], vals[2])
        mystr = '{} {} {}'.format(
            vals[0], get_symbol(vals[1].__doc__), vals[2])

    return mystr, res[idx]


def get_symbol(opdoc):
    """
    function to estimate symbol from operator.__doc__

    Parameters
    ----------
    opdoc : str
        operator.__doc__.

    Returns
    -------
    sym : str
        corresponding sym.

    """
    # sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', getattr(operator, op).__doc__)
    sym = re.sub(r'.*\w\s?(\S+)\s?\w.*', '\\1', opdoc)
    if re.match('^\\W+$', sym):
        return sym


def process_season(data, seas, field, norm_factor):
    """
    Function to process a season

    Parameters
    ----------
    data : pandas df
        Data to process.
    seas : int
        season number.
    field : str
        Field of interest.
    norm_factor : float
        normalization factor.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    idx = data['season'] == seas
    mysel = data[idx]

    n_nosel = int(len(mysel)/norm_factor)
    print('no sel', n_nosel)
    ra = get_pulls(mysel)
    ra['sel_str'] = 'nosel'
    ra['field'] = field
    ra['season'] = seas
    # dfa = pd.concat((dfa, ra))

    ro = get_nsn(len(mysel), len(mysel), norm_factor)
    ro['sel_str'] = 'nosel'
    ro['field'] = field
    ro['season'] = seas
    # dfb = pd.concat((dfb, ro))
    # get_pulls(mysel)
    for i in range(1, len(sellist)+1):
        # ro = [field, int(seas)]
        mystr, sel = select_str(mysel, sellist[:i])
        rasel = get_pulls(sel)
        rasel['sel_str'] = mystr
        rasel['field'] = field
        rasel['season'] = seas
        ra = pd.concat((ra, rasel))
        rosel = get_nsn(len(sel), len(mysel), norm_factor)
        rosel['sel_str'] = mystr
        rosel['field'] = field
        rosel['season'] = seas
        # dfb = pd.concat((dfb, ro))
        ro = pd.concat((ro, rosel))

    # merge the two Dataframes

    df_effi = ro.merge(ra,
                       left_on=['field', 'season', 'sel_str'],
                       right_on=['field', 'season', 'sel_str'],
                       suffixes=['', ''])

    return df_effi


def process_season_field_pixel(grp, norm_factor):
    """
    Function to process a season

    Parameters
    ----------
    data : pandas df
        Data to process.
    norm_factor : float
        normalization factor.

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    n_nosel = int(len(grp)/norm_factor)
    print('no sel', n_nosel)
    ra = get_pulls(grp)
    ra['sel_str'] = 'nosel'
    # dfa = pd.concat((dfa, ra))
    print('hh', ra)
    ro = get_nsn(len(grp), len(grp), norm_factor)
    ro['sel_str'] = 'nosel'
    # dfb = pd.concat((dfb, ro))
    # get_pulls(mysel)
    for i in range(1, len(sellist)+1):
        # ro = [field, int(seas)]
        mystr, sel = select_str(grp, sellist[:i])
        rasel = get_pulls(sel)
        rasel['sel_str'] = mystr
        ra = pd.concat((ra, rasel))
        rosel = get_nsn(len(sel), len(grp), norm_factor)
        rosel['sel_str'] = mystr
        # dfb = pd.concat((dfb, ro))
        ro = pd.concat((ro, rosel))

    # merge the two Dataframes

    df_effi = ro.merge(ra,
                       left_on=['sel_str'],
                       right_on=['sel_str'],
                       suffixes=['', ''])

    return df_effi


def process_db(dbDir, dbName, runType, fields,
               norm_factor, zmin=0.01, zmax=1.1):
    """
    Function to process OS data

    Parameters
    ----------
    dbDir : str
        Data dir.
    dbName : str
        OS to process.
    runType : str
        run type.
    fields : list(str)
        List of fields to process.
    norm_factor : float
        normalization factor.
    zmin: float, optional.
        redshift min for data. The default is 0.01.
    zmax: float, optional.
        redshift max for data. The default is 1.11.   

    Returns
    -------
    df_effi : pandas df
        processed data.

    """

    df_effi = pd.DataFrame()

    for field in fields:
        data = load_data(dbDir, dbName, runType, field)
        data = complete_df(data)

        idxz = data['z'] >= zmin
        idxz &= data['z'] <= zmax

        data = data[idxz]
        # print(field, len(data), len(data)/norm_factor)

        seasons = data['season'].unique()

        idx = data['healpixID'] == 108958

        print('fff', data.columns)
        df_effi = data[idx].groupby(['healpixID', 'season']).apply(
            lambda x: process_season_field_pixel(x, norm_factor), include_groups=False).reset_index()
        """
        for seas in seasons:
            # print('processing', zmin, zmax, seas)
            dd = process_season(data, seas, field, norm_factor)
            df_effi = pd.concat((df_effi, dd))
        """
    df_effi['dbName'] = dbName
    df_effi['zmin'] = np.round(zmin, 2)
    df_effi['zmax'] = np.round(zmax, 2)
    df_effi['field'] = field

    return df_effi


parser = OptionParser(description='Script to analyze SN selection criteria')

parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str,
                  default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--runType', type=str,
                  default='DDF_spectroz',
                  help='run type [%default]')
parser.add_option('--timescale', type=str,
                  default='season',
                  help='timescale [%default]')
parser.add_option('--seasons', type=str,
                  default='1',
                  help='seasons/years to process [%default]')
parser.add_option('--fields', type=str,
                  default='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFS_a,EDFS_b',
                  help='fields to process [%default]')
parser.add_option('--norm_factor', type=float,
                  default=30.,
                  help='normalization factor [%default]')
parser.add_option("--selconfig", type=str,
                  default='G10_JLA', help="sel config name [%default]")
parser.add_option("--zrange", type=int,
                  default=0, help="to process data per zrange [%default]")
parser.add_option('--outDir', type=str,
                  default='../effi_pull_stat',
                  help='output Dir dir[%default]')

opts, args = parser.parse_args()

dbDir = opts.dbDir
dbName = opts.dbName
runType = opts.runType
runType = opts.runType
timescale = opts.timescale
seasons = opts.seasons.split(',')
seasons = list(map(int, seasons))
fields = opts.fields.split(',')
norm_factor = opts.norm_factor
selconfig = opts.selconfig
zrange = opts.zrange
outDir = opts.outDir

# create output dir (if necessary)
checkDir(outDir)

# selection criteria
sellist = selection_criteria()[selconfig]

# add criteria
sellist.append(('Nfilt_2', operator.ge, 3, 7))
# sellist.append(('Nfilt_5', operator.ge, 2, 7))
# sellist.append(('sigmaC', operator.le, 0.04, 7))

print(sellist)
rb = []
# dfa = pd.DataFrame()
# dfb = pd.DataFrame()

zmin = 0.0
zmax = 1.1
deltaz = 1.1

if zrange:
    deltaz = 0.10

zvals = np.arange(zmin, zmax, deltaz)

# zvals[0] += 0.01
print(zvals)

for vv in zvals:
    zmi = vv
    if zmi < 0.001:
        zmi = 0.01
    zma = vv+deltaz
    df_effi = process_db(dbDir, dbName, runType, fields,
                         norm_factor, zmin=zmi, zmax=zma)


# save the data
outName = '{}/{}.hdf5'.format(outDir, dbName)
df_effi.to_hdf(outName, key='effi_pull')
