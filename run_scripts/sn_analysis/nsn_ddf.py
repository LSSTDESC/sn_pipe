#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:25:03 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
from sn_plotter_analysis.sn_analyser_summary import process_DDF_db
from sn_plotter_analysis.sn_analyser_ddf import get_val
from sn_plotter_analysis.sn_analyser_tools import Estimate_NSN
from sn_plotter_analysis.sn_analyser_tools import clean_level
from sn_tools.sn_io import checkDir
from sn_plotter_analysis.sn_analyser_ddf import pixelSize


def get_nsn(data, norm_factor, nside=128):
    """
    Function to estimate the number of SNe Ia + errors

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    norm_factor : TYPE
        DESCRIPTION.

    Returns
    -------
    res_fi : TYPE
        DESCRIPTION.

    """

    nsn = Estimate_NSN(norm_factor=norm_factor)

    # get nsn - no cuts
    resa = nsn(data)
    resa = clean_level(resa)

    # get nsn - z >= 0.8
    idx = data['z'] >= 0.8
    sel = data[idx]
    resb = nsn(sel)
    resb = clean_level(resb)
    resb = resb.rename(columns={'nsn': 'nsn_z_08', 'err_nsn': 'err_nsn_z_08'})

    # get nsn - z >= 0.8 and sigmaC <= 0.04
    idx = data['z'] >= 0.8
    idx &= data['sigmaC'] <= 0.04
    sel = data[idx]
    resc = nsn(sel)
    resc = clean_level(resc)
    resc = resc.rename(
        columns={'nsn': 'nsn_z_08_sigmaC', 'err_nsn': 'err_nsn_z_08_sigmaC'})

    cols = ['dbName', 'field', 'healpixID', 'season', 'year']
    res_fi = resa.merge(resb, left_on=cols, right_on=cols, suffixes=['', ''])

    res_fi = res_fi.merge(resc, left_on=cols, right_on=cols, suffixes=['', ''])

    res_fi['survey_area'] = pixelSize(nside)

    return res_fi


parser = OptionParser(
    description='Script to estimate SN - DDF after selection')
parser.add_option('--dbDir', type=str,
                  default='../Output_SN_DD_sigmaInt_0.0_Hounsell_z_smflux_notelrot_G10_JLA',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str,
                  default='ddf_list.csv',
                  help='OS DD list[%default]')
parser.add_option('--norm_factor', type=int,
                  default=30,
                  help='normalization factor [%default]')
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
parser.add_option('--nside', type=int,
                  default=128,
                  help='nside healpix parameter [%default]')
parser.add_option('--outDir', type=str,
                  default='../sn_summary_ddf',
                  help='output dir for the file to store [%default]')
parser.add_option('--fileName', type=str,
                  default='sn_summary_ddf.hdf5',
                  help='sn file name to draw [%default]')


opts, args = parser.parse_args()

dbDir = opts.dbDir
norm_factor = opts.norm_factor
runType = opts.runType
dbList = opts.dbList
timeslots = opts.timeslots
timescale = opts.timescale
timeslots = get_val(timeslots)
nside = opts.nside
outputDir = opts.outDir
fileName = opts.fileName
dataType = opts.dataType


# create dir if necessary
checkDir(outputDir)

# load os to process
conf_df = pd.read_csv(dbList, comment='#')

for i, row in conf_df.iterrows():
    dbName = row['dbName']
    ddf = process_DDF_db(dbName, dataType, dbDir, runType,
                         timescale, timeslots, norm_factor)

    df_nsn = get_nsn(ddf, norm_factor, nside)

    outt = '{}/{}'.format(outputDir, dbName)
    checkDir(outt)
    fName = '{}/{}'.format(outt, fileName)
    df_nsn.to_hdf(fName, key='ddf')
