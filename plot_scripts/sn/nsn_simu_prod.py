#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:13:13 2024

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
import pandas as pd
import glob


def load_csv(fDir, fieldType):
    """
    Function to load csv files

    Parameters
    ----------
    fDir : str
        file directory.
    fieldType : str
        field type.

    Returns
    -------
    df : pandas df
        output data.

    """

    fis = glob.glob('{}/*{}*.csv'.format(fDir, fieldType))

    df = pd.DataFrame()

    for fi in fis:
        dfa = pd.read_csv(fi, comment='#')
        df = pd.concat((df, dfa))

    return df


parser = OptionParser(
    description='Script to estimate the number of simulated SNe Ia')

parser.add_option('--fileDir', type=str,
                  default='../nsn_simu_prod',
                  help='files location dir[%default]')

opts, args = parser.parse_args()

fileDir = opts.fileDir

df_wfd = load_csv(fileDir, 'WFD')
df_ddf = load_csv(fileDir, 'DD')

nsn_wfd = df_wfd['nsn_simu'].sum()
nsn_ddf = df_ddf['nsn_simu'].sum()
nsn_tot = nsn_wfd+nsn_ddf

print('number of SNe Ia simulated:', nsn_tot, 'DDF:', nsn_ddf, 'WFD:', nsn_wfd)
