#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:41:11 2025

@author: philippe.gris@clermont.in2p3.fr
"""
import pandas as pd
from sn_tools.sn_batchutils import BatchIt
from optparse import OptionParser


def get_combi(params, relat_err=[0.5, 1, 3, 5, 10, 15, 20]):
    """
    Function to get combi of parameters

    Parameters
    ----------
    params : dict
        parameters.
    relat_err : list(float), optional
        relative error for the parameters. The default is [0.5, 1, 3, 5, 10, 15, 20].

    Returns
    -------
    pandas df
      combination of parameters
    """

    df_err = pd.DataFrame(relat_err, columns=['relat_error'])

    df_err['relat_error'] /= 100.

    df_dict = {}
    for vv in ['pwv', 'ozone', 'aerosol']:
        dfa = pd.DataFrame([params[vv]], columns=[vv])

        dfa = dfa.merge(df_err, how='cross')
        dfa['sigma_{}'.format(vv)] = dfa[vv]*dfa['relat_error']

        dfa = dfa.drop(columns=[vv, 'relat_error'])
        df_dict[vv] = dfa

    df = df_dict['pwv'].merge(df_dict['ozone'], how='cross')
    df = df.merge(df_dict['aerosol'], how='cross')
    df['num_combi'] = df.index+1

    return df


parser = OptionParser(
    description='Script to estimate zp and mean_wave from atmos parameters')
parser.add_option('--telDir', type=str, default='throughputs',
                  help='tel main dir [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--throughDir', type=str, default='baseline',
                  help='throughput dir [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--pwv', type=float, default=4.0,
                  help='airmass value [%default]')
parser.add_option('--ozone', type=float, default=300,
                  help='ozone value [%default]')
parser.add_option('--aerosol', type=float, default=0.1,
                  help='airmass value [%default]')
parser.add_option('--beta', type=float, default=0.2,
                  help='beta value [%default]')
parser.add_option('--ntrial', type=int, default=20,
                  help='number of trials [%default]')

opts, args = parser.parse_args()

params = vars(opts)

df = get_combi(params)

print(df)
ntrial = params['ntrial']
airmass_step = 0.1
script = 'run_scripts/telescope/zero_points_atmos.py'
outDir = '/sps/lsst/users/gris/zp_atmos'
dd = {}

for vv in ['pwv', 'ozone', 'aerosol']:
    dd['{}_min'.format(vv)] = params[vv]

dd['ntrial'] = ntrial
dd['outDir'] = outDir
dd['airmass_step'] = airmass_step

for i, row in df.iterrows():
    num_combi = int(row['num_combi'])
    processName = 'zp_atmos_{}'.format(num_combi)
    mybatch = BatchIt(processName=processName)

    for tt in ['pwv', 'ozone', 'aerosol']:
        dd['sigma_{}_min'.format(tt)] = row['sigma_{}'.format(tt)]

    dd['outName'] = 'zp_atmos_config{}.hdf5'.format(num_combi)

    mybatch.add_batch(script, dd)

    mybatch.go_batch()
