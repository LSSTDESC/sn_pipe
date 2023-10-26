#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:34:59 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import numpy as np
from optparse import OptionParser
from sn_tools.sn_obs import season
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=Warning)


def get_info(grp, DDlist, col='fiveSigmaDepth'):

    r = []
    for dd in DDlist:
        idx = grp['note'] == dd
        sel = grp[idx]
        vv = str(sel[col].values[0])
        if len(vv) < 5:
            vv += '0'
        r.append(str(vv))

    res = '/'.join(r)

    return pd.DataFrame({'m5': [res]})


def print_latex(grp, bands='ugrizy'):

    seas = grp.name
    for io, b in enumerate(bands):
        idx = grp['filter'] == b
        sel = grp[idx]
        tp = ' &{} &{} \\\\'.format(b, sel['m5'].values[0])
        if io == 2:
            tp = '{} {}'.format(seas, tp)

        print(tp)
    print('\\hline')


parser = OptionParser(description='Script to analyze Observing Strategy')

parser.add_option('--dbDir', type=str, default='../DB_Files',
                  help='OS location dir [%default]')
parser.add_option('--dbName', type=str,
                  default='DDF_DESC_0.80_SN_rand_m5_0.07',
                  help='db name [%default]')
parser.add_option('--dbExtens', type=str, default='npy',
                  help='db extension [%default]')
parser.add_option('--DDlist', type=str,
                  default='DD:COSMOS,DD:XMM_LSS,DD:ECDFS,DD:ELAISS1,DD:EDFS_a,DD:EDFS_b',
                  help='FF fields to consider[%default]')
opts, args = parser.parse_args()
DDlist = opts.DDlist.split(',')

fullName = '{}/{}.{}'.format(opts.dbDir, opts.dbName, opts.dbExtens)

data = np.load(fullName, allow_pickle=True)

# select DDF
idx = np.in1d(data['note'], DDlist)
data = data[idx]

# get seasons
data_seas = None
for ddf in DDlist:
    idx = data['note'] == ddf
    sel = data[idx]
    seas_sel = season(sel)
    if data_seas is None:
        data_seas = seas_sel
    else:
        data_seas = np.concatenate((data_seas, seas_sel))


df = pd.DataFrame.from_records(data_seas)

dfres = df.groupby(['note', 'filter', 'season'])[
    'fiveSigmaDepth'].median().reset_index()

dfres = dfres.round({'fiveSigmaDepth': 2})

tt = dfres.groupby(['season', 'filter']).apply(
    lambda x: get_info(x, DDlist)).reset_index()


caption = '\\fivesig~values used in the simulations.'
caption += ' These values are extracted from the LSST simulation baseline\\_v3.0\\_10yrs.'
label = 'tab:fivesigmasim'
print('\\begin{table}[!htbp]')
print('\\begin{center}')
print('\\caption{}\\label{}'.format('{'+caption+'}', '{'+label+'}'))
print('\\begin{tabular}{c|c|c}')
print('\\hline')
print('\\hline')
print('season & band & \\fivesig~single exposure\\\\')
print('\\hline')

bb = tt.groupby(['season']).apply(lambda x: print_latex(x))
print('\\hline')
print('\\end{tabular}')
print('\\end{center}')
print('\\end{table}')
