#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:57:11 2023

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
from sn_analysis import plt
import operator
from sn_analysis.sn_calc_plot import zlimit
from sn_analysis.sn_zlim import get_data, gime_zlim, plot_2D
from optparse import OptionParser
import glob
import pandas as pd


def zlim_multi(df_dict, combis, nproc=8):

    from sn_tools.sn_utils import multiproc
    params = {}
    params['df_dict'] = df_dict
    params['combis'] = combis
    res = multiproc(list(df_dict.keys()), params, zlim_indiv, nproc)

    return res


def zlim_indiv(keys, params, j, output_q):

    r = []
    df_dict = params['df_dict']
    combis = params['combis']

    for key in keys:
        vals = df_dict[key]
        zlimb = zlimit(vals)
        kkey = int(key.split('_')[1])
        combi_name = 'combi_{}'.format(kkey)
        idx = combis['tagName'] == combi_name
        Nvisits = combis[idx]['Nvisits'].values[0]
        #print('allo', combis[idx]['Nvisits'], combi_name)
        r.append((kkey, zlimb, Nvisits))

    res = np.rec.fromrecords(r, names=['config', 'zlim', 'Nvisits'])

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


parser = OptionParser()

parser.add_option("--inputDir", type="str",
                  default='Output_SN/Fakes', help="input dir [%default]")
parser.add_option("--combiFile", type="str",
                  default='combi1.csv', help="combi file [%default]")

opts, args = parser.parse_args()

inputDir = opts.inputDir
combiFile = opts.combiFile

fis = glob.glob('{}/SN*.hdf5'.format(inputDir))

df_dict = get_data(inputDir, fis)

dict_sel = {}

dict_sel['G10'] = [('n_epochs_m10_p35', operator.ge, 4),
                   ('n_epochs_m10_p5', operator.ge, 1),
                   ('n_epochs_p5_p20', operator.ge, 1),
                   ('n_bands_m8_p10', operator.ge, 2),
                   ('sigmaC', operator.le, 0.04),
                   ]

combis = pd.read_csv(combiFile)
ccols = []
for b in 'grizy':
    ccols.append('Nvisits_{}'.format(b))

combis['Nvisits'] = combis[ccols].sum(axis=1)

print(combis['Nvisits'])
"""
dictb = {}
dictb['combi_212'] = df_dict['combi_212']
"""
res = zlim_multi(df_dict, combis, nproc=1)

"""
Nvisits_ref = 292.
idx = np.abs(res['Nvisits']-Nvisits_ref) < 1.
sel = res[idx]
imax = np.argmax(sel['zlim'])
print(sel)
print(sel[imax])
confName = 'combi_{}'.format(sel[imax]['config'])
print('allo', confName)
ido = combis['tagName'] == confName
cols = []
for b in 'ugrizy':
    cols.append('Nvisits_{}'.format(b))
print(combis[ido][cols])
"""

fig, ax = plt.subplots()
ax.plot(res['Nvisits'], res['zlim'], 'ko')
plt.show()
