#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:07:50 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import numpy as np
import pandas as pd
from optparse import OptionParser

parser = OptionParser('script to transform npy effi(z) arry to cvs files')

parser.add_option("--dataDir", type=str,
                  default='../desi_desc_efficiencies_v2_with_desi2_strategies',
                  help="data dir[%default]")
parser.add_option("--fName", type=str,
                  default='host_efficiency_desi.npy',
                  help="file name to process [%default]")
parser.add_option("--outDir", type=str,
                  default='input/cosmology/host_effi',
                  help="output dir dir[%default]")
parser.add_option("--tagName", type=str,
                  default='DESI',
                  help="output tag name: host_effi_tagName [%default]")


opts, args = parser.parse_args()

nname = '{}/{}'.format(opts.dataDir, opts.fName)

redshift, efficiency = np.load(nname)

df = pd.DataFrame(redshift, columns=['z'])
df['effi'] = efficiency

fullpath = '{}/host_effi_{}.csv'.format(opts.outDir, opts.tagName)
df.to_csv(fullpath, index=False)
