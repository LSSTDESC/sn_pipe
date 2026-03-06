#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:29:38 2026

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_tools.sn_utils import X0_norm
from sn_tools.sn_io import check_get_dir           
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--model', type=str, default='salt3',
                  help='sncosmo model [%default]')
parser.add_option('--version', type=str, default='2.0',
                  help='sncosmo model version [%default]')
parser.add_option('--absmag', type=float, default=-19.0906,
                  help='sn abs mag [%default]')
parser.add_option('--web_path', type=str, 
                  default='https://me.lsst.eu/gris/DESC_SN_pipeline',
                  help='web path to grab SALT2 files [%default]')
parser.add_option('--salt2Dir', type=str, 
                  default='SALT2_Files',
                  help='SALT2 files directory [%default]')

opts, args = parser.parse_args()

model = opts.model
version=opts.version
version='2.0'
absmag=np.round(opts.absmag,4)
web_path = opts.web_path
salt2Dir = opts.salt2Dir

outFile = 'x0_norm_{}_{}.npy'.format(absmag,model)

check_get_dir(web_path, 'SALT2_Files', salt2Dir)
X0_norm(salt2Dir=salt2Dir, model=model, version=version,
        absmag=absmag, outfile=outFile)