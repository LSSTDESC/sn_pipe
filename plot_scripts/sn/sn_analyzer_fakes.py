#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:30:27 2023

@author: philippe.gris@clermont.in2p3.fr
"""
import matplotlib.pyplot as plt
import operator
import glob
from optparse import OptionParser
from sn_analysis.sn_zlim import plot_delta_zlim, plot_delta_nsn, get_data


parser = OptionParser()

parser.add_option("--inputDir", type="str",
                  default='Output_SN/Fakes', help="input dir [%default]")
parser.add_option("--plot_delta_zlim", type=int, default=0,
                  help="to plot delta_zlim [%default]")
parser.add_option("--plot_delta_nsn", type=int, default=1,
                  help="to plot delta_nsn [%default]")
parser.add_option("--zmin", type=float, default=0.8,
                  help="zmin to plot delta_nsn [%default]")

opts, args = parser.parse_args()

theDir = opts.inputDir
zmin = opts.zmin

fis = glob.glob('{}/SN_*.hdf5'.format(theDir))
# theFile = 'SN_conf_z_moon_-1_full_salt3.hdf5'


df_dict = get_data(theDir, fis, nproc=1)
print(df_dict.keys())

dict_sel = {}

dict_sel['G10'] = [('n_epochs_m10_p35', operator.ge, 4),
                   ('n_epochs_m10_p5', operator.ge, 1),
                   ('n_epochs_p5_p20', operator.ge, 1),
                   ('n_bands_m8_p10', operator.ge, 2),
                   ('sigmaC', operator.le, 0.04),
                   ]

dict_sel['metric'] = [('n_epochs_bef', operator.ge, 4),
                      ('n_epochs_aft', operator.ge, 10),
                      ('n_epochs_phase_minus_10', operator.ge, 1),
                      ('n_epochs_phase_plus_20', operator.ge, 1),
                      ('sigmaC', operator.le, 0.04),
                      ]


if opts.plot_delta_zlim:
    plot_delta_zlim(df_dict, dict_sel, 'metric')

if opts.plot_delta_nsn:
    plot_delta_nsn(df_dict, dict_sel, 'metric', zmin=zmin)

plt.show()
