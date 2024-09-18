#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:11:34 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from getObsAtmo.getObsAtmo import ObsAtmo
from sn_telmodel.sn_transtools import get_trans
import numpy as np
from sn_tools.sn_io import checkDir


def go_to_file(df, fName, dirOut=''):
    """
    Function to print trans results in file

    Parameters
    ----------
    df : pandas df
        Data to write.
    fName : str
        output file name.
    dirOut : str, optional
        output dir. The default is ''.

    Returns
    -------
    None.

    """

    fName_out = '{}/{}'.format(dirOut, fName)
    fout = open(fName_out, 'w')
    print('# Wavelength(nm)  Transmission(0-1)', file=fout)
    for i, row in df.iterrows():
        # fout.write(row['wl'], row['trans'])
        print(row['wl'], row['trans'], file=fout)

    fout.close()


airmass = np.arange(1.0, 2.6, 0.1)
# airmass = [1.2]
pwvs = [4.]
ozs = [300]
taus = [0.0]
beta = 1.4

outputDir = 'throughputs_1.9/atmos_new'

checkDir(outputDir)
# emulate LSST
emul = ObsAtmo('LSST', 743.0)

for am in airmass:
    am = np.round(am, 1)
    for pwv in pwvs:
        for oz in ozs:
            for tau in taus:
                trans = get_trans(
                    am, pwv, oz, tau, beta, colname=['wl', 'trans'], emul=emul)
                fName = 'airmass_{}_pwv_{}_oz_{}_aero_{}_beta_{}.dat'.format(
                    am, pwv, oz, tau, beta)
                trans = trans.round({'wl': 1, 'trans': 8})
                go_to_file(trans, fName, outputDir)
