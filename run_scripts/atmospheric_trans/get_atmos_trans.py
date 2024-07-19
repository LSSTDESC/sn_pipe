#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:10:32 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from getObsAtmo.getObsAtmo import ObsAtmo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_trans(am, pmw, oz, tau=0., beta=1.4, colname=['Wavelength(nm)', 'Throughput(0-1)']):

    wl = list(np.arange(300., 1100., 0.1))
    transm = emul.GetAllTransparencies(wl, am, pwv, oz, tau=tau, beta=beta)
    df = pd.DataFrame(wl, columns=[colname[0]])
    df[colname[1]] = transm

    return df


emul = ObsAtmo('LSST', 743.0)

am = 1.2  # set the airmass
pwv = 4.0  # set the precipitable water vapor in mm
oz = 300.  # set the ozone depth on DU
tau = 0.1  # the vertical aerosol depth of each component at lambda0 vavelength, default set to 0.0 for no aerosol component
beta = 1.4  # the angstrom exponent. Must be positive in the range 0., 3.

df = get_trans(am, pwv, oz, tau=tau, beta=beta, colname=['wl', 'trans'])

dfb = get_trans(am, pwv, oz, tau=0., beta=beta, colname=['wl', 'trans'])
# emul.plot_transmission()  # plot the transmission

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(df['wl'], df['trans'])
ax.plot(dfb['wl'], dfb['trans'])

plt.show()
