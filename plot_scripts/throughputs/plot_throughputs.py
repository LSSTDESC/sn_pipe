#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:04:53 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_telmodel.sn_throughputs import Throughputs
import matplotlib.pyplot as plt


def plots(throughputs):
    """
    Function to plot throughputs (telescope+atmosphere)

    Parameters
    ----------
    throughputs : Throughputs
        Throughputs class.

    Returns
    -------
    None.

    """

    # optical components
    throughputs.plot_components()

    # atmospheric transmission
    throughputs.load_atmosphere()

    throughputs.plot_atmospheric_transmission(plt)

    # plot throughputs

    throughputs.plot_throughputs(plt)

    # plot darksky
    throughputs.plot_darksky(plt)


throughputs = Throughputs()

# plot throughputs
# plots(throughputs)


# get etc
throughputs.etc()

# reset data
throughputs.reset_data()

# new atmos
throughputs.new_atmosphere(airmass=1.5)

# get etc

throughputs.etc()

plt.show()
