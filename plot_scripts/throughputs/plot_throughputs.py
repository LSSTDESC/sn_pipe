#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:04:53 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_telmodel.sn_throughputs_new import Throughputs
import matplotlib.pyplot as plt


throughputs = Throughputs()

throughputs.load_atmosphere()

throughputs.plot_atmospheric_transmission(plt)

plt.show()
