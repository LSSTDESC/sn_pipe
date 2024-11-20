#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:15:04 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_telmodel.sn_atmosphere import Atmos_Transmission
import matplotlib.pyplot as plt

atmos_trans = Atmos_Transmission()

atmos_trans.load_atmosphere()

atmos_trans.plot_atmospheric_transmission()

plt.show()
