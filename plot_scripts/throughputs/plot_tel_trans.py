#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:41:08 2024

@author: philippe.gris@clermont.in2p3;fr
"""

from sn_telmodel.sn_telescope import Telescope
import matplotlib.pyplot as plt


telescope = Telescope()
telescope.plot_components()

plt.show()
