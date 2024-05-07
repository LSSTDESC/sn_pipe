#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:30:54 2024

@author: philippe.gris@clermont.in2p3.fr
"""

import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
from sn_plotters_flux_pixels.plot_flux_pixels import plot_frac_flux

parser = OptionParser()
parser.add_option("--fNames", type=str,
                  default='psf_pixel_single_gauss_summary.npy,psf_pixel_moffat_summary.npy',
                  help="PSF files [%default]")
parser.add_option("--labels", type=str,
                  default='single gaussian PSF,Moffat PSF',
                  help="PSF files [%default]")

opts, args = parser.parse_args()
fNames = opts.fNames.split(',')
labels = opts.labels.split(',')

fig, ax = plt.subplots(figsize=(12, 8))
arra = np.load(fNames[0])
arrb = np.load(fNames[1])

print(arra.dtype)


plot_frac_flux(arra, fig=fig, ax=ax, label=labels[0], ls='solid')
plot_frac_flux(arrb, fig=fig, ax=ax, label=labels[1], ls='dashed')

ax.set_xlabel(r'seeing ["]')
ax.set_ylabel(r'Max frac pixel flux')
ax.grid(visible=True)
ax.set_ylim([0, None])
ax.legend()
plt.show()
