#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:17:02 2024

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
from sn_saturation.psf_plot import plot_pixel

parser = OptionParser()
parser.add_option("--psf_type", type=str, default='single_gauss',
                  help="PSF (single_gauss/moffat)[%default]")

opts, args = parser.parse_args()

psf_type = opts.psf_type

plot_pixel(0.5, psf_type, 'xpixel', 0., 'ypixel', 0., 'xc',
           'yc', 'Moffat', type_plot='contour')
