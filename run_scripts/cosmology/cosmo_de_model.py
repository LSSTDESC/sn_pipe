#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:29:11 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser

parser = OptionParser(description='Script to analyze SN selection criteria')

parser.add_option('--de_params', type=str,
                  default='w0,wa',
                  help='DE eos parameters [%default]')
parser.add_option('--de_values', type=str,
                  default='-1.,0.',
                  help='DE eos parameter values [%default]')
parser.add_option('--de_class', type=str,
                  default='w0waCDM',
                  help='DE class to use [%default]')
parser.add_option('--class_loc', type=str,
                  default='astropy.cosmology',
                  help='DE class location [%default]')
parser.add_option('--de_model', type=str,
                  default='CPL',
                  help='DE eos model [%default]')
parser.add_option('--H0', type=float,
                  default='CPL',
                  help='DE eos model [%default]')
parser.add_option('--Om0', type=float,
                  default=0.30,
                  help='Omega_matter [%default]')

opts, args = parser.parse_args()

opts_dict = vars(opts)

print(opts_dict)