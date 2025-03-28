#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 13:09:04 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from sn_tools.sn_utils import SimuParameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


params = {}

params['modelPar'] = {}

params['modelPar']['dirFile'] = 'reference_files'
params['modelPar']['nameFile'] = 'x1_color_G10.csv'
params['modelPar']['x1sigma'] = 0
params['modelPar']['colorsigma'] = 0


params['minRFphase'] = -20
params['maxRFphase'] = 60.
params['minRFphaseQual'] = -10.
params['maxRFphaseQual'] = 35.

params['z'] = {}
params['z']['rate'] = 'Hounsell'
params['z']['type'] = 'random'
params['z']['minsimu'] = 0.01
params['z']['maxsimu'] = 1.1
params['z']['min'] = 0.01
params['z']['max'] = 1.1
params['z']['step'] = 0.01
params['z']['weight'] = 'sn_rate'

params['NSNfactor'] = 10
params['NSNabsolute'] = 0
params['differentialFlux'] = False
params['type'] = 'SN_Ia'


params['daymax'] = {}
params['daymax']['type'] = 'random'
params['daymax']['step'] = 10

params['x1'] = {}
params['x1']['type'] = 'random'
params['x1']['min'] = -3.0
params['x1']['max'] = 3.0
params['x1']['step'] = 0.01

params['color'] = {}
params['color']['type'] = 'random'
params['color']['min'] = -0.3
params['color']['max'] = 0.3
params['color']['step'] = 0.01

cosmo_parameters = {}
cosmo_parameters['H0'] = 70.
cosmo_parameters['Om'] = 0.3

simupars = SimuParameters(params, cosmo_parameters)

mjds = np.arange(6000., 6000+210., 10)
obs = pd.DataFrame(mjds, columns=['mjd'])


sp = simupars.simuparams(obs)

print(sp, len(sp))

fig, ax = plt.subplots(figsize=(12, 8))

ax.hist(sp['z'], histtype='step', bins=20)

plt.show()
