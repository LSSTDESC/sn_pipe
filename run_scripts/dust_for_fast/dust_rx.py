#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:54:44 2026

@author: philippe.gris@clermont.in2p3.fr
"""
from rubin_sim.phot_utils import Bandpass
from rubin_sim.phot_utils import Sed
import os
import pandas as pd
from optparse import OptionParser

def r_x(r_v=3.1, bandpass_dict=None, ref_ebv=1.0,tel_dir='throughputs_1.9'):
    """
    Calculate extinction values

    Parameters
    ----------
    r_v : float, optional
        Extinction law parameters. The default is 3.1.
    bandpass_dict : dict, optional
        A dict with keys of filtername and values of
        rubin_sim.phot_utils.Bandpass objects.
        Default of None will load the standard ugrizy bandpasses. 
        The default is None.
    ref_ebv : float, optional
        The reference E(B-V) value to use. The default is 1.0.
    tel_dir : str, optional
        Throughput dir. The default is 'throughputs_1.9'.

    Returns
    -------
    r_x : float
        r_x value.
   
    Note
    ----
    The value that dust_values calls "ax1" is equivalent  to r_x in any filter.
    And  r_x * ebv = A_x (the extinction due to dust in any bandpass).
    DustValues.r_x is also provided as a copy of DustValues.ax1 ..
    eventually ax1 may be deprecated in favor of r_x.
    
    function inspired from the DustValues class in 
    https://github.com/lsst/rubin_sim/blob/main/rubin_sim/phot_utils/photometric_parameters.py
    """
    
    # Calculate dust extinction values
    ax1 = {}
    if bandpass_dict is None:
        bandpass_dict = {}
        root_dir = os.path.join(tel_dir, "baseline")
        for f in ["u", "g", "r", "i", "z", "y"]:
            bandpass_dict[f] = Bandpass()
            bandpass_dict[f].read_throughput(os.path.join(root_dir, f"total_{f}.dat"))

        for filtername in bandpass_dict:
            wavelen_min = bandpass_dict[filtername].wavelen.min()
            wavelen_max = bandpass_dict[filtername].wavelen.max()
            testsed = Sed()
            testsed.set_flat_sed(wavelen_min=wavelen_min, wavelen_max=wavelen_max, wavelen_step=1.0)
            # Calculate non-dust-extincted magnitude
            flatmag = testsed.calc_mag(bandpass_dict[filtername])
            # Add dust
            #a, b = testsed.setup_ccm_ab()
            a,b = testsed.setup_o_donnell_ab()
            testsed.add_dust(a, b, ebv=ref_ebv, r_v=r_v)
            # Calculate difference due to dust when EBV=1.0
            # (m_dust = m_nodust - Ax, Ax > 0)
            ax1[filtername] = testsed.calc_mag(bandpass_dict[filtername]) - flatmag
        # Add the R_x term, to start to transition toward this name.
        r_x = ax1.copy()
        
        return r_x
    
desc = 'script to estimate delta_mag due to galactic interstellar extinction'
parser = OptionParser(description=desc)

parser.add_option("--inputFile", type=str, default='dustmap_64.hdf5',
                  help="file with pixels and e(B-V) of MW values [%default]")    

opts, args = parser.parse_args()

theFile = opts.inputFile

data = pd.read_hdf(theFile)

#grab r_x

vv = r_x()

for key,vals in vv.items():
    data['delta_mag_{}'.format(key)] = vals*data['ebvofMW']
    
fName = theFile.split('/')[-1].split('.hdf5')[0]

outName = '{}_delta_mag_dust.hdf5'.format(fName)

data.to_hdf(outName,key='mag_dust')

    