#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:02:11 2025

@author: philippe.gris@clermont.in2p3.fr
"""

from optparse import OptionParser
import numpy as np
from sn_telmodel.sn_throughputs import get_telescope
import pandas as pd
from rubin_sim.skybrightness import SkyModel
from rubin_scheduler.utils import Site
import matplotlib.pyplot as plt

parser = OptionParser(
    description='Script to estimate m5 from atmos parameters')

parser.add_option('--fileDir', type=str, default='../DB_Files',
                  help='OS file dir [%default]')
parser.add_option('--dbName', type=str, default='baseline_v4.3.1_10yrs',
                  help='OS name [%default]')
parser.add_option('--tag', type=str, default='1.9',
                  help='tag version of the throughputs [%default]')
parser.add_option('--tel_dir', type=str, default='throughputs',
                  help='main throughputs location dir [%default]')
parser.add_option('--throughputsDir', type=str, default='baseline',
                  help='throughputs location dir [%default]')
parser.add_option('--atmosDir', type=str, default='atmos',
                  help='atmosphere location dir [%default]')
parser.add_option('--airmass', type=float, default=1.2,
                  help='airmass value [%default]')
parser.add_option('--aerosol', type=float, default=0.04,
                  help='aerosol value [%default]')
parser.add_option('--pwv', type=float, default=5.0,
                  help='precipitable water vapor value [%default]')
parser.add_option('--ozone', type=float, default=320.,
                  help='ozone value [%default]')
parser.add_option('--gain', type=float, default=2.5,
                  help='electronic gain [%default]')
parser.add_option('--pressure', type=float, default=743.,
                  help='pressure on the Cerro Pachon [%default]')
parser.add_option('--atmos_type', type=str, default='obsatmo',
                  help='how is the atmos estimated (obsatmo, from_file) [%default]')
opts, args = parser.parse_args()

fileDir = opts.fileDir
dbName = opts.dbName
tag = opts.tag
tel_dir = opts.tel_dir
throughputsDir = opts.throughputsDir
atmosDir = opts.atmosDir
airmass = opts.airmass
pwv = opts.pwv
ozone = opts.ozone
aerosol = opts.aerosol
gain = opts.gain
pressure = opts.pressure
atmos_type = opts.atmos_type

# throughputs instance
telb = '{}_{}'.format(tel_dir, tag)
through_dir = '{}/{}'.format(telb, throughputsDir)
atmos_dir = '{}/{}'.format(telb, atmosDir)
throughput = get_telescope(tel_dir=telb,
                           through_dir=through_dir,
                           atmos_dir=atmos_dir,
                           atmos_type=atmos_type,
                           tag=tag, load_components=True,
                           airmass=airmass, aerosol=aerosol,
                           pwv=pwv, ozone=ozone, gain=gain, pressure=pressure)
# setup rubin tools
throughput.etc()
# configure your Site, will default to Rubin if you use Site()
site = Site(name='LSST')
# site = Site(latitude=33.35, longitude=116.85, height=1706)

# sky model (Rubin)
sky_model = SkyModel(observatory=site,
                     twilight=True,
                     zodiacal=False,
                     moon=False,
                     airglow=False,
                     merged_spec=False,
                     lower_atm=False,
                     upper_atm=False)


# grab OS data
fName = '{}/{}.npy'.format(fileDir, dbName)
os_data = np.load(fName)

io = 0
for x in os_data:
    print(x.dtype.names)
    io += 1
    throughput.reset_data()
    b = x['filter']
    fwhmeff = x['seeingFwhmEff']
    m5 = x['fiveSigmaDepth']
    exptime = x['exptime']
    nexp = x['numExposures']
    airmass = x['airmass']
    sky = x['sky']
    moonPhase = x['moonPhase']
    lon = x['RA']
    lat = x['Dec']
    mjd = x['mjd']

    print(lon, lat)
    sky_model.set_ra_dec_mjd(lon=lon, lat=lat, mjd=mjd, degrees=True)

    # get sky spectrum
    wave, flux = sky_model.return_wave_spec()

    # print('fff', flux)
    """
    fig, ax = plt.subplots()
    throughput.plot_darksky(plt, fig, ax)
    """
    # load the night sky
    # throughput.load_nightsky(wave, flux)

    """
    throughput.plot_darksky(plt, fig, ax)

    plt.show()
    """

    throughput.data['FWHMeff'][b] = fwhmeff
    throughput.data['FWHMeff'][b] = 0.83
    """
    throughput.load_atmosphere(
        airmass=airmass, pwv=pwv, ozone=ozone, aerosol=aerosol)
    """
    throughput.new_atmosphere(
        airmass=1.2, atmos_type='from_file')

    # throughput.plot_throughputs(plt)

    # plt.show()
    sky_new = throughput.mag_sky(b)
    # m5_new = throughput.m5(b, exptime*nexp, nexp)
    m5_new = throughput.m5(b, 30, 1)

    # sky_last = throughput.mag_sky(b)
    print('new sky', sky_new, m5_new, exptime, nexp)

    """
    print(sky_model.get_computed_vals())

    print(np.deg2rad(lon), np.deg2rad(lat), b, airmass, fwhmeff, m5, m5_new, m5-m5_new,
          sky, sky_new, sky-sky_new, moonPhase)

    # sky model (Rubin)
    """
    siteb = Site(name='LSST')
    sky_modelb = SkyModel(observatory=siteb, mags=True,
                          twilight=True,
                          zodiacal=False,
                          moon=False,
                          airglow=False,
                          merged_spec=False,
                          lower_atm=False,
                          upper_atm=False,
                          scattered_star=False)
    sky_modelb.set_ra_dec_mjd(lon=lon, lat=lat, mjd=mjd, degrees=True)
    # sky_modelb.set_ra_dec_mjd(lon=270., lat=30., mjd=61200.75, degrees=True)
    mags = sky_modelb.return_mags()

    print(mags[b])

    # throughput.data['FWHMeff'][b] = fwhmeff
    # throughput.etc()

    if io > 2:
        break
