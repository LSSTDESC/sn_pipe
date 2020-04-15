import os
from sn_design_dd_survey.ana_file import Anadf
import h5py
from astropy.table import Table, vstack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser


def plotLC(filename, x1=-2.0, color=0.2, corrFisher=True):
    """
    Method to plot sigmac vs z and SNRb vs sigmaC

    Parameters
    ---------------
    filename: str
      name of LC file
    x1: float, opt
      SN x1 (default: -2.0)
    color: float, opt
      SN color (default: 0.2)
    corrFisher: bool, opt
      if True Fisher elements are corrected so as to use the background dominated limit


    """

    f = h5py.File(filename, 'r')
    # get the keys
    keys = list(f.keys())

    print(keys)
    lc = Table.read(filename, path=keys[0])
    lc = pd.DataFrame(np.copy(lc))

    lc['band'] = lc['band'].map(lambda x: x.decode()[-1])
    # lc['band'] = lc['band'].astype('|S')
    print(lc.dtypes, lc['band'])

    lc['sigma_corr'] = (lc['flux_e_sec']/lc['snr_m5'])/(lc['flux_5']/5.)

    if corrFisher:
        for vv in ['F_x0x0', 'F_x0x1', 'F_x0daymax',
                   'F_x0color', 'F_x1x1', 'F_x1daymax',
                   'F_x1color', 'F_daymaxdaymax',
                   'F_daymaxcolor', 'F_colorcolor']:
            lc[vv] = lc[vv]*(lc['flux_e_sec']/lc['snr_m5']
                             )**2/(lc['flux_5']/5.)**2

    x1, color = -2.0, 0.2
    idx = np.abs(lc['x1']-x1) < 1.e-5
    idx &= np.abs(lc['color']-color) < 1.e-5

    Anadf(lc[idx]).plotzlim()


parser = OptionParser()

parser.add_option("--fake_config", type="str", default='Fake_cadence.yaml',
                  help="config file name for fake obs[%default]")
parser.add_option("--fake_output", type="str", default='Fake_DESC',
                  help="output file namefor fake_obs[%default]")
parser.add_option("--RAmin", type=float, default=-0.05,
                  help="RA min for obs area [%default]")
parser.add_option("--RAmax", type=float, default=0.05,
                  help="RA max for obs area [%default]")
parser.add_option("--Decmin", type=float, default=-0.05,
                  help="Dec min for obs area [%default]")
parser.add_option("--Decmax", type=float, default=0.05,
                  help="Dec max for obs area [%default]")
parser.add_option("--lc_outDir", type="str", default='MetricOutput',
                  help="output dir [%default]")

opts, args = parser.parse_args()

fake_config = opts.fake_config
fake_output = opts.fake_output
RAmin = opts.RAmin
RAmax = opts.RAmax
Decmin = opts.Decmin
Decmax = opts.Decmax
lc_outDir = opts.lc_outDir

metric = 'NSN'
fieldType = 'Fake'

# first step: create fake data from yaml configuration file
cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
    fake_config, fake_output)

os.system(cmd)
# from this: use the nsn metric to produce lc

cmd = 'python run_scripts/metrics/run_metrics_fromnpy.py'
cmd += ' --dbDir .'
cmd += ' --dbName {}'.format(fake_output)
cmd += ' --dbExtens npy'
cmd += ' --nproc 1'
cmd += ' --metric {} --fieldType {} --templateDir ../Templates'.format(
    metric, fieldType)
cmd += ' --proxy_level 2 --RAmin {} --RAmax {} --Decmin {} --Decmax {}'.format(
    RAmin, RAmax, Decmin, Decmax)
cmd += ' --coadd 0 --T0s one --output lc --saveData 1'
cmd += ' --outDir {}'.format(lc_outDir)

os.system(cmd)

# finally plot sigmaC vs z and SNRband vs sigmaC

filename = '{}/{}/{}/'.format(lc_outDir, fake_output, metric)
filename += '{}_{}Metric_{}_nside_64_coadd_0_{}_{}_{}_{}_npixels_0_0.hdf5'.format(
    fake_output, metric, fieldType, RAmin, RAmax, Decmin, Decmax)

plotLC(filename, corrFisher=True)

plt.show()
