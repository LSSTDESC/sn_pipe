#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:24:13 2025

@author: philippe.gris@clermont.in2p3.fr
"""
from optparse import OptionParser
from sn_scheduler.generate_obs import load_data, process, ana_moon
import pandas as pd
from sn_tools.sn_io import checkDir

parser = OptionParser(description='Script to build a realistic ddf survey')

parser.add_option('--dirFiles', type=str,
                  default='../sn_ddf_scheduler',
                  help='Location dir of the ddf summary file [%default]')
parser.add_option('--ddf_rubin_scheduler_data', type=str,
                  default='../../rubin_sim_data/scheduler/ddf_grid.npz',
                  help='Location dir of the ddf summary file [%default]')
parser.add_option("--mjd_min", type=int, default=60980,
                  help="survey start [%default]")
parser.add_option("--inputconfigDir", type=str, default='input/scheduler',
                  help="input config dir of the survey [%default]")
parser.add_option("--ddf_survey", type=str, default='ddf_desc_0.70_sn',
                  help="survey to implement [%default]")
parser.add_option("--udf", type=str, default='COSMOS,XMM_LSS',
                  help="ultra-deep fields [%default]")
parser.add_option("--ddf", type=str, default='ELAISS1,ECDFS,EDFS_a,EDFS_b',
                  help="deep fields [%default]")
parser.add_option('--outDir', type=str,
                  default='../observations_sn_scheduler',
                  help='output dir of the produced files [%default]')

opts, args = parser.parse_args()

dirFiles = opts.dirFiles
ddf_rubin_scheduler_data = opts.ddf_rubin_scheduler_data
mjd_min = opts.mjd_min
ddf_survey = opts.ddf_survey
udfs = opts.udf.split(',')
ddfs = opts.ddf.split(',')
inputconfigDir = opts.inputconfigDir
outDir = opts.outDir

# check if output dir exist
checkDir(outDir)
# set a dict for field types
field_type = {}

for vv in udfs:
    field_type[vv] = 'UD'

for vv in ddfs:
    field_type[vv] = 'DF'


# load data

ddf_scheduler = load_data(dirFiles)

# load survey
surName = '{}/{}.csv'.format(inputconfigDir, ddf_survey)
survey_df = pd.read_csv(surName, comment='#', index_col=False)

res_survey = process(ddf_scheduler, survey_df, field_type)

# plot_moon(res_survey)

res_survey = ana_moon(res_survey)

outName = '{}/{}.hdf5'.format(outDir, ddf_survey)
res_survey.to_hdf(outName, key='ddf')
"""
# load rubin scheduler data
ddf_data = np.load(ddf_rubin_scheduler_data)
ddf_grid = pd.DataFrame.from_records(ddf_data["ddf_grid"].copy())
ddf_data.close()

print(ddf_grid.columns)
idx = ddf_grid['mjd'] >= mjd_min
ddf_grid = ddf_grid[idx]
ddf_grid['night'] = ddf_grid['mjd']-mjd_min+1
ddf_grid['night'] = ddf_grid['night'].astype(int)

fig, ax = plt.subplots()
ax.plot(ddf_grid['night'], ddf_grid['COSMOS_m5_g'], 'k.')

plt.show()
"""
