from sn_design_dd_survey.wrapper import Data, Nvisits_cadence, Mod_z
from sn_design_dd_survey.budget import DD_Budget
from sn_design_dd_survey.snr import SNR, SNR_plot
from sn_design_dd_survey.signal_bands import RestFrameBands
from sn_design_dd_survey.showvisits import ShowVisits
from sn_design_dd_survey.templateLC import templateLC
from sn_design_dd_survey import plt

#from sn_DD_opti.showvisits import GUI_Visits
#from sn_DD_opti.budget import GUI_Budget

import os
import multiprocessing
import pandas as pd
import numpy as np

# Step 1: Load the data needed for analysis
# ----------------------------------------

blue_cutoff = 380.
red_cutoff = 800.
x1 = -2.0
color = 0.2
simulator = 'sn_fast'
ebv = 0.
error_model = 1
bands = 'grizy'
zmin = 0.0
zmax = 0.9
zstep = 0.01
bands = 'grizy'
cadence = 3.
cadences = dict(zip(bands, [cadence]*len(bands)))
cutoff = '{}_{}'.format(blue_cutoff, red_cutoff)
if error_model:
    cutoff = 'error_model'

plot_input = False
plot_snr = False
plot_nvisits = False
plot = False

theDir = 'input/sn_studies'
# simulation of template LC here
prodid = '{}_{}_{}_ebv_{}_{}_cad_{}'.format(
    simulator, x1, color, ebv, cutoff, int(cadence))
fname = 'LC_{}_0.hdf5'.format(prodid)
if not os.path.isfile('{}/{}'.format(theDir, fname)):
    templateLC(x1, color, simulator, ebv, blue_cutoff, red_cutoff,
               error_model, zmin, zmax, zstep, theDir, bands, cadences, prodid)

m5file = 'medValues_flexddf_v1.4_10yrs_DD.npy'

data = Data(theDir, fname, m5file, x1, color,
            blue_cutoff, red_cutoff, error_model, bands=bands)

print(type(data))

if plot_input:
    # few plots related to data

    data.plotzlim()
    data.plotFracFlux()
    data.plot_medm5()

    # this is to plot restframebands cutoff
    mybands = RestFrameBands(blue_cutoff=blue_cutoff,
                             red_cutoff=red_cutoff)
    print(mybands.zband)
    mybands.plot()
    plt.show()

fracSB = 'fracSignalBand.npy'
if not os.path.isfile(fracSB):
    fracSignalBand = data.fracSignalBand.fracSignalBand
    np.save(fracSB, fracSignalBand.to_records(index=False))

# SNR vs m5 reference file needed
SNR_m5_file = 'SNR_m5_error_model_snrmin_1_cad_{}.npy'.format(int(cadence))
if not os.path.isfile(SNR_m5_file):
    cmd = 'python run_scripts/sn_studies/snr_m5.py --inputDir {} --refLC {} --outFile {}'.format(
        theDir, fname, SNR_m5_file)
    os.system(cmd)

# Step 2: get the SNR requirements (minimal per band) with sigma_C<0.04
# ----------------------------------------------------------------------

# Create dir for SNR output
# ---------------------------
SNRDir = 'SNR_files'
if not os.path.isdir(SNRDir):
    os.makedirs(SNRDir)

# criteria used for SNR choice
# can be :
# Nvisits, Nvisits_y -> minimal number of visits (total) or in the y band
# fracflux -> SNR distribution (per band) according to the flux distribution
# -------------------------------------------------------------------------


# choose one SNR distribution
SNR_par = dict(zip(['max', 'step', 'choice'], [50., 1., 'Nvisits']))

zref = np.round(np.arange(0.1, np.max(data.lc['z'])+0.1, 0.05), 2)
zref = np.round(np.arange(0.1, np.max(data.lc['z'])+0.1, 0.1), 2)
zref = np.round(np.array([0.8]), 2)

snr_calc = SNR(SNRDir, data, SNR_par,
               SNR_m5_file=SNR_m5_file, zref=zref,
               save_SNR_combi=True, verbose=False, nproc=3)

# plot the results

# load m5 reference values - here: med value per field per filter per season
# m5_type = 'median_m5_field_filter_season'  # m5 values
plot_snr = False
if plot_snr:
    snrplot = SNR_plot('SNR_files', -2.0, 0.2, 1.0, 380.,
                       800., 3., theDir, m5file, 'median_m5_filter')
    snrplot.plotSummary()
    snrplot.plotSummary_band()
    # for combi in ['fracflux_rizy', 'Nvisits_rizy', 'Nvisits_y_rizy']:
    for combi in ['Nvisits_rizy', 'Nvisits_y_rizy']:
        snrplot.plotIndiv(combi)
        snrplot.plotIndiv(combi, legy='Filter allocation')

        plt.show()


# snr_calc.plot()
# plt.show()

# Step 3: Estimate the DD-DESC budget
# ----------------------------------------------------------------------

cadence = -1  # define the cadence -1 = all cadences

# load m5 reference values - here: med value per field per filter per season
m5_type = 'median_m5_field_filter_season'  # m5 values
myvisits_seasons = Nvisits_cadence(
    snr_calc.SNR, cadence, theDir, m5file, m5_type, 'Nvisits', bands)

# this is to plot the variation of the number visits vs season

if plot_nvisits:
    print('plot number of visits vs season')
    myvisits_seasons.plot()

    plt.show()

# load m5 reference file - here median per filter (over fields and seasons)
m5_type = 'median_m5_filter'

myvisits_ref = Nvisits_cadence(
    snr_calc.SNR, cadence, theDir, m5file, m5_type, 'Nvisits', bands)


# visualization of the results is done with the sn_DD_Opti package
nvisits_cadence = 'Nvisits_cadence_Nvisits_median_m5_filter.npy'
nvisits_cadence_season = 'Nvisits_cadence_Nvisits_median_m5_field_filter_season.npy'

dir_config = 'sn_DD_Opti/input'
dir_config = '.'

# this is to display the number of visits vs z for a given cadence
#GUI_Visits(nvisits_cadence, cadence=3, dir_config=dir_config)

# this is to display budget vs zlim (and filter allocation)
"""
GUI_Budget(nvisits_cadence,
           nvisits_cadence_season,
           runtype='Nvisits_single', dir_config='.')

plt.show()
"""
