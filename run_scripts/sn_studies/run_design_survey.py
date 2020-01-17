from sn_design_dd_survey.wrapper import Data, Nvisits_cadence
from sn_design_dd_survey.budget import DD_Budget
from sn_design_dd_survey.snr import SNR, SNR_plot
from sn_design_dd_survey.signal_bands import RestFrameBands
from sn_design_dd_survey import plt

import os
import multiprocessing


# Step 1: Load the data needed for analysis
# ----------------------------------------

blue_cutoff = 380.
red_cutoff = 800.
x1 = -2.0
color = 0.2
bands = 'rizy'

theDir = 'input/sn_studies'
fname = 'Fakes_NSNMetric_Fake_lc_nside_64_coadd_0_0.0_360.0_-1.0_-1.0_0.hdf5'
m5file = 'medValues.npy'

data = Data(theDir, fname, m5file, x1, color,
            blue_cutoff, red_cutoff, bands=bands)


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
SNR_par = dict(zip(['max', 'step', 'choice'], [80., 2., 'Nvisits']))

snr_calc = SNR(SNRDir, data, SNR_par)

# plot the results

# load m5 reference values - here: med value per field per filter per season
# m5_type = 'median_m5_field_filter_season'  # m5 values
"""
snrplot = SNR_plot('SNR_files', -2.0, 0.2, 2.0, 380.,
                   800., 3., theDir, m5file, 'median_m5_filter')
snrplot.plotSummary()
snrplot.plotSummary_band()
for combi in ['fracflux_rizy', 'Nvisits_rizy', 'Nvisits_y_rizy']:
    snrplot.plotIndiv(combi)
    snrplot.plotIndiv(combi, legy='Filter allocation')

plt.show()
"""

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
"""
myvisits_seasons.plot()

plt.show()
"""
# load m5 reference file - here median per filter (over fields and seasons)
m5_type = 'median_m5_filter'

myvisits_ref = Nvisits_cadence(
    snr_calc.SNR, cadence, theDir, m5file, m5_type, 'Nvisits', bands)

"""
myvisits_ref.plot()

plt.show()
"""
configName = 'input/sn_studies/DD_scen1.yaml'
#configName = 'input/sn_studies/DD_scen2.yaml'
#configName = 'input/sn_studies/DD_scen3.yaml'

dd_budget = 0.06
mybud = DD_Budget(configName, myvisits_ref.nvisits_cadence,
                  myvisits_seasons.nvisits_cadence,
                  runtype='Nvisits_single')

mybud.plot_budget_zlim(dd_budget=-1)
mybud.plot_budget_visits(fieldName='COSMOS', season=1, dd_budget=-1)

mybud.plot_budget_zlim(dd_budget=dd_budget)
mybud.plot_budget_visits(fieldName='COSMOS', season=1, dd_budget=dd_budget)


# mybud.printVisits(dd_budget)

plt.show()
