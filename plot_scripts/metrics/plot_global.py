from optparse import OptionParser
from sn_plotter_metrics.globalPlot import PlotHist, PlotTime, PlotStat
import pandas as pd
import matplotlib.pyplot as plt

parser = OptionParser(description='Display Global metric results')
parser.add_option("--listPlot", type="str",
                  default='plot_scripts/input/cadenceCustomize.csv', help="db name [%default]")
parser.add_option("--dirFile", type="str", default='',
                  help="file directory [%default]")

opts, args = parser.parse_args()

dbDir = opts.dirFile

forPlot = pd.read_csv(opts.listPlot, comment='#')


# histograms of few OS
myhist = PlotHist(dbDir, forPlot)
# variables that can be displayed
print(myhist.data.columns)
# histogram of the number of filter changes per night
#myhist.plot('nfc_noddf', '# filter changes /night')
"""
for b in 'ugrizy':
    myhist.plot('med_fiveSigmaDepth_{}'.format(b),'$m_5^{}$'.format(b))
"""

#myhist.plot('med_cloud', 'cloud')
myhist.plot('med_moonAlt','moon alt')
myhist.plot('med_moonPhase','moon phase')
myhist.plot('med_moonDistance','moon distance')
for b in 'ugrizy':
    myhist.plot('nvisits_{}'.format(b),'nvisits_{}'.format(b))
#myhist.plotBarh('nfc_noddf', '# filter changes /night')
#plotHist('obs_area', forPlot, 'Observed area [deg2]/night')
plt.show()


# correlation plots for a given OS
dbName = 'alt_sched'
#dbName = 'rolling_fpo_6nslice0.9_v1.6_10yrs'
myplot = PlotTime(dbDir, dbName, forPlot)
myplot.plot('night', 'night', 'nfc_noddf', '# filter changes',nightEnd=3650)
myplot.plot('med_moonAlt', 'moon_alt', 'nfc_noddf', '# filter changes',nightEnd=3650)
for b in 'ugrizy':
    myplot.plot('med_moonDistance', 'moon_dist', 'nvisits_{}'.format(b), '# Nvisits_{}'.format(b),nightEnd=3650)
    
plt.show()
# Summary plots
mysum = PlotStat(dbDir, forPlot)
# get the list of variables to display
mysum.listvar()
# plot some of these
mysum.plotBarh('nfc', 'Median number of filter changes per night',xmin=5)
for band in 'ugrizy':
    #mysum.plotBarh('frac_{}'.format(band),
    #               'Filter allocation - {} band'.format(band))
    mysum.plotBarh('med_fiveSigmaDepth_{}'.format(band),
                   'Median m5 - {} band'.format(band),xmin=21.)


plt.show()
