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
myhist.plotBarh('nfc_noddf', '# filter changes /night')
#plotHist('obs_area', forPlot, 'Observed area [deg2]/night')

# correlation plots for a given OS
dbName = 'alt_sched'
myplot = PlotTime(dbDir, dbName, forPlot)
myplot.plot('night', 'night', 'nfc_noddf', '# filter changes')
myplot.plot('med_moonAlt', 'moon_alt', 'nfc_noddf', '# filter changes')

# Summary plots
mysum = PlotStat(dbDir, forPlot)
# get the list of variables to display
mysum.listvar()
# plot some of these
mysum.plotBarh('nfc_med', 'Median number of filter changes per night')
for band in 'ugrizy':
    mysum.plotBarh('frac_{}'.format(band),
                   'Filter allocation - {} band'.format(band))


plt.show()
