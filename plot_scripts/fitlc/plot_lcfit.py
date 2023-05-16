import matplotlib.pyplot as plt
from optparse import OptionParser
import pandas as pd
from sn_plotter_fitlc.fitlcPlot import FitPlots


parser = OptionParser()

parser.add_option("--prodids", type="str",
                  default='input/plots/prodids_fit.csv',
                  help="prodid [%default]")

opts, args = parser.parse_args()

thelist = opts.prodids

data = pd.read_csv(thelist, delimiter=',', comment='#')

dictfiles = {}

for indx, val in data.iterrows():
    dictfiles[val['plotname']
              ] = '{}/{}'.format(val['dirfile'], val['filename'])

print(data)
fitplot = FitPlots(dictfiles)
fitplot.plot2D(fitplot.SN_table, 'z', 'Cov_colorcolor',
               '$z$', '$\sigma_{C}$', compare=False, zmin=0.2)

# SNR plot here
# fitplot.plot_snr_sigmaC(fitplot.SN_table)

plt.show()
