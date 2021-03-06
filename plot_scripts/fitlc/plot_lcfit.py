import matplotlib.pyplot as plt
from optparse import OptionParser
import pandas as pd
from sn_plotter_fitlc.fitlcPlot import FitPlots


parser = OptionParser()

parser.add_option("--prodids", type="str", default='input/plots/podids_fit.csv',
                  help="prodid [%default]")
parser.add_option("--fileDir", type="str", default='',
                  help="dir location of the results [%default]")


opts, args = parser.parse_args()

theDir = opts.fileDir
thelist = opts.prodids

data = pd.read_csv(thelist, delimiter=',', comment='#')

if theDir != '':
    data['dirfile'] = theDir

dictfiles = {}

for indx, val in data.iterrows():
    dictfiles[val['nickname']] = '{}/Fit_{}.hdf5'.format(val['dirfile'], val['prodid'])

print(data)
fitplot = FitPlots(dictfiles)
fitplot.plot2D(fitplot.SN_table, 'z', 'Cov_colorcolor',
               '$z$', '$\sigma_{color}$', compare=False)

plt.show()
