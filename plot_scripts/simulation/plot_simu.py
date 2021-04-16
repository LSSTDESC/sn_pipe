import h5py
import numpy as np
from astropy.table import Table, vstack
import pprint
from optparse import OptionParser
from sn_plotter_simu.simuPlot import SimuPlot

parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="OS name [%default]")
parser.add_option("--fileDir", type="str", default='Output_Simu',
                  help="dir location of the results [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

splot = SimuPlot(opts.fileDir, opts.dbName)

# get the simulation parameters
simupars = splot.simuPars

#plt.show()

print('Number of simulated supernovae', len(simupars))
# get columns

cols = simupars.columns

print(cols)
import matplotlib.pyplot as plt
plt.plot(simupars['pixRA'],simupars['pixDec'],'ko')
plt.show()
print(splot.simuPars)

# plotting SN parameters
splot.plotParameters()

# display LC loop
splot.plotLoopLC()


# splot.checkLC()
