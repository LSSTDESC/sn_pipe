from optparse import OptionParser
from sn_plotter_simu.simuPlot import SimuPlot
from sn_plotter_simu import plt

parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="OS name [%default]")
parser.add_option("--fileDir", type="str", default='Output_Simu',
                  help="dir location of the results [%default]")
parser.add_option("--tagName", type="str", default='SNIa',
                  help="tag name for production [%default]")


opts, args = parser.parse_args()

print('Start processing...', opts)

splot = SimuPlot(opts.fileDir, opts.dbName, opts.tagName)
# plt.show()
# get the simulation parameters
simupars = splot.simuPars

# plt.show()

print('Number of simulated supernovae', len(simupars))
# get columns

cols = simupars.columns

print(cols)
#plt.plot(simupars['pixRA'], simupars['pixDec'], 'ko')
print(splot.simuPars)
# plt.show()

# plotting SN parameters
splot.plotParameters(season=3)

# display LC loop
splot.plotLoopLC(pause_time=10)
# splot.plotLoopLC_errmod()
plt.show()

# splot.checkLC()
