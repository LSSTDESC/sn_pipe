from optparse import OptionParser
from sn_plotter_simu.simuPlot import SimuPlot
from sn_plotter_simu import plt
from sn_tools.sn_io import checkDir
import numpy as np

parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched',
                  help="OS name [%default]")
parser.add_option("--fileDir", type="str", default='Output_Simu',
                  help="dir location of the results [%default]")
parser.add_option("--tagName", type="str", default='SNIa',
                  help="tag name for production [%default]")
parser.add_option("--pause_time", type=int, default=-1,
                  help="display time (sec) [%default]")
parser.add_option("--save_fig", type=int, default=0,
                  help="to save figures [%default]")
parser.add_option("--dir_fig", type=str, default='LC_figs',
                  help="dir of saved figs [%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

splot = SimuPlot(opts.fileDir, opts.dbName, opts.tagName)
# plt.show()
# get the simulation parameters
simupars = splot.simuPars
simupars.round({'z':2,'daymax':1})
vvals = ['SNID','z','daymax','x1','color']
#print(simupars[vvals])

simupars.convert_bytestring_to_unicode()
simupars[vvals].pprint_all()



print('Number of simulated supernovae', len(simupars))
# get columns

cols = simupars.columns

# plotting SN parameters
# splot.plotParameters(season=3)

# display LC loop
if opts.save_fig:
    checkDir(opts.dir_fig)
    
while 1:
    answer = input('SN to plot? ')

    snids = answer.split(',')
    print('snids',snids)
    idx = np.in1d(simupars['SNID'],snids)
    selpars = simupars[idx]
    print('sel',selpars)
    splot.plotLoopLC(selpars,
                 save_fig=opts.save_fig, dir_fig=opts.dir_fig)

# splot.plotLoopLC_errmod()
#plt.show()

# splot.checkLC()
