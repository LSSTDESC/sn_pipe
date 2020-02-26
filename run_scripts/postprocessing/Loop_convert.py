import os
import numpy as np
import multiprocessing
from optparse import OptionParser
import pandas as pd

def loop(toproc,metricName,unique,remove_galactic,j=0, output_q=None):

    for index, row in toproc.iterrows():
        cmd = 'python run_scripts/postprocessing/convert_to_npy.py --metric {} --dbName {} --unique {} --remove_galactic {}'.format(metricName,row['dbName'],unique,remove_galactic)
        print('cmd',cmd)
        os.system(cmd)
        #break

parser = OptionParser()

parser.add_option("--metricName", type="str", default='SNR',
                  help="metric to process  [%default]")
parser.add_option("--nproc", type="int", default=8,
                  help="nb of proc to use  [%default]")
parser.add_option("--unique", type="int", default=1,
                  help="to keep only uniques [%default]")
parser.add_option("--simuVersion", type="str", default='fbs14',
                  help="simulation version[%default]")
parser.add_option("--remove_galactic", type="int", default=1,
                  help="remove galactic plane[%default]")

opts, args = parser.parse_args()

metricName = opts.metricName
nproc = opts.nproc
unique = opts.unique
remove_galactic = opts.remove_galactic

filename = 'plot_scripts/cadenceCustomize_{}.csv'.format(opts.simuVersion)

# forPlot = pd.read_csv(filename).to_records()
forPlot = pd.read_csv(filename)

nvals = int(len(forPlot))
delta = nvals
if nproc > 1:
    delta = int(delta/(nproc))

tabvals= range(0, nvals, delta)
if nvals not in tabvals:
    tabvals = np.append(tabvals, nvals)

tabvals = tabvals.tolist()


if nproc >=7:
    if tabvals[-1]-tabvals[-2] <= 10:
        tabvals.remove(tabvals[-2])

print(tabvals, len(tabvals))
result_queue = multiprocessing.Queue()

for j in range(len(tabvals)-1):
    ida = tabvals[j]
    idb = tabvals[j+1]

    p = multiprocessing.Process(name='Subprocess-'+str(j), target=loop, args=(
        forPlot[ida:idb], metricName,unique,remove_galactic,j, result_queue))
    p.start()


