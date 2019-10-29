from optparse import OptionParser
import matplotlib.pyplot as plt
from sn_metrics.sn_global_metric import SNGlobalMetric
from sn_tools.sn_obs import renameFields
import numpy as np
import multiprocessing
from sn_tools.sn_io import Read_Sqlite
import os

def loop(obs, nights,j=0, output_q=None):
    restot = None
    for night in nights:
        idx = obs['night'] == night
        sel = obs[idx]
        res = sn_metric.run(sel)
        if restot is None:
            restot = res
        else:
            restot = np.concatenate((restot,res))
    if output_q is not None:
        return output_q.put({j:restot})
    else:
        return restot



parser = OptionParser()

parser.add_option("--dbName", type="str", default='alt_sched', help="db name [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--dbDir", type="str", default='', help="db dir [%default]")
parser.add_option("--outDir", type="str", default='', help="output directory [%default]")
parser.add_option("--nproc", type="int", default='1', help="nproc for multiprocessing [%default]")

opts, args = parser.parse_args()

dbDir = opts.dbDir
if dbDir == '':
    dbDir='/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'

dbName = opts.dbName
dbExtens = opts.dbExtens

outDir = opts.outDir
if outDir == '':
    outDir='/sps/lsst/users/gris/MetricOutput'

nproc = opts.nproc

# metric instance
sn_metric = SNGlobalMetric()

#load observations
dbFullName = '{}/{}.{}'.format(dbDir, dbName,dbExtens)
# if extension is npy -> load
if dbExtens == 'npy':
    observations = np.load(dbFullName)
else:
    #db as input-> need to transform as npy
    print('looking for',dbFullName)
    keymap = {'observationStartMJD': 'mjd',
              'filter': 'band',
              'visitExposureTime': 'exptime',
              'skyBrightness': 'sky',
              'fieldRA': 'Ra',
              'fieldDec': 'Dec',}

    reader = Read_Sqlite(dbFullName)
    #sql = reader.sql_selection(None)
    observations = reader.get_data(cols=None, sql='',
                           to_degrees=False,
                           new_col_names=keymap)

obs = renameFields(observations)

# Grab the number of nights
nights = np.unique(obs['night'])
nnights = int(len(nights))
delta = nnights
if nproc > 1:
    delta = int(delta/(nproc))

tabnight = range(0,nnights,delta)
if nnights not in tabnight:
    tabnight = np.append(tabnight,nnights)

tabnight= tabnight.tolist()

if tabnight[-1]-tabnight[-2]<= 100:
    tabnight.remove(tabnight[-2])

print(tabnight)
result_queue = multiprocessing.Queue()
for j in range(len(tabnight)-1):
    ida = tabnight[j]
    idb = tabnight[j+1]
    p=multiprocessing.Process(name='Subprocess-'+str(j),target=loop,args=(obs,nights[ida:idb],j,result_queue))
    p.start()
    

resultdict = {}
for i in range(len(tabnight)-1):
    resultdict.update(result_queue.get())

for p in multiprocessing.active_children():
    p.join()

restot = None

for key,vals in resultdict.items():
    if restot is None:
        restot = vals
    else:
        restot = np.concatenate((restot,vals))

outDir = '{}/{}/Global'.format(outDir,dbName)
if not os.path.isdir(outDir):
    os.makedirs(outDir)

np.save('{}/{}_SNGlobal.npy'.format(outDir,dbName),np.copy(restot))
