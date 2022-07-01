from sn_tools.sn_batchutils import BatchIt
from sn_tools.sn_io import checkDir
from optparse import OptionParser
import pandas as pd
import numpy as np

parser = OptionParser()

parser.add_option("--dirFile", type="str",
                  default='/sps/lsst/users/gris/MetricOutput_fbs171_circular_dust', help="metric file dir [%default]")
parser.add_option("--cvsList", type="str", default='WFD_fbs_2.0.csv',
                  help="list odf DBs [%default]")
parser.add_option("--metricName", type="str",
                  default='NSNY', help="metric name [%default]")
parser.add_option("--dirObspixels", type="str",
                  default='/sps/lsst/users/gris/ObsPixelized_circular_fbs171', help="obs pixel dir [%default]")
parser.add_option("--nproc", type=int,
                  default=8, help="number of proc for multiprocessing [%default]")
parser.add_option("--outputDir", type="str",
                  default='/sps/lsst/users/gris/batchCheck', help="output directory [%default]")


opts, args = parser.parse_args()

print(opts)
dirFile = opts.dirFile
cvsList = opts.cvsList
metricName = opts.metricName
dirObspixels = opts.dirObspixels
nproc = opts.nproc
outputDir = opts.outputDir


checkDir(outputDir)

# read cvs file
dbs = pd.read_csv(cvsList, comment='#')

arr_spl = np.array_split(dbs, 12)
params={}
params['outputDir'] = outputDir
params['nproc'] = nproc
params['dirFile'] = dirFile
params['metricName'] = metricName
params['dirObspixels'] = dirObspixels

thescript = 'for_batch/check/check_prod_metric_pixels.py'
for i,vv in enumerate(arr_spl):
    cvsfi = '{}/{}_{}.csv'.format(outputDir,cvsList.split('.csv')[0],i)
    pd.DataFrame(vv).to_csv(cvsfi,index=False)
    processName = 'check_batch_{}'.format(cvsfi.split('/')[-1].split('.csv')[0])
    bb = BatchIt(processName=processName)
    params['cvsList'] = cvsfi
    bb.add_batch(thescript,params)
    bb.go_batch()
                                          
