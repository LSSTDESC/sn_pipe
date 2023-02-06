import os
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_batchutils import BatchIt

dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

def bbatch(scriptref,params,fieldNames,tagName,mem='40G'):

    processName = 'metric_DD_{}_{}_{}_{}_{}'.format(params['dbName'],params['project_FP'],params['VRO_FP'],params['telrot'],tagName)
    mybatch = BatchIt(processName=processName,mem=mem)
    for fieldName in fieldNames:
        #params['fieldName'] = ','.join(fieldNames)
        params['fieldName'] = fieldName
        mybatch.add_batch(scriptref, params)

    mybatch.go_batch()
    

parser = OptionParser()

parser.add_option('--dbList', type='str', default='for_batch/input/List_Db_DD.csv',help='list of dbNames to process  [%default]')
parser.add_option('--outDir', type='str', default='/sps/lsst/users/gris/MetricOutput_DD_new_128',help='output Dir to  [%default]')
parser.add_option('--mode', type='str', default='batch',help='running mode batch/interactive [%default]')
parser.add_option('--snrmin', type=float, default=1.,help='min snr for LC point fit [%default]')
parser.add_option('--pixelmap_dir', type='str', default='None',help='pixelmap directory [%default]')
parser.add_option('--nproc', type=int, default=8,help='number of proc [%default]')
parser.add_option('--ebvofMW', type=float, default=-1.0,help='E(B-V) [%default]')
parser.add_option('--fieldNames', type=str, default='COSMOS,CDFS,ELAISS1,XMM-LSS,EDFSa,EDFSb',help='DD fields to process [%default]')
parser.add_option('--nclusters', type=int, default=6,help='total number of DD in this OS [%default]')
parser.add_option('--nside', type=int, default=128,help='healpix nside parameter [%default]')
parser.add_option("--RAmin", type=float, default=0.,
                  help="RA min for obs area - for WDF only[%default]")
parser.add_option("--RAmax", type=float, default=360.,
                  help="RA max for obs area - for WDF only[%default]")
parser.add_option("--Decmin", type=float, default=-1.,
                  help="Dec min for obs area - for WDF only[%default]")
parser.add_option("--Decmax", type=float, default=-1.,
                  help="Dec max for obs area - for WDF only[%default]")
parser.add_option("--select_epochs", type=int, default=0,
                  help="to select LC shapes according to epochs [%default]")
parser.add_option("--n_bef", type=int, default=4,
                  help="n points (epochs) before max [%default]")
parser.add_option("--n_aft", type=int, default=8,
                  help="n points (epochs) after max [%default]")
parser.add_option("--project_FP", type=str, default='gnomonic',
                  help="type of projection in FP [%default]")
parser.add_option("--VRO_FP", type=str, default='circular',
                  help="FP model [%default]")
parser.add_option("--telrot", type=int, default=0,
                  help="telescope rotation angle [%default]")
parser.add_option("--addInfo", type=int, default=0,
                  help="to get additionnal infos from obs [%default]")
parser.add_option("--tagName", type=int, default=0,
                  help="to tag script name (avoid duplicates) [%default]")


opts, args = parser.parse_args()

toprocess = pd.read_csv(opts.dbList, comment='#')

fieldNames = opts.fieldNames.split(',')

ppNames = ['outDir','pixelmap_dir','nclusters','nside','nproc','ebvofMW','RAmin','RAmax','Decmin','Decmax','select_epochs','n_bef','n_aft','project_FP','VRO_FP','telrot','addInfo']

params=dict_filter(opts.__dict__,ppNames)
scriptref = 'run_scripts/metrics/run_metrics.py NSNY'
params['fieldType'] = 'DD'
params['zmax'] = 1.1
params['npixels'] = -1
params['saveData'] = 1
tagName = opts.tagName

for i,row in toprocess.iterrows():
    for vv in ['dbDir','dbName','dbExtens']:
        params[vv] = row[vv]
    bbatch(scriptref,params,fieldNames,tagName)
