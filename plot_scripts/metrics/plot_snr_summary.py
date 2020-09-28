from sn_tools.sn_io import loopStack
import glob
from optparse import OptionParser
import numpy as np
from astropy.table import Table, vstack
import pandas as pd

def process(grp,dirFile, metricName, fieldtype, nside,var):
    
    metricValues = load(dirFile, grp.name, metricName, fieldtype, nside)
    med = summaryOS(metricValues,var)

    return pd.DataFrame({'frac':[med]})
    
def load(dirFile, dbName, metricName, fieldtype, nside):

    search_name = '{}/{}/{}/*{}Metric_{}_nside_{}*.hdf5'.format(
    dirFile, dbName, metricName, metricName, fieldtype, nside)
    print('search name', search_name)
    fileNames = glob.glob(search_name)

    # fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
    print(fileNames)

    metricValues = loopStack(fileNames, 'astropyTable')
    metricValues.convert_bytestring_to_unicode()

    return metricValues

def summaryOS(data,what):

    #loop on the season and remove seasons with too few pixels
    
    tab = Table()
    print(data.columns)
    for band, season in np.unique(data[['band', 'season']]):
        idx = (data['band'] == band) & (data['season'] == season)
        sel = data[idx]
        npixels = len(sel)
        if npixels < 10000:
            continue
        tab = vstack([tab,sel])
   
    return np.median(tab[what])

parser = OptionParser(description='Display SNR metric results')
parser.add_option("--dbList", type="str",
                  default='cadenceCustomize_fbs14.csv', help="db list [%default]")
parser.add_option("--dirFile", type="str", default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--fieldtype", type="str",
                  default='WFD', help="band [%default]")
parser.add_option("--metricName", type="str",
                  default='SNRr', help="metric name[%default]")
parser.add_option("--var", type="str",
                  default='frac_obs_SNCosmo', help="column name for the processing[%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside from healpix [%default]")

    
opts, args = parser.parse_args()

dirFile = opts.dirFile
dbList= opts.dbList
metricName = opts.metricName
fieldtype = opts.fieldtype
nside = opts.nside
var = opts.var

toprocess = pd.read_csv(dbList, comment='#')


df = toprocess.groupby(['dbName']).apply(lambda x: process(x,dirFile, metricName, fieldtype, nside,var)).reset_index()

print(df)

df.to_csv('Metric_{}.cvs'.format(metricName),index=False)
"""                                             
df = pd.DataFrame()
for io, val in toprocess.iterrows():
    print(type(val))
    po = pd.DataFrame(val)
    metricValues = load(dirFile, val['dbName'], metricName, fieldtype, nside)
    po['frac'] = summaryOS(metricValues,'frac_obs_SNCosmo')
    df = pd.concat((df,po))
    if io > 2:
        break

print(df.columns)
"""
