from sn_tools.sn_io import loopStack
import glob
from optparse import OptionParser
import numpy as np
from astropy.table import Table, vstack
import pandas as pd
import multiprocessing

def processMulti(dbList, dirFile, metricName, fieldtype, nside,var):
    """
    Function to process data using multiprocessing

    Parameters
    ----------
    dbList: list(str)
      list of dbNames to process
    dirFile: str
      location dir of the files
    metricName: str
      metric to consider
    fieldtype: str
      type of field (WFD or DD)
    nside: int
     healpix nside parameter
    var: str
     name of the variable to consider

    Returns
    -------
    pandas df with the result

    """

    nproc = 8
    nz = len(dbList)
    t = np.linspace(0, nz,nproc+1, dtype='int')
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=processLoop,
                                 args=(dbList[t[j]:t[j+1]], dirFile, metricName, fieldtype, nside,var,j, result_queue))
                 for j in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    # get the results in a dict                                                                                           

    for i in range(nproc):
        resultdict.update(result_queue.get())
        
    for p in multiprocessing.active_children():
        p.join()

    df = pd.DataFrame()
    # gather the results                                                                                                  
    for key, vals in resultdict.items():
        df = pd.concat((df, vals), ignore_index=True)


    return df




def processLoop(dbNames,dirFile, metricName, fieldtype, nside,var, j=0, output_q=None):
    """
    Function to process multiple dbs

    Parameters
    ----------
    dbNames: list(str)
      list of dbNames to process
    dirFile: str
      location dir of the files
    metricName: str
      metric to consider
    fieldtype: str
      type of field (WFD or DD)
    nside: int
     healpix nside parameter
    var: str
     name of the variable to consider
    j: int,opt
      multiproc process num (default: 0)
    output_q: multiprocessing queue,opt
      (default: None)

    Returns
    -------
    pandas df with the result

    """
    res = pd.DataFrame()

    for dbName in dbNames:
        ro = process(dbName,dirFile, metricName, fieldtype, nside,var)
        res = pd.concat((res,ro))

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def process(dbName,dirFile, metricName, fieldtype, nside,var):
    """
    Function  to process data

    Parameters
    ----------
    dbName: str
      dbName to process
    dirFile: str
      directory of the files to process
    metricName: str
      name of the metric to consider
    fieldtype: str
      type of field (DD or WFD)
    nside: int
      healpix nside parameter
    var: str
     name of the variable to estimate

    Returns
    -------
    pandas df with the result


    """
    metricValues = load(dirFile, dbName, metricName, fieldtype, nside)
    med = summaryOS(metricValues,var)

    return pd.DataFrame({'frac':[med],
                        'dbName': [dbName]})
    
def load(dirFile, dbName, metricName, fieldtype, nside):
    """
    Function to load data

    Parameters
    ----------
    dirFile: str
      location dir of the files to load
    dbName: str
      db name of the file to load
    metricName: str
      name of the metric to consider
    fieldtype: str
     field type (DD or WFD)
    nside: int
      healpix nside value

    Returns
    -------
    astropy table of data

    """
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
parser.add_option("--simutag", type="str", default='fbs14',
                  help="simulation tag [%default]")


    
opts, args = parser.parse_args()

dirFile = opts.dirFile
dbList= opts.dbList
metricName = opts.metricName
fieldtype = opts.fieldtype
nside = opts.nside
var = opts.var
simutag = opts.simutag

toprocess = pd.read_csv(dbList, comment='#')


df = processMulti(toprocess['dbName'].tolist(),dirFile, metricName, fieldtype, nside,var)

print(df)

df.to_csv('Metric_{}_{}.csv'.format(metricName,simutag),index=False)
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
