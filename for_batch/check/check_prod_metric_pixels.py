import glob
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_io import loopStack
import numpy as np
from sn_tools.sn_utils import multiproc
import time


def npixels(dirFile, dirObspixels, metricName, dbName):

    df = getFiles(dirFile, metricName, dbName)

    print(df)
    time_ref = time.time()
    npixels = df.groupby(['RA_min_max', 'Dec_min_max']).apply(
        lambda x: countPixels(x)).reset_index()

    print(npixels)

    npixels_obs = countObspixels(dirObspixels, dbName)
    print(npixels_obs)
    print('timing', time.time()-time_ref)

    dfm = npixels.merge(npixels_obs, left_on=['RA_min_max', 'Dec_min_max'], right_on=[
                        'RA_min_max', 'Dec_min_max'])

    dfm['diff_pixels'] = dfm['npixels_obs']-dfm['npixels']

    print(dfm[['RA_min_max', 'Dec_min_max', 'diff_pixels']])


def countPixels(grp):
    """
    metricValues = loopStack(grp['fullName'].to_list(), 'astropyTable')

    return pd.DataFrame({'npixels': [len(np.unique(metricValues['healpixID']))]})
    """
    params = {}
    res = multiproc(grp, params, countPixels_loop, 4)['npixels'].sum()
    return pd.DataFrame({'npixels': [res]})


def countPixels_loop(grps, params, j=0, output_q=None):

    npixels = 0
    for io, grp in grps.iterrows():
        metricValues = loopStack([grp['fullName']], 'astropyTable')
        npixels += len(np.unique(metricValues['healpixID']))

    res = pd.DataFrame({'npixels': [npixels]})

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def getFiles(dirFile, metricName, dbName):
    """
    Method to get the files corresponding to dbName

    Parameters
    ---------------
    dirFile: str
      location dir of the metric files
    metricName: str
      metric name
    dbName: str
      db file name

    Returns
    -----------

    """
    path = '{}/{}/{}/{}*.hdf5'.format(dirFile, dbName, metricName, dbName)

    fis = glob.glob(path)

    prefix = '{}_{}Metric_WFD_nside_64_coadd_1_'.format(dbName, metricName)
    r = []
    for fi in fis:
        fispl = fi.split('/')[-1]
        fisplb = fispl.split(prefix)[1]
        ra = fisplb.split('_')[:2]
        dec = fisplb.split('_')[2:4]
        rastr = '_'.join(ra)
        decstr = '_'.join(dec)
        r.append((fispl, ra[0], ra[1], dec[0], dec[1], rastr, decstr, fi))

    df = pd.DataFrame(
        r, columns=['fName', 'RAmin', 'RAmax', 'Decmin', 'Decmax', 'RA_min_max', 'Dec_min_max', 'fullName'])
    df['metricName'] = metricName
    df['dirFile'] = dirFile

    return df


def countObspixels(dirObspixels, dbName):

    path = '{}/{}/*.npy'.format(dirObspixels, dbName)

    fis = glob.glob(path)
    print('ahahaha', fis, path)
    params = {}
    params['dbName'] = dbName
    return multiproc(fis, params, countObspixels_loop, 3)


def countObspixels_loop(fis, params, j=0, output_q=None):

    dbName = params['dbName']
    prefix = '{}_WFD_nside_64_'.format(dbName)
    r = []
    for fi in fis:
        tab = np.load(fi, allow_pickle=True)
        npixels = len(np.unique(tab['healpixID']))
        fispl = fi.split('/')[-1]
        fisplb = fispl.split(prefix)[1]
        ra = fisplb.split('_')[:2]
        dec = fisplb.split('_')[2:4]
        rastr = '_'.join(ra)
        decstr = '_'.join(dec)
        r.append((npixels, rastr, decstr))

    res = pd.DataFrame(r, columns=['npixels_obs', 'RA_min_max', 'Dec_min_max'])

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


parser = OptionParser()

parser.add_option("--dirFile", type="str",
                  default='../../MetricOutput_fbs171_circular_dust', help="metric file dir [%default]")
parser.add_option("--cvsList", type="str", default='WFD_fbs_1.7_1.csv',
                  help="list odf DBs [%default]")
parser.add_option("--metricName", type="str",
                  default='NSN', help="metric name [%default]")
parser.add_option("--dirObspixels", type="str",
                  default='../../ObsPixelized_circular_fbs171', help="obs pixel dir [%default]")


opts, args = parser.parse_args()

print('Start processing...')

dbs = pd.read_csv(opts.cvsList, comment='#')

print(dbs)

for index, row in dbs.iterrows():
    dbName = row['dbName']
    npixels(opts.dirFile, opts.dirObspixels, opts.metricName, dbName)

    break
