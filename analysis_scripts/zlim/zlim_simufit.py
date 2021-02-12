from sn_tools.sn_tools.sn_io import loopStack
import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from scipy import interpolate
from optparse import OptionParser
import matplotlib
matplotlib.use('agg')


def loadFile(dbDir, dbName):
    """
    function to load file

    Parameters
    ---------------
    dbDir: str
      location dir of the file
    dbName: str
      name of the file to process

    Returns
    -----------
    astropyTable of the data

    """
    fullName = '{}/{}/*.hdf5'.format(dbDir, dbName)
    fis = glob.glob(fullName)

    tab = loopStack(fis, 'astropyTable')

    return tab


def multiprocess(tab, nproc=3):

    # multiprocessing parameters
    healpixID_list = list(np.unique(tab['healpixID']))
    nz = len(healpixID_list)
    t = np.linspace(0, nz, nproc+1, dtype='int')

    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=process,
                                     args=(tab, healpixID_list[t[j]:t[j+1]], j, result_queue))
             for j in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    # get the results in a dict

    for i in range(nproc):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    restot = None

    # gather the results
    for key, vals in resultdict.items():
        if restot is None:
            restot = vals
        else:
            restot = np.concatenate((restot, vals))

    return restot


def process(tab, healpixID_list, j=0, output_q=None, sigmaC=0.04):

    r = []
    for healpixID in healpixID_list:
        idx = tab['healpixID'] == healpixID
        idx &= tab['fitstatus'] == 'fitok'
        sel = tab[idx]
        if len(sel) > 0:
            for season in np.unique(sel['season']):
                idxb = sel['season'] == season
                idxb &= np.sqrt(sel['Cov_colorcolor']) <= sigmaC
                selb = sel[idxb]
                selb.sort(keys=['z'])
                if len(selb) > 0:
                    n_bins = 100
                    fig, ax = plt.subplots()
                    n, bins, patches = ax.hist(selb['z'], n_bins, density=True, histtype='step',
                                               cumulative=True, label='Empirical')
                    bin_center = (bins[:-1] + bins[1:]) / 2
                    interp = interpolate.interp1d(n, bin_center)
                    r_50 = interp(0.5).item()
                    r_95 = interp(0.95).item()
                    r.append((healpixID, season, r_50, r_95))
                    plt.close(fig)
    res = np.rec.fromrecords(
        r, names=['healpixID', 'season', 'z_50', 'z_95'])

    print('finished', j, len(res))
    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


parser = OptionParser()

parser.add_option("--dbDir", type='str', default='../../Fit_sncosmo',
                  help="location dir of the files to process[%default]")
parser.add_option("--dbName", type='str', default='baseline_nexp2_v1.7_10yrs',
                  help="name of the file to process [%default]")

opts, args = parser.parse_args()


# load file
tab = loadFile(opts.dbDir, opts.dbName)

print(len(tab))

res = multiprocess(tab, nproc=4)

print(res)
