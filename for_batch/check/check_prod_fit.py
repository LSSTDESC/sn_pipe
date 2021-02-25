from sn_tools.sn_io import loopStack
import glob
from optparse import OptionParser
import multiprocessing
import numpy as np
import pandas as pd

def analysis(simus,nproc):
    """
    Function to analyze simu/fit files using multiprocessing

    Parameters
    ----------
    simus: list(str)
      list of files to process
    nproc: int
     number of procs to use
    
    Returns
    -------
    pandas df with nsims and nfits as cols

    """

    # multiprocessing parameters
    nz = len(simus)
    t = np.linspace(0, nz, nproc+1, dtype='int')
    print('multi', nz, t)
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=check_count,
                                     args=(simus[t[j]:t[j+1]], j, result_queue))
                 for j in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    # get the results in a dict

    for i in range(nproc):
        resultdict.update(result_queue.get())
        
    for p in multiprocessing.active_children():
        p.join()

    restot = pd.DataFrame()

    # gather the results
    for key, vals in resultdict.items():
        restot = pd.concat((restot, vals), sort=False)
        
    return restot



def check_count(simus,j=0,output_q=None):
    """
    Function to analyse simu and associated fitted files

    Parameters
    ----------
    simus: list(str)
      list of simu files to process

    Returns
    -------
    pandas df with nsims_tot and nfits_tot


    """

    nsims_tot = 0
    nfits_tot = 0
    for sim in simus:
        sims = loopStack([sim],objtype='astropyTable')
        nsims = len(sims)
        sim_last = sim.split('/')[-1]
        #fitName = sim_last.replace('Simu','Fit').replace('.hdf5','_sn_cosmo.hdf5')
        fitName = sim_last.replace('Simu','Fit').replace('.hdf5','_*.hdf5')
        #print('fitname',fitName)
        fitName = '{}/{}/{}'.format(fitDir,dbName,fitName)
        fitList = glob.glob(fitName)
        fits = loopStack(fitList,'astropyTable')
        nfits = len(fits)
        if nfits/nsims<0.9:
            print(sim.split('/')[-1],len(sims),len(fits),nfits/nsims)
        nsims_tot += nsims
        nfits_tot += nfits
        #break
        

    res = pd.DataFrame({'nsims':[nsims_tot],'nfits':[nfits_tot]})

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res

parser = OptionParser()

parser.add_option("--simDir", type="str", default='/sps/lsst/users/gris/DD/Simu',
                  help="simulation directory [%default]")
parser.add_option("--fitDir", type="str", default='/sps/lsst/users/gris/DD/Fit',
                  help="fit directory [%default]")
parser.add_option("--dbName", type="str", default='descddf_v1.5_10yrs',
                  help="db name [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="nproc for multiprocessing [%default]")

opts, args = parser.parse_args()

simDir = opts.simDir
fitDir = opts.fitDir
dbName = opts.dbName
nproc = opts.nproc

#search_path_simu = '{}/{}/Simu*COSMOS*allSN*.hdf5'.format(simDir,dbName)                                                                                   
search_path_simu = '{}/{}/Simu*.hdf5'.format(simDir,dbName)
print('looking for',search_path_simu)
simus = glob.glob(search_path_simu)

print(simus)

res = analysis(simus,nproc)
#print(res)
res = res[['nsims','nfits']].sum()

print('summary',res['nsims'],res['nfits'],res['nfits']/res['nsims'])
