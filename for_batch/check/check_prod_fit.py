from sn_tools.sn_io import loopStack
import glob
from optparse import OptionParser
import multiprocessing
import numpy as np
import pandas as pd

def analysis(simus,verbose,nproc):
    """
    Function to analyze simu/fit files using multiprocessing

    Parameters
    ----------
    simus: list(str)
      list of files to process
    verbose: bool
      to set the verbose mode
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
                                     args=(simus[t[j]:t[j+1]],verbose, j, result_queue))
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
    resbad = pd.DataFrame()

    # gather the results
    for key, vals in resultdict.items():
        restot = pd.concat((restot, vals[0]), sort=False)
        resbad = pd.concat((resbad, vals[1]), sort=False)
    return restot,resbad



def check_count(simus,verbose=False,j=0,output_q=None):
    """
    Function to analyse simu and associated fitted files

    Parameters
    ----------
    simus: list(str)
      list of simu files to process
    verbose: bool, opt
      to set the verbose mode or not

    Returns
    -------
    pandas df with nsims_tot and nfits_tot


    """

    nsims_tot = 0
    nfits_tot = 0
    r =[]
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
        if verbose:
            print(sim.split('/')[-1],len(sims),len(fits),nfits/nsims)
        if nfits/nsims<0.9:
            print(sim.split('/')[-1],len(sims),len(fits),nfits/nsims)
            r.append(sim.split('/')[-1])
        nsims_tot += nsims
        nfits_tot += nfits
        #break
        
    rbad = pd.DataFrame({'file':r})
    res = pd.DataFrame({'nsims':[nsims_tot],'nfits':[nfits_tot]})

    if output_q is not None:
        return output_q.put({j: (res,rbad)})
    else:
        return (res,rbad)

parser = OptionParser()

parser.add_option("--simDir", type="str", default='/sps/lsst/users/gris/DD/Simu',
                  help="simulation directory [%default]")
parser.add_option("--fitDir", type="str", default='/sps/lsst/users/gris/DD/Fit',
                  help="fit directory [%default]")
parser.add_option("--dbName", type="str", default='descddf_v1.5_10yrs',
                  help="db name [%default]")
parser.add_option("--nproc", type=int, default=8,
                  help="nproc for multiprocessing [%default]")
parser.add_option("--verbose", type=int, default=0,
                  help="to activate the verbose mode [%default]")
parser.add_option("--tagprod", type='str', default='faintSN,allSN',
                  help="production to check [%default]")

opts, args = parser.parse_args()

simDir = opts.simDir
fitDir = opts.fitDir
dbName = opts.dbName
nproc = opts.nproc
verbose = opts.verbose
tagprod = opts.tagprod.split(',')

#search_path_simu = '{}/{}/Simu*allSN*.hdf5'.format(simDir,dbName)                                           
for tag in tagprod:                                        
    search_path_simu = '{}/{}/Simu*{}*.hdf5'.format(simDir,dbName,tag)
    print('looking for',search_path_simu)
    simus = glob.glob(search_path_simu)

    print(simus)

    res,resbad = analysis(simus,verbose,nproc)
    #print(res)
    res = res[['nsims','nfits']].sum()

    print('summary',res['nsims'],res['nfits'],res['nfits']/res['nsims'])
    print('To reprocess',resbad)
    resbad.to_csv('bad_fits.csv',index=False)
