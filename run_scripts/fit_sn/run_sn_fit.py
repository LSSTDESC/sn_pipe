import yaml
import argparse
import time
import h5py
from astropy.table import Table, vstack
import numpy as np
import multiprocessing
from optparse import OptionParser
from sn_fit.process_fit import Fitting
from sn_fit.mbcov import MbCov
import glob
import os
import sn_fit_input as simu_fit
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse
from sn_tools.sn_io import loopStack

def procsimus(simu_names, fit, nproc=1,j=0, output_q=None):
    """
    Method to process (fit) a set of LC
    elementary cell of the multiprocessing effort

    Parameters
    ---------------
    lc_name: str
      name of the file containing LCs
    simul: 
    fit: fit_xxx class
      instance of the class used to fit LCs
    j:  int,opt
       internal parameter for multiprocessing (default: -1)
    output_q: multiprocessing.queue, opt
       queue for multiprocessing (default: None)

    Returns
    ----------
    res: astropy table with fit results

    """
    res = Table()
    time_ref = time.time()
    for name in simu_names:
        lc_name = name.replace('Simu_','LC_')
        f = h5py.File(name, 'r')
        print(f.keys())
        # reading the simu file
        simul = Table()
        for i, key in enumerate(f.keys()):
            simul = vstack([simul, Table.read(name, path=key)])

        result_queue = multiprocessing.Queue()
        nlc = len(simul)
        t = np.linspace(0, nlc, nproc+1, dtype='int')
        print(t)
        
        procs = [multiprocessing.Process(name='Subprocess-'+str(i), target=procelem, args=(
            simul[t[i]:t[i+1]], lc_name,fit, i, result_queue)) for i in range(nproc)]

        for p in procs:
            p.start()

        resultdict = {}
        for ja in range(nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

       
        for ja in range(nproc):
            res = vstack([res, resultdict[ja]])
            

    print('done', j,time.time()-time_ref)
    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res

def procelem(simul, lc_name,fit,j=0, output_q=None):

    res = Table()
    for ilc,simu in enumerate(simul):
        #print('fitting',j,ilc)
        lc = None
        lc = Table.read(lc_name, path='lc_{}'.format(simu['index_hdf5']))
        lc.convert_bytestring_to_unicode()
        if simu['status'] == 1:
            resfit = fit(lc)
            res = vstack([res, resfit])

    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res   
            

def multiproc(simu_name,fit, covmb=None, nproc=1,nproc_indiv=1):
    """
    Method to grab and fit LC using multiprocessing

    Parameters
    ---------------
    simu_name: str
      name of the file with SN parameters (astropy table)
    lc_name: str
       name of the file containing the light curves to fit
    fit: Fitting instance
       where the fit is performed
    covmb: MbCov class, opt
      MbCov class used to estimate Mb covariance data (default: None)
    nproc: int, opt
      number of procs for multiprocessing (default: 1)

    Returns
    -----------
    astropy table with fitted parameters

    """
    
    # getting the simu files
    """
    f = h5py.File(simu_name, 'r')
    print(f.keys())
    # reading the simu file
    simu = Table()
    for i, key in enumerate(f.keys()):
        simu = vstack([simu, Table.read(simu_name, path=key)])
    """
    #simu = loopStack(files,'astropyTable')
    # multiprocessing parameters
    nlc = len(simu_name)
    t = np.linspace(0, nlc, nproc+1, dtype='int')
    print(t)

    names = None
    val = []
    inum = -1
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(i), target=procsimus, args=(
        simu_name[t[i]:t[i+1]], fit, nproc_indiv,i, result_queue)) for i in range(nproc)]

    for p in procs:
        p.start()

    resultdict = {}
    for j in range(nproc):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    val = Table()
    for j in range(nproc):
        val = vstack([val, resultdict[j]])

    print('dumping results', len(val.columns))
    if len(val) > 0:
        inum += 1
        fit.dump(val, inum)


def process(config, files,nproc=1,covmb=None):
    """
    Method to process(fit) a set of LCs

    Parameters
    --------------
    config: dict
      parameters used for the processing (fit)
    covmb: MbCov class, opt
      MbCov call used to estimated mb covariance data (default: None)

    """

    # this is for output
    save_status = config['Output']['save']
    outdir = config['Output']['directory']

    # prodid parameter
    prodid = config['ProductionID']

    # input files:
    # simu_file: astropy table with SN parameters
    # lc file: astropy tables with light curves

    dirSimu = config['Simulations']['dirname']
    prodidSimu = config['Simulations']['prodid']
    """
    simu_name = '{}/Simu_{}.hdf5'.format(dirSimu, prodidSimu)
    lc_name = '{}/LC_{}.hdf5'.format(dirSimu, prodidSimu)
    """
    
    # Fitting instance

    fit = Fitting(config, covmb=covmb)
    # Loop on the simu_file to grab and fit simulated LCs
    multiproc(simu_name=files, fit=fit,
              covmb=covmb, nproc=nproc,nproc_indiv=config['Multiprocessing']['nproc'])

# get all possible simulation parameters and put in a dict
path = simu_fit.__path__
confDict = make_dict_from_config(path[0],'config_simulation.txt')

parser = argparse.ArgumentParser(
    description='Run a LC fitter on a set of LC curves.')

parser = OptionParser()
# parser for fit parameters : 'dynamical' generation
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0],vals[1]))
    parser.add_option('--{}'.format(key),help='{} [%default]'.format(vals[2]),default=vv,type=vals[0],metavar='')

#add nproc for 'global' multiprocessing
parser.add_option('--nproc' ,help='nproc [%default]',default=1,type=int,metavar='')
    
opts, args = parser.parse_args()

print('Start processing...')

#load the new values
newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key]=(vals[0],newval)

# new dict with configuration params
yaml_params = make_dict_from_optparse(newDict)

covmb = None
mbCalc = yaml_params['Fitter']['covmb']

if mbCalc:
    salt2Dir = 'SALT2_Files'
    covmb = MbCov(salt2Dir, paramNames=dict(
        zip(['x0', 'x1', 'color'], ['x0', 'x1', 'c'])))

# create outputdir if necessary
outDir = yaml_params['Output']['directory']
if not os.path.isdir(outDir):
    os.makedirs(outDir)

prodid = yaml_params['ProductionID']
yaml_name = '{}/Fit_{}.yaml'.format(outDir, prodid)
with open(yaml_name, 'w') as f:
    data = yaml.dump(yaml_params, f)

dirFiles = yaml_params['Simulations']['dirname']
search_path = '{}/Simu_{}*.hdf5'.format(dirFiles, prodid)
files = glob.glob(search_path)

process(yaml_params, files,nproc=opts.nproc,covmb=covmb)
