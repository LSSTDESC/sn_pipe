import yaml
import argparse
import time
import h5py
from astropy.table import Table, vstack
import numpy as np
import multiprocessing
from optparse import OptionParser
from sn_fit.sn_fit import Fitting
from sn_fit.mbcov import MbCov
import glob


def makeYaml(input_file, dbDir, prodid, outDir, nproc, covmb=0, display=0):
    """
    Function to replace generic parameters of a yaml file

    Parameters
    --------------
    input_file: str
       input yaml file to modify
    dbDir: str
       database directory location
    prodid: str
       production id
    outDir: str
      output directory - where to put the results
    nproc: int
      number of procs for multiprocessing
    covmb: int, opt
      bool to estimate mb covariance data (default: 0)
    display: int, opt
      bool to display LC fit in real-time (default:0)

    Returns
    -----------
    dict with modified parameters

    """
    with open(input_file, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('prodid', prodid)
    filedata = filedata.replace('outDir', outDir)
    filedata = filedata.replace('dbDir', dbDir)
    filedata = filedata.replace('nnproc', str(nproc))
    filedata = filedata.replace('covmbcalc', str(covmb))
    filedata = filedata.replace('displayval', str(display))

    return yaml.load(filedata, Loader=yaml.FullLoader)


def procelem(lc_name, simul, fit, j=0, output_q=None):
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
    for simu in simul:
        lc = None
        lc = Table.read(lc_name, path='lc_{}'.format(simu['id_hdf5']))
        resfit = fit(lc)
        res = vstack([res, resfit])

    print('done', j)
    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def multiproc(simu_name, lc_name, fit, covmb=None, nproc=1):
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

    # getting the simu file
    f = h5py.File(simu_name, 'r')
    print(f.keys())
    # reading the simu file
    for i, key in enumerate(f.keys()):
        simu = Table.read(simu_name, path=key)

    # multiprocessing parameters
    nlc = len(simu)
    t = np.linspace(0, nlc, nproc+1, dtype='int')
    print(t)

    names = None
    val = []
    inum = -1
    result_queue = multiprocessing.Queue()

    procs = [multiprocessing.Process(name='Subprocess-'+str(i), target=procelem, args=(
        lc_name, simu[t[i]:t[i+1]], fit, i, result_queue)) for i in range(nproc)]

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

    print('dumping results')
    if len(val) > 0:
        inum += 1
        fit.dump(val, inum)


def process(config, covmb=None):
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

    simu_name = config['Simulations']['dirname']+'/Simu_'+prodid+'.hdf5'
    lc_name = config['Simulations']['dirname']+'/LC_'+prodid+'.hdf5'

    # Fitting instance
    fit = Fitting(config, covmb=covmb)

    # Loop on the simu_file to grab and fit simulated LCs
    multiproc(simu_name=simu_name, lc_name=lc_name, fit=fit,
              covmb=covmb, nproc=config['Multiprocessing']['nproc'])


parser = argparse.ArgumentParser(
    description='Run a LC fitter on a set of LC curves.')

parser = OptionParser()

parser.add_option("--dirFiles", type="str", default='/sps/lsst/users/gris/Output_Simu_pipeline_0',
                  help="location dir of the files[%default]")
parser.add_option("--prodid", type="str", default='Test',
                  help="db name [%default]")
parser.add_option("--outDir", type="str", default='/sps/lsst/users/gris/Output_Fit_0',
                  help="output dir [%default]")
parser.add_option("--nproc", type="int", default=1,
                  help="number of proc [%default]")
parser.add_option("--mbcov", type="int", default=0,
                  help="mbcol calc [%default]")
parser.add_option("--display", type="int", default=0,
                  help="to display fit in real-time[%default]")
parser.add_option("--prefix", type=str, default='sncosmo_DD',
                  help="prefix for input file[%default]")


opts, args = parser.parse_args()

dirFiles = opts.dirFiles
outDir = opts.outDir
prodid = opts.prodid
nproc = opts.nproc
mbCalc = opts.mbcov
display = opts.display


covmb = None
if mbCalc:
    salt2Dir = 'SALT2_Files'
    covmb = MbCov(salt2Dir, paramNames=dict(
        zip(['x0', 'x1', 'color'], ['x0', 'x1', 'c'])))

# prefix = 'sncosmo_DD'
files = glob.glob('{}/Simu_{}_{}*.hdf5'.format(dirFiles, opts.prefix, prodid))

for fi in files:
    prodid = '{}_{}'.format(opts.prefix, fi.split(
        '{}_'.format(opts.prefix))[-1].split('.hdf5')[0])
    # make and load config file
    config = makeYaml('input/fit_sn/param_fit_gen.yaml',
                      dirFiles, prodid, outDir, nproc, mbCalc, display)
    print(config)
    # now run
    process(config, covmb=covmb)
