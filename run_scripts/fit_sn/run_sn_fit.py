import yaml
import argparse
import time
import h5py
from astropy.table import Table, vstack
from importlib import import_module
from sn_tools.sn_telescope import Telescope
import os
import numpy as np
import multiprocessing
from optparse import OptionParser
from sn_fit.mbcov import MbCov
import glob


alpha = 0.14
beta = 3.1


def makeYaml(input_file, dbDir, prodid, outDir, nproc):

    with open(input_file, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('prodid', prodid)
    filedata = filedata.replace('outDir', outDir)
    filedata = filedata.replace('dbDir', dbDir)
    filedata = filedata.replace('nnproc', str(nproc))
    return yaml.load(filedata, Loader=yaml.FullLoader)


def dump(outname, names, val, inum):

    # dump the results in a hdf5 file
    # res = Table(np.rec.fromrecords(val,names = names))
    Table(rows=val, names=names).write(
        outname, 'fit_lc_{}'.format(inum), append=True, compression=True)


def procelem(lc_name, simul, fit, covmb, j=0, output_q=None):

    res = []
    for simu in simul:
        lc = None
        # if simu['n_lc_points'] > 0:
        lc = Table.read(lc_name, path='lc_{}'.format(simu['id_hdf5']))
        resfit = fit(lc)
        idx = resfit['fitstatus'] == 'fitok'
        if len(resfit[idx]) > 0:
            if covmb is not None:
                covDict = mbcovCalc(covmb, resfit)
                for key in covDict.keys():
                    resfit[key] = [covDict[key]]

            res = vstack([res, resfit])

    # dump('{}_{}'.format(fit.fit_out,j),names,val,j)

    print('done', j)
    if output_q is not None:
        return output_q.put({j: res})
    else:
        return res


def mbcovCalc(covmb, vals):

    cov = np.ndarray(shape=(3, 3), dtype=float, order='F')
    cov[0, 0] = vals['Cov_x0x0'].data
    cov[1, 1] = vals['Cov_x1x1'].data
    cov[2, 2] = vals['Cov_colorcolor'].data
    cov[0, 1] = vals['Cov_x0x1'].data
    cov[0, 2] = vals['Cov_x0color'].data
    cov[1, 2] = vals['Cov_x1color'].data
    cov[2, 1] = cov[1, 2]
    cov[1, 0] = cov[0, 1]
    cov[2, 0] = cov[0, 2]

    params = dict(zip(['x0', 'x1', 'c'], [vals['x0_fit'].data,
                                          vals['x1_fit'].data, vals['color_fit'].data]))

    resu = covmb.mbCovar(params, cov, ['x0', 'x1', 'c'])
    sigmu_sq = resu['Cov_mbmb']
    sigmu_sq += alpha**2 * vals['Cov_x1x1'].data + \
        beta**2 * vals['Cov_colorcolor'].data
    sigmu_sq += 2.*alpha*resu['Cov_x1mb']
    sigmu_sq += -2.*alpha*beta*vals['Cov_x1color'].data
    sigmu_sq += -2.*beta*resu['Cov_colormb']
    sigmu = 0.
    if sigmu_sq >= 0.:
        sigmu = np.sqrt(sigmu_sq)

    resu['sigma_mu'] = sigmu.item()
    return resu


def multiproc(simu_name='', lc_name='', fit=None, covmb=None, nproc=1):

    f = h5py.File(simu_name, 'r')
    print(f.keys())
    for i, key in enumerate(f.keys()):
        simu = Table.read(simu_name, path=key)

    nref = 2000
    nlc = len(simu)
    print('total number of LC', nlc)
    delta = nlc
    if nproc > 1:
        delta = int(delta/(nproc))

    batch = range(0, nlc, delta)
    if nlc not in batch:
        batch = np.append(batch, nlc)
    print(batch)
    names = None
    val = []
    inum = -1
    result_queue = multiprocessing.Queue()

    for i in range(len(batch)-1):
        ida = batch[i]
        idb = batch[i+1]
        if ida > 0 and (ida % nref) == 0:
            print('Running', ida)
        p = multiprocessing.Process(name='Subprocess-'+str(i), target=procelem, args=(
            lc_name, simu[ida:idb], fit, covmb, i, result_queue))
        p.start()
        print('start', i)

    resultdict = {}
    for j in range(len(batch)-1):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    val = Table()
    print('dumping results')
    for j in range(len(batch)-1):
        val = vstack([val, resultdict[j]])

    print('finishing')
    if len(val) > 0:
        inum += 1
        fit.dump(val, inum)


class Fit_All:
    def __init__(self, telescope, output_config, display_lc, fitter_config, prodid):

        module = import_module(fitter_config['name'])
        self.fitter = module.Fit_LC(
            model=fitter_config['model'], version=fitter_config['version'], telescope=telescope, display=display_lc)

        if output_config['save']:
            self.prepareSave(output_config['directory'], prodid)

        self.val = []

    def __call__(self, lc, j=-1, output_q=None):

        resfit = self.fitter(lc)

        if output_q is not None:
            output_q.put({j: resfit})
        else:
            return resfit

    def prepareSave(self, outdir, prodid):

        if not os.path.exists(outdir):
            print('Creating output directory', outdir)
            os.makedirs(outdir)

        self.fit_out = outdir+'/Fit_'+prodid+'.hdf5'
        if os.path.exists(self.fit_out):
            os.remove(self.fit_out)

    def dump(self, tab, inum):

        # dump the results in a hdf5 file
        # res = Table(np.rec.fromrecords(val,names = names))
        # tab = Table(rows=val, names=names)
        tab['fitstatus'] = tab['fitstatus'].astype(
            h5py.special_dtype(vlen=str))
        tab.write(self.fit_out, 'fit_lc_{}'.format(
            inum), append=True, compression=True)


parser = argparse.ArgumentParser(
    description='Run a LC fitter from a configuration file')
parser.add_argument('config_filename',
                    help='Configuration file in YAML format.')


def run(dirFiles, prodid, outDir, nproc, covmb=None):
    # YAML input file.
    # config = yaml.load(open(config_filename))
    config = makeYaml('input/fit_sn/param_fit_gen.yaml',
                      dirFiles, prodid, outDir, nproc)
    print(config)

    # load telescope
    tel_par = config['Instrument']

    # this is for output
    save_status = config['Output']['save']
    outdir = config['Output']['directory']
    prodid = config['ProductionID']

    simu_name = config['Simulations']['dirname']+'/Simu_'+prodid+'.hdf5'
    lc_name = config['Simulations']['dirname']+'/LC_'+prodid+'.hdf5'

    telescope = Telescope(atmos=True, airmass=1.2)

    fit = Fit_All(telescope, config['Output'],
                  config['Display'], config['Fitter'], prodid)

    # Loop on the simu_file to grab simulated LCs
    multiproc(simu_name=simu_name, lc_name=lc_name, fit=fit,
              covmb=covmb, nproc=config['Multiprocessing']['nproc'])


parser = OptionParser()

parser.add_option("--dirFiles", type="str", default='',
                  help="location dir of the files[%default]")
parser.add_option("--prodid", type="str", default='Test',
                  help="db name [%default]")
parser.add_option("--outDir", type="str", default='',
                  help="output dir [%default]")
parser.add_option("--nproc", type="int", default=1,
                  help="number of proc [%default]")
parser.add_option("--mbcov", type="int", default=0,
                  help="mbcol calc [%default]")
parser.add_option("--prefix", type=str, default='sncosmo_DD',
                  help="prefix for input file[%default]")


opts, args = parser.parse_args()

dirFiles = opts.dirFiles
if dirFiles == '':
    dirFiles = '/sps/lsst/users/gris/Output_Simu_pipeline_0'
outDir = opts.outDir
if outDir == '':
    outDir = '/sps/lsst/users/gris/Output_Fit_0'

prodid = opts.prodid
nproc = opts.nproc
mbCalc = opts.mbcov

salt2Dir = 'SALT2_Files'

covmb = None
if mbCalc:
    covmb = MbCov(salt2Dir, paramNames=dict(
        zip(['x0', 'x1', 'color'], ['x0', 'x1', 'c'])))

# prefix = 'sncosmo_DD'
files = glob.glob('{}/Simu_{}_{}*.hdf5'.format(dirFiles, opts.prefix, prodid))

for fi in files:
    prodid = '{}_{}'.format(opts.prefix, fi.split(
        '{}_'.format(opts.prefix))[-1].split('.hdf5')[0])
    print('hhh', prodid)
    run(dirFiles, prodid, outDir, nproc, covmb=covmb)
