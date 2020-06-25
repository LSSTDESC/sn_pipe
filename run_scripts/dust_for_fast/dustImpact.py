import os
import numpy as np
from optparse import OptionParser
import glob
import h5py
from astropy.table import Table, vstack
from scipy.interpolate import interp1d
import multiprocessing


class DustEffects:
    """
    class analyzing results of simulation+dust files

    Parameters
    ---------------
    x1: float
      SN x1
    color: float
      SN color
    dirSimu: str
       location dir of simulation file
    nproc: int, opt
       nproc for multiprocessing (default: 4)

    """

    def __init__(self, x1, color, dirSimu, nproc=4):

        self.x1 = np.round(x1, 2)
        self.color = np.round(color, 2)
        self.nproc = nproc

        # save the final Table in a hdf5 file
        # the name of the file will be Dust_x1_color.hdf5

        self.outName = 'Dust_{}_{}'.format(self.x1, self.color)

        # get all the files
        lc_dust_files = glob.glob('{}/LC*.hdf5'.format(dirSimu))

        # among the dust files, there should be one finishing with ebvofMW_0.hdf5
        # this will be the reference LC file to estimate dust effects

        simu_dust_ref = glob.glob(
            '{}/Simu*_ebvofMW_0.0.hdf5'.format(dirSimu))[0]
        lc_dust_ref = glob.glob('{}/LC*_ebvofMW_0.0.hdf5'.format(dirSimu))[0]

        # get astropy Table of Simu_dust_ref
        simu_ref = self.loadSimu(simu_dust_ref)

        # get the effects of dust
        self.multiprocess(simu_ref, lc_dust_ref, lc_dust_files)

        # merge files in one astropy Table
        tabtot = Table()
        finalOutName = '{}.hdf5'.format(self.outName)
        check_rm(finalOutName)
        for io in range(nproc):
            print('processing', io)
            fName = 'Dust_{}_{}_{}.hdf5'.format(x1, color, io)
            tabtot = vstack([tabtot, loadStack(fName)])
            # remove the file since no more needed
            os.system('rm {}'.format(fName))

        tabtot.write(finalOutName, 'dust', compression=True)

    def multiprocess(self, simu_ref, lc_dust_ref, lc_dust_files):
        """
        Method to estimate dust effects using multiprocessing

        Parameters
        --------------
        simu_ref: astropy Table
          astropy Table with a list of simulated LCs
        lc_ref: str
           name (full path) of the reference LC file
        lcfiles: list(str)
           list of names (full path) of simulated LCs

        Returns
        -----------
        astropy Table with results
        """

        # now estimate dust effects using multiprocessing
        nz = len(simu_ref)
        t = np.linspace(0, nz, self.nproc+1, dtype='int')
        print('hello', nz, t)

        result_queue = multiprocessing.Queue()

        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.dustImpact,
                                         args=(simu_ref[t[j]:t[j+1]], lc_dust_ref, lc_dust_files, j, result_queue))
                 for j in range(self.nproc)]

        for p in procs:
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(self.nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = []

        # gather the results
        for key, vals in resultdict.items():
            restot.append(vals)

        return restot

    def loadSimu(self, simuFile):
        """
        Method to load astropy Table in hdf5 file

        Parameters
        ---------------
        simuFile: str
          name of the file to load

        Returns
        -----------
        astropy Table of the loaded file

        """
        # getting the simu file
        f = h5py.File(simuFile, 'r')
        # print(f.keys())
        # reading the simu file
        for i, key in enumerate(f.keys()):
            simu = Table.read(simuFile, path=key)

        return simu

    def dustImpact(self, simu_ref, lc_ref, lcs, jtag, output_q=None):
        """
        Method to estimate dust effects

        Parameters
        ---------------
        simu_ref: astropy Table
          astropy Table with a list of simulated LCs
        lc_ref: str
           name(full path) of the reference LC file
        lcs: list(str)
           list of names(full path) of simulated LCs
        jtag: int
          tag for multiprocessing

        """

        outName = '{}_{}.hdf5'.format(self.outName, jtag)

        check_rm(outName)
        # effects will be given as a function of the phase
        phases = np.around(np.arange(-20., 55., 0.1), 2)

        print('processing', len(simu_ref))
        # loop on simu file
        for tt in simu_ref:
            # get the reference LC file
            lcref = Table.read(lc_ref, path='lc_{}'.format(tt['index_hdf5']))
            # print(lcref.columns)
            lctot = Table(lcref)

            # loop on all simulated LC files
            for lcName in lcs:
                # grab LCs
                lcb = Table.read(lcName, path='lc_{}'.format(tt['index_hdf5']))
                key = 'dust_{}_{}'.format(np.round(lcb.meta['z'], 2), np.round(
                    lcb.meta['ebvofMW'], 3))
                # print('processing ', key, tt['index_hdf5'])
                # loop on all bands - effects are given per band, phase and ebvofMW values
                # final result: astropy Table
                tabtot = Table()
                for band in 'grizy':
                    selref = self.select(lcref, band)
                    if len(selref) > 0:
                        selb = self.select(lcb, band)
                        # interpolator of the flux ratio (lc/lcref) vs phase
                        interpo = interp1d(selb['phase'], selb['flux_e_sec'] /
                                           selref['flux_e_sec'], bounds_error=False, fill_value=0.)
                        # fill the Table for output
                        tab = Table()
                        tab['phase'] = phases
                        ratios = interpo(phases)
                        ratios[np.isnan(ratios)] = 0.
                        tab['ratio'] = ratios
                        tab['band'] = band
                        tab['z'] = np.round(lcb.meta['z'], 2)
                        tab['ebvofMW'] = np.round(
                            lcb.meta['ebvofMW'], 3)
                        tabtot = vstack([tabtot, tab])

                # write the Table
                tabtot.write(outName, key,
                             compression=True, append=True)

        if output_q is not None:
            return output_q.put({jtag: 1})
        else:
            return 1

    def select(self, lc, band, prefix='LSST::'):
        """
        Method to select LC according to the band

        Parameters
        ---------------
        lc: astropy Table
          MC to consider
        band: str
           band to select in LC
        prefix: str, opt
          prefix in the band name (default: LSST:: )

        Returns
        -----------
        astropy Table of selected LC

        """
        idx = lc['band'] == '{}{}'.format(prefix, band)
        return lc[idx]


def check_rm(fName):
    """
    Function checking whether a file exists
    and remove it if it is the case

    Parameters
    ---------------
    fName: str
      file name(full path)
    """
    # If this file already exist, remove it!
    if os.path.isfile(fName):
        os.system('rm {}'.format(fName))


def loadStack(fName):
    """
    Method to load astropy Table in hdf5 file
    and stack tables therein

    Parameters
    ---------------
    fName: str
      name of the file to load

    Returns
    -----------
    astropy Table of the loaded file

    """
    # getting the file
    f = h5py.File(fName, 'r')
    # print(f.keys())
    # reading the file
    tabtot = Table()
    for i, key in enumerate(f.keys()):
        tabtot = vstack([tabtot, Table.read(fName, path=key)])

    return tabtot


parser = OptionParser()

parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1 [%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color [%default]")
parser.add_option("--zmin", type=float, default=0.01,
                  help="redshift min value [%default]")
parser.add_option("--zmax", type=float, default=1.0,
                  help="redshift max value [%default]")
parser.add_option("--zstep", type=float, default=0.01,
                  help="redshift step value[%default]")

opts, args = parser.parse_args()

print('Start processing...', opts)

zmin = opts.zmin
zmax = opts.zmax
zstep = opts.zstep

fakeFile = 'input/Fake_cadence/Fake_cadence.yaml'

x1 = np.round(opts.x1, 2)
color = np.round(opts.color, 2)

outSimu = 'Output_Simu_dust'
# first step: simulation with some dust varying values
cmd = 'python run_scripts/fakes/full_simulation.py'
cmd += ' --simulator sn_cosmo'
cmd += ' --x1 {}'.format(x1)
cmd += ' --color {}'.format(color)
cmd += ' --zmin {}'.format(opts.zmin)
cmd += ' --zmax {}'.format(opts.zmax)
cmd += ' --zstep {}'.format(opts.zstep)
cmd += ' --outDir_simu {}'.format(outSimu)
cmd += ' --fake_config {}'.format(fakeFile)


# ebvofMW values
ebvals = list(np.arange(0.0, 0.06, 0.005))

for ebv in ebvals:
    comd = cmd
    comd += ' --ebvofMW {}'.format(ebv)
    print(comd)
    os.system(comd)

# second step: result analysis
nproc = 4
dustprocess = DustEffects(x1, color, outSimu, nproc=nproc)
