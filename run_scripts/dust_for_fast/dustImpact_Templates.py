import h5py
import glob
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from optparse import OptionParser


class DustEffects_Templates:
    """
    class to parametrize dust effects on flux and Fisher matrix elements

    Parameters
    ---------------
    x1: float
      SN x1
    color: float
      SN color
    bluecutoff: float
      blue cut-off
    redcutoff: float
      red cut-off
    dirFile: str
      dir where LC files are located
    zrange: array
      redshift values to consider
    """

    def __init__(self, x1, color, bluecutoff, redcutoff, error_model, dirFile, zrange):

        self.x1 = x1
        self.color = color
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        error_model = error_model

        cutoff = 'error_model'
        if error_model < 1:
            cutoff = '{}_{}'.format(bluecutoff, redcutoff)


        prefix = 'LC_{}_{}_{}_ebvofMW'.format(
            x1, color, cutoff)

        self.process(dirFile, cutoff,prefix, zrange)

    def loadData(self, ff, zrange=None):
        """
        Method to load the data in ff

        Parameters
        ----------------
        ff: str
           file name (full path)

        Returns
        ----------
        pandas df of data

        """

        lcFile = h5py.File(ff, 'r')
        keys = list(lcFile.keys())
        df = pd.read_hdf(ff, key=keys[0], mode='r')

        print('hello',ff)
        ebvofMW = ff.split('/')[-1].split('ebvofMW_')[1]
        ebvofMW = float(ebvofMW.split('_vstack')[0])

        df['ebvofMW'] = ebvofMW
        df = df.round({'z': 2, 'phase': 2, 'ebvofMW': 3})

        df = df.sort_values(by=['z'])

        if zrange is not None:
            idx = df['z'].isin(zrange)
            df = df[idx]

        return df

    def getVals(self, grp, phases):
        """
        Method to estimate dust effects on flux and Fisher elements

        Parameters
        ---------------
        grp: pandas group
          data to consider
        phases: numpy array
          phases for interpolation

        Returns
        -----------
        pandas df with the ratio wrt the case ebvofMW=0.0

        """

        resdf = pd.DataFrame(phases, columns=['phase'])
        resdf['ebvofMW'] = np.mean(grp['ebvofMW_y'])

        for vv in ['flux_e_sec', 'dx0', 'dcolor', 'dx1', 'ddaymax']:
            interpo = interp1d(grp['phase'], grp['{}_y'.format(vv)] /
                               grp['{}_x'.format(vv)], bounds_error=False, fill_value=0.)
            resdf['ratio_{}'.format(vv)] = interpo(phases)

        resdf = resdf.fillna(0.0)
        resdf = resdf.rename(columns={'ratio_flux_e_sec': 'ratio_flux'})
        return resdf

    def process(self, thedir, cutoff, prefix, zRange=None):
        """
        Method to process the data
        The idea is to loop on input files and estimate 
        the ratio of flux and Fisher elements wrt a reference file corresponding
        to ebvofMW=0.0

        Parameters
        ---------------
        thedir: str
          location directory of the files
        cutoff: str
          cutoff value (ie error_model or bluecutoff_redcutoff)
        prefix: str
          a tag for the files to process
        zrange: array, opt
          redshift values to consider (default: None = all values considered)

        """
        dirTemplates = '{}/Template_LC_{}_ebvofMW*'.format(thedir,cutoff)

        # get the reference table
        reffile = glob.glob('{}/{}_0.0_vstack.hdf5'.format(dirTemplates, prefix))

        print(reffile)
        dataref = self.loadData(reffile[0], zrange)

        # get the list of files to process
        fis = glob.glob('{}/{}_*_vstack.hdf5'.format(dirTemplates, prefix))

        print(fis)
        # phases array: common to all process
        phases = np.around(np.arange(-20., 55., 0.1), 2)

        # dftot: where results will be stored
        dftot = pd.DataFrame()

        # loop on files
        for ff in fis:
            # loading the file
            datadf = self.loadData(ff, zrange)

            # merge with reference
            dfmerge = dataref.merge(
                datadf, on=['z', 'phase', 'band'])

            # estimate ratios in self.getVals
            res = dfmerge.groupby(['band', 'z']).apply(
                lambda x: self.getVals(x, phases)).reset_index()

            """
            print(res)
            
            import matplotlib.pyplot as plt
            plt.plot(res['z'],res['ratio_flux'],'ko')
            plt.show()
            """
            # stack the results
            dftot = pd.concat((dftot, res))

        # save result as an astropy Table
        outName = 'Dust_{}_{}_{}.hdf5'.format(
            self.x1, self.color, cutoff)
        Table.from_pandas(dftot).write(outName, 'dust',
                                       compression=True, append=True)


parser = OptionParser()

parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1 [%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color [%default]")
parser.add_option("--zmin", type=float, default=0.0,
                  help="redshift min value [%default]")
parser.add_option("--zmax", type=float, default=0.8,
                  help="redshift max value [%default]")
parser.add_option("--zstep", type=float, default=0.05,
                  help="redshift step value[%default]")
parser.add_option("--zfiltering", type=int, default=0,
                  help="to filter z values [%default]")
parser.add_option("--bluecutoff", type=float, default=380.0,
                  help="blue cutoff value [%default]")
parser.add_option("--redcutoff", type=float, default=800.0,
                  help="red cutoff value [%default]")
parser.add_option("--dirFiles", type=str, default='../Templates',
                  help="location dir of the files to process [%default]")
parser.add_option("--error_model", type=int, default=1,
                  help="error model bool [%default]")

opts, args = parser.parse_args()

zrange = None
if opts.zfiltering:
    zrange = np.arange(opts.zmin, opts.zmax, opts.zstep)
    zrange = list(np.round(zrange, 2))
    zrange[0] = 0.01

DustEffects_Templates(opts.x1, opts.color, opts.bluecutoff,
                      opts.redcutoff, opts.error_model,opts.dirFiles, zrange)
