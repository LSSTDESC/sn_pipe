import h5py
import glob
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


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
    """

    def __init__(self, x1, color, bluecutoff, redcutoff, dirFile):

        self.x1 = x1
        self.color = color
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff

        prefix = 'LC_{}_{}_{}_{}_ebvofMW'.format(
            x1, color, bluecutoff, redcutoff)

        self.process(dirFile, prefix)

    def loadData(self, ff):
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

        ebvofMW = ff.split('ebvofMW_')[1]
        ebvofMW = float(ebvofMW.split('_vstack')[0])

        df['ebvofMW'] = ebvofMW
        df = df.round({'z': 2, 'phase': 2, 'ebvofMW': 3})
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

    def process(self, thedir, prefix):

        fis = glob.glob('{}/{}_*_vstack.hdf5'.format(thedir, prefix))

        reffile = glob.glob('{}/{}_0.0_vstack.hdf5'.format(thedir, prefix))

        print(reffile)
        dataref = self.loadData(reffile[0])

        phases = np.around(np.arange(-20., 55., 0.1), 2)

        dftot = pd.DataFrame()
        for ff in fis:
            datadf = self.loadData(ff)

            dfmerge = dataref.merge(
                datadf, on=['z', 'phase', 'band'])

            res = dfmerge.groupby(['band', 'z']).apply(
                lambda x: self.getVals(x, phases)).reset_index()

            print(res)

            dftot = pd.concat((dftot, res))

        outName = 'Dust_{}_{}_{}_{}.hdf5'.format(
            self.x1, self.color, self.bluecutoff, self.redcutoff)
        Table.from_pandas(dftot).write(outName, 'dust',
                                       compression=True, append=True)


thedir = '../Templates'
x1 = -2.0
color = 0.2
bluecutoff = 380.0
redcutoff = 800.0
DustEffects_Templates(x1, color, bluecutoff, redcutoff, thedir)
