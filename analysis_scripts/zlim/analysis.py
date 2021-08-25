import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib
matplotlib.use('agg')


def loadData(dbDir, dbName, tagName):
    """
    function to load data from file

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
    from sn_tools.sn_io import loopStack
    import glob
    fullName = '{}/{}/*{}*.hdf5'.format(dbDir, dbName, tagName)
    fis = glob.glob(fullName)

    print('loading', fis)
    tab = loopStack(fis, 'astropyTable')

    return tab


class zlimit:
    """
    class to estimate the redshift limit from (fully) fitted LC curves

    Parameters
    ---------------
    data: astropytable
       data to process (default: 1)
    sigmaC: float, opt
      max sigmaC selection value (default: 0.04) 
    nproc: int, opt
      number of proc for multiprocessing

    """

    def __init__(self, data, sigmaC=0.04, nproc=1):

        self.data = data
        self.sigmaC = sigmaC
        self.nproc = nproc

    def __call__(self):
        """
        Method to process data using multiprocessing

        Returns
        ----------
        numpy array with the following cols: HealpixID, season, zlim_50, zlim_95

        """
        from sn_tools.sn_utils import multiproc
        healpixIDs = np.unique(self.data['healpixID'])
        params = {}
        params['data'] = self.data
        params['sigmaC'] = self.sigmaC

        res = multiproc(healpixIDs, params, self.zlim, self.nproc)

        return res

    def zlim(self, healpixID_list, params={'sigmaC': 0.04}, j=0, output_q=None):
        """
        Method to estimate redshift limits

        Parameters
        ---------------
        healpixID_list: list(int)
          list of healpixIDs to process
        params: dict
           dict of parameters
        j: int, opt
          tag for multiprocessing (default: 0)
        output_q: multiprocessing queue
          default: None

        Returns
        -----------
        numpy array with the following cols: HealpixID, season, zlim_50, zlim_95
        """
        sigmaC = params['sigmaC']
        tab = params['data']

        # select good data here
        idx = tab['fitstatus'] == 'fitok'
        idx &= np.sqrt(tab['Cov_colorcolor']) <= sigmaC

        tab = tab[idx]

        r = []
        for healpixID in healpixID_list:
            idx = tab['healpixID'] == healpixID
            sel = tab[idx]
            if len(sel) > 0:
                for season in np.unique(sel['season']):
                    idxb = sel['season'] == season
                    selb = sel[idxb]
                    if len(selb) > 0:
                        r_50, r_95 = self.zlim_from_cumul(selb)
                        r.append((healpixID, season, r_50, r_95))
        res = None
        if len(r) > 0:
            res = np.rec.fromrecords(
                r, names=['healpixID', 'season', 'zlim_50', 'zlim_95'])
        print('finished', j, len(res))
        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def zlim_from_cumul(self, data):
        """
        Method to estimate zlimit from the cumulative of redshift distrib

        Parameters
        ---------------
        data: array
          dat to process

        Returns
        -----------
        z_50: float
          50th percentile of the cumulative distribution
        z_95: float
          95th percentile of the cumulative distribution

        """
        data.sort(keys=['z'])
        n_bins = 100
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(data['z'], n_bins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')
        bin_center = (bins[:-1] + bins[1:]) / 2
        interp = interpolate.interp1d(n, bin_center)
        r_50 = interp(0.5).item()
        r_95 = interp(0.95).item()
        # plt.show()
        plt.close(fig)

        return r_50, r_95
