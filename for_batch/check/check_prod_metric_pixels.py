import glob
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_io import loopStack
import numpy as np
from sn_tools.sn_utils import multiproc
import time
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import numpy.lib.recfunctions as rf


class Check_Pixels:
    """
    class to check if the number of pixels processed by the metric is equal to the expected one (from obs pixel map)

    """

    def __init__(self, dirFile, dirObspixels, metricName, dbName, nproc=3):

        self.dirFile = dirFile
        self.dirObspixels = dirObspixels
        self.metricName = metricName
        self.dbName = dbName
        self.nproc = nproc

    def __call__(self):

        # get metric files infos
        df_metric = self.getMetricFiles()

        # get number of pixels processed by the metric
        # npixels_metric = df_metric.groupby(['RA_min_max', 'Dec_min_max']).apply(
        #    lambda x: self.countPixels(x)).reset_index()

        # get list of pixels processed by the metric
        listpixels_metric = df_metric.groupby(['RA_min_max', 'Dec_min_max']).apply(
            lambda x: self.listPixels(x)).reset_index()

        print('hhh', listpixels_metric.groupby(
            ['RA_min_max', 'Dec_min_max']).size())

        # get number of files from obs pixel files
        #npixels_obs = self.countObspixels()

        # get the list of pixels
        listpixels_obs = self.listObspixels()

        # ebv cut
        idx = listpixels_obs['ebvofMW'] < 0.25
        listpixels_obs = listpixels_obs[idx]
        print('bobo', listpixels_obs.groupby(
            ['RA_min_max', 'Dec_min_max']).size())

        for rr in listpixels_obs['RA_min_max'].unique():
            ida = listpixels_obs['RA_min_max'] == rr
            sela = listpixels_obs[ida]
            idb = listpixels_metric['RA_min_max'] == rr
            selb = listpixels_metric[idb]
            diff = list(set(sela['healpixID'].to_list()) -
                        set(selb['healpixID'].to_list()))
            intersect = list(set(sela['healpixID'].to_list()).intersection(
                set(selb['healpixID'].to_list())))
            print(rr, diff, len(diff), intersect[0])
        """
        for rr in listpixels_obs['RA_min_max'].unique():
            ida = listpixels_obs['RA_min_max']==rr
            sela = listpixels_obs[ida]
            for rrb in listpixels_obs['RA_min_max'].unique():
                idb = listpixels_obs['RA_min_max']==rrb
                selb = listpixels_obs[idb]
                diff = list(set(sela['healpixID'].to_list()).intersection(set(selb['healpixID'].to_list())))
                print(rr,rrb, len(diff))
        """
        """
        dfm = npixels_metric.merge(npixels_obs, left_on=['RA_min_max', 'Dec_min_max'], right_on=[
            'RA_min_max', 'Dec_min_max'])

        dfm['diff_pixels'] = dfm['npixels_obs']-dfm['npixels_metric']

        #idx = dfm['diff_pixels'] > 0
        print(dfm)
        """

        # return dfm[idx]

        return -1

    def getMetricFiles(self):
        """
        Method to grab all metric files

        Returns
        -----------
        pandas df with all infos
        """
        path = '{}/{}/{}/{}*.hdf5'.format(self.dirFile,
                                          self.dbName, self.metricName, self.dbName)

        fis = glob.glob(path)

        prefix = '{}_{}Metric_WFD_nside_64_coadd_1_'.format(
            self.dbName, self.metricName)
        r = []
        for fi in fis:
            ra, dec, rastr, decstr, fispl = self.getInfo(fi, prefix)
            r.append((fispl, ra[0], ra[1], dec[0], dec[1], rastr, decstr, fi))

        df = pd.DataFrame(
            r, columns=['fName', 'RAmin', 'RAmax', 'Decmin', 'Decmax', 'RA_min_max', 'Dec_min_max', 'fullName'])
        df['metricName'] = self.metricName
        df['dirFile'] = self.dirFile

        return df

    def countPixels(self, grp):
        """
        Method to estimate the number of pixels grp

        Parameters
        ---------------
        grp: pandas df
          data to process

        Returns
        ----------
        pandas df with the number of pixels as col.

        """
        """
        metricValues = loopStack(grp['fullName'].to_list(), 'astropyTable')

        return pd.DataFrame({'npixels': [len(np.unique(metricValues['healpixID']))]})
        """
        params = {}
        res = multiproc(grp, params, self.countPixels_loop,
                        self.nproc)['npixels_metric'].sum()
        return pd.DataFrame({'npixels_metric': [res]})

    def countPixels_loop(self, grps, params, j=0, output_q=None):
        """
        Method to estimate the number of pixel per grp

        Parameters
        ---------------
        grps: list(pandas group)
          data to process
        params: dict
          dict of parameters
        j: int, opt
           multiproc num (default: 0)
        output_q: multiprocessing queue, opt
          queue where results will be launched (default: None)

        Returns
        -----------
        dic(pandas df) if output_q is not None, pandas df if output_q is None

        """
        npixels = 0
        for io, grp in grps.iterrows():
            metricValues = loopStack([grp['fullName']], 'astropyTable')
            npixels += len(np.unique(metricValues['healpixID']))

        res = pd.DataFrame({'npixels_metric': [npixels]})

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def listPixels(self, grp):
        """
        Method to estimate the list of pixels

        Parameters
        ---------------
        grp: pandas df
          data to process

        Returns
        ----------
        pandas df with the number of pixels as col.

        """
        """
        metricValues = loopStack(grp['fullName'].to_list(), 'astropyTable')

        return pd.DataFrame({'npixels': [len(np.unique(metricValues['healpixID']))]})
        """
        params = {}
        res = multiproc(grp, params, self.listPixels_loop,
                        self.nproc)
        return res

    def listPixels_loop(self, grps, params, j=0, output_q=None):
        """
        Method to estimate the number of pixel per grp

        Parameters
        ---------------
        grps: list(pandas group)
          data to process
        params: dict
          dict of parameters
        j: int, opt
           multiproc num (default: 0)
        output_q: multiprocessing queue, opt
          queue where results will be launched (default: None)

        Returns
        -----------
        dic(pandas df) if output_q is not None, pandas df if output_q is None

        """

        ll = []
        for io, grp in grps.iterrows():
            metricValues = loopStack([grp['fullName']], 'astropyTable')
            ll += np.unique(metricValues['healpixID']).tolist()

        res = pd.DataFrame({'healpixID': ll})

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def countObspixels(self):
        """
        Method to estimate the number of pixels corresponding to all the obs pixel files

        Returns
        ----------
        pandas df with the col RA_min_max, Dec_min_max, npixels_obs
        """
        path = '{}/{}/*.npy'.format(self.dirObspixels, self.dbName)

        fis = glob.glob(path)
        params = {}
        params['dbName'] = self.dbName
        return multiproc(fis, params, self.countObspixels_loop, self.nproc)

    def countObspixels_loop(self, fis, params, j=0, output_q=None):
        """
        Method to estimate the number of pixels corresponding to a set of files

        Parameters
        ---------------
        fis: list(str)
           list of files
        params: dict
           dict of parameters
        j: int, opt
          multiproc number (default: 0)
        output_q: multi proc queue (default: None)

        Returns
        -----------
        pandas df with the number of pixels
        """
        dbName = params['dbName']
        prefix = '{}_WFD_nside_64_'.format(dbName)
        r = []

        for fi in fis:
            tab = np.load(fi, allow_pickle=True)
            # get ebv here
            ebvs = self.ebvofMW(tab['pixRA'], tab['pixDec'])
            tab = rf.append_fields(tab, 'ebvofMW', ebvs)
            idx = tab['ebvofMW'] < 0.25
            npixels = len(np.unique(tab[idx]['healpixID']))
            ra, dec, rastr, decstr, fispl = self.getInfo(fi, prefix)
            r.append((npixels, rastr, decstr))
        res = pd.DataFrame(
            r, columns=['npixels_obs', 'RA_min_max', 'Dec_min_max'])

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def listObspixels(self):
        """
        Method to estimate the list of pixels of all obs pixel files

        Returns
        ----------
        pandas df with the col RA_min_max, Dec_min_max, healpixID, ebvOfMW
        """
        path = '{}/{}/*.npy'.format(self.dirObspixels, self.dbName)

        fis = glob.glob(path)
        params = {}
        params['dbName'] = self.dbName
        return multiproc(fis, params, self.listObspixels_loop, self.nproc)

    def listObspixels_loop(self, fis, params, j=0, output_q=None):
        """
        Method to estimate the number of pixels corresponding to a set of files

        Parameters
        ---------------
        fis: list(str)
           list of files
        params: dict
           dict of parameters
        j: int, opt
          multiproc number (default: 0)
        output_q: multi proc queue (default: None)

        Returns
        -----------
        pandas df with healpixID and ebvofMW
        """
        dbName = params['dbName']
        prefix = '{}_WFD_nside_64_'.format(dbName)

        res = pd.DataFrame()

        for fi in fis:
            tab = np.load(fi, allow_pickle=True)
            ra, dec, rastr, decstr, fispl = self.getInfo(fi, prefix)
            tab_un = np.unique(tab[['healpixID', 'pixRA', 'pixDec']])
            ro = pd.DataFrame(
                tab_un['healpixID'].tolist(), columns=['healpixID'])
            ro['ebvofMW'] = self.ebvofMW(
                tab_un['pixRA'], tab_un['pixDec']).tolist()
            ro['RA_min_max'] = rastr
            ro['Dec_min_max'] = decstr
            res = pd.concat((res, ro))

        if output_q is not None:
            return output_q.put({j: res})
        else:
            return res

    def getInfo(self, fi, prefix):
        """
        Method to extract infos from a string

        Parameters
        ---------------
        fi: str
          string to process
        prefix: str
          str used in process

        Returns
        ----------
        a list of infos related to the string

        """
        fispl = fi.split('/')[-1]
        fisplb = fispl.split(prefix)[1]
        ra = fisplb.split('_')[:2]
        dec = fisplb.split('_')[2:4]
        rastr = '_'.join(ra)
        decstr = '_'.join(dec)

        return ra, dec, rastr, decstr, fispl

    def ebvofMW(self, pixRA, pixDec):

        coords = SkyCoord(pixRA, pixDec, unit='deg')
        try:
            sfd = SFDQuery()
        except Exception as err:
            from dustmaps.config import config
            config['data_dir'] = 'dustmaps'
            import dustmaps.sfd
            dustmaps.sfd.fetch()
            # dustmaps('dustmaps')
        sfd = SFDQuery()
        ebv = sfd(coords)
        return ebv


class Script:

    def __init__(self, df, dirFile, dirObspixels, metricName, dbName):

        self.dirFile = dirFile
        self.dirObspixels = dirObspixels
        self.metricName = metricName
        self.dbName = dbName

        scriptmain = 'tt'
        for io, row in df.iterrows():
            scriptn = '{}_{}'.format(scriptmain, io)
            RAs = row['RA_min_max'].split('_')
            Decs = row['Dec_min_max'].split('_')
            RAmin = float(RAs[0])
            RAmax = float(RAs[1])
            Decmin = float(Decs[0])
            Decmax = float(Decs[1])
            cmd = self.script_for_batch(RAmin, RAmax, Decmin, Decmax, scriptn)
            print(cmd)
            break

    def script_for_batch(self, RAmin, RAmax, Decmin, Decmax, scriptname):

        cmd = 'qsub - P P_lsst - l sps = 1, ct = 20: 00: 00, h_vmem = 16G - j y - o / pbs/throng/lsst/users/gris/sn_pipe_last/sn_pipe/logs/{}.log - pe multicores 8 << EOF \n'.format(
            scriptname)
        cmd += '#!/bin/env bash' + '\n'
        cmd += 'cd /pbs/throng/lsst/users/gris/sn_pipe_last/sn_pipe'+'\n'
        cmd += 'echo \'sourcing setups\'' + '\n'
        cmd += 'source setup_release.sh Linux -5'
        cmd += 'echo \'sourcing done\'' + '\n'
        cmd_m = 'python run_scripts/metrics/run_metrics.py '
        cmd_m += ' --dbDir / sps/lsst/cadence/LSST_SN_PhG/cadence_db/fbs_1.7_1/db '
        cmd_m += ' --dbName {}'.format(self.dbName)
        cmd_m += ' --dbExtens db --nproc 8 --nside 64 --simuType 0 --outDir {}'.format(
            self.dirFiles)
        cmd_m += ' --fieldType WFD --saveData 1 --metric {} --coadd 1'.format(
            self.metricName)
        cmd_m += ' --RAmin {} --RAmax {} --Decmin {} --Decmax {}'.format(
            RAmin, RAmax, Decmin, Decmax)
        cmd_m += ' --pixelmap_dir {}'.format(self.dirObspixels)
        cmd_m += ' --npixels -1 --proxy_level 2 \n'
        cmd += cmd_m
        cmd += 'EOF'


parser = OptionParser()

parser.add_option("--dirFile", type="str",
                  default='/sps/lsst/users/gris/MetricOutput_fbs171_circular_dust', help="metric file dir [%default]")
parser.add_option("--cvsList", type="str", default='WFD_fbs_1.7_1.csv',
                  help="list odf DBs [%default]")
parser.add_option("--metricName", type="str",
                  default='NSN', help="metric name [%default]")
parser.add_option("--dirObspixels", type="str",
                  default='/sps/lsst/users/gris/ObsPixelized_circular_fbs171', help="obs pixel dir [%default]")
parser.add_option("--nproc", type=int,
                  default=8, help="number of proc for multiprocessing [%default]")

opts, args = parser.parse_args()

print('Start processing...')

dbs = pd.read_csv(opts.cvsList, comment='#')

print(dbs)

for index, row in dbs.iterrows():
    dbName = row['dbName']
    # npixels(opts.dirFile, opts.dirObspixels, opts.metricName, dbName)
    check = Check_Pixels(opts.dirFile, opts.dirObspixels,
                         opts.metricName, dbName, opts.nproc)
    pixels = check()
    print('res', pixels)
    break
    """
    if len(pixels) >= 1:
        Script(pixels, opts.dirFile, opts.dirObspixels,
               opts.metricName, dbName)
    break
    """
