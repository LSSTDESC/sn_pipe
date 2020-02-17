from sn_tools.sn_obs import renameFields, getFields
from optparse import OptionParser
import numpy as np
from sn_tools.sn_io import Read_Sqlite
from sn_tools.sn_cadence_tools import AnaOS
import os
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd


class Process:
    def __init__(self, dbDir, dbExtens, outName, toprocess, nproc=8):
        """
        Class to process cadence files
        Results are stored in outName as a numpy array

        Parameters
        ---------------
        dbDir: str
         directory where the files are located
        dbExtens:
         extension (db or npy) of the files
        outName: str
         name of the output file where the results of the processing are stored
        toprocess: pandas DataFrame
         df with (at least) a column dbName: list of cadences to process
        nproc: int, opt
         number of proc to use for the processing

        """

        # Process using multiprocessing
        res = self.multi(dbDir, dbExtens, toprocess, nproc)

        # save the results
        np.save(outName, res)

    def multi(self, dbDir, dbExtens, toprocess, nproc):
        """
        Method to perform multiprocessing

        Parameters
        ---------------
        dbDir: str
         directory where the files are located
        dbExtens:
         extension (db or npy) of the files
        toprocess: pandas DataFrame
         df with (at least) a column dbName: list of cadences to process
        nproc: int, opt
         number of proc to use for the processing

        Returns
        -----------
        numpy array with processed data

        """

        lengths = [len(val) for val in toprocess['dbName']]
        ljustdb = np.max(lengths)

        # multiprocessing parameters
        nz = len(toprocess)
        tabmulti = np.linspace(0, nz-1, nproc+1, dtype='int')
        result_queue = multiprocessing.Queue()

        # loop on processings
        for j in range(len(tabmulti)-1):
            ida = tabmulti[j]
            idb = tabmulti[j+1]

            p = multiprocessing.Process(name='Subprocess-'+str(j), target=self.anaOS_multi,
                                        args=(toprocess[ida:idb], dbDir, dbExtens, ljustdb, j, result_queue))
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(len(tabmulti)-1):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = None

        # gather the results
        for key, vals in resultdict.items():
            if restot is None:
                restot = vals
            else:
                restot = np.concatenate((restot, vals))
                # res=pd.concat([res,vals],sort_index=False)
                # restot = None

        return restot

    def anaOS_multi(self, toprocess, dbDir, dbExtens, ljustdb, j=0, output_q=None):
        """
        Method to process the data

        Parameters
        ---------------
        toprocess: pandas DataFrame
         df with (at least) a column dbName: list of cadences to process
        dbDir: str
         directory where the files are located
        dbExtens:
         extension (db or npy) of the files
        ljustdb: int
         internal parameter for AnaOS



        Returns
        -----------
        numpy array with processed data


        """

        restot = None
        for io, proc in enumerate(toprocess):
            n_cluster = 5
            dbName = proc['dbName'].decode()
            if 'euclid' in dbName:
                n_cluster = 6
            ana = AnaOS(dbDir, dbName, dbExtens, n_cluster, ljustdb).stat
            if ana is not None:
                if restot is None:
                    restot = ana
                else:
                    restot = np.concatenate((restot, ana))

        if output_q is not None:
            return output_q.put({j: restot})
        else:
            return restot


def mystr(thestr):

    return thestr.ljust(9)


def Nvisits_night_test(df, cadence=3.,
                       season_length=6,
                       nseasons=10,
                       nfields=5,
                       frac_DD=0.06):
    """"
    Function to estimate the number of visits per observing night for
    a set(cadence, season_length,nseasons, nfields)
    assuming that the number of DD visits is equal to 6% of the
    total number of visits of a given scheduler simulation

    Parameters
    ----------------
    tab: pandas df
     data to process
    cadence: float, opt
     cadence of observations (default: 3 days-1)
    season_length: float, opt
     season length of observations (default: 6. months)
    nseasons: int, opt
     number of seasons (default: 10)
    nfields: int, opt
     number of fields to consider (default: 5)
    frac_DD: float, opt
     DD fraction for the estimation (default: 0.06)

    """

    print(df)
    df['Field'] = df['Field'].str.strip()
    df['cadence'] = df['cadence'].str.strip()

    rs = df.groupby(['cadence']).apply(lambda x: pd.DataFrame({'Nvisits_DD': frac_DD*(x[x['Field'] == 'DD']
                                                                                       ['Nvisits'].values+x[x['Field'] == 'WFD']['Nvisits'].values)})).reset_index()

    rs['Nvisits_DD'] /= season_length*nseasons*nfields*30./cadence

    print(rs[['cadence', 'Nvisits_DD']])


class PlotStat:
    def __init__(self, df):
        """
        Class to display some results related to DDFs

        Parameters
        ----------------
        df: pandas DataFrame
         df with the following cols:
         cadence: cadence name 
         Field: field name 
         Nvisits: number of visits

        """

        # DD fields names

        self.fields = ['COSMOS', 'ELAIS', 'XMM-LSS',
                       'CDFS', 'SPT', 'ADFS1', 'ADFS2']

        # estimate some stat from initial df
        rbtot = self.calc(df)

        # plots
        self.plots(rbtot)

    def calc(self, df):
        """
        Method to estimate, for each cadence:
         - DD fraction
         - number of visits per field (in total and per band)

        Parameters
        ----------------
        df: pandas DataFrame
         df with the following cols:
         cadence: cadence name 
         Field: field name 
         Nvisits: number of visits

        Returns
        ------------
        pandas df with the following cols:
         cadence: cadence name
         frac_DD: DD fraction
         for field in self.fields:
          field: number of visits
          for band in 'ugrizy':
           field_band: number of visits

        """

        df['Field'] = df['Field'].str.strip()
        df['cadence'] = df['cadence'].str.strip()

        rbtot = df.groupby(['cadence']).apply(lambda x: pd.DataFrame({'frac_DD': x[x['Field'] == 'DD']
                                                                      ['Nvisits'].values/(x[x['Field'] == 'DD']
                                                                                          ['Nvisits'].values+x[x['Field'] == 'WFD']['Nvisits'].values)})).reset_index()

        # rbtot = pd.DataFrame()
        for field in self.fields:
            for b in ['', '_u', '_g', '_r', '_i', '_z', '_y']:
                rb = df.groupby(['cadence']).apply(
                    lambda x: self.Nvisits(x, '{}{}'.format(field, b))).reset_index()

                if rbtot.empty:
                    rbtot = rb.copy()
                else:
                    rbtot = rbtot.merge(
                        rb, left_on=['cadence', 'level_1'], right_on=['cadence', 'level_1'])

        return rbtot

    def Nvisits(self, grp, field):
        """
        Method to estimate the number of visits of a group

        Parameters
        ----------------
        grp: pandas df group
         data to process
        field: str
         name of the field to consider

        Returns
        -----------
        pandas df with one row: number of visits (colname=field)

        """

        idx = grp['Field'] == field

        val = 0
        if len(grp[idx]) > 0:
            val = grp[idx]['Nvisits'].values

        df = pd.DataFrame(columns=[field])
        df.loc[0] = val

        return df

    def plots(self, df):
        """
        Method performing a set of plots

        Parameters
        ----------------
        df: pandas df
          df with variables to plot

        """

        # sort the df by DD fraction
        df = df.sort_values(by=['frac_DD'])

        # plot DD fraction
        self.plotDD(df, 'frac_DD', 'DD frac')

        # loop on fields to plot filter allocation

        for field in ['COSMOS']:
            for band in 'ugrizy':
                cola = '{}'.format(field)
                colb = '{}_{}'.format(field, band)
                dfb = df[['cadence', cola, colb]]
                bandfrac = '{}_frac_{}'.format(field, band)
                dfb.loc[:, bandfrac] = dfb[colb]/dfb[cola]
                self.plotDD(dfb, bandfrac,
                            '{} - {} band frac'.format(field, band))

    def plotDD(self, df, what, leg):
        """

        Method to plot(barh)  some variables: cadence vs what

        Parameters
        ----------------
        df: pandas df
         df with data to plot
        what: str
         x-axis variable to plot
        leg: str
         x-axis legend

        """

        fig, ax = plt.subplots()
        fontsize = 12
        ax.barh(df['cadence'], df[what])

        xmin, xmax = ax.get_xlim()
        ax.set_xlim([0.0, xmax])
        ax.set_xlabel(leg, fontsize=fontsize)
        # ax.yaxis.label.set_size(3.)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize-5.)
        plt.grid(axis='x')
        plt.tight_layout()


parser = OptionParser()
parser.add_option("--dbList", type="str", default='List.txt',
                  help="db name [%default]")
parser.add_option("--dbExtens", type="str", default='npy',
                  help="db extension [%default]")
parser.add_option("--dbDir", type="str", default='',
                  help="db dir [%default]")

opts, args = parser.parse_args()

print('Start processing...')

dbDir = opts.dbDir
if dbDir == '':
    dbDir = '/sps/lsst/cadence/LSST_SN_CADENCE/cadence_db'

dbList = opts.dbList
dbExtens = opts.dbExtens


outName = 'Nvisits.npy'

if not os.path.isfile(outName):
    toprocess = np.genfromtxt(dbList, dtype=None, names=[
        'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])
    Process(dbDir, dbExtens, outName, toprocess, nproc=8)


tab = pd.DataFrame(np.load(outName))

# this is a check: number of visits per obs night with
# cadence, season_length, nseasons and nfields chosen
# Nvisits_night_test(tab)

# Estimate DD fraction for cadences considered in dbList


PlotStat(tab)

plt.show()
