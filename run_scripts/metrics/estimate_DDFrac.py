from sn_tools.sn_obs import renameFields, getFields
from optparse import OptionParser
import numpy as np
from sn_tools.sn_io import Read_Sqlite
from sn_tools.sn_cadence_tools import AnaOS
from sn_tools.sn_obs import DDFields
import os
import matplotlib.pyplot as plt
import pandas as pd
from sn_tools.sn_utils import MultiProc


def func(proc):

    print('there man', proc)
    n_cluster = proc['nproc']
    #dbName = proc['dbName'].decode()
    dbName = proc['dbName']
    dbDir = proc['dbDir']
    dbExtens = proc['dbExtens']
    fields = DDFields()

    print('dbDir', dbDir)
    print('dbName', dbName)
    print('go', dbDir, dbName, dbExtens, n_cluster)
    ana = AnaOS(dbDir, dbName, dbExtens, n_cluster, fields).stat

    return ana


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

    df['Nvisits_DD'] = frac_DD*(df['DD']+df['WFD'])
    df['Nvisits_DD'] /= season_length*nseasons*nfields*30./cadence

    print(df[['cadence', 'Nvisits_DD']])


def plots(df, fields=['COSMOS']):
    """
    Function performing a set of plots

    Parameters
    ----------------
    df: pandas df
     df with variables to plot
    fields: list(str)
     fields to display

    """

    # sort the df by DD fraction
    df = df.sort_values(by=['frac_DD'])

    # plot DD fraction
    df['cadence'] = df['cadence'].map(lambda x: x.split('_v1')[0])
    plotDD(df, 'frac_DD', 'DD frac')

    # loop on fields to plot filter allocation

    for field in fields:
        for band in 'ugrizy':
            cola = '{}'.format(field)
            colb = '{}_{}'.format(field, band)
            dfb = df[['cadence', cola, colb]]
            bandfrac = '{}_frac_{}'.format(field, band)
            dfb.loc[:, bandfrac] = dfb[colb]/dfb[cola]
            plotDD(dfb, bandfrac,
                   '{} - {} band frac'.format(field, band))
        for val in ['area', 'width_RA', 'width_Dec']:
            plotDD(df, '{}_{}'.format(field, val),
                   '{} - {}'.format(field, val))


def plotDD(df, what, leg):
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
    fontsize = 15

    df = df.sort_values(by=what)
    ax.barh(df['cadence'], df[what])

    xmin, xmax = ax.get_xlim()
    ax.set_xlim([0.0, xmax])
    ax.set_xlabel(leg, fontsize=fontsize)
    # ax.yaxis.label.set_size(3.)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.grid(axis='x')
    plt.tight_layout()


parser = OptionParser()
parser.add_option("--dbList", type="str", default='List.csv',
                  help="db name [%default]")

opts, args = parser.parse_args()

print('Start processing...')

dbDir = opts.dbDir
dbList = opts.dbList
dbExtens = opts.dbExtens

outName = 'Nvisits.npy'

# if not os.path.isfile(outName):
"""
toprocess = np.genfromtxt(dbList, dtype=None, names=[
'dbName', 'simuType', 'nside', 'coadd', 'fieldType', 'nproc'])
if len(toprocess.shape) == 1:
toprocess = np.array([toprocess])
"""
toprocess = pd.read_csv(dbList, comment='#')

print('toprocess', toprocess)
#data = MultiProc(toprocess, params, func, nproc=1).data
res = pd.DataFrame()
for i, vv in toprocess.iterrows():
    data = func(vv)
    res = pd.concat((res, data))

np.save(outName, res.to_records(index=False))


tab = pd.DataFrame(np.load(outName, allow_pickle=True))

# this is a check: number of visits per obs night with
# cadence, season_length, nseasons and nfields chosen
# Nvisits_night_test(tab)


# Plots
plots(tab)

plt.show()
