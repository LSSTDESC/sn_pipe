import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
from sn_tools.sn_io import loopStack
import matplotlib.pylab as plt
import argparse
from optparse import OptionParser
import glob
import healpy as hp
import numpy.lib.recfunctions as rf
import pandas as pd
from sn_tools.sn_cadence_tools import Match_DD
from sn_tools.sn_obs import DDFields
import os
import pandas as pd
from sn_tools.sn_utils import MultiProc
import csv


def write_to_csv(what, df, fieldType='DD', bands=['all']):
    """"
    Method to write metric results to csv files

    Parameters
    ---------------
    what: str
      the variable (df col)
    df: pandas df
      data to process
    fieldType: str, opt
      field type (DD or WFD)
    bands: list(str)
      bands to consider

    Returns
    ----------
    csv file with name  "{}_{}.csv".format(what, fieldType)

    """

    sel = df

    csvfile = "{}_{}.csv".format(what, fieldType)
    dbNames = sel['cadence'].unique()
    dbNamew = [dbName.split('_')[0] for dbName in dbNames]
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        db = [' ']
        db += dbNamew
        writer.writerow(db)
        for fieldname in sel['fieldname'].unique():
            io = sel['fieldname'] == fieldname
            sela = sel[io]
            rr = [fieldname]
            for vv in dbNames:
                ik = sela['cadence'] == vv
                selabb = sela[ik]
                rband = []
                for b in bands:
                    ib = selabb['filter'] == b
                    selfi = selabb[ib]
                    ro = selfi[what].values
                    if len(ro) > 0:
                        rband.append(np.round(ro[0], 1))
                    else:
                        rband.append(0)
                rr += ["/".join(map(str, rband))]
            writer.writerow(rr)


def func(tab, params):
    """
    Function analysing cadence files generated from the Cadence metric

    Parameters
    ----------------
    tab: array
     input data (dbName)
    params: dict
     parameters of the function


    """
    print('processing here', tab)
    dbName = tab['dbName']
    dirFile = params['dirFile']
    fieldType = params['fieldType']
    nside = params['nside']
    ljustdb = params['ljustdb']
    fsearch = '{}/{}/Cadence/*CadenceMetric_{}*_nside_{}_*.hdf5'.format(
        dirFile, dbName, fieldType, nside)

    fileNames = glob.glob(fsearch)
    print('files', fileNames)
    if not fileNames:
        return None
    metricValues = loopStack(fileNames).to_records(index=False)
    fields_DD = DDFields()

    print('hhboooo', len(metricValues), len(
        np.unique(metricValues['healpixID'])))
    df = pd.DataFrame(metricValues)
    df.loc[:, 'cadence'] = dbName
    #tab = Match_DD(fields_DD,df).to_records(index=False)
    tab = Match_DD(fields_DD, df)

    return tab


parser = OptionParser(
    description='Display Cadence metric results for DD fields')
parser.add_option("--dirFile", type="str", default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--dbList", type="str", default='plot_scripts/cadenceCustomize_fbs14.csv',
                  help="dbList  [%default]")
parser.add_option("--nside", type="int", default=128,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type [%default]")
parser.add_option("--globalPlot", type="int", default=0,
                  help="to display (per DD field) cadence, season_lenght, max_gap[%default]")
parser.add_option("--visitsPlot", type="int", default=0,
                  help="to display (per DD field) Nvisits/night and per band[%default]")
parser.add_option("--obsPlot", type="int", default=1,
                  help="to display (per OS) cadence, season_length,filter allocation [%default]")
parser.add_option("--fields_to_Display", type="str", default='COSMOS',
                  help="name of the field to display - for opts globalPlot and visitsPlot [%default]")
parser.add_option("--os_to_Display", type="str", default='descddf,agnddf',
                  help="name of the OS to display - for opts obsPlot [%default]")
parser.add_option("--outName", type="str", default='DD_Cadence_Summary.npy',
                  help="name for output file [%default]")
parser.add_option("--write_to_csv", type="int", default=0,
                  help="to write stat on csv files [%default]")


opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
nside = opts.nside
fieldType = opts.fieldType
dbList = opts.dbList
write_csv = opts.write_to_csv

# get the list to process
toprocess = pd.read_csv(dbList).to_records(index=False)

print('alore', toprocess)
outName = opts.outName

lengths = [len(val) for val in toprocess['dbName']]
ljustdb = np.max(lengths)

# summary file: to generate if needed

if not os.path.isfile(outName):

    params = {}
    params['dirFile'] = dirFile
    params['fieldType'] = fieldType
    params['nside'] = nside
    params['ljustdb'] = ljustdb

    data = MultiProc(toprocess, params, func, nproc=4).data

    np.save(outName, data.to_records(index=False))

# load the summary file

#metricTot = data.to_records(index=False)
metricTot = np.load(outName, allow_pickle=True)
fontsize = 15
fields_DD = DDFields()

# list of fields to display
fields = opts.fields_to_Display.split(',')

# Plots: season_length, cadence, max_gap (median over seasons) per field
print('ooooooo', metricTot.dtype)

# estimate the area covered here
ido = metricTot['filter'] == 'all'
sel = pd.DataFrame(metricTot[ido])

print(sel.columns)
sel = sel.groupby(['fieldname', 'season', 'filter', 'cadence']).apply(lambda x: pd.DataFrame({'pixArea': [x['pixArea'].sum()],
                                                                                              'pixDec': [x['pixDec'].median()], })).reset_index()
#print(sel.groupby(['fieldname','season']).apply(lambda x : x['pixArea'].sum()).reset_index())
print(sel)
plt.plot(sel['pixDec'], sel['pixArea'], 'ko')
if opts.globalPlot:
    sn_plot.plotDDCadence_barh(
        metricTot, 'season_length', 'season length [days]', bands=['all'], fields=fields)
    sn_plot.plotDDCadence_barh(metricTot, 'cadence_mean', 'cadence [days]', bands=[
                               'all'], fields=fields)
    sn_plot.plotDDCadence_barh(metricTot, 'gap_max', 'max gap [days]', bands=[
                               'all'], fields=fields)
    sn_plot.plotDDCadence_barh(sel.to_records(
        index=False), 'pixArea', 'area [deg2]', bands=['all'], fields=fields)


# Plots: number of visits per night

if opts.visitsPlot:
    sn_plot.plotDDCadence_barh(metricTot,
                               'visitExposureTime', 'N$_{visits}$/night',
                               bands=['g', 'r', 'i', 'z', 'y'], fields=fields, scale=30.)

# These are plot for a given observing strategy

# choose the variables to draw
if opts.obsPlot:
    vars = ['visitExposureTime', 'cadence_mean', 'numExposures']
    legends = ['Exposure Time [sec]/night',
               'cadence [days]', 'N$_visits$/observing night']
    todraw = dict(zip(vars, legends))

    dbChoice = opts.os_to_Display.split(',')
    pattern = '|'.join(dbChoice)

    df = pd.DataFrame(toprocess)

    idx = df.dbName.str.contains(pattern)
    seldf = df[idx]

    for dbName in seldf['dbName']:
        for key, vals in todraw.items():
            sn_plot.plotDDCadence(metricTot, dbName, key,
                                  vals, ljustdb, fields_DD)

if write_csv:

    dfa = pd.DataFrame(np.copy(metricTot))

    df = dfa.groupby(['cadence', 'fieldname', 'filter']).median().reset_index()

    df['cadence'] = df['cadence'].map(lambda x: '_'.join(x.split('_')[:4]))

    print(df.columns)

    #cadence_mean, season_length, gap_max
    for val in ['cadence_mean', 'season_length', 'gap_max']:
        write_to_csv(val, df)

    # number of exposures per observing night
    df['numExposures'] = df['numExposures'].astype(int)
    write_to_csv('numExposures', df, bands='ugrizy')

plt.show()
