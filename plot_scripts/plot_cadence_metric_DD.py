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
from sn_tools.sn_cadence_tools import DDFields,Match_DD
import os
import pandas as pd
from sn_tools.sn_utils import MultiProc



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

    dbName = tab['dbName']
    dirFile = params['dirFile']
    fieldType = params['fieldType']
    nside = params['nside']
    ljustdb = params['ljustdb']
    fsearch = '{}/{}/Cadence/*CadenceMetric_{}*_nside_{}_*.hdf5'.format(
            dirFile, dbName, fieldType, nside)
    
    fileNames = glob.glob(fsearch)
    if not fileNames:
        return None
    metricValues = loopStack(fileNames).to_records(index=False)
    fields_DD = DDFields()

    df =pd.DataFrame(metricValues)
    df.loc[:,'cadence'] = dbName
    tab = Match_DD(fields_DD,df).to_records()
    
    return tab

parser = OptionParser(
    description='Display Cadence metric results for DD fields')
parser.add_option("--dirFile", type="str", default='',
                  help="file directory [%default]")
parser.add_option("--dbList", type="str", default='',
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

opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'

nside = opts.nside
fieldType = opts.fieldType
dbList = opts.dbList


# get the list to process
toprocess = pd.read_csv(dbList).to_records()

outName = 'DD_Cadence_Summary.npy'

lengths = [len(val) for val in toprocess['dbName']]
ljustdb = np.max(lengths)

# summary file: to generate if needed

if not os.path.isfile(outName):
  
    params = {}
    params['dirFile'] = dirFile
    params['fieldType'] = fieldType
    params['nside'] = nside
    params['ljustdb'] = ljustdb

    data = MultiProc(toprocess,params,func,nproc=8).data

    np.save(outName,data)

# load the summary file

metricTot = np.load(outName)
fontsize = 15
fields_DD = DDFields()

# list of fields to display
fields=opts.fields_to_Display.split(',')

# Plots: season_length, cadence, max_gap (median over seasons) per field
if opts.globalPlot:
    sn_plot.plotDDCadence_barh(metricTot,'season_length','season length [days]',bands=['all'],fields=fields)
    sn_plot.plotDDCadence_barh(metricTot,'cadence_mean','cadence [days]',bands=['all'],fields=fields)
    sn_plot.plotDDCadence_barh(metricTot,'gap_max','max gap [days]',bands=['all'],fields=fields)


# Plots: number of visits per night 

if opts.visitsPlot:
    sn_plot.plotDDCadence_barh(metricTot, 
                               'visitExposureTime', 'N$_{visits}$/night', 
                               bands=['g','r','i','z','y'],fields=fields,scale=30.)

## These are plot for a given observing strategy

# choose the variables to draw
if opts.obsPlot:
    vars = ['visitExposureTime', 'cadence_mean', 'numExposures']
    legends = ['Exposure Time [sec]/night',
           'cadence [days]', 'N$_visits$/observing night']
    todraw = dict(zip(vars, legends))

    
    dbChoice  = opts.os_to_Display.split(',')
    pattern = '|'.join(dbChoice)

    df = pd.DataFrame(toprocess)

    idx = df.dbName.str.contains(pattern)
    seldf = df[idx]

    for dbName in seldf['dbName']:
        for key, vals in todraw.items():
            sn_plot.plotDDCadence(metricTot, dbName, key,
                                  vals, ljustdb, fields_DD)
plt.show()



