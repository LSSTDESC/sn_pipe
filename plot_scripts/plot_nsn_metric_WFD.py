import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
import sn_plotters.sn_NSNPlotters as nsn_plot
from sn_tools.sn_io import loopStack
import matplotlib.pylab as plt
import argparse
from optparse import OptionParser
import glob
from sn_tools.sn_obs import dataInside
import healpy as hp
import numpy.lib.recfunctions as rf
import pandas as pd

def match_colors(data):

    #print('here',data)
    x1_colors = [(-2.0,0.2),(0.0,0.0)]
    corr = dict(zip(x1_colors,['faint','medium']))
    r = []
    for (healpixID,season) in np.unique(data[['healpixID','season']]):
        idx = data['healpixID'] == healpixID
        idx &= data['season'] == season
        zlim = {}
        nsn_med={}
        nsn = {}
        seldata = data[idx]
        pixRa = np.unique(seldata['pixRa'])[0]
        pixDec = np.unique(seldata['pixDec'])[0]
        good_event = True
        for (x1,color) in x1_colors:
            idxb = np.abs(seldata['x1']-x1)<1.e-5
            idxb &= np.abs(seldata['color']-color)<1.e-5
            selb = seldata[idxb]
            if len(selb)>0:
                zlim[corr[(x1,color)]] = selb['zlim'][0]
                nsn_med[corr[(x1,color)]] = selb['nsn_med'][0]
                nsn[corr[(x1,color)]] = selb['nsn'][0]
            else:
                #zlim[corr[(x1,color)]] = -1.
                
                good_event = False
        if good_event:        
            r.append((healpixID,season,pixRa,pixDec,zlim['faint'],zlim['medium'],nsn_med['faint'],nsn_med['medium'],nsn['faint'],nsn['medium']))

   
    return np.rec.fromrecords(r, names=['healpixID','season',
                                        'pixRa','pixDec',
                                        'zlim_faint','zlim_medium',
                                        'nsn_med_zfaint','nsn_med_zmedium',
                                        'nsn_zfaint','nsn_zmedium'])
    


def append(metricTot,sel):

    if metricTot is None:
        metricTot = np.copy(sel)
    else:
        metricTot = np.concatenate((metricTot,np.copy(sel)))

    return metricTot

def getFields(elaisRa=0.0):

    r = []
    r.append(('COSMOS '.ljust(7), 1, 2786, 150.36, 2.84))
    r.append(('XMM-LSS', 2, 2412, 34.39, -5.09))
    r.append(('CDFS'.ljust(7), 3, 1427, 53.00, -27.44))
    r.append(('ELAIS'.ljust(7), 4, 744, elaisRa, -45.52))
    r.append(('SPT'.ljust(7), 5, 290, 349.39, -63.32))
    #r.append(('Fake'.ljust(7), 6, 111, 0.0, 0.0))
    r.append(('ADFS'.ljust(7), 6, 290,61.00, -48.0))
    fields_DD = np.rec.fromrecords(
        r, names=['fieldname', 'fieldnum', 'fieldId', 'Ra', 'Dec'])
    return fields_DD

def getVals(fields_DD, tab, cadence, nside=64, plotting=False):
    
    pixArea = hp.nside2pixarea(nside, degrees=True)

    if plotting:
        fig, ax = plt.subplots()

    r = []
    dataTot = None

    print(tab)
    print(tab.dtype)
    for field in fields_DD:
        dataSel = dataInside(
            tab, field['Ra'], field['Dec'], 10., 10., 'pixRa', 'pixDec')
        if dataSel is not None:                            
            dataSel = match_colors(dataSel)
            if dataSel is not None:
                dataSel = rf.append_fields(dataSel,'fieldname',[field['fieldname']]*len(dataSel))
                dataSel = rf.append_fields(dataSel,'fieldnum',[int(field['fieldnum'])]*len(dataSel))
                dataSel = rf.append_fields(dataSel,'cadence',[cadence]*len(dataSel))
                dataSel = rf.append_fields(dataSel,'nside',[nside]*len(dataSel))
                dataSel = rf.append_fields(dataSel,'pixArea',[pixArea]*len(dataSel))
                if dataTot is None:
                    dataTot = dataSel
                else:
                    dataTot = np.concatenate((dataTot,dataSel))

    print(dataTot)
    return dataTot

parser = OptionParser(description='Display Cadence metric results for DD fields')
parser.add_option("--dirFile", type="str", default='', help="file directory [%default]")
parser.add_option("--nside", type="int", default=128, help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD', help="field type - DD, WFD, Fake [%default]")
opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'

nside = opts.nside
fieldType = opts.fieldType
metricName = 'NSN'

dbNames = ['altLike_v1.3_10yrs']

metricTot = None
metricTot_med = None
pixArea = hp.nside2pixarea(nside,degrees=True)
x1 = -2.0
color = 0.2

for dbName in dbNames:
    search_path = '{}/{}/{}/*NSNMetric_{}*_nside_{}_*'.format(dirFile,dbName,metricName,fieldType,nside)
    print('looking for',search_path)
    fileNames = glob.glob(search_path)
    #fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
    print(fileNames)
    metricValues = np.array(loopStack(fileNames,'astropyTable'))
   


print(metricValues.dtype)
x1_colors = [(0.0,0.0),(-2.0,0.2)]
        
nsn = {}

for (x1,color) in x1_colors:
    nsn[(x1,color)] = 0.

for season in np.unique(metricValues['season']):
    idx = (metricValues['season']==season)&(metricValues['status']==1)
    sel = metricValues[idx]

    for (x1,color) in x1_colors:
        idb = np.abs(sel['x1']-x1)<1.e-5
        idb &= np.abs(sel['color']-x1)<1.e-5
        if len(sel[idb]) > 1:
            nsn[(x1,color)]+=np.sum(sel[idb]['nsn'])
            print(season,x1,color,np.sum(sel[idb]['nsn']))
        else:
            print(season,x1,color,0.)
print(nsn)
