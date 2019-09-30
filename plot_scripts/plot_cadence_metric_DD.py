import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
from sn_tools.sn_io import loopStack
import matplotlib.pylab as plt
import argparse
from optparse import OptionParser
import glob
from sn_tools.sn_obs import dataInside
import healpy as hp
import numpy.lib.recfunctions as rf
import pandas as pd

def getFields(elaisRa=0.0):

    r = []
    r.append(('COSMOS '.ljust(7), 1, 2786, 150.36, 2.84))
    r.append(('XMM-LSS', 2, 2412, 34.39, -5.09))
    r.append(('CDFS'.ljust(7), 3, 1427, 53.00, -27.44))
    r.append(('ELAIS'.ljust(7), 4, 744, elaisRa, -45.52))
    r.append(('SPT'.ljust(7), 5, 290, 349.39, -63.32))

    fields_DD = np.rec.fromrecords(
        r, names=['fieldname', 'fieldnum', 'fieldId', 'Ra', 'Dec'])
    return fields_DD

def getVals(fields_DD, tab, cadence, nside=64, plotting=False):
    pixArea = hp.nside2pixarea(nside, degrees=True)

    if plotting:
        fig, ax = plt.subplots()

    r = []
    dataTot = None
    for field in fields_DD:
        dataSel = dataInside(
            tab, field['Ra'], field['Dec'], 10., 10., 'pixRa', 'pixDec')
        if dataSel is not None:
            dataSel = rf.append_fields(dataSel,'fieldname',[field['fieldname']]*len(dataSel))
            dataSel = rf.append_fields(dataSel,'fieldnum',[int(field['fieldnum'])]*len(dataSel))
            dataSel = rf.append_fields(dataSel,'cadence',[cadence]*len(dataSel))
            if dataTot is None:
                dataTot = dataSel
            else:
                dataTot = np.concatenate((dataTot,dataSel))

    
    return dataTot

parser = OptionParser(description='Display Cadence metric results for DD fields')
parser.add_option("--dirFile", type="str", default='', help="file directory [%default]")
parser.add_option("--nside", type="int", default=128, help="nside for healpixels [%default]")
opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'

nside = opts.nside
fieldtype = 'DD'

dbNames = ['ddf_pn_0.23deg_1exp_pairsmix_10yrs']
dbNames = ['kraken_2026','ddf_pn_0.23deg_1exp_pairsmix_10yrs_nodither']
#dbNames = ['ddf_0.70deg_1exp_pairsmix_10yrs']
#dbNames = ['ddf_0.23deg_1exp_pairsmix_10yrs']
#dbNames = ['ddf_pn_0.70deg_1exp_pairsmix_10yrs']
colors = ['k', 'r', 'b']
markers = ['s', '*', 'o']
mfc = ['None', 'None', 'None']

lengths = [len(val) for val in dbNames]
adjl = np.max(lengths)

metricTot = None
pixArea = hp.nside2pixarea(nside,degrees=True)


for dbName in dbNames:
    fileNames = glob.glob('{}/{}/*CadenceMetric_{}*_nside_{}_*'.format(dirFile,dbName,fieldtype,nside))
    #fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
    print(fileNames)
    #metricValues = np.load(fileName)
    metricValues = loopStack(fileNames).to_records(index=False)
    fields_DD = getFields(10.)
    tab = getVals(fields_DD, metricValues, dbName.ljust(adjl), nside,True)
    """
    idx = metricValues['filter']=='all'
    sel = metricValues[idx]
    #print(sel[['pixRa','pixDec','filter']])
    print(sel.dtype)
    for (pixRa,pixDec) in np.unique(sel[['pixRa','pixDec']]):
        idd = np.abs(sel['pixRa']-pixRa)<1.e-5
        idd &= np.abs(sel['pixDec']-pixDec)<1.e-5
        print(len(sel[idd]))
        print(sel[idd][['pixRa','pixDec','healpixID','season','filter']])
        break
    #    print(val)
    #print(test)
    print(len(np.unique(sel[['pixRa','pixDec']])),len(sel))
    print(tab.dtype,np.sum(sel['pixArea']))
    idb = fields_DD['fieldname'] == 'COSMOS'.ljust(7)
    #sn_plot.plotMollview(nside,sel,'season_length','season_length','days',1.,'',dbName,saveFig=False,seasons=-1,type='mollview', fieldzoom=fields_DD[idb])
    """
    if metricTot is None:
        metricTot = tab
    else:
        metricTot = np.concatenate((metricTot,tab))


fontsize = 15
fields_DD = getFields()
print(metricTot[['cadence','filter']])

#grab median values
df = pd.DataFrame(np.copy(metricTot)).groupby(['healpixID','fieldnum','filter','cadence']).median().reset_index()

#print(df)
metricTot = df.to_records(index=False)
idx = metricTot['filter']=='all'
sel = metricTot[idx]


figleg = 'nside = {}'.format(nside)
sn_plot.plotDDLoop(nside,dbNames,sel,'season_length','season length [days]',markers,colors,mfc,adjl,fields_DD,figleg)
sn_plot.plotDDLoop(nside,dbNames,sel,'cadence_mean','cadence [days]',markers,colors,mfc,adjl,fields_DD,figleg)



"""
for band in 'grizy':
    idx = metricTot['filter']==band
    sel = metricTot[idx]
    figlegb = '{} - {} band'.format(figleg,band)
    sn_plot.plotDDLoop(nside,dbNames,sel,'visitExposureTime','Exposure Time [s]',markers,colors,mfc,adjl,fields_DD,figlegb)
"""
filtercolors = 'cgyrm'
filtermarkers = ['o','*','s','v','^']
mfiltc = ['None']*len(filtercolors)

print(metricTot.dtype)
vars = ['visitExposureTime','cadence_mean','gap_max','gap_5']
legends = ['Exposure Time [sec]/night','cadence [days]','max gap [days]','frac gap > 5 days']

todraw=dict(zip(vars,legends))
for dbName in dbNames:
    for key, vals in todraw.items():
        sn_plot.plotDDCadence(metricTot, dbName, key, vals,adjl,fields_DD)


"""

for season in np.unique(sel['season']):
    idf = (sel['season'] == season)&(sel['season_length']>10.)
    selb = sel[idf]
    plt.plot(selb['fieldnum'],selb['season_length'],marker='.',lineStyle='None',label='season {}'.format(season))

plt.legend()
plt.xticks(fields_DD['fieldnum'], fields_DD['fieldname'], fontsize=fontsize)
"""
plt.show()




print(metricTot.dtype)

