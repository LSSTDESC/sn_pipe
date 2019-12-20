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
from sn_tools.sn_cadence_tools import DDFields
import os
import multiprocessing

def anadB(dbNames,dirFile,ljustdb,j,output_q):

    metricTot = None
    for dbNam in dbNames:
        dbName = dbNam.decode()
        fsearch = '{}/{}/Cadence/*CadenceMetric_{}*_nside_{}_*'.format(dirFile,dbName,fieldType,nside)
        print('searching',fsearch)
        fileNames = glob.glob(fsearch)
        print(fileNames)
        if not fileNames:
            continue
        metricValues = loopStack(fileNames).to_records(index=False)
        print(metricValues)
        print(metricValues.dtype)
        fields_DD = DDFields()
    
        tab = getVals(fields_DD, metricValues, dbName.ljust(ljustdb), nside,False)

        if tab is not None:
            if metricTot is None:
                metricTot = tab
            else:
                metricTot = np.concatenate((metricTot,tab))
    
    if output_q is not None:
        return output_q.put({j:metricTot})
    else:
        return metricTot

def getVals(fields_DD, tab, cadence, nside=64, plotting=False):

    pixArea = hp.nside2pixarea(nside, degrees=True)

    """
    if plotting:
        fig, ax = plt.subplots()
    """

    r = []
    dataTot = None
    for field in fields_DD:
        dataSel = dataInside(
            tab, field['RA'], field['Dec'], 10., 10., 'pixRa', 'pixDec')
        if dataSel is not None:
            dataSel = rf.append_fields(dataSel,'fieldname',[field['name'].ljust(7)]*len(dataSel))
            dataSel = rf.append_fields(dataSel,'fieldnum',[int(field['fieldnum'])]*len(dataSel))
            dataSel = rf.append_fields(dataSel,'cadence',[cadence]*len(dataSel))
            if dataTot is None:
                dataTot = dataSel
            else:
                print(dataTot,dataSel)
                dataTot = np.concatenate((dataTot,dataSel))

    
    return dataTot

parser = OptionParser(description='Display Cadence metric results for DD fields')
parser.add_option("--dirFile", type="str", default='', help="file directory [%default]")
parser.add_option("--dbList", type="str", default='', help="dbList  [%default]")
parser.add_option("--nside", type="int", default=128, help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD', help="field type [%default]")

opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
if dirFile == '':
    dirFile = '/sps/lsst/users/gris/MetricOutput'

nside = opts.nside
fieldType = opts.fieldType
dbList = opts.dbList

dbNames = ['ddf_pn_0.23deg_1exp_pairsmix_10yrs']
dbNames = ['kraken_2026','ddf_pn_0.23deg_1exp_pairsmix_10yrs']
dbNames = ['Fake_DESC']
dbNames=['agnddf_illum60_v1.3_10yrs']
dbNames=['descddf_illum60_v1.3_10yrsno']
dbNames = ['descddf_illum60_v1.3_10yrs',
           'descddf_illum30_v1.3_10yrs',
           'descddf_illum7_v1.3_10yrs',
           'descddf_illum15_v1.3_10yrs',
           'descddf_illum10_v1.3_10yrs',
           'descddf_illum3_v1.3_10yrs',
           'descddf_illum4_v1.3_10yrs',
           'descddf_illum5_v1.3_10yrs',
           'euclid_ddf_v1.3_10yrs']

nodither = ''

if nodither != '':
    dbNames = [dbName+'nodither' for dbName in dbNames]
"""
,'descddf_illum30_v1.3_10yrs','descddf_illum7
_v1.3_10yrs','descddf_illum15_v1.3_10yrs','descddf_illum10_v1.3_10yrs','descddf_illum3_v1.3_10yrs','euclid_ddf_v1.3_10yrs','descddf_illum
4_v1.3_10yrs','descddf_illum5_v1.3_10yrs',
"""
#dbNames += ['ddf_0.70deg_1exp_pairsmix_10yrs']
#dbNames += ['ddf_0.23deg_1exp_pairsmix_10yrs']
#dbNames += ['ddf_pn_0.70deg_1exp_pairsmix_10yrs']
colors = ['k', 'r', 'b','m','c','g','k','r','b']
markers = ['s', '*', 'o','.','p','P','>','<','^']
mfc = ['None', 'None', 'None','None','None', 'None','None','None','None']

lengths = [len(val) for val in dbNames]
adjl = np.max(lengths)

metricTot = None
pixArea = hp.nside2pixarea(nside,degrees=True)

toprocess = np.genfromtxt(dbList,dtype=None,names=['dbName','simuType','nside','coadd','fieldType','nproc'])

outName = 'DD_Cadence_Summary.npy'
 
lengths = [len(val) for val in toprocess['dbName']]
ljustdb = np.max(lengths)

if not os.path.isfile(outName):

   
    nz = len(toprocess)
    nproc = 8
    tabmulti = np.linspace(0,nz-1,nproc+1,dtype='int')
    result_queue = multiprocessing.Queue()
    for j in range(len(tabmulti)-1):
    #for j in [6]:
        ida = tabmulti[j]
        idb = tabmulti[j+1]
   
        p = multiprocessing.Process(name='Subprocess-'+str(j), target=anadB, args=(toprocess[ida:idb]['dbName'],dirFile,ljustdb,j, result_queue))
        p.start()
 
    resultdict = {}
    for i in range(len(tabmulti)-1):
        resultdict.update(result_queue.get())

    for p in multiprocessing.active_children():
        p.join()

    restot = None
    for key,vals in resultdict.items():
        if restot is None:
            restot = vals
        else:
            restot = np.concatenate((restot,vals))
        #res=pd.concat([res,vals],sort_index=False)
        #restot = None
    
    np.save(outName,restot)
"""
for dbName in dbNames:
    fsearch = '{}/{}/Cadence/*CadenceMetric_{}*_nside_{}_*'.format(dirFile,dbName,fieldType,nside)
    print('searching',fsearch)
    fileNames = glob.glob(fsearch)
    #fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
    print(fileNames)
    #metricValues = np.load(fileName)
    metricValues = loopStack(fileNames).to_records(index=False)
    print(metricValues)
    print(metricValues.dtype)
    fields_DD = getFields(10.)
    #tab = None
    tab = getVals(fields_DD, metricValues, dbName.ljust(adjl), nside,False)
    
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
    
    if tab is not None:
        if metricTot is None:
            metricTot = tab
        else:
            metricTot = np.concatenate((metricTot,tab))
    
"""
metricTot = np.load(outName)
fontsize = 15
fields_DD = DDFields()


#grab median values
#df = pd.DataFrame(np.copy(metricTot)).groupby(['healpixID','fieldnum','filter','cadence']).median().reset_index()

"""
df = pd.DataFrame(np.copy(metricTot)).groupby(['fieldnum','filter','cadence']).median().reset_index()
print(metricTot[['cadence','filter']])

#print(df)
metricTot = df.to_records(index=False)
"""

idx = metricTot['filter']=='all'
sel = metricTot[idx]

print(len(np.unique(metricTot['cadence'])))
print(test)
print(metricTot.dtype)
figleg = 'nside = {}'.format(nside)

if nodither != '':
    figleg += '- {}'.format(nodither)
#sn_plot.plotDDLoop(nside,dbNames,sel,'season_length','season length [days]',markers,colors,mfc,ljustdb,fields_DD,figleg)
#sn_plot.plotDDLoop(nside,dbNames,sel,'cadence_mean','cadence [days]',markers,colors,mfc,ljustdb,fields_DD,figleg)
#sn_plot.plotDDLoop(nside,dbNames,sel,'gap_max','max gap [days]',markers,colors,mfc,ljustdb,fields_DD,figleg)

"""
sn_plot.plotDDLoop_barh(sel,'season_length','season length [days]')

sn_plot.plotDDLoop_barh(sel,'cadence_mean','cadence [days]')
plt.show()

sn_plot.plotDDLoop_barh(sel,'gap_max','max gap [days]')

plt.show()
"""

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
#vars = ['visitExposureTime','cadence_mean','gap_max','gap_5']
#legends = ['Exposure Time [sec]/night','cadence [days]','max gap [days]','frac gap > 5 days']
vars = ['visitExposureTime','cadence_mean','numExposures']
legends = ['Exposure Time [sec]/night','cadence [days]','N$_exposures$/observing night']

#sn_plot.plotDDCadence_barh(metricTot,'numExposures','N$_{exposures}$ per night of observation')
#sn_plot.plotDDCadence_barh(metricTot,'visitExposureTime','Exposure Time [sec]/night')
sn_plot.plotDDCadence_barh(metricTot,'visitExposureTime','N$_{visits}$/night',scale=30.)

todraw=dict(zip(vars,legends))
for dbName in dbNames:
    #if 'descddf_illum10' in dbName:
    if 'baseline_2snap' in dbName:
        for key, vals in todraw.items():
            print(metricTot)
            sn_plot.plotDDCadence(metricTot, dbName, key, vals,ljustdb,fields_DD)
        #break
    
whichdb = 'baseline_2snap'
whichdb = 'euclid'
dbName_f = ''
for dbName in np.unique(metricTot['cadence']):
    if whichdb in dbName:
        print('found')
        dbName_f = dbName

for key, vals in todraw.items():
    print(metricTot)
    sn_plot.plotDDCadence(metricTot,dbName_f, key, vals,ljustdb,fields_DD)
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

