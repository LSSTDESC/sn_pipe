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
from sn_tools.sn_cadence_tools import DDFields
import os
import csv
import pandas as pd


class Summary:
    def __init__(self, dirFile, metricName,
                 fieldType, nside, forPlot, fname=None):

        x1_colors = [(-2.0, 0.2), (0.0, 0.0)]
        self.corr = dict(zip(x1_colors, ['faint', 'medium']))
        df = self.process_loop(dirFile, metricName, fieldType,
                               nside, forPlot)

        self.data = self.identify_DD(df)

    def identify_DD(self, df):

        fields_DD = DDFields()
        dfb = pd.DataFrame()
        for field in fields_DD:
            dataSel = dataInside(
                df.to_records(), field['RA'], field['Dec'], 10., 10., 'pixRa', 'pixDec')
            if dataSel is not None:
                print(type(dataSel))
                dfSel = pd.DataFrame(np.copy(dataSel))
                print(field.dtype)
                dfSel['fieldname'] = field['name']
                dfSel['fieldId'] = field['fieldId']
                dfSel['RA'] = field['RA']
                dfSel['Dec'] = field['Dec']
                dfSel['fieldnum'] = field['fieldnum']
                dfb = pd.concat([dfb, dfSel], sort=False)

        return dfb

    def process_loop(self, dirFile, metricName, fieldType, nside, forPlot):

        metricTot = None

        df = pd.DataFrame()

        io = -1
        for dbName in forPlot['dbName']:
            io += 1
            dfi = self.process(dirFile, dbName, metricName,
                               fieldType, nside, forPlot)
            df = pd.concat([df, dfi], sort=False)
            if io >= 2:
                break

        return df

    def process(self, dirFile, dbName, metricName, fieldType, nside, forPlot):

        search_path = '{}/{}/{}/*{}Metric_{}*_nside_{}_*.hdf5'.format(
            dirFile, dbName, metricName, metricName, fieldType, nside)
        print('looking for', search_path)
        vars = ['pixRa', 'pixDec', 'healpixID', 'season', 'status']

        fileNames = glob.glob(search_path)
        print(fileNames)
        finaldf = pd.DataFrame()
        if fileNames:
            # plt.plot(metricValues['pixRa'],metricValues['pixDec'],'ko')
            # plt.show()
            metricValues = loopStack(fileNames, 'astropyTable').to_pandas()
            print(metricValues.columns)
            newdf = {}
            for key, vals in self.corr.items():
                idx = np.abs(key[0]-metricValues['x1']) < 1.e-5
                idx &= np.abs(key[1]-metricValues['color']) < 1.e-5
                sel = metricValues[idx]
                print(sel[['x1', 'color']])
                sel['zlim_{}'.format(vals)] = sel['zlim']
                sel['nsn_zlim_{}'.format(vals)] = sel['nsn']
                sel['nsn_med_zlim_{}'.format(vals)] = sel['nsn_med']
                sel = sel.drop(
                    columns=['x1', 'color', 'zlim', 'nsn', 'nsn_med', 'index'])
                newdf[vals] = sel

            finaldf = newdf['faint'].merge(
                newdf['medium'], left_on=vars, right_on=vars)

        return finaldf


def summary_old(forPlot):

    for val in forPlot:
        dbName = val['dbName']
        search_path = '{}/{}/{}/*NSNMetric_{}*_nside_{}_*'.format(
            dirFile, dbName, metricName, fieldType, nside)
        print('looking for', search_path)
        fileNames = glob.glob(search_path)
        # fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
        print(fileNames)
        #metricValues = np.load(fileName)
        #metricValues = loopStack(fileNames).to_records(index=False)
        metricValues = np.array(loopStack(fileNames, 'astropyTable'))
        # plt.plot(metricValues['pixRa'],metricValues['pixDec'],'ko')
        # plt.show()

        tab = getVals(fields_DD, metricValues, dbName.ljust(adjl), nside)

        # plt.plot(sel['pixRa'],sel['pixDec'],'ko')
        # plt.show()
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

        #sn_plot.plotMollview(nside,sel,'zlim','zlim - faint SN','',0.,'',dbName,saveFig=False,seasons=-1,type='mollview')
        # plt.show()

        metricTot = append(metricTot, tab)


def summary(forPlot, fname):

    metricTot = None
    for dbName in forPlot['dbName']:
        #dbName = val['dbName']
        search_path = '{}/{}/{}/*NSNMetric_{}*_nside_{}_*.hdf5'.format(
            dirFile, dbName, metricName, fieldType, nside)
        print('looking for', search_path)
        fileNames = glob.glob(search_path)
        # fileName='{}/{}_CadenceMetric_{}.npy'.format(dirFile,dbName,band)
        print(fileNames)
        if fileNames:
            # plt.plot(metricValues['pixRa'],metricValues['pixDec'],'ko')
            # plt.show()
            metricValues = np.array(loopStack(fileNames, 'astropyTable'))

            tab = getVals(fields_DD, metricValues, dbName.ljust(adjl), nside)

            # plt.plot(sel['pixRa'],sel['pixDec'],'ko')
            # plt.show()

            metricTot = append(metricTot, tab)

    # return metricTot
    np.save(fname, np.copy(metricTot))


def match_colors(data):

    # print('here',data)
    x1_colors = [(-2.0, 0.2), (0.0, 0.0)]
    corr = dict(zip(x1_colors, ['faint', 'medium']))
    r = []
    for (healpixID, season) in np.unique(data[['healpixID', 'season']]):
        idx = data['healpixID'] == healpixID
        idx &= data['season'] == season
        zlim = {}
        nsn_med = {}
        nsn = {}
        seldata = data[idx]
        pixRa = np.unique(seldata['pixRa'])[0]
        pixDec = np.unique(seldata['pixDec'])[0]
        good_event = True
        for (x1, color) in x1_colors:
            idxb = np.abs(seldata['x1']-x1) < 1.e-5
            idxb &= np.abs(seldata['color']-color) < 1.e-5
            selb = seldata[idxb]
            if len(selb) > 0:
                zlim[corr[(x1, color)]] = selb['zlim'][0]
                nsn_med[corr[(x1, color)]] = selb['nsn_med'][0]
                nsn[corr[(x1, color)]] = selb['nsn'][0]
            else:
                #zlim[corr[(x1,color)]] = -1.

                good_event = False
        if good_event:
            r.append((healpixID, season, pixRa, pixDec,
                      zlim['faint'], zlim['medium'], nsn_med['faint'], nsn_med['medium'], nsn['faint'], nsn['medium']))

    return np.rec.fromrecords(r, names=['healpixID', 'season',
                                        'pixRa', 'pixDec',
                                        'zlim_faint', 'zlim_medium',
                                        'nsn_med_zfaint', 'nsn_med_zmedium',
                                        'nsn_zfaint', 'nsn_zmedium'])


def append(metricTot, sel):

    if metricTot is None:
        metricTot = np.copy(sel)
    else:
        metricTot = np.concatenate((metricTot, np.copy(sel)))

    return metricTot


def getFields(elaisRa=0.0):

    r = []
    r.append(('COSMOS '.ljust(7), 1, 2786, 150.36, 2.84))
    r.append(('XMM-LSS', 2, 2412, 34.39, -5.09))
    r.append(('CDFS'.ljust(7), 3, 1427, 53.00, -27.44))
    r.append(('ELAIS'.ljust(7), 4, 744, elaisRa, -45.52))
    r.append(('SPT'.ljust(7), 5, 290, 349.39, -63.32))
    #r.append(('Fake'.ljust(7), 6, 111, 0.0, 0.0))
    r.append(('ADFS'.ljust(7), 6, 290, 61.00, -48.0))
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
            tab, field['RA'], field['Dec'], 10., 10., 'pixRa', 'pixDec')
        if dataSel is not None:
            dataSel = match_colors(dataSel)
            if dataSel is not None:
                dataSel = rf.append_fields(dataSel, 'fieldname', [
                                           field['name'].ljust(7)]*len(dataSel))
                dataSel = rf.append_fields(dataSel, 'fieldnum', [
                                           int(field['fieldnum'])]*len(dataSel))
                dataSel = rf.append_fields(
                    dataSel, 'cadence', [cadence]*len(dataSel))
                dataSel = rf.append_fields(
                    dataSel, 'nside', [nside]*len(dataSel))
                dataSel = rf.append_fields(
                    dataSel, 'pixArea', [pixArea]*len(dataSel))
                if dataTot is None:
                    dataTot = dataSel
                else:
                    dataTot = np.concatenate((dataTot, dataSel))

    print(dataTot)
    return dataTot


parser = OptionParser(
    description='Display Cadence metric results for DD fields')
parser.add_option("--dirFile", type="str",
                  default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=128,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--simuVersion", type="str", default='fbs14',
                  help="simulation version[%default]")
parser.add_option("--x1", type="float", default=-2.0,
                  help="SN x1 for display[%default]")
parser.add_option("--color", type="float", default=0.2,
                  help="SN color for display[%default]")

opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
nside = opts.nside
fieldType = opts.fieldType
metricName = 'NSN'

# Loading input file with the list of cadences to take into account and siaplay features
filename = 'plot_scripts/cadenceCustomize_{}.csv'.format(opts.simuVersion)

#forPlot = pd.read_csv(filename).to_records()
forPlot = pd.read_csv(filename)

print(forPlot)


# get DD fields
#fields_DD = getFields(5.)
fields_DD = DDFields()

lengths = [len(val) for val in forPlot['dbName']]
adjl = np.max(lengths)


metricTot = None
metricTot_med = None

# get pixelArea
pixArea = hp.nside2pixarea(nside, degrees=True)

# get x1 and color for SN
x1 = opts.x1
color = opts.color

# Summary: to reproduce the plots faster
"""
fname = 'Summary_DD_{}.npy'.format(opts.simuVersion)
if not os.path.isfile(fname):
    summary(forPlot, fname)
"""
mysum = Summary(dirFile, 'NSN',
                'DD', nside, forPlot).data

# print(test)
metricTot = mysum.to_records()

print('oo', metricTot.dtype, type(metricTot))
# print(test)
fieldnames = np.unique(metricTot['fieldname'])
#nsn_plot.plot_DDSummary(metricTot, dict(zip(dbNames,mmarkers)),dict(zip(fieldnames,colors)),colors_cad)
nsn_plot.plot_DDSummary(metricTot, forPlot)
plt.show()

fontsize = 15
fields_DD = getFields()
# print(metricTot[['cadence','filter']])

# grab median values
"""
df = pd.DataFrame(np.copy(metricTot)).groupby(['healpixID','fieldnum','filter','cadence']).median().reset_index()

#print(df)
metricTot = df.to_records(index=False)
idx = metricTot['filter']=='all'
sel = metricTot[idx]
"""

figleg = 'nside = {}'.format(nside)
# sn_plot.plotDDLoop(nside,dbNames,metricTot,'zlim_faint','$z_{lim}^{faint}$',markers,colors,mfc,adjl,fields_DD,figleg)
#sn_plot.plotDDLoop(nside,dbNames,sel,'cadence_mean','cadence [days]',markers,colors,mfc,adjl,fields_DD,figleg)

#fig,ax = plt.subplots()


# print(metricTot.dtype,type(metricTot))


# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'zlim_faint','zlim_medium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'nsn_med_zfaint','nsn_med_zmedium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDFit(metricTot,'nsn_med_zmedium','nsn_zmedium')
# sn_plot.plotDDFit(metricTot,'zlim_faint','zlim_medium')

# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'nsn_med_zfaint','nsn_med_zmedium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDFit(metricTot,'nsn_med_zmedium','nsn_med_zfaint','zlim_medium','zlim_faint')


df = pd.DataFrame(metricTot)

sums = df.groupby(['fieldnum', 'fieldname', 'cadence', 'nside', 'season'])[
    'pixArea'].sum().reset_index()

idx = sums['pixArea'] > 1.
sn_plot.plotDDLoop(nside, dbNames, sums[idx], 'pixArea', 'area [deg2]',
                   mmarkers, colors_cadb, mfc_cad, adjl, fields_DD, figleg)


plt.show()


"""
for band in 'grizy':
    idx = metricTot['filter']==band
    sel = metricTot[idx]
    figlegb = '{} - {} band'.format(figleg,band)
    sn_plot.plotDDLoop(nside,dbNames,sel,'visitExposureTime','Exposure Time [s]',markers,colors,mfc,adjl,fields_DD,figlegb)
"""
filtercolors = 'cgyrm'
filtermarkers = ['o', '*', 's', 'v', '^']
mfiltc = ['None']*len(filtercolors)

vars = ['visitExposureTime', 'cadence_mean', 'gap_max', 'gap_5']
legends = ['Exposure Time [sec]/night', 'cadence [days]',
           'max gap [days]', 'frac gap > 5 days']

todraw = dict(zip(vars, legends))
for dbName in dbNames:
    for key, vals in todraw.items():
        sn_plot.plotDDCadence(metricTot, dbName, key, vals, adjl, fields_DD)


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
