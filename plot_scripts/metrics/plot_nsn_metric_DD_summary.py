import numpy as np
import sn_plotter_metrics.cadencePlot as sn_plot
import sn_plotter_metrics.nsnPlot as nsn_plot
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
import csv
import pandas as pd


class Summary:
    def __init__(self, dirFile, metricName='NSN',
                 fieldType='DD', fieldNames=['COSMOS'], nside=128, forPlot=pd.DataFrame(), outName=''):
        """
        Class to transform input data and match to DD fields

        Parameters
        ---------------
        dirFile: str
          directory of the files to process
        metricName: str, opt
          name of the metric to consider (default: NSN)
        fieldType: str,opt
          field type to consider (default: DD)
        fieldNames: list(str), opt
          fieldNames to process (default: ['COSMOS'])
        nside: int, opt
          nside healpix parameter (default: 128)
        forPlot: pandas df, opt
          list of cadences to process and associated plot parameters (default: empty df)
        simuVersion: str, opt
          tag for output file name for summary of the results(default: '')

        Returns
        ----------


        """

        #fname = 'Summary_{}_{}.npy'.format(fieldType, simuVersion)

        fields_DD = DDFields()
        # if not os.path.isfile(outName):
        # get pixelArea
        self.pixArea = hp.nside2pixarea(nside, degrees=True)
        x1_colors = [(-2.0, 0.2), (0.0, 0.0)]
        self.corr = dict(zip(x1_colors, ['faint', 'medium']))
        self.data = self.process_loop(dirFile, metricName, fieldType, fieldNames,
                                      nside, forPlot).to_records()

        """
        self.data = Match_DD(fields_DD, df).to_records()
        """

        #np.save(outName, self.data)

        # else:
        #    self.data = np.load(outName, allow_pickle=True)

    def process_loop(self, dirFile, metricName, fieldType, fieldNames, nside, forPlot):
        """
        Method to loop on all the files and process the data

        Parameters
        --------------
        dirFile: str
          directory of the files to process
        metricName: str, opt
          name of the metric to consider (default: NSN)
        fieldType: str,opt
          field type to consider (default: DD)
        nside: int, opt
          nside healpix parameter (default: 128)
        forPlot: pandas df, opt
          list of cadences to process and associated plot parameters (default: empty df)

        Returns
        ----------
        pandas df with the following cols:
         pixRA: RA of the sn pixel location
         pixDec: Dec of the sn pixel location
         healpixID: healpixID of the sn pixel location
         season: season number
         status:  status of the processing
         zlim_faint: redshift limit for faint sn
         nsn_zfaint:  number of sn with z<= zfaint
         nsn_med_zfaint: number of medium sn with z<= zfaint
         zlim_medium: redshift limit for medium sn
         nsn_zmedium: number of sn with z<= zmedium
         nsn_med_zmedium: number of medium sn with z<= zmedium
         cadence: cadence name

        """

        metricTot = None

        df = pd.DataFrame()

        io = -1
        for dbName in forPlot['dbName']:
            io += 1
            dfi = self.process(dirFile, dbName, metricName,
                               fieldType, fieldNames, nside)
            df = pd.concat([df, dfi], sort=False)

        return df

    def process(self, dirFile, dbName, metricName, fieldType, fieldNames, nside):

        restot = pd.DataFrame()
        for fieldName in fieldNames:
            res = self.process_field(
                dirFile, dbName, metricName, fieldType, fieldName, nside)
            restot = pd.concat((restot, res))

        return restot

    def process_field(self, dirFile, dbName, metricName, fieldType, fieldName, nside):
        """
        Single file processing
        This method load the files corresponding to dbName and transform it
        so as to have all infos on one line.


        Parameters
        ----------------
        dirFile: str
         directory where the files are located
        dbName: str
          name of the cadence to processe
        metricName: str
          name of the metric of interest
        fieldType: str
          field type: DD or WFD
        nside: int
          nside for healpix tessallation


        Returns
        -----------
         pandas df with the following cols:
          pixRA: RA of the sn pixel location
          pixDec: Dec of the sn pixel location
          healpixID: healpixID of the sn pixel location
          season: season number
          status:  status of the processing
          zlim_faint: redshift limit for faint sn
          nsn_zfaint:  number of sn with z<= zfaint
          nsn_med_zfaint: number of medium sn with z<= zfaint
          zlim_medium: redshift limit for medium sn
          nsn_zmedium: number of sn with z<= zmedium
          nsn_med_zmedium: number of medium sn with z<= zmedium
          cadence: cadence name

        """

        search_path = '{}/{}/{}_{}/*{}Metric_{}*_nside_{}_*.hdf5'.format(
            dirFile, dbName, metricName,  fieldName, metricName, fieldType, nside)
        print('looking for', search_path)
        vars = ['pixRA', 'pixDec', 'healpixID', 'season', 'status']
        # vars = ['healpixID', 'season']
        fileNames = glob.glob(search_path)
        print(fileNames)
        finaldf = pd.DataFrame()
        if fileNames:
            # plt.plot(metricValues['pixRA'],metricValues['pixDec'],'ko')
            # plt.show()
            metricValues = loopStack(fileNames, 'astropyTable').to_pandas()
            metricValues = metricValues.round({'pixRA': 3, 'pixDec': 3})
            metricValues['cadence'] = dbName
            metricValues['fieldname'] = fieldName
            metricValues['pixArea'] = self.pixArea
            metricValues['filter'] = 'grizy'
            print(metricValues.columns)
            return metricValues
        """
        newdf = {}
            
            for key, vals in self.corr.items():
                idx = np.abs(key[0]-metricValues['x1']) < 1.e-5
                idx &= np.abs(key[1]-metricValues['color']) < 1.e-5
                sel = metricValues[idx]
                sel.loc[:, 'zlim_{}'.format(vals)] = sel['zlim']
                sel.loc[:, 'nsn_z{}'.format(vals)] = sel['nsn']
                sel.loc[:, 'nsn_med_z{}'.format(vals)] = sel['nsn_med']
           
            
            newdf[vals] = sel.drop(
                columns=['x1', 'color', 'zlim', 'nsn', 'nsn_med'])

            finaldf = newdf['faint'].merge(
                newdf['medium'], left_on=vars, right_on=vars)

        finaldf['cadence'] = dbName
        
        return finaldf
        """


parser = OptionParser(
    description='Display (NSN,zlim) metric results for DD fields')
parser.add_option("--dirFile", type="str",
                  default='/sps/lsst/users/gris/MetricOutput',
                  help="file directory [%default]")
parser.add_option("--nside", type="int", default=64,
                  help="nside for healpixels [%default]")
parser.add_option("--fieldType", type="str", default='DD',
                  help="field type - DD, WFD, Fake [%default]")
parser.add_option("--dbList", type="str", default='plot_scripts/cadenceCustomize_fbs14.csv',
                  help="list of cadences to display[%default]")
parser.add_option("--snType", type="str", default='faint',
                  help="SN type: faint or medium[%default]")
parser.add_option("--outName", type="str", default='Summary_DD_fbs14.npy',
                  help="output name for the summary[%default]")
parser.add_option("--fieldNames", type="str", default='COSMOS,CDFS,XMM-LSS,ELAIS,ADFS1,ADFS2',
                  help="fields to process [%default]")


opts, args = parser.parse_args()

# Load parameters
dirFile = opts.dirFile
nside = opts.nside
fieldType = opts.fieldType
metricName = 'NSN'
snType = opts.snType
outName = opts.outName
fieldNames = opts.fieldNames.split(',')

# Loading input file with the list of cadences to take into account and siaplay features
filename = opts.dbList

# forPlot = pd.read_csv(filename).to_records()
forPlot = pd.read_csv(filename, comment='#')

print(forPlot)


# get DD fields
# fields_DD = getFields(5.)
"""
fields_DD = DDFields()

lengths = [len(val) for val in forPlot['dbName']]
adjl = np.max(lengths)


metricTot = None
metricTot_med = None
"""


# Summary: to reproduce the plots faster

metricTot = Summary(dirFile, 'NSN',
                    'DD', fieldNames, nside, forPlot, outName).data

print(metricTot.dtype)
print('oo', np.unique(metricTot[['cadence', 'fieldname']]), type(metricTot))

#nsn_plot.plot_DDArea(metricTot, forPlot, sntype='faint')

#nsn_plot.plot_DDSummary(metricTot, forPlot, sntype=snType)
#nsn_plot.plot_DD_Moll(metricTot, 'ddf_dither0.00_v1.7_10yrs', 1, 128)
nsn_plot.plot_DD_Moll(metricTot, 'descddf_v1.5_10yrs', 1, 128)
plt.show()

print(test)
fontsize = 15
fields_DD = DDFields()
# print(metricTot[['cadence','filter']])

# grab median values
"""
df = pd.DataFrame(np.copy(metricTot)).groupby(
    ['healpixID','fieldnum','filter','cadence']).median().reset_index()

# print(df)
metricTot = df.to_records(index=False)
idx = metricTot['filter']=='all'
sel = metricTot[idx]
"""

figleg = 'nside = {}'.format(nside)
# sn_plot.plotDDLoop(nside, dbNames, metricTot, 'zlim_faint',
#                   '$z_{lim}^{faint}$', markers, colors, mfc, adjl, fields_DD, figleg)
# sn_plot.plotDDLoop(nside,dbNames,sel,'cadence_mean','cadence [days]',markers,colors,mfc,adjl,fields_DD,figleg)

# fig,ax = plt.subplots()


# print(metricTot.dtype,type(metricTot))


# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'zlim_faint','zlim_medium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'nsn_med_zfaint','nsn_med_zmedium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDFit(metricTot,'nsn_med_zmedium','nsn_zmedium')
# sn_plot.plotDDFit(metricTot,'zlim_faint','zlim_medium')

# sn_plot.plotDDLoopCorrel(nside,dbNames,metricTot,'nsn_med_zfaint','nsn_med_zmedium','$z_{lim}^{faint}$','$z_{lim}^{med}$',markers,colors,mfc,adjl,fields_DD,figleg)

# sn_plot.plotDDFit(metricTot,'nsn_med_zmedium','nsn_med_zfaint','zlim_medium','zlim_faint')


df = pd.DataFrame(metricTot)
dbNames = np.unique(df['cadence'])
"""
print(df.columns)
#sums = df.groupby(['cadence','season'])['pixArea'].sum().reset_index()
sums = df.groupby(['fieldname', 'cadence', 'season'])['pixArea'].sum().reset_index()

idx = sums['pixArea'] > 1.
sn_plot.plotDDLoop(nside, dbNames, sums[idx], 'pixArea', 'area [deg2]',
                   mmarkers, colors_cadb, mfc_cad, adjl, fields_DD, figleg)

"""
# plt.show()


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

"""
vars = ['visitExposureTime', 'cadence_mean', 'gap_max', 'gap_5']
legends = ['Exposure Time [sec]/night', 'cadence [days]',
           'max gap [days]', 'frac gap > 5 days']
"""
vars = ['N_total', 'cadence', 'gap_max', 'gap_med']
legends = ['Number of visits', 'cadence [days]',
           'max gap [days]', 'med gap [day]']

print(df.columns)


todraw = dict(zip(vars, legends))
for dbName in dbNames:
    for key, vals in todraw.items():
        sn_plot.plotDDCadence_new(df, dbName, key, vals)


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
