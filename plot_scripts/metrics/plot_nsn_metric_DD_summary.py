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
import multiprocessing
import scipy.stats


def dumpcsv_medcad(metricTot):

    metricTot = pd.DataFrame(metricTot)
    idx = metricTot['nsn_med_faint'] >= 0
    sel = metricTot[idx]
    r = pd.DataFrame()

    for fieldName in np.unique(sel['fieldname']):
        ij = sel['fieldname'] == fieldName
        selb = sel[ij]
        for cad in range(1, 9, 1):
            r = pd.concat(
                (r, fill(selb, tagprod=fieldName,
                         cadence=dict(zip('grizy', [cad]*5)))))

    print(r)
    r.to_csv('config_DD_obs_medcad.csv', index=False)


def dumpcsv_pixels(metricTot, x1=-2.0, color=0.2, ebvofMW=0.0, snrmin=1.0, error_model=1, errmodrel=0.1, bluecutoff=380., redcutoff=800., simulator='sn_fast', fitter='sn_fast', Nvisits=dict(zip('grizy', [1, 1, 2, 2, 2]))):

    metricTot = pd.DataFrame(metricTot)
    idx = metricTot['nsn_med_faint'] >= 0
    ro = metricTot[idx]
    ro['healpixID'] = ro['healpixID'].astype(int)
    ro['season'] = ro['season'].astype(int)
    ro['tagprod'] = ro['healpixID'].astype(str) + '_' + \
        ro['season'].astype(str)
    ro['x1'] = x1
    ro['color'] = color
    ro['ebvofMW'] = ebvofMW
    ro['snrmin'] = snrmin
    ro['error_model'] = error_model
    ro['errmodrel'] = errmodrel
    ro['bluecutoff'] = bluecutoff
    ro['redcutoff'] = redcutoff
    ro['simulator'] = simulator
    ro['fitter'] = fitter
    for b in Nvisits.keys():
        ro['N{}'.format(b)] = Nvisits[b]
        cadname = 'cadence_{}'.format(b)
        ro[cadname] = ro['cadence']
        #ro[cadname] = ro[cadname].astype(int)
    for b in Nvisits.keys():
        nn = 'm5_{}'.format(b)
        ro = ro.rename(columns={'m5_med_{}'.format(b): nn})
        ro = ro.round({nn: 2})

    for b in 'grizy':
        ro = ro.round({'cadence_{}'.format(b): 2})

    print(ro.columns)
    tocsv_vars = ['tagprod', 'x1', 'color', 'ebvofMW', 'snrmin', 'error_model',
                  'errmodrel', 'bluecutoff', 'redcutoff', 'simulator', 'fitter',
                  'Ng', 'Nr', 'Ni', 'Nz', 'Ny', 'm5_g', 'm5_r', 'm5_i', 'm5_z', 'm5_y',
                  'cadence_g', 'cadence_r', 'cadence_i', 'cadence_z', 'cadence_y',
                  'season', 'healpixID', 'season_length']

    max_rows = 500
    dataframes = []
    df = ro[tocsv_vars]
    while len(df) > max_rows:
        top = df[:max_rows]
        dataframes.append(top)
        df = df[max_rows:]
    else:
        dataframes.append(df)

    for i, dd in enumerate(dataframes):
        dd.to_csv('config_DD_obs_pixels_{}.csv'.format(i), index=False)


def fill(selb, tagprod='', x1=-2.0, color=0.2, ebvofMW=0.0, snrmin=1.0, error_model=1, errmodrel=0.1, bluecutoff=380., redcutoff=800., simulator='sn_fast', fitter='sn_fast', Nvisits=dict(zip('grizy', [1, 1, 2, 2, 2])), cadence=dict(zip('grizy', [1, 1, 1, 1, 1]))):

    fmed = []
    r = pd.DataFrame()
    for b in 'grizy':
        fmed.append('m5_med_{}'.format(b))

    ro = selb.groupby(['season'])[fmed].median().reset_index()

    ro['tagprod'] = tagprod
    ro['x1'] = x1
    ro['color'] = color
    ro['ebvofMW'] = ebvofMW
    ro['snrmin'] = snrmin
    ro['error_model'] = error_model
    ro['errmodrel'] = errmodrel
    ro['bluecutoff'] = bluecutoff
    ro['redcutoff'] = redcutoff
    ro['simulator'] = simulator
    ro['fitter'] = fitter
    for b in Nvisits.keys():
        ro['N{}'.format(b)] = Nvisits[b]
        ro['cadence_{}'.format(b)] = cadence[b]

    for b in Nvisits.keys():
        nn = 'm5_{}'.format(b)
        ro = ro.rename(columns={'m5_med_{}'.format(b): nn})
        ro = ro.round({nn: 2})

    ro['season'] = ro['season'].astype(int)
    ro['tagprod'] = ro['tagprod'] + '_' + \
        ro['season'].astype(str)+'_'+ro['cadence_z'].astype(str)

    return ro


def plotAllBinned(metricTot, xp='cadence', yp='nsn_med_faint', legx='cadence [day]', legy='$N_{SN} ^ {z < z_{complete}}$'):

    metricTot = pd.DataFrame(metricTot)
    idx = metricTot['nsn_med_faint'] >= 0
    sel = metricTot[idx]
    # plt.plot(sel['cadence'], sel['nsn_med_faint'], 'ko')
    print(sel[['cadence', 'nsn_med_faint']])
    fig, ax = plt.subplots(ncols=2, nrows=6, sharex=True)
    fig.subplots_adjust(hspace=0)
    zlim_theo = pd.read_csv('config_z_0.csv', comment="#")
    zlim_theo['fieldName'] = zlim_theo['tagprod'].str.split(
        '_').str.get(0)
    ff = ['COSMOS', 'XMM-LSS', 'ELAIS', 'CDFS', 'ADFS1', 'ADFS2']
    ipos = [0, 1, 2, 3, 4, 5, 6]
    iposi = dict(zip(ff, ipos))
    for fieldName in np.unique(sel['fieldname']):
        # fig, ax = plt.subplots()
        ipp = iposi[fieldName]
        ij = sel['fieldname'] == fieldName
        selb = sel[ij]
        plotBinned(ax[ipp, 0], selb, xp=xp, yp=yp, label=fieldName)
        plotBinned(ax[ipp, 1], selb, xp=xp,
                   yp='nsn_med_faint', label=fieldName)
        # ax[ipp,0].plot(selb['cadence'],selb['zlim_faint'],'k.')
        idx = zlim_theo['fieldName'] == fieldName
        selz = zlim_theo[idx]
        mink = selz.groupby(['cadence_z']).min().reset_index()
        maxk = selz.groupby(['cadence_z']).max().reset_index()
        # mink = mink.sort_values(by=['cadence_z'],ascending=False)
        print(mink)
        print(maxk)
        # maxk = pd.concat((maxk,mink))
        print(maxk)
        # ax.plot(selz['cadence_z'], selz['zlim'], 'ko')
        ax[ipp, 0].fill_between(
            mink['cadence_z'], mink['zlim'], maxk['zlim'], color='yellow')
        ax[ipp, 0].grid()
        # break

        # ax.set_xlabel(legx, fontweight='bold')
        # ax.set_ylabel(legy, fontweight='bold')
    # ax.legend()

    plt.show()


def plotBinned(ax, metricTot, xp='cadence', yp='nsn_med_faint', label=''):

    x = metricTot[xp]
    y = metricTot[yp]

    means_result = scipy.stats.binned_statistic(
        x, [y, y**2], bins=8, range=(0.5, 8.5), statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

    ax.errorbar(x=bin_centers, y=means, yerr=standard_deviations,
                marker='.', label=label)


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

        # fname = 'Summary_{}_{}.npy'.format(fieldType, simuVersion)

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

        # np.save(outName, self.data)

        # else:
        #    self.data = np.load(outName, allow_pickle=True)

    def process_loop(self, dirFile, metricName, fieldType, fieldNames, nside, forPlot, nproc=8):
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

        nz = len(forPlot['dbName'])
        t = np.linspace(0, nz, nproc+1, dtype='int')
        # print('multi', nz, t)
        result_queue = multiprocessing.Queue()
        """
        self.process(dirFile, dbName, metricName,
                               fieldType, fieldNames, nside)
        """
        procs = [multiprocessing.Process(name='Subprocess-'+str(j), target=self.process,
                                         args=(dirFile, forPlot['dbName'][t[j]:t[j+1]], metricName,
                                               fieldType, fieldNames, nside, j, result_queue))
                 for j in range(nproc)]

        for p in procs:
            p.start()

        resultdict = {}
        # get the results in a dict

        for i in range(nproc):
            resultdict.update(result_queue.get())

        for p in multiprocessing.active_children():
            p.join()

        restot = pd.DataFrame()

        # gather the results
        for key, vals in resultdict.items():
            restot = pd.concat((restot, vals), sort=False)

        return restot
        """
        metricTot = None

        df = pd.DataFrame()

        for dbName in forPlot['dbName']:
            dfi = self.process(dirFile, dbName, metricName,
                               fieldType, fieldNames, nside)
            df = pd.concat([df, dfi], sort=False)

        return df
        """

    def process(self, dirFile, dbNames, metricName, fieldType, fieldNames, nside, j=0, output_q=None):

        restot = pd.DataFrame()
        for dbName in dbNames:
            for fieldName in fieldNames:
                res = self.process_field(
                    dirFile, dbName, metricName, fieldType, fieldName, nside)
                restot = pd.concat((restot, res))

        if output_q is not None:
            return output_q.put({j: restot})
        else:
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
            metricValues['dbName'] = dbName
            metricValues['fieldname'] = fieldName
            metricValues['pixArea'] = self.pixArea
            metricValues['filter'] = 'grizy'
            dbName_split = dbName.split('_')
            n = len(dbName_split)
            metricValues['dbName_plot'] = '_'.join(dbName_split[0:n-2])

            print(metricValues.columns)
            return metricValues
        """
        newdf = {}

            for key, vals in self.corr.items():
                idx = np.abs(key[0]-metricValues['x1']) < 1.e-5
                idx &= np.abs(key[1]-metricValues['color']) < 1.e-5
                sel = metricValues[idx]
                sel.loc[:, 'zlim_{}'.format(vals)] = sel['zlim']
                sel.loc[:, 'n_z{}'.format(vals)] = sel['nsn']
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
# plotAllBinned(metricTot)
dumpcsv_pixels(metricTot)
#plotAllBinned(metricTot, yp='zlim_faint', legy='$z_{complete}^{0.95}$')

plt.show()
print(test)
"""
print('oo', np.unique(
    metricTot[['dbName', 'fieldname', 'zlim_faint']]), type(metricTot))
"""
# fieldNames = ['COSMOS', 'CDFS', 'XMM-LSS', 'ELAIS', 'ADFS1', 'ADFS2']
# fieldNames = ['COSMOS']
# nsn_plot.plot_DDArea(metricTot, forPlot, sntype='faint')

nsn_plot.plot_DDSummary(metricTot, forPlot, sntype=snType,
                        fieldNames=fieldNames, nside=nside)
# nsn_plot.plot_DD_Moll(metricTot, 'ddf_dither0.00_v1.7_10yrs', 1, 128)
# nsn_plot.plot_DD_Moll(metricTot, 'descddf_v1.5_10yrs', 1, 128)
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
# sums = df.groupby(['cadence','season'])['pixArea'].sum().reset_index()
sums = df.groupby(['fieldname', 'cadence', 'season'])[
                  'pixArea'].sum().reset_index()

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
    sn_plot.plotDDLoop(nside,dbNames,sel,'visitExposureTime',
                       'Exposure Time [s]',markers,colors,mfc,adjl,fields_DD,figlegb)
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
    plt.plot(selb['fieldnum'],selb['season_length'],marker='.',
             lineStyle='None',label='season {}'.format(season))

plt.legend()
plt.xticks(fields_DD['fieldnum'], fields_DD['fieldname'], fontsize=fontsize)
"""
plt.show()


print(metricTot.dtype)
