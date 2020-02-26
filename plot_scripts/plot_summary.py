import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
import glob
from sn_tools.sn_io import loopStack
import csv
import pandas as pd

class PlotSummary:
    def __init__(self, x1=-2.0, color=0.2, namesRef=['SNCosmo'],
                 SNR = dict(zip('griz', [30., 40., 30., 20.])),
                 dt_range = [0.5, 30.],
                 mag_range = [21., 25.5]):

        refDir = 'reference_files'
        x1 = x1
        color = color
        self.namesRef = namesRef
        Li_files = []
        mag_to_flux_files = []

        #SNR = dict(zip('griz', [25., 25., 30., 35.]))  # SNR for DD
        #dt_range = [0.5, 25.]  # DD dt range
        #mag_range = [23., 27.5]  # DD mag range
        
        #SNR = dict(zip('griz', [30., 40., 30., 20.]))  # SNR for WFD
        #dt_range = [0.5, 30.]  # WFD dt range
        #mag_range = [21., 25.5]  # WFD mag range
        self.SNR = SNR
        self.dt_range = dt_range
        self.mag_range = mag_range

        for name in namesRef:
            self.Li_files = ['{}/Li_{}_{}_{}.npy'.format(refDir,name,x1,color)]
            self.mag_to_flux_files = ['{}/Mag_to_Flux_{}.npy'.format(refDir,name)]


    def loadFile(self,dirFile, dbName, fieldtype,metricName, nside=64):
    
        fileName  ='{}/{}/{}/{}_{}*.hdf5'.format(dirFile,dbName,metricName,dbName,metricName)
        """
        if fieldtype != '':
            fileName  ='{}/{}/{}_{}_{}_nside_{}_*.hdf5'.format(dirFile,dbName,dbName,metricName,fieldtype,nside)
        
        if band != '':
            fileName  ='{}/{}/{}_{}_nside_{}_{}*.hdf5'.format(dirFile,dbName,dbName,metricName,nside,band)
        """
        print('looking for',fileName)
        
        fileNames = glob.glob(fileName)

        metricValues = loopStack(fileNames, 'astropyTable')
        #print(fileNames,len(fileNames))
        """
        metricValues = loopStack(fileNames,'astropyTable')
        """
        #metricValues = np.load(fileNames[0])
        """
        if not os.path.isfile(fileName):
            return None
        return np.load(fileName)
        """
        return metricValues    

    def loadFile_old(self,dirFile, dbName, fieldtype, band, metricName, nside=64):
        if fieldtype != '':
            fileName  ='{}/{}/{}_{}_{}_nside_{}_*.hdf5'.format(dirFile,dbName,dbName,metricName,fieldtype,nside)
        if band != '':
            fileName  ='{}/{}/{}_{}_nside_{}_{}*.hdf5'.format(dirFile,dbName,dbName,metricName,nside,band)

        print('looking for',fileName)
        fileNames = glob.glob(fileName)
        print(fileNames,len(fileNames))
        metricValues = loopStack(fileNames,'astropyTable')
        """
        if not os.path.isfile(fileName):
            return None
        return np.load(fileName)
        """
        return metricValues


    def getMetricValues(self, dirFile, dbName, band,m5_str='m5_median'):
        
        print('loading here',dirFile, dbName, band)
        metricValuesCad = self.loadFile(dirFile, dbName, 'WFD','Cadence')
        metricValuesSNR =  self.loadFile(dirFile, dbName,'','SNR{}'.format(band))
        print('hello',metricValuesSNR.dtype)
        #idx = metricValuesSNR['band'] == str.encode(band)
        #metricValuesSNR = np.copy(metricValuesSNR[idx])
        #print('yes',metricValuesSNR['band'])

        if metricValuesCad is None or metricValuesSNR is None:
            return None
        resCadence = sn_plot.plotCadence(band,self.Li_files,self.mag_to_flux_files,
                            self.SNR[band],
                            metricValuesCad,
                            self.namesRef,
                            mag_range=self.mag_range, 
                            dt_range=self.dt_range,
                            dbName=dbName,      
                                         display=False,m5_str=m5_str)
        

        r = []
        for name in self.namesRef:
            med_zlim = np.median(resCadence['zlim_{}'.format(name)])
            med_detect = np.median(metricValuesSNR['frac_obs_{}'.format(name)])
            r.append((band,dbName, med_zlim, med_detect))

        return r

def isInZone(x1,x2,y1,y2,x,y):
    
    if x<x1 or x>x2 or y<y1 or y>y2:
        return True
    return False

def plotBandSimple(ax,band,medVals,shiftx = -0.003, shifty = -0.004):

    ida = medVals['band'] == band
    medValues = medVals[ida]

    tot_label=[]
   
    print(medValues['dbName'])
    diffx = np.max(medValues['zlim'])-np.min(medValues['zlim'])
    diffy = np.max(medValues['detect_rate'])-np.min(medValues['detect_rate'])

    
    shifty = -0.03*diffy
    shiftx = -0.01*diffx
 
    if band == 'r':
        shifty = -0.03*diffy
        shiftx = -0.01*diffx
    
    if band == 'z':
        shifty = -0.04*diffy
        shiftx = -0.01*diffx

    for dbName in forPlot['dbName']:
        idx = medValues['dbName'] == dbName
        sel = medValues[idx]
        if len(sel) ==0:
            continue

        print(dbName,band,sel['zlim'][0],sel['detect_rate'][0])
        idxp = np.where(forPlot['dbName'] == dbName)
        color = forPlot[idxp]['color'][0]
        marker = forPlot[idxp]['marker'][0]
        name = forPlot[idxp]['newName'][0]
        namefig = forPlot[idxp]['newName'][0]
        ax.plot(sel['zlim'],sel['detect_rate'],color=color,marker=marker)
        if band=='r' and namefig in ['opsim_large_rolling_3yr']:
            ax.text(sel['zlim']-shiftx,sel['detect_rate'],namefig,ha='left',va='center',color=color)
            continue
        if band=='r' and namefig in ['opsim_single_visits','opsim_single_exp']:
            ax.text(sel['zlim']-shiftx,sel['detect_rate']+shifty,namefig,ha='left',va='center',color=color)
            continue
        if band=='z' and namefig in ['opsim_single_exp','opsim_single_visits','opsim_large_single_visits']:
            ax.text(sel['zlim']-shiftx,sel['detect_rate']-shifty,namefig,ha='center',va='center',color=color)
            continue
        if band=='z' and namefig in ['opsim_baseline']:
            ax.text(sel['zlim']-shiftx,sel['detect_rate'],namefig,ha='left',va='center',color=color)
            continue
        if band=='z' and namefig in ['opsim_extra_ddf']:
            ax.text(sel['zlim']+shiftx,sel['detect_rate'],namefig,ha='right',va='center',color=color)
            continue
        ax.text(sel['zlim']+shiftx,sel['detect_rate']+shifty,namefig,ha='center',va='center',color=color)
        

    ax.set_xlabel('z$_{lim}$')
    ax.set_ylabel('SNR rate')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([xmin-0.01,xmax+0.01])
    plt.grid(linestyle='--')

def plotBand(ax,band,medVals,label='True', shiftx = -0.003, shifty = -0.004,exclude=False,x1=0, x2=0, y1=0, y2=0, zoom=False):

    ida = medVals['band'] == band
    medValues = medVals[ida]

    tot_label=[]
   
    print(medValues['dbName'])
    diffx = np.max(medValues['zlim'])-np.min(medValues['zlim'])
    diffy = np.max(medValues['detect_rate'])-np.min(medValues['detect_rate'])

    shifty = -0.005*diffy
    shiftx = -0.01*diffx

    if zoom:
       diffx = x2-x1
       diffy = y2-y1
    
       print(diffx,diffy)
       shifty = 0.07*diffy
       shiftx = -0.005*diffx

    tagig = {}
    for dbName in forPlot['dbName']:
        idx = medValues['dbName'] == dbName
        sel = medValues[idx]
        if len(sel) ==0:
            continue

        idxp = np.where(forPlot['dbName'] == dbName)
        color = forPlot[idxp]['color'][0]
        marker = forPlot[idxp]['marker'][0]
        name = forPlot[idxp]['newName'][0]
        namefig = forPlot[idxp]['newName'][0]
        namepl = forPlot[idxp]['Namepl'][0]
        groupid = forPlot[idxp]['group'][0]

        if zoom:
            if isInZone(x1,x2,y1,y2,sel['zlim'],sel['detect_rate']):
                continue
            
        if namepl == 'none':
            namepl = ''
        
        if groupid == 'none':
            ax.plot(sel['zlim'],sel['detect_rate'],color=color,marker=marker)
            if (not zoom and isInZone(x1,x2,y1,y2,sel['zlim'],sel['detect_rate'])) or (zoom and not isInZone(x1,x2,y1,y2,sel['zlim'],sel['detect_rate'])):
                ax.text(sel['zlim']+shiftx,sel['detect_rate']+shifty,namefig,ha='center',va='center')

        else:
            if label and groupid not in tagig.keys():
                tot_label.append(ax.errorbar(sel['zlim'],sel['detect_rate'],color=color,marker=marker,label=groupid,linestyle='None',markerfacecolor='None'))
                tagig[groupid] = 0
            else:
                ax.plot(sel['zlim'],sel['detect_rate'],color=color,marker=marker,markerfacecolor='None')
            if (not zoom and isInZone(x1,x2,y1,y2,sel['zlim'],sel['detect_rate'])) or (zoom and not isInZone(x1,x2,y1,y2,sel['zlim'],sel['detect_rate'])):
                if namepl in ['large_single_visit','single_exp']:
                    ax.text(sel['zlim'],sel['detect_rate']-shifty,namepl,ha='center',va='bottom')
                else:
                    if namepl=='baseline' and color=='b':
                        ax.text(sel['zlim']+shiftx,sel['detect_rate'],namepl,ha='right',va='center') 
                    else:
                        ax.text(sel['zlim'],sel['detect_rate']+shifty,namepl,ha='center',va='top')
    if label:
        labs = [l.get_label() for l in tot_label]
        #ax.legend(tot_label, labs, ncol=3,loc='best',prop={'size':12},frameon=True)
        ax.legend(tot_label, labs, ncol=1,loc='center left', bbox_to_anchor=(1,0.5),prop={'size':12},frameon=True)
        ax.set_xlabel('z$_{lim}$')
        ax.set_ylabel('Detection rate')
        xmin, xmax = ax.get_xlim()
        ax.set_xlim([xmin-0.001,xmax+0.001])
    plt.grid(linestyle='--')
    
    

dirFile = '/sps/lsst/users/gris/MetricOutput'
#dirFile = '/sps/lsst/users/gris/MetricSummary'


plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['figure.figsize'] = (10, 7)

#forPlot = np.loadtxt('plot_scripts/cadenceCustomizefb13.txt',
#                     dtype={'names': ('dbName', 'newName', 'group','Namepl','color','marker'),'formats': ('U39', 'U39','U12','U18','U7','U1')})

simuVersion = 'fbs14'
filename = 'plot_scripts/cadenceCustomize_{}.csv'.format(simuVersion)

# forPlot = pd.read_csv(filename).to_records()
forPlot = pd.read_csv(filename)

print(forPlot)

plotSum = PlotSummary()

bands = 'rz'
if not os.path.isfile('Summary.npy'):
    medList = []
    for band in bands:
        for dbName in forPlot['dbName']:
            print('processing',dbName)
            res = plotSum.getMetricValues(dirFile, dbName,band,m5_str='m5_median')
            if res is not None:
                medList += res
    
    medValues = np.array(medList, dtype=[('band','U1'),('dbName','U39'),('zlim','f8'),('detect_rate','f8')])

    np.save('Summary.npy',np.copy(medValues))

print(forPlot)
bands = 'rz'
medValues = np.load('Summary.npy')

zoom={}

zoom['g']=[0.2835, 0.288, 0.42, 0.426]
zoom['r']=[0.332,0.334,0.4515,0.457]
zoom['i']=[0.305,0.31,0.457,0.463]
zoom['z']=[0.269, 0.285, 0.42, 0.438]

zoomval=dict(zip('griz',[4.,5.,4.,4.]))
window={}

window['z']= {}
window['z']['ax'] = [0.2,0.4,0.39,0.50]
window['z']['ax2'] = [0.2,0.4,0.52,0.56]

for band in bands:
    x1, x2, y1, y2 = zoom[band][0], zoom[band][1],zoom[band][2],zoom[band][3]# specify the limits
    fig, ax = plt.subplots()
    """
    fig,(ax,ax2) = plt.subplots(2, 1, sharey=True)
    
    ax.set_xlim([window[band]['ax'][0],window[band]['ax'][1]])
    ax.set_ylim([window[band]['ax'][2],window[band]['ax'][3]])
    ax2.set_xlim([window[band]['ax2'][0],window[band]['ax2'][1]])
    ax2.set_ylim([window[band]['ax2'][2],window[band]['ax'][3]])
    """
    fig.suptitle('{} band'.format(band))
    fig.subplots_adjust(right=0.9)
    
    #plotBand(ax,band,medValues,x1=x1, x2=x2, y1=y1, y2=y2)
    #plotBand(ax2,band,medValues,x1=x1, x2=x2, y1=y1, y2=y2)
    plotBandSimple(ax,band,medValues)
    if band == 'y':
        axins = zoomed_inset_axes(ax, zoomval[band], loc=4) # zoom-factor: 2.5, location: upper-left
        
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        mark_inset(ax, axins, loc1=1,loc2=3, fc="none", ec="0.5")
    
        plotBand(axins,band,medValues,label=False,shiftx=-0.001,shifty=-0.001,exclude=True,x1=x1, x2=x2, y1=y1, y2=y2,zoom=True)
        # Turn off tick labels
        axins.set_yticklabels([])
        axins.set_xticklabels([])
    
plt.show()

#ref_opsim = np.genfromtxt('opsim_runs.csv',delimiter=',')

print('oo',medValues.dtype)
idx = medValues['band'] == 'z'
sel = medValues[idx]
print(sel['dbName'])

rows = []
rows.append(['','name','group','zlim_z','snr_rate'])
with open('opsim_runs.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['num'],row['name'],row['group']) 
        name = row['name'].split('.db')[0]
        idb = sel['dbName'] == name
        selb = sel[idb]
        print(row['num'],row['name'],row['group'],np.round(selb['zlim'][0],4),np.round(selb['detect_rate'][0],4))
        zlim = np.round(selb['zlim'][0],4)
        detect_rate = np.round(selb['detect_rate'][0],4)
        rowb = [row['num'],row['name'],row['group'],str(zlim),str(detect_rate)]
        rows.append(rowb)

with open('person.csv', 'w') as csvFile:
    writer = csv.writer(csvFile,delimiter=',')
    for myrow in rows:
        print(myrow)
        writer.writerow(myrow)

#print(ref_opsim)
