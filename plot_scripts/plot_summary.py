import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset
import glob
from sn_tools.sn_io import loopStack
import csv
import pandas as pd
from optparse import OptionParser

class Summary:
    def __init__(self, x1=-2.0, color=0.2, namesRef=['SNCosmo'],
                 SNR = dict(zip('griz', [30., 40., 30., 20.])),
                 dt_range = [0.5, 30.],
                 mag_range = [21., 25.5]):

        """
        class to load metric data and estimate medians
    
        Parameters
        ----------------
        x1: float, opt
         SN x1 (default: -2.0)
        color: float, opt
         SN color (default: 0.2)
        namesRef: list(str)
         list of reference names for reference LC (default: ['SNCosmo'])
        SNR: dict, opt
         SNR cut per band (default: dict(zip('griz', [30., 40., 30., 20.])))
       dt_range: list(float), opt
        cadence range to consider (default: [0.5, 30.])
       mag_range: list(float), opt
        m5 range to consider (default: [21.0, 25.5])

        """

        refDir = 'reference_files'
        x1 = x1
        color = color
        self.namesRef = namesRef
        Li_files = []
        mag_to_flux_files = []

        self.SNR = SNR
        self.dt_range = dt_range
        self.mag_range = mag_range

        # load reference files
        for name in namesRef:
            self.Li_files = ['{}/Li_{}_{}_{}.npy'.format(refDir,name,x1,color)]
            self.mag_to_flux_files = ['{}/Mag_to_Flux_{}.npy'.format(refDir,name)]


    def loadFile(self,dirFile, dbName, fieldtype,metricName, nside=64):
        """
        Method to load the files to be processed

        Parameters
        ---------------
        dirFile: str
         location dir of the file
        dbName: str 
          cadence name 
        fieldtype: str
          fieldtype (DD or WFD)
        metricName:str
          metric name
        nside: int, opt
          nside value for healpix (default: 64)

        Returns
        -----------
        numpy array with the metric results

        """
        
        fileName  ='{}/{}/{}_{}*.npy'.format(dirFile,dbName,dbName,metricName)
        print('looking for',fileName)
        
        fileNames = glob.glob(fileName)

        print('found',fileNames)
        metricValues = np.load(fileNames[0],allow_pickle=True)

        print('iii',metricValues)
        return metricValues 


    def getMetricValues(self, dirFile, dbName, bands,m5_str='m5_median'):
        """
        Method to get medians of the metric values
        
        Parameters
        ----------------
        dirFile: str
         location dir of the file
        dbName: str 
          cadence name 
        bands: str
          filters to consider
        m5_str: str, opt
          m5 to consider (default: m5_median)

        Returns
        ----------

        """

        print('loading here',dirFile, dbName, bands)
        metricValuesCad = self.loadFile(dirFile, dbName, 'WFD','Cadence')
        for band in bands:
            metricValuesSNR =  self.loadFile(dirFile, dbName,'','SNR{}'.format(band))

        #metricValuesSNR =  self.loadFile(dirFile, dbName,'WFD','ObsRate')
        if metricValuesCad is None or metricValuesSNR is None:
            return None
        
        r = []
        for band in bands:
            # This is to estimate the redshift limit per band from the Cadence metric
            resCadence = sn_plot.plotCadence(band,self.Li_files,self.mag_to_flux_files,
                            self.SNR[band],
                            metricValuesCad,
                            self.namesRef,
                            mag_range=self.mag_range, 
                            dt_range=self.dt_range,
                            dbName=dbName,      
                                         display=False,m5_str=m5_str)


            for name in self.namesRef:
                med_zlim = np.median(resCadence['zlim_{}'.format(name)])
                med_detect = np.median(metricValuesSNR['frac_obs_{}'.format(name)])
                r.append((band,dbName, med_zlim, med_detect))

        return r

def isInZone(x1,x2,y1,y2,x,y):
    
    if x<x1 or x>x2 or y<y1 or y>y2:
        return True
    return False

def plotBand(band,medVals,shiftx = -0.003, shifty = -0.004):
    """
    Method to plot final results
    

    """


    fig, ax = plt.subplots()
    ida = medVals['band'] == band
    medValues = medVals[ida]

    tot_label=[]
   
    print(medValues['dbName'])
    for dbName in forPlot['dbName']:
        idx = medValues['dbName'] == dbName
        sel = medValues[idx]
        if len(sel) ==0:
            continue

        print(dbName,band,sel['zlim'][0],sel['detect_rate'][0])
 
        idxp =forPlot['dbName'] == dbName
        color = forPlot[idxp]['color'].values[0]
        marker = forPlot[idxp]['marker'].values[0]
        name = forPlot[idxp]['newName'].values[0]
        namefig = forPlot[idxp]['newName'].values[0]

        ax.plot(sel['zlim'],sel['detect_rate'],color=color,marker=marker)
    

    ax.set_xlabel('z$_{lim}$')
    ax.set_ylabel('SNR rate')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([xmin-0.01,xmax+0.01])
    plt.grid(linestyle='--')

parser = OptionParser(description='Display summary plot')
parser.add_option("--dirFile", type="str", default='/sps/lsst/users/gris/MetricSummary', help="metric file directory [%default]")
parser.add_option("--configFile", type="str", default='plot_scripts/cadenceCustomize_fbs14.csv', help="config file (for plots) directory [%default]")

opts, args = parser.parse_args()

dirFile = opts.dirFile
filename = opts.configFile

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['figure.figsize'] = (10, 7)

forPlot = pd.read_csv(filename,comment='#')

print(forPlot)

Sum = Summary()

bands = 'r'
if not os.path.isfile('Summary.npy'):
    medList = []
    for dbName in forPlot['dbName']:
        print('processing',dbName)
        res = Sum.getMetricValues(dirFile, dbName,bands,m5_str='m5_median')
        if res is not None:
            medList += res
    
    medValues = np.array(medList, dtype=[('band','U1'),('dbName','U39'),('zlim','f8'),('detect_rate','f8')])

    np.save('Summary.npy',np.copy(medValues))

print(forPlot)
bands = 'r'
medValues = np.load('Summary.npy')

# now plot
for band in bands:
    plotBand(band,medValues)
   
    
plt.show()

#This is to save results in a csv file

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
