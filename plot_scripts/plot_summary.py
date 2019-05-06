import numpy as np
import sn_plotters.sn_cadencePlotters as sn_plot
import matplotlib.pylab as plt

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

    def loadFile(self,dirFile, dbName, band, metricName):
        fileName  ='{}/{}_{}_{}.npy'.format(dirFile,dbName,metricName,band)
        return np.load(fileName)

    def getMetricValues(self, dirFile, dbName, band):
        
        metricValuesCad = self.loadFile(dirFile, dbName, band, 'CadenceMetric')
        resCadence = sn_plot.plotCadence(band,self.Li_files,self.mag_to_flux_files,
                            self.SNR[band],
                            metricValuesCad,
                            self.namesRef,
                                         mag_range=self.mag_range, dt_range=self.dt_range, display=False)
        metricValuesSNR =  self.loadFile(dirFile, dbName, band, 'SNRMetric')

        r = []
        for name in self.namesRef:
            med_zlim = np.median(resCadence['zlim_{}'.format(name)])
            med_detect = np.median(metricValuesSNR['frac_obs_{}'.format(name)])
            r.append((band,dbName, med_zlim, med_detect))

        return r

            


def plotBand(band,medVals):

    ida = medVals['band'] == band
    medValues = medVals[ida]

    fig, ax = plt.subplots()

    fig.suptitle('{} band'.format(band))
    tot_label=[]
    shiftx = 0.002
    shifty = 0.005
    for dbName in forPlot['dbName']:
        idx = medValues['dbName'] == dbName
        sel = medValues[idx]
        idxp = np.where(forPlot['dbName'] == dbName)
        color = forPlot[idxp]['color'][0]
        marker = forPlot[idxp]['marker'][0]
        name = forPlot[idxp]['newName'][0]
        namefig = forPlot[idxp]['dbName'][0]
        #tot_label.append(ax.errorbar(sel['zlim'],sel['detect_rate'],color=color,marker=marker,label=name,linestyle='None'))
        ax.plot(sel['zlim'],sel['detect_rate'],color=color,marker=marker)
        if name not in ['opsim_single_visits','fb_rolling']:
            ax.text(sel['zlim']-shiftx,sel['detect_rate']-shifty,namefig)
        else:
            ax.text(sel['zlim']+shiftx/2.,sel['detect_rate'],namefig) 
    labs = [l.get_label() for l in tot_label]
    ax.legend(tot_label, labs, ncol=4,loc='best',prop={'size':12},frameon=False)
    ax.set_xlabel('z$_{lim}$')
    ax.set_ylabel('Detection rate')
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([xmin,xmax+0.01])
    plt.grid(linestyle='--')
    

dirFile = '/sps/lsst/users/gris/MetricOutput'



plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['figure.figsize'] = (10, 7)

forPlot = np.loadtxt('plot_scripts/cadenceCustomize.txt',
                     dtype={'names': ('dbName', 'newName', 'color','marker'),'formats': ('U17', 'U25', 'U1','U1')})

plotSum = PlotSummary()

bands = 'griz'

medList = []
for band in bands:
    for dbName in forPlot['dbName']:
        print('processing',dbName)
        medList += plotSum.getMetricValues(dirFile, dbName,band)
    
medValues = np.array(medList, dtype=[('band','U1'),('dbName','U17'),('zlim','f8'),('detect_rate','f8')])

print(forPlot)

for band in bands:
    plotBand(band,medValues)

plt.show()
