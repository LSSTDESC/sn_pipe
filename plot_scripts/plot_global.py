import numpy as np
import matplotlib.pyplot as plt

def plotBarh(sel, plotstr, title,forPlot):
    sel.sort(order=plotstr)
    fig, ax = plt.subplots(figsize=(12,10))
    fig.suptitle(title)
    icol = []
    myrange = np.arange(len(sel))
    print(forPlot['dbName'])
    for val in myrange:
        idf = forPlot['dbName'] == sel['dbName'][val].strip()
        print('hallo',val,sel['dbName'][val],forPlot['color'][idf])
        if len(forPlot['color'][idf])>0:
            icol.append(forPlot['color'][idf][0])
    icol = np.array(icol)
    print(len(icol),len(sel))
    ax.barh(myrange,sel[plotstr],color=icol)
    #plt.barh(sel['ilist'],sel['Nvisits_frac'])

    plt.yticks(myrange,sel['dbName'])
    #plt.yticks(np.arange(len(sel)),sel['dbName'])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([0.05,xmax])
    plt.grid(axis='x')
    plt.tight_layout()


def plotHist(plotstr, forPlot, legx,legy='Number of Entries'):
    
    fig, ax = plt.subplots()
    for dbName in forPlot['dbName']:
        tab = np.load('{}/{}_SNGlobal.npy'.format(dbDir,dbName))
        ax.hist(tab[plotstr],histtype='step',lw=2,label=dbName)
    
    ax.legend(ncol=1,loc='best',prop={'size':12},frameon=True)
    ax.set_xlabel(legx, fontsize=12)
    ax.set_ylabel(legy, fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

dbDir = '/sps/lsst/users/gris/MetricOutput'

forPlot = np.loadtxt('plot_scripts/cadenceCustomize.txt',
                     dtype={'names': ('dbName', 'newName', 'group','Namepl','color','marker'),'formats': ('U33', 'U33','U12','U18','U6','U1')})

r = []

plotHist('nfc',forPlot,'# filter changes /night')
plotHist('obs_area',forPlot,'Observed area [deg2]/night')

for dbName in forPlot['dbName']:
    tab = np.load('{}/{}_SNGlobal.npy'.format(dbDir,dbName))
    r.append((dbName,np.median(tab['nfc']),np.median(tab['obs_area'])))

res = np.rec.fromrecords(r,names=['dbName','nfc_med','obs_area_med'])

plotBarh(res,'nfc_med','Median number of filter changes per night',forPlot)
plotBarh(res,'obs_area_med','Median observed area per night',forPlot)


plt.show()
