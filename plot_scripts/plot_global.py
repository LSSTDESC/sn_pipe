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

def plotHistTime(ax, axb,dbName, plotstr, forPlot, legx,legy):
    
    
    tab = np.load('{}/{}_SNGlobal.npy'.format(dbDir,dbName))
    print('hello',dbName,tab.dtype)
    idx = tab['night']<365
    sel = tab[idx]
    #ax.plot(sel['night'],sel[plotstr],label=dbName,linestyle='',marker='o')

    idx = forPlot['dbName'] == dbName
    mark = forPlot[idx]['marker'][0]

    sel.sort(order='night')
    bin = 1
    nt = int(365/bin)
    r=[]
    medref = np.median(sel[plotstr])
    for i in range(nt):
        selb = sel[i*bin:(i+1)*bin]
        r.append((np.median(selb['night']),np.median(selb[plotstr]),np.median(selb['med_moonAlt']),np.median(selb['nddf'])))
     
    rr = np.rec.fromrecords(r, names=['night',plotstr,'med_moonAlt','med_nddf'])
    #ax.hist2d(sel['night'],sel[plotstr],bins=100)
    if dbName != 'dec_1exp_pairsmix_10yrs':
        ax.plot(rr['night'],rr[plotstr],label=dbName)
    else:
        ax.plot(rr['night'],rr[plotstr],label=dbName,marker=mark,ls='none') 

    ax.legend(ncol=1,loc='best',prop={'size':12},frameon=True)
    ax.set_xlabel(legx, fontsize=12)
    ax.set_ylabel(legy, fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)

    idx = rr['med_nddf']<1.
    axb.plot(rr[idx][plotstr],rr[idx]['med_moonAlt'],label=dbName,marker=mark,ls='none')
    

dbDir = '/sps/lsst/users/gris/MetricOutput'

forPlot = np.loadtxt('plot_scripts/cadenceCustomize.txt',
                     dtype={'names': ('dbName', 'newName', 'group','Namepl','color','marker'),'formats': ('U33', 'U33','U12','U18','U6','U1')})

r = []

plotHist('nfc_noddf',forPlot,'# filter changes /night')
plotHist('obs_area',forPlot,'Observed area [deg2]/night')


fig, ax = plt.subplots()
figb, axb = plt.subplots()
plotHistTime(ax,axb,'alt_sched','nfc_noddf',forPlot,'night','# filter changes')
plotHistTime(ax,axb,'altsched_1exp_pairsmix_10yrs','nfc_noddf',forPlot,'night','# filter changes')
#plotHistTime(ax,'dec_1exp_pairsmix_10yrs','nfc',forPlot,'night','# filter changes')

"""
for dbName in forPlot['dbName']:
    tab = np.load('{}/{}_SNGlobal.npy'.format(dbDir,dbName))
    rint = [dbName,np.median(tab['nfc']),np.median(tab['obs_area'])]
    names = ['dbName','nfc_med','obs_area_med']
    for band in 'ugrizy':
        rint += [np.sum(tab['nvisits_{}'.format(band)])/np.sum(tab['nvisits'])]
        names += ['frac_{}'.format(band)]
    r.append((rint))

res = np.rec.fromrecords(r,names=names)

plotBarh(res,'nfc_med','Median number of filter changes per night',forPlot)
plotBarh(res,'obs_area_med','Median observed area per night',forPlot)
for band in 'ugrizy':
    plotBarh(res,'frac_{}'.format(band),'Filter allocation - {} band'.format(band),forPlot)
"""


plt.show()
