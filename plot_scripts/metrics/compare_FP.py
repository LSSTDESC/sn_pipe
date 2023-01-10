import pandas as pd
from sn_plotter_metrics.plot4metric import plot_vs_OS_dual,plot_series_fields,plot_field,plot_pixels
import numpy as np

def get_diff(dfa, dfb,right_on=['dbName','family'],left_on=['dbName','family']):
    """
    

    Parameters
    ----------
    dfa : TYPE
        DESCRIPTION.
    dfb : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    dfmerge = dfa.merge(dfb, right_on=right_on,left_on=left_on)
    dfmerge['diff_nsn'] = dfmerge['nsn_x']-dfmerge['nsn_y']
    dfmerge['diff_nsn_rel'] = (dfmerge['nsn_x']-dfmerge['nsn_y'])/dfmerge['nsn_x']
    dfmerge['diff_zcomp'] = dfmerge['zcomp_x']-dfmerge['zcomp_y']
    
    return dfmerge
   

dbList = pd.read_csv('DD_fbs_2.99_plot.csv')

print(dbList)

csvFiles = dict(zip(['gnomonic_circular_0',  'gnomonic_realistic_0'],
                    ['metric_summary_DD_gnomonic_circular_0_fields_pixels.csv', 'metric_summary_DD_gnomonic_realistic_0_fields_pixels.csv']))



df = {}

for key, vals in csvFiles.items():
    df[key] = pd.read_csv(vals)
    print(df[key])

#get the runtype
run_type = 'global'
checkstr ='\t'.join(list(csvFiles.values()))
for ll in ['fields','season','pixels']:
    if ll in checkstr:
        run_type = ll


ref_str = 'gnomonic_circular_0'
val_str = 'gnomonic_realistic_0'

dfa = df[val_str]
dfb = df[ref_str]

vary=['diff_nsn_rel','diff_zcomp']
legy=[r'$\Delta N_{SN}=\frac{N_{SN}^{realistic}-N_{SN}^{circular}}{N_{SN}^{circular}}$','$\Delta z_{complete}=Z_{complete}^{realistic}-z_{complete}^{circular}$']

if run_type == 'global':
  
    res = get_diff(dfa,dfb)
    plot_vs_OS_dual(res,varx='family',vary=vary,legy=legy)

if run_type == 'fields':
    
    res = get_diff(dfa,dfb,left_on=['dbName','family','fieldname'],right_on=['dbName','family','fieldname'])
    res['field'] = res['fieldname']
    plot_series_fields(res,what=vary,leg=legy)

if run_type == 'season':
    res = get_diff(dfa,dfb,left_on=['dbName','family','fieldname','season'],right_on=['dbName','family','fieldname','season'])
    print(res.columns)
    res = res.merge(dbList, left_on=['dbName','family'],right_on=['dbName','family'])
    print(res.columns)
    for field in ['COSMOS']:
        idx = res['fieldname'] == field
        sel = res[idx]
        print(sel)
        plot_field(sel,yvars=vary,ylab=legy)  

if run_type == 'pixels':
   vv = ['dbName','family','fieldname','season','healpixID','pixRA','pixDec']
   res = get_diff(dfa,dfb,left_on=vv,right_on=vv)
   print(res['dbName'].unique(),res.columns)
   dbName = 'draft_connected_v2.99_10yrs'
   field = 'COSMOS'
   season = 3
   idx = res['dbName'] == dbName
   idx &= res['fieldname'] == field
   idx &= res['season'] == season
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=(14,8))
   figtitle = '{} - {} \n season {}'.format(dbName,field,season)
   plot_pixels(res[idx],yvar='nsn_x',fig=fig,ax=ax,figtitle=figtitle,marker='s',color='k',showIt=False,label='{} FP'.format(val_str.split('_')[1]))
   plot_pixels(res[idx],yvar='nsn_y',fig=fig,ax=ax,figtitle=figtitle,marker='*',color='r',mfc='r',label='{} FP'.format(ref_str.split('_')[1]))
   
"""
fig, ax = plt.subplots()
for i, row in dbList.iterrows():
    print('ttt', row['dbName'], row['marker'], row['color'], row['mfc'])
    for key, vals in df.items():
        idx = vals['dbName'] == row['dbName']
        sel = vals[idx]
        ax.plot(sel['zcomp'], sel['nsn'],
                marker=row['marker'], color=row['color'])

plt.show()
"""
