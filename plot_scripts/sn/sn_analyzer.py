from sn_analysis.sn_tools import *
from sn_analysis.sn_calc_plot import Calc_zlim, histSN_params, select
from sn_analysis.sn_calc_plot import plot_effi, effi, plot_NSN


from optparse import OptionParser
import numpy as np

parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str, default='../Output_SN',
                  help='OS location dir[%default]')
parser.add_option('--dbName', type=str, default='dd6_v2.99_10yrs',
                  help='OS name[%default]')
parser.add_option('--prodType', type=str, default='DDF_spectroz',
                  help='type prod (DDF_spectroz/DDF_photoz) [%default]')
parser.add_option('--norm_factor', type=float, default=50.,
                  help='normalization factor on prod [%default]')
opts, args = parser.parse_args()


dbDir = opts.dbDir
dbName = opts.dbName
prodType = opts.prodType
norm_factor = opts.norm_factor


#res_fast = load_complete('Output_SN_fast', dbName)

listDDF = 'COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'
# listDDF = 'COSMOS'
res = load_complete_dbSimu(dbDir, dbName, prodType, listDDF=listDDF)
fields = listDDF.split(',')


"""
print('aooo', len(res), res.columns, np.unique(res['healpixID']))
histSN_params(res)
plotSN_2D(res)
print(res.columns)
"""

dict_sel = {}

dict_sel['G10'] = [('n_epochs_m10_p35', operator.ge, 4),
                   ('n_epochs_m10_p5', operator.ge, 1),
                   ('n_epochs_p5_p20', operator.ge, 1),
                   ('n_bands_m8_p10', operator.ge, 2),
                   ('sigmaC', operator.le, 0.04),
                   ]

dict_sel['metric'] = [('n_epochs_bef', operator.ge, 4),
                      ('n_epochs_aft', operator.ge, 10),
                      ('n_epochs_phase_minus_10', operator.ge, 1),
                      ('n_epochs_phase_plus_20', operator.ge, 1),
                      ('sigmaC', operator.le, 0.04),
                      ]

sel = select(res, dict_sel['G10'])

for field in fields:
    idx = res['field'] == field
    idxb = sel['field'] == field
    print(field, len(res[idx])/norm_factor, len(sel[idxb])/norm_factor)


for field in fields:
    idx = sel['field'] == field
    sela = sel[idx]
    for season in np.unique(sela['season']):
        idxb = sela['season'] == season
        selb = sela[idxb]
        print(field, season, len(selb)/norm_factor)

print(len(sel)/norm_factor)

histSN_params(sel)
plotSN_2D(sel)
plot_NSN(sel, norm_factor=norm_factor)
plt.show()

"""
sel = select(res, dict_sel['G10'])
fig, ax = plt.subplots()
xmin = sel['n_epochs_aft'].min()
xmax = sel['n_epochs_aft'].max()
idx = sel['z'] > 0.5
ax.hist(sel[idx]['n_epochs_aft'], histtype='step', bins=range(xmin, xmax))

fig, ax = plt.subplots()
xmin = sel['n_epochs_bef'].min()
xmax = sel['n_epochs_bef'].max()
ax.hist(sel[idx]['n_epochs_bef'], histtype='step', bins=range(xmin, xmax))

plt.show()
"""
combi = []
for nbef in range(3, 5):
    for naft in range(3, 11):
        combi.append((nbef, naft))

combi = [(3, 8), (4, 10), (3, 6)]
for (nbef, naft) in combi:
    print(nbef, naft)
    seltype = 'metric_{}_{}'.format(nbef, naft)
    dict_sel[seltype] = [('n_epochs_phase_minus_10', operator.ge, 1),
                         ('n_epochs_phase_plus_20', operator.ge, 1),
                         ('n_epochs_bef', operator.ge, nbef),
                         ('n_epochs_aft', operator.ge, naft),
                         # ('sigmaC', operator.le, 0.04),
                         ]

fig, ax = plt.subplots(figsize=(10, 8))
for key, vals in dict_sel.items():
    sel = select(res, vals)
 #   selfast = select(res_fast, vals)
 #   print(key, len(sel), len(selfast))
    print(key, len(sel))
    plot_effi(res, sel, leg=key, fig=fig, ax=ax)
#    plot_effi(res, selfast, leg=key, fig=fig, ax=ax)

ax.legend()

# plt.show()

"""
plotSN_2D(sel, vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, vary='sigmat0', legy='$\sigma_{T_0}$')
plotSN_2D_binned(sel, bins=np.arange(0.1, 0.7, 0.01),
                 vary='sigmaC', legy='$\sigma_C$')
plotSN_2D(sel, varx='n_epochs_aft',
          legx='N$_{epochs}^{aft}$', vary='sigmaC', legy='$\sigma_C$')

"""


"""
plotSN_2D(sel, varx='n_epochs_bef',
          legx='N$_{epochs}^{bef}$', vary='sigmaC', legy='$\sigma_C$')
"""
plt.show()
"""
plotSN_effi(sel)
plt.show()
"""
