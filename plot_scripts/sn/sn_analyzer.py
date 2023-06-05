import matplotlib.pyplot as ply
from sn_analysis.sn_tools import *
from sn_analysis.sn_calc_plot import Calc_zlim, histSN_params, select
from sn_analysis.sn_calc_plot import plot_effi, effi, plot_NSN


from optparse import OptionParser
import numpy as np


class sn_load_select:
    def __init__(self, dbDir, dbName, prodType,
                 listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'):
        """
        class to load and select sn data and make some stats

        Parameters
        ----------
        dbDir : str
            loc dir of the data to process.
        dbName : str
            db name.
        prodType : str
            production type (DDF_spectroz, DDF_photoz).
        listDDF : str, optional
            List of DDF to process.
            The default is 'COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'.

        Returns
        -------
        None.

        """

        fields = listDDF.split(',')
        # load the data
        data = load_complete_dbSimu(dbDir, dbName, prodType, listDDF=listDDF)

        # selectc only fitted LC
        idx = data['fitstatus'] == 'fitok'
        res = data[idx]

        res['dbName'] = dbName
        self.data = res

    def sn_selection(self, dict_sel={}):
        """
        Method to select data

        Parameters
        ----------
        dict_sel : dict, optional
            Selection criteria. The default is {}.

        Returns
        -------
        sel : dict
            Selected data.

        """

        sel = {}
        # select data here
        for selvar in dict_sel.keys():
            sel[selvar] = select(self.data, dict_sel[selvar])

        return sel


class sn_analyze(sn_load_select):
    def __init__(self, dbDir, dbName, prodType,
                 listDDF='COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb',
                 dict_sel={}):

        super().__init__(dbDir, dbName, prodType, listDDF)

        # get selected data
        self.data_dict = self.sn_selection(dict_sel)
        # self.dbName = dbName

    def get_nsn(self, grpby=['field'], norm_factor=1):
        """
        Method to get the number of SN

        Parameters
        ----------
        grpby : list(str), optional
            Used for the groupby df analysis. The default is ['field'].

        Returns
        -------
        nsn_fields : pandas df
            result with NSN col added (in addition to grpby).

        """

        nsn_fields = pd.DataFrame()
        for key, vals in self.data_dict.items():
            # dd['NSN'] = vals.groupby(['field']).transform('size').reset_index()
            dd = vals.groupby(grpby).size().to_frame('NSN').reset_index()
            dd['NSN'] /= norm_factor
            dd['selconfig'] = key
            nsn_fields = pd.concat((nsn_fields, dd))

        # nsn_fields['dbName'] = self.dbName
        return nsn_fields


parser = OptionParser(description='Script to analyze SN prod')

parser.add_option('--dbDir', type=str, default='../Output_SN',
                  help='OS location dir[%default]')
parser.add_option('--dbList', type=str, default='config_ana.csv',
                  help='OS name[%default]')
parser.add_option('--prodType', type=str, default='DDF_spectroz',
                  help='type prod (DDF_spectroz/DDF_photoz) [%default]')
parser.add_option('--norm_factor', type=float, default=50.,
                  help='normalization factor on prod [%default]')
opts, args = parser.parse_args()


dbDir = opts.dbDir
dbList = opts.dbList
prodType = opts.prodType
norm_factor = opts.norm_factor


# res_fast = load_complete('Output_SN_fast', dbName)

listDDF = 'COSMOS,CDFS,XMM-LSS,ELAISS1,EDFSa,EDFSb'

dict_sel = {}

dict_sel['G10_sigmaC'] = [('n_epochs_m10_p35', operator.ge, 4),
                          ('n_epochs_m10_p5', operator.ge, 1),
                          ('n_epochs_p5_p20', operator.ge, 1),
                          ('n_bands_m8_p10', operator.ge, 2),
                          ('sigmaC', operator.le, 0.04)]

dict_sel['G10_sigmaC_z0.7'] = [('n_epochs_m10_p35', operator.ge, 4),
                               ('n_epochs_m10_p5', operator.ge, 1),
                               ('n_epochs_p5_p20', operator.ge, 1),
                               ('n_bands_m8_p10', operator.ge, 2),
                               ('sigmaC', operator.le, 0.04),
                               ('z', operator.ge, 0.7)]

dict_sel['G10_JLA'] = [('n_epochs_m10_p35', operator.ge, 4),
                       ('n_epochs_m10_p5', operator.ge, 1),
                       ('n_epochs_p5_p20', operator.ge, 1),
                       ('n_bands_m8_p10', operator.ge, 2),
                       ('sigmat0', operator.le, 2.),
                       ('sigmax1', operator.le, 1.)]

dict_sel['G10_JLA_z0.7'] = [('n_epochs_m10_p35', operator.ge, 4),
                            ('n_epochs_m10_p5', operator.ge, 1),
                            ('n_epochs_p5_p20', operator.ge, 1),
                            ('n_bands_m8_p10', operator.ge, 2),
                            ('sigmat0', operator.le, 2.),
                            ('sigmax1', operator.le, 1.),
                            ('z', operator.ge, 0.7)]
"""
dict_sel['metric'] = [('n_epochs_bef', operator.ge, 4),
                      ('n_epochs_aft', operator.ge, 10),
                      ('n_epochs_phase_minus_10', operator.ge, 1),
                      ('n_epochs_phase_plus_20', operator.ge, 1),
                      ('sigmaC', operator.le, 0.04),
                      ]
"""

# load the data
dd = pd.read_csv(dbList, comment='#')
"""
sn_field = pd.DataFrame()
sn_field_season = pd.DataFrame()

for io, row in dd.iterrows():
    dbName = row['dbName']
    myclass = sn_analyze(dbDir, dbName,
                         prodType, listDDF, dict_sel)

    # estimate the number of SN
    fa = myclass.get_nsn(
        grpby=['dbName', 'field'], norm_factor=norm_factor)
    sn_field = pd.concat((sn_field, fa))
    fb = myclass.get_nsn(grpby=['dbName', 'field', 'season'],
                         norm_factor=norm_factor)

    sn_field_season = pd.concat((sn_field_season, fb))

# save in hdf5 files
sn_field.to_hdf('sn_field.hdf5', key='sn')
sn_field_season.to_hdf('sn_field_season.hdf5', key='sn')

print(sn_field)
print(sn_field_season)
"""

sn_field = pd.read_hdf('sn_field.hdf5')
sn_field_season = pd.read_hdf('sn_field_season.hdf5')
sn_tot = sn_field.groupby(['dbName', 'selconfig'])['NSN'].sum().reset_index()
sn_tot_season = sn_field_season.groupby(['dbName', 'selconfig', 'season'])[
    'NSN'].sum().reset_index()

sn_tot = sn_tot.merge(dd, left_on=['dbName'], right_on=['dbName'])

sn_tot_season = sn_tot_season.merge(
    dd, left_on=['dbName'], right_on=['dbName'])
print(sn_tot)
print(sn_tot_season)


dict_sel = {}

dict_sel['select'] = [('selconfig', operator.eq, 'G10_sigmaC')]

plot_NSN(sn_tot_season,
         xvar='season', xlabel='season',
         yvar='NSN', ylabel='$N_{SN}$',
         bins=None, norm_factor=1, loopvar='dbName', dict_sel=dict_sel)

plot_NSN(sn_tot_season,
         xvar='season', xlabel='season',
         yvar='NSN', ylabel='$\Sigma N_{SN}$',
         bins=None, norm_factor=1, loopvar='dbName', dict_sel=dict_sel, cumul=True)


plt.show()
print(test)


# listDDF = 'COSMOS'
res = load_complete_dbSimu(dbDir, dbName, prodType, listDDF=listDDF)
fields = listDDF.split(',')


"""
print('aooo', len(res), res.columns, np.unique(res['healpixID']))
histSN_params(res)
plotSN_2D(res)
print(res.columns)
"""


# selectc only fitted LC
idx = res['fitstatus'] == 'fitok'
res = res[idx]


sel = {}
# select data here
sel['noCut'] = res
for selvar in ['G10_sigmaC', 'G10_JLA']:
    sel[selvar] = select(res, dict_sel[selvar])

nsn_fields = pd.DataFrame()
for key, vals in sel.items():
    # dd['NSN'] = vals.groupby(['field']).transform('size').reset_index()
    dd = vals.groupby(['field']).size().to_frame('NSN').reset_index()
    dd['NSN'] /= norm_factor
    dd['selconfig'] = key
    nsn_fields = pd.concat((nsn_fields, dd))

print(nsn_fields)
print(test)

# print the results
pp = {}
for field in fields:
    idx = res['field'] == field
    pp['nSN_tot'] = len(res[idx])/norm_factor
    idxb = sel['field'] == field
    idxc = selb['field'] == field
    print(field, len(res[idx])/norm_factor, len(sel[idxb]) /
          norm_factor, len(selb[idxc])/norm_factor)


for field in fields:
    idx = sel['field'] == field
    sela = sel[idx]
    for season in np.unique(sela['season']):
        idxb = sela['season'] == season
        selc = sela[idxb]
        print(field, season, len(selc)/norm_factor)

print(len(sel)/norm_factor)

histSN_params(sel)
plotSN_2D(sel)
plot_NSN(sel, norm_factor=norm_factor)
idx = sel['z'] >= 0.7
idxb = selb['z'] >= 0.7

print('high-z', len(sel[idx])/norm_factor, len(selb[idxb])/norm_factor)
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
