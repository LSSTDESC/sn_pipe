# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sn_plotter_metrics import plt
from scipy.interpolate import interp1d


@dataclass
class DDF:
    Nf: float  # number of fields
    Ns: float  # number of seasons/field
    Nv: float  # number of visits/season
    """
    cad: float  # cadence of observation
    sl: float  # season length
    zlim: float  # redshift completeness
    """


def plot(restot, varx='Nv_DD_night',
         legx='N$_{visits}^{DD}/obs. night}$',
         vary='Nv_UD_night',
         legy='N$_{visits}^{UD}/obs. night}$', ll={}, llb={},
         scenario={}, figtitle='', ll_err={}, ll_zcomp={}):

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(figtitle)
    fig.subplots_adjust(right=0.8)
    ls = dict(zip([1, 2, 3], ['solid', 'dotted', 'dashed']))
    mark = dict(zip([2, 3, 4], ['s', 'o', '^']))

    for (Nf_UD, Ns_UD) in np.unique(restot[['Nf_UD', 'Ns_UD']]):
        idx = restot['Nf_UD'] == Nf_UD
        idx &= restot['Ns_UD'] == Ns_UD
        sel = restot[idx]
        lstyle = ls[Nf_UD]
        mmark = mark[Ns_UD]
        label = '$(N_f^{UD},N_{season}^{UD})$'
        lab = '{} = ({},{})'.format(label, Nf_UD, Ns_UD)
        ax.plot(sel[varx], sel[vary], label=lab, marker=mmark,
                linestyle=lstyle, mfc='None', ms=7, color='k')
        if scenario:
            nv_UD = scenario[(Nf_UD, Ns_UD)][0]
            name = scenario[(Nf_UD, Ns_UD)][1]
            interp = interp1d(sel[vary], sel[varx], bounds_error=False)
            nv_DD = interp(nv_UD)
            ax.plot([nv_DD], [nv_UD], marker='o', ms=20,
                    color='b', mfc='None', markeredgewidth=3.)
            ax.text(1.05*nv_DD, 1.05*nv_UD, name, color='b', fontsize=12)
            print(name, int(nv_DD), int(nv_UD))
    xmin = np.min(restot[varx])
    xmax = np.max(restot[varx])
    ymin = np.min(restot[vary])
    ymax = np.max(restot[vary])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    if ll_err:
        for key, vals in ll_err.items():
            xminb = vals[0]
            xmaxb = vals[1]
            #poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
            ax.fill_between([xminb, xmaxb], ymin, ymax,
                            color='yellow', alpha=0.2)
    if ll_zcomp:
        for key, vals in ll_zcomp.items():
            yminb = vals[0]
            ymaxb = vals[1]
            print('alors', key, yminb, ymaxb)
            #poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
            ax.fill_between([xmin, xmax], yminb, ymaxb,
                            color='yellow', alpha=0.2)

    ax.set_xlabel(r'{}'.format(legx))
    ax.set_ylabel(r'{}'.format(legy))

    coltext = 'r'
    if ll:
        xmin, xmax = ax.get_xlim()
        k = 0
        for key, vals in ll.items():
            x = vals[0]
            ymin = vals[1]
            ymax = vals[2]
            if k == 0:
                tt = 1.01*ymin
                k = tt-ymin
            ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
            ax.text(x, ymin+k, key, fontsize=12, color=coltext)

    if llb:
        ymin, ymax = ax.get_ylim()
        k = 0
        for key, vals in llb.items():
            x = vals[0]
            xmin = vals[1]
            xmax = vals[2]
            if k == 0:
                tt = 1.01*xmin
                k = tt-xmin
            ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
            ax.text(xmin+k, x, key, fontsize=12, rotation=270,
                    color=coltext, va='top')

    ax.legend(bbox_to_anchor=(1.0, 0.55), ncol=1, frameon=False, fontsize=13)
    ax.grid()


def get_Nv_UD(Nf_UD, Ns_UD, Nv_UD, Nf_DD, Ns_DD, Nv_DD, k):
    """
    Function to estimate the number of DD visits per season

    Parameters
    ----------
    Nf_UD : int
        nb UD fields.
    Ns_UD : int
        nb season per UD field.
    Nv_UD : int
        nb visits per season and per UD field.
    Nf_DD : int
        nb DD fields.
    Ns_DD : int
        nb season per DD field.
    Nv_DD : int
        nb visits per season and per DD field.
    k : float
        equal to Nf_UD/Nf_DD.

    Returns
    -------
    Nv_DD : float
        nb visits per DD field and per season.

    """

    # UD = DDF(Nf_UD, Ns_UD, Nv_UD, cad_UD, sl_UD, -1)
    # DD = DDF(NDD-Nf_UD, Ns_DD, -1, cad_DD, sl_DD, -1)
    UD = DDF(Nf_UD, Ns_UD, Nv_UD)
    DD = DDF(NDD-Nf_UD, Ns_DD, Nv_DD)

    Nv_DD = budget_DD*Nv_LSST-UD.Nf*UD.Ns*UD.Nv
    Nv_DD /= (NDD-UD.Nf)*DD.Ns+k*UD.Nf*UD.Ns

    return Nv_DD


# load zlim vs Nvisits

dfa = pd.read_csv('Nvisits_zcomp_paper.csv', comment='#')
zlim_nvisits = interp1d(dfa['nvisits'], dfa['zcomp'],
                        bounds_error=False, fill_value=0.)
nvisits_zlim = interp1d(dfa['zcomp'], dfa['nvisits'],
                        bounds_error=False, fill_value=0.)

Nv_LSST = 2100000  # total number of visits
budget_DD = 0.07  # DD budget
NDD = 5  # 5 DDFs

sl_UD = 180.
cad_UD = 2.
cad_DD = 4
sl_DD = sl_UD

Nf_UD = 1
Nv_UD = -1
Ns_UD = 3
Ns_DD = (50-Nf_UD*Ns_UD)/(5-Nf_UD)

r = []
Nf_combi = [(1, 3), (2, 2), (2, 3), (2, 4)]

for combi in Nf_combi:
    Nf_UD = combi[0]
    Ns_UD = combi[1]
    for k in np.arange(1., 22., 1.):
        res = get_Nv_UD(Nf_UD, Ns_UD, Nv_UD, NDD-Nf_UD, Ns_DD, -1, k)
        print(k, res, k*res, cad_DD, sl_DD, cad_UD, sl_UD, Nf_UD, Ns_UD)
        r.append((k, res, k*res, res*cad_DD/sl_DD,
                  k*res*cad_UD/sl_UD, Nf_UD, Ns_UD,
                  zlim_nvisits(res*cad_DD/sl_DD)))

restot = np.rec.fromrecords(r, names=[
    'k', 'Nv_DD', 'Nv_UD', 'Nv_DD_night',
    'Nv_UD_night', 'Nf_UD', 'Ns_UD', 'zcomp'])
zc = '$z_{complete}$'

ll = dict(zip(['{}=0.80'.format(zc), '{}=0.75'.format(zc),
               '{}=0.70'.format(zc), '{}=0.66'.format(zc)],
              [[68, 242, 242],
              [70, 176, 176],
              [72, 134, 134],
              [74, 91, 91]]))

nvisits = '$N_{visits}^{LSST}$'
ftit = 'DD budget={}% - {}={} million'.format(int(100*budget_DD),
                                              nvisits, Nv_LSST/1.e6)
cadud = '$cad^{UD}$'
slud = '$sl^{UD}$'
caddd = '$cad^{DD}$'
sldd = '$sl^{DD}$'
ffig = '{} \n'.format(ftit)
ffig += '{}={} days, {}={} days, season length={} days'.format(cadud, cad_UD,
                                                               caddd, int(
                                                                   cad_DD),
                                                               int(sl_UD))

ffigb = '{} \n'.format(ftit)
ffigb += '{}={} days, season length={} days'.format(cadud, cad_UD,
                                                    int(sl_UD))
# ll = {}
plot(restot, ll=ll, figtitle=ffig)


zcomp = ['0.80', '0.75', '0.70', '0.66']
xaxis_leg = dict(zip(zcomp, [3000, 3010, 3020, 3030]))
xaxis_leg = dict(zip(zcomp, [3300, 3300, 3300, 3300]))

ll = {}
for z in zcomp:
    a = '{}={}'.format(zc, z)
    nv = cad_UD*nvisits_zlim(float(z))
    ll[a] = [xaxis_leg[z], int(nv), int(nv)]

N = 14852/9
Nb = 13545/9
Nc = 16285/9
llb = dict(zip(['PZ_y1', 'PZ_y2_y10', 'WL_10xWFD'],
               [[85, 1070, 1070],
               [85, N, N],
               [85, 800, 800]]))
ll_err = {}
ll_err['PZ_y2_y10'] = (Nb, Nc)

ll_zcomp = {}

for z in [0.75, 0.70]:
    zmin = z-0.01
    zmax = z+0.01
    ll_zcomp['z_{}'.format(z)] = (
        cad_UD*nvisits_zlim(zmin), cad_UD*nvisits_zlim(zmax))

field_season = [(1, 3), (2, 2), (2, 3), (2, 4)]
names = ['DDF_SCOC', 'DDF_DESC_0.80', 'DDF_DESC_0.75', 'DDF_DESC_0.70']
zvals = [0.66, 0.80, 0.75, 0.70]

scenario = {}

for i in range(len(field_season)):
    nv_UD = cad_UD*nvisits_zlim(zvals[i])
    name = names[i]
    scenario[field_season[i]] = [nv_UD, name]
#scenario = dict(zip(field_season, cad_UD*nvisits_zlim(zvals)))

"""

ll = dict(zip(['{}=0.80'.format(zc), '{}=0.75'.format(zc),
               '{}=0.70'.format(zc), '{}=0.66'.format(zc)],
              [[3000, 242, 242],
              [3010, 176, 176],
              [3020, 134, 134],
              [3030, 91, 91]]))
"""
plot(restot, varx='Nv_DD',
     legx='N$_{visits}^{DD}/season}$',
     ll=ll, llb=llb, scenario=scenario, figtitle=ffigb,
     ll_err=ll_err, ll_zcomp=ll_zcomp)

"""
ll = {}
plot(restot, varx='Nv_DD',
     legx='N$_{visits}^{DD}/season}$', vary='Nv_UD',
     legy='N$_{visits}^{UD}/season}$',
     figtitle=ftit)
"""

ll = dict(zip(['k=3.333'], [[1500, 3.3, 3.3]]))
"""
ll = {}
plot(restot, varx='Nv_DD',
     legx='N$_{visits}^{DD}/season}$',
     vary='k',
     legy='$k=\\frac{N_{visits}^{UD}}{N_{visits}^{DD}}$', ll=ll, figtitle=ftit)
"""
plt.show()
