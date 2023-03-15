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
         legy='N$_{visits}^{UD}/obs. night}$', zcomp_req={}, pz_wl_req={},
         scenario={}, figtitle='', pz_wl_req_err={}, zcomp_req_err={}):

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(figtitle)
    fig.subplots_adjust(right=0.8)
    ls = dict(zip([1, 2, 3], ['solid', 'dotted', 'dashed']))
    mark = dict(zip([2, 3, 4], ['s', 'o', '^']))
    vx = -1
    vy = -1
    for_res = []
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
            if pz_wl_req and Nf_UD >= 2:
                nv_DD_n = pz_wl_req['PZ_y2_y10'][1]
                interpb = interp1d(sel[varx], sel[vary], bounds_error=False)
                nv_UD_n = interpb(nv_DD_n)
                nv_UD = np.mean([nv_UD, nv_UD_n])
                nv_DD = np.mean([nv_DD, nv_DD_n])

            print('scenario', name, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD))
            for_res.append((name, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD)))
            ax.plot([nv_DD], [nv_UD], marker='o', ms=20,
                    color='b', mfc='None', markeredgewidth=3.)
            ax.plot([nv_DD], [nv_UD], marker='.', ms=5,
                    color='b', mfc='None', markeredgewidth=3.)
            # ax.text(1.05*nv_DD, 1.05*nv_UD, name, color='b', fontsize=12)
            if vx < 0:
                vx = np.abs(nv_DD-0.85*nv_DD)
            if vy < 0:
                vy = np.abs(nv_UD-0.85*nv_UD)
            ax.text(nv_DD-vx, nv_UD-vy, name, color='b', fontsize=12)
            # print(name, int(nv_DD), int(nv_UD), vx, vy)
    xmin = np.min(restot[varx])
    xmax = np.max(restot[varx])
    ymin = np.min(restot[vary])
    ymax = np.max(restot[vary])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    if pz_wl_req_err:
        for key, vals in pz_wl_req_err.items():
            xminb = vals[0]
            xmaxb = vals[1]
            # poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
            ax.fill_between([xminb, xmaxb], ymin, ymax,
                            color='yellow', alpha=0.2)
    if zcomp_req_err:
        for key, vals in zcomp_req_err.items():
            yminb = vals[0]
            ymaxb = vals[1]
            # poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
            ax.fill_between([xmin, xmax], yminb, ymaxb,
                            color='yellow', alpha=0.2)

    ax.set_xlabel(r'{}'.format(legx))
    ax.set_ylabel(r'{}'.format(legy))

    coltext = 'r'
    if zcomp_req:
        xmin, xmax = ax.get_xlim()
        k = 0
        for key, vals in zcomp_req.items():
            x = vals[0]
            ymin = vals[1]
            ymax = vals[2]
            if k == 0:
                tt = 1.01*ymin
                k = tt-ymin
            ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
            ax.text(0.895*xmax, ymin+k, key, fontsize=12, color=coltext)

    if pz_wl_req:
        ymin, ymax = ax.get_ylim()
        k = 0
        for key, vals in pz_wl_req.items():
            x = vals[0]
            xmin = vals[1]
            xmax = xmin
            if k == 0:
                tt = 1.01*xmin
                k = tt-xmin
            ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
            ax.text(xmin+k, x, key, fontsize=12, rotation=270,
                    color=coltext, va='top')

    ax.legend(bbox_to_anchor=(1.0, 0.55), ncol=1, frameon=False, fontsize=13)
    ax.grid()

    res = None
    if for_res:
        res = np.rec.fromrecords(for_res, names=['name', 'Nf_UD', 'Ns_UD',
                                                 'nvisits_UD_night',
                                                 'nvisits_DD_season'])
    return res


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


class DD_Scenario:
    def __init__(self, Nv_LSST=2100000,  # total number of visits
                 budget_DD=0.07,  # DD budget
                 NDD=5,  # 5 DDFs
                 sl_UD=180.,  # season length UD fields
                 cad_UD=2.,  # cadence of observation UD fields
                 cad_DD=4.,  # cadence of observation DD fields
                 sl_DD=180.,  # season length DD fields
                 nvisits_zcomp_file='Nvisits_zcomp_paper.csv',  # nvisits vs zcomp
                 Nf_combi=[(1, 3), (2, 2), (2, 3), (2, 4)],
                 zcomp=[0.66, 0.80, 0.75, 0.70],
                 scen_names=['DDF_SCOC', 'DDF_DESC_0.80',
                             'DDF_DESC_0.75',
                             'DDF_DESC_0.70']):

        self.Nv_LSST = Nv_LSST
        self.budget_DD = budget_DD
        self.NDD = NDD
        self.sl_UD = sl_UD
        self.cad_UD = cad_UD
        self.cad_DD = cad_DD
        self.sl_DD = sl_DD
        self.Nf_combi = Nf_combi
        self.zcomp = zcomp
        self.scen_names = scen_names

        # load zlim vs nvisits
        dfa = pd.read_csv(nvisits_zcomp_file, comment='#')

        # interpolators
        self.zlim_nvisits = interp1d(dfa['nvisits'], dfa['zcomp'],
                                     bounds_error=False, fill_value=0.)
        self.nvisits_zlim = interp1d(dfa['zcomp'], dfa['nvisits'],
                                     bounds_error=False, fill_value=0.)

    def get_Nv_UD(self, Nf_UD, Ns_UD, Nv_UD, Nf_DD, Ns_DD, Nv_DD, k):
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
        DD = DDF(self.NDD-Nf_UD, Ns_DD, Nv_DD)

        Nv_DD = self.budget_DD*self.Nv_LSST-UD.Nf*UD.Ns*UD.Nv
        Nv_DD /= (self.NDD-UD.Nf)*DD.Ns+k*UD.Nf*UD.Ns

        return Nv_DD

    def get_combis(self):

        r = []
        for combi in self.Nf_combi:
            Nf_UD = combi[0]
            Ns_UD = combi[1]
            Ns_DD = (50-Nf_UD*Ns_UD)/(5-Nf_UD)

            for k in np.arange(1., 22., 1.):
                res = self.get_Nv_UD(Nf_UD, Ns_UD, -1,
                                     self.NDD-Nf_UD, Ns_DD, -1, k)
                print(k, res, k*res, self.cad_DD, self.sl_DD,
                      self.cad_UD, self.sl_UD, Nf_UD, Ns_UD)
                r.append((k, res, k*res, res*self.cad_DD/self.sl_DD,
                          k*res*self.cad_UD/self.sl_UD, Nf_UD, Ns_UD,
                          self.zlim_nvisits(res*self.cad_DD/self.sl_DD)))

        restot = np.rec.fromrecords(r, names=[
            'k', 'Nv_DD', 'Nv_UD', 'Nv_DD_night',
            'Nv_UD_night', 'Nf_UD', 'Ns_UD', 'zcomp'])

        return restot

    def get_scenario(self):

        scenario = {}

        for i in range(len(self.Nf_combi)):
            nv_UD = self.cad_UD*self.nvisits_zlim(self.zcomp[i])
            name = self.scen_names[i]
            scenario[self.Nf_combi[i]] = [int(np.round(nv_UD, 0)), name]

        return scenario

    def get_zcomp_req(self):

        zc = '$z_{complete}^{UD}$'

        zcomp_req = {}
        for z in self.zcomp:
            nv = self.cad_UD*self.nvisits_zlim(z)
            key = '{}={:0.2f}'.format(zc, np.round(z, 2))
            zcomp_req[key] = int(np.round(nv, 2))

        return zcomp_req

    def get_zcomp_req_err(self, zcomp=[0.80, 0.75, 0.70]):

        zcomp_req_err = {}

        for z in zcomp:
            zmin = z-0.01
            zmax = z+0.01
            zcomp_req_err['z_{}'.format(z)] = (
                self.cad_UD*self.nvisits_zlim(zmin),
                self.cad_UD*self.nvisits_zlim(zmax))

        return zcomp_req_err

    def plot(self, restot, varx='Nv_DD_night',
             legx='N$_{visits}^{DD}/obs. night}$',
             vary='Nv_UD_night',
             legy='N$_{visits}^{UD}/obs. night}$', scenario={}, figtitle='',
             zcomp_req={}, pz_wl_req={},
             pz_wl_req_err={}, zcomp_req_err={}):

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(figtitle)
        fig.subplots_adjust(right=0.8)
        ls = dict(zip([1, 2, 3], ['solid', 'dotted', 'dashed']))
        mark = dict(zip([2, 3, 4], ['s', 'o', '^']))
        vx = -1
        vy = -1
        for_res = []
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
                ax.plot([nv_DD], [nv_UD], marker='*', ms=20,
                        color='g', mfc='None', markeredgewidth=2)
                if pz_wl_req and Nf_UD >= 2:
                    nv_DD_n = pz_wl_req['PZ_y2_y10'][1]
                    interpb = interp1d(
                        sel[varx], sel[vary], bounds_error=False)
                    nv_UD_n = interpb(nv_DD_n)
                    ax.plot([nv_DD_n], [nv_UD_n], marker='*', ms=20,
                            color='g', mfc='None', markeredgewidth=2)
                    nv_UD = np.mean([nv_UD, nv_UD_n])
                    nv_DD = np.mean([nv_DD, nv_DD_n])

                print('scenario', name, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD))
                for_res.append((name, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD)))
                ax.plot([nv_DD], [nv_UD], marker='o', ms=15,
                        color='b', mfc='None', markeredgewidth=3.)
                ax.plot([nv_DD], [nv_UD], marker='.', ms=5,
                        color='b', mfc='None', markeredgewidth=3.)
                # ax.text(1.05*nv_DD, 1.05*nv_UD, name, color='b', fontsize=12)
                if vx < 0:
                    vx = np.abs(nv_DD-0.85*nv_DD)
                if vy < 0:
                    vy = np.abs(nv_UD-0.85*nv_UD)
                ax.text(nv_DD-vx, nv_UD-vy, name, color='b', fontsize=12)
                # print(name, int(nv_DD), int(nv_UD), vx, vy)
        xmin = np.min(restot[varx])
        xmax = np.max(restot[varx])
        ymin = np.min(restot[vary])
        ymax = np.max(restot[vary])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        if pz_wl_req_err:
            for key, vals in pz_wl_req_err.items():
                xminb = vals[0]
                xmaxb = vals[1]
                # poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
                ax.fill_between([xminb, xmaxb], ymin, ymax,
                                color='yellow', alpha=0.2)
        if zcomp_req_err:
            for key, vals in zcomp_req_err.items():
                yminb = vals[0]
                ymaxb = vals[1]
                # poly = [(xmin,yminb),(xmax,yminb),(xmax,ymaxb),(xmin,ymaxb)]
                ax.fill_between([xmin, xmax], yminb, ymaxb,
                                color='yellow', alpha=0.2)

        ax.set_xlabel(r'{}'.format(legx))
        ax.set_ylabel(r'{}'.format(legy))

        coltext = 'r'
        if zcomp_req:
            xmin, xmax = ax.get_xlim()
            k = 0
            for key, vals in zcomp_req.items():
                ymin = vals
                ymax = vals
                if k == 0:
                    tt = 1.05*ymin
                    k = tt-ymin
                ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
                ax.text(0.895*xmax, ymin+k, key, fontsize=12, color=coltext)

        if pz_wl_req:
            ymin, ymax = ax.get_ylim()
            k = 0
            for key, vals in pz_wl_req.items():
                x = vals[0]
                xmin = vals[1]
                xmax = xmin
                if k == 0:
                    tt = 1.01*xmin
                    k = tt-xmin
                ax.plot([xmin, xmax], [ymin, ymax], ls='dotted', color=coltext)
                ax.text(xmin+k, x, key, fontsize=12, rotation=270,
                        color=coltext, va='top')

        ax.legend(bbox_to_anchor=(1.0, 0.55),
                  ncol=1, frameon=False, fontsize=13)
        ax.grid()

        res = None
        if for_res:
            res = np.rec.fromrecords(for_res, names=['name', 'Nf_UD', 'Ns_UD',
                                                     'nvisits_UD_night',
                                                     'nvisits_DD_season'])
        return res


myclass = DD_Scenario()
restot = myclass.get_combis()
zcomp_req = myclass.get_zcomp_req()
zcomp_req_err = myclass.get_zcomp_req_err()
scenario = myclass.get_scenario()
N = 14852/9
Nb = 13545/9
Nc = 16285/9
pz_wl_req = dict(zip(['PZ_y1', 'PZ_y2_y10', 'WL_10xWFD'],
                     [[85, 1070],
                     [85, N],
                     [85, 800]]))
pz_wl_req_err = {}
pz_wl_req_err['PZ_y2_y10'] = (Nb, Nc)

nvisits = '$N_{visits}^{LSST}$'
cadud = '$cad^{UD}$'
ftit = 'DD budget={}% - {}={} million'.format(int(100*myclass.budget_DD),
                                              nvisits, myclass.Nv_LSST/1.e6)
ffig = '{} \n'.format(ftit)
ffig += '{}={} days, season length={} days'.format(cadud, myclass.cad_UD,
                                                   int(myclass.sl_UD))

myclass.plot(restot, varx='Nv_DD',
             legx='N$_{visits}^{DD}/season}$', figtitle=ffig)

res = myclass.plot(restot, varx='Nv_DD',
                   legx='N$_{visits}^{DD}/season}$', scenario=scenario,
                   zcomp_req=zcomp_req, zcomp_req_err=zcomp_req_err,
                   pz_wl_req=pz_wl_req, pz_wl_req_err=pz_wl_req_err, figtitle=ffig)

print(res)
plt.show()
print(test)

"""
res = plot(restot, varx='Nv_DD',
           legx='N$_{visits}^{DD}/season}$',
           zcomp_req=zcomp_req, pz_wl_req=pz_wl_req, scenario=scenario, figtitle=ffigb,
           pz_wl_req_err=pz_wl_req_err, zcomp_req_err=zcomp_req_err)
"""


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
    Ns_DD = (50-Nf_UD*Ns_UD)/(5-Nf_UD)
    for k in np.arange(1., 22., 1.):
        res = get_Nv_UD(Nf_UD, Ns_UD, Nv_UD, NDD-Nf_UD, Ns_DD, -1, k)
        print(k, res, k*res, cad_DD, sl_DD, cad_UD, sl_UD, Nf_UD, Ns_UD)
        r.append((k, res, k*res, res*cad_DD/sl_DD,
                  k*res*cad_UD/sl_UD, Nf_UD, Ns_UD,
                  zlim_nvisits(res*cad_DD/sl_DD)))

restot = np.rec.fromrecords(r, names=[
    'k', 'Nv_DD', 'Nv_UD', 'Nv_DD_night',
    'Nv_UD_night', 'Nf_UD', 'Ns_UD', 'zcomp'])

zc = '$z_{complete}^{UD}$'

zcomp_req = dict(zip(['{}=0.80'.format(zc), '{}=0.75'.format(zc),
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
plot(restot, zcomp_req=zcomp_req, figtitle=ffig)


zcomp = ['0.80', '0.75', '0.70', '0.66']
xaxis_leg = dict(zip(zcomp, [3000, 3010, 3020, 3030]))
xaxis_leg = dict(zip(zcomp, [3300, 3300, 3300, 3300]))

zcomp_req = {}
for z in zcomp:
    a = '{}={}'.format(zc, z)
    nv = cad_UD*nvisits_zlim(float(z))
    zcomp_req[a] = [xaxis_leg[z], int(nv), int(nv)]

N = 14852/9
Nb = 13545/9
Nc = 16285/9
pz_wl_req = dict(zip(['PZ_y1', 'PZ_y2_y10', 'WL_10xWFD'],
                     [[85, 1070],
                     [85, N],
                     [85, 800]]))
pz_wl_req_err = {}
pz_wl_req_err['PZ_y2_y10'] = (Nb, Nc)

zcomp_req_err = {}

for z in [0.75, 0.70]:
    zmin = z-0.01
    zmax = z+0.01
    zcomp_req_err['z_{}'.format(z)] = (
        cad_UD*nvisits_zlim(zmin), cad_UD*nvisits_zlim(zmax))

field_season = [(1, 3), (2, 2), (2, 3), (2, 4)]
names = ['DDF_SCOC', 'DDF_DESC_0.80', 'DDF_DESC_0.75', 'DDF_DESC_0.70']
zvals = [0.66, 0.80, 0.75, 0.70]

scenario = {}

for i in range(len(field_season)):
    nv_UD = cad_UD*nvisits_zlim(zvals[i])
    name = names[i]
    scenario[field_season[i]] = [nv_UD, name]
# scenario = dict(zip(field_season, cad_UD*nvisits_zlim(zvals)))

"""

ll = dict(zip(['{}=0.80'.format(zc), '{}=0.75'.format(zc),
               '{}=0.70'.format(zc), '{}=0.66'.format(zc)],
              [[3000, 242, 242],
              [3010, 176, 176],
              [3020, 134, 134],
              [3030, 91, 91]]))
"""
res = plot(restot, varx='Nv_DD',
           legx='N$_{visits}^{DD}/season}$',
           zcomp_req=zcomp_req, pz_wl_req=pz_wl_req, scenario=scenario, figtitle=ffigb,
           pz_wl_req_err=pz_wl_req_err, zcomp_req_err=zcomp_req_err)

print('jjj', res)

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
