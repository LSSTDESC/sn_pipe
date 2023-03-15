import numpy as np
import pandas as pd
from sn_plotter_metrics import plt
from scipy.interpolate import interp1d
from dataclasses import dataclass


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


class FiveSigmaDepth_Nvisits:
    def __init__(self, dbDir='../DB_Files',
                 dbName='draft_connected_v2.99_10yrs.npy',
                 requirements='pz_requirements.csv'):
        """
        class to estimate Nvisits from m5 and m5 from Nvisits

        Parameters
        ----------
        dbDir : str, optional
            location dir of the db to load. The default is '../DB_Files'.
        dbName : str, optional
            db Name to load. The default is 'draft_connected_v2.99_10yrs.npy'.
        requirements : str, optional
            csv file of requirements. The default is 'pz_requirements.csv'.

        Returns
        -------
        None.

        """

        # load data
        self.data = self.load_DDF(dbDir, dbName)

        # get frac events Moon-on
        nref = len(self.data)
        idx = self.data['moonPhase'] <= 20.
        frac_moon = len(self.data[idx])/len(self.data)

        self.frac_moon = np.round(frac_moon, 2)

        # load requirements

        self.req = self.load_req(requirements)

        # get m5 single exp. median
        self.msingle = pd.DataFrame.from_records(self.get_median_m5())

        self.msingle_calc, self.summary = self.get_Nvisits(
            self.msingle, self.req)

        print(self.summary)

    def load_req(self, requirements):
        """
        Method to load the requirements file

        Parameters
        ----------
        requirements : str
            requirement file name.

        Returns
        -------
        df_pz : pandas df
            array with requirement parameters

        """

        df_pz = pd.read_csv(requirements, comment='#')

        ll = df_pz['m5_y2_y10'].to_list()
        delta_mag = 0.05
        ll = list(map(lambda x: x - delta_mag, ll))
        df_pz['m5_y2_y10_m'] = ll

        ll = list(map(lambda x: x + 2*delta_mag, ll))
        df_pz['m5_y2_y10_p'] = ll

        return df_pz

    def load_DDF(self, dbDir, dbName, DDList=['COSMOS', 'ECDFS',
                                              'EDFS_a', 'EDFS_b',
                                              'ELAISS1', 'XMM_LSS']):
        """
        Method to load DDFs

        Parameters
        ----------
        dbDir : str
            location dir of the database.
        dbName : str
            db name (OS) to load.
        DDList : list(str), optional
            list of DDFs to consider. The default is ['COSMOS', 'ECDFS',
                                                      'EDFS_a', 'EDFS_b',
                                                      'ELAISS1', 'XMM_LSS'].

        Returns
        -------
        data : array
            DDF observations.

        """

        fullPath = '{}/{}'.format(dbDir, dbName)
        tt = np.load(fullPath)

        print(np.unique(tt['note']))
        data = None
        for field in DDList:
            idx = tt['note'] == 'DD:{}'.format(field)
            if data is None:
                data = tt[idx]
            else:
                data = np.concatenate((data, tt[idx]))

        return data

    def get_median_m5_field(self):
        """
        Method to get m5 per band and per DD field

        Parameters
        ----------
        None

        Returns
        -------
        msingle : array
            median m5 per band and per field.

        """

        r = []
        for field in np.unique(self.data['note']):
            idxa = self.data['note'] == field
            sela = self.data[idxa]
            for b in 'ugrizy':
                idxb = sela['band'] == b
                selb = sela[idxb]
                print(b, np.median(selb['fiveSigmaDepth']))
                r.append(
                    (b, np.median(selb['fiveSigmaDepth']),
                     field.split(':')[-1]))

        msingle = np.rec.fromrecords(
            r, names=['band', 'm5_med_single', 'field'])

        return msingle

    def get_median_m5(self):
        """
        Method to get the median m5 per band (all fields)

        Parameters
        ----------
        None

        Returns
        -------
        msingle : array
            median m5 values (per band).

        """

        r = []

        for b in 'ugrizy':
            idxb = self.data['band'] == b
            selb = self.data[idxb]
            r.append((b, np.median(selb['fiveSigmaDepth'])))

        msingle = np.rec.fromrecords(r, names=['band', 'm5_med_single'])

        return msingle

    def get_Nvisits(self, msingle, df_pz):
        """
        Method to estimate the number of visits depending on m5

        Parameters
        ----------
        msingle : pandas df
            array with m5 single exp. values.
        df_pz : pandas df
            array with config (target) m5 values

        Returns
        -------
        msingle : pandas df
            array with m5 single exp. values+ target
        summary : pandas df
            array with sum of visits (over field and band)

        """

        msingle = msingle.merge(df_pz, left_on=['band'], right_on=['band'])

        llv = []

        ccols = df_pz.columns.to_list()
        ccols.remove('band')
        ccols = list(map(lambda it: it.split('m5_')[1], ccols))

        for vv in ccols:
            diff = msingle['m5_{}'.format(vv)]-msingle['m5_med_single']
            Nv = 'Nvisits_{}'.format(vv)
            msingle[Nv] = 10**(0.8 * diff)
            llv.append(Nv)
        if 'field' in msingle.columns:
            summary = msingle.groupby(['field'])[llv].sum().reset_index()
        else:
            summary = msingle[llv].sum()

        return msingle, summary

    def get_Nvisits_from_frac(self, Nvisits,
                              col='Nvisits_y2_y10'):
        """
        Method to estimate the number of visits per band from a ref

        Parameters
        ----------
        Nvisits : int
            number of visits (total).
        col : str, optional
            ref col to estimate filter allocation.
            The default is 'Nvisits_y2_y10'.

        Returns
        -------
        df : pandas df
            array with the number of visits per band.

        """

        ntot = self.msingle_calc[col].sum()
        r = []

        for b in 'ugrizy':
            idx = self.msingle_calc['band'] == b
            frac = self.msingle_calc[idx][col].values/ntot
            r.append((b, frac[0]*Nvisits))

        df = pd.DataFrame(r, columns=['band', 'Nvisits'])

        return df

    def m5_from_Nvisits(self, Nvisits):
        """


        Parameters
        ----------
        Nvisits : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """

        df = self.get_Nvisits_from_frac(Nvisits)
        df = df.merge(self.msingle, left_on=['band'], right_on=['band'])
        df = df.merge(self.req, left_on=['band'], right_on=['band'])
        df['m5'] = df['m5_med_single']+1.25*np.log10(df['Nvisits'])
        df['delta_m5'] = df['m5']-df['m5_y2_y10']

        return df


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

        # interpolators per band
        self.nvisits_zlim_band = {}

        for b in 'grizy':
            self.nvisits_zlim_band[b] = interp1d(dfa['zcomp'], dfa[b],
                                                 bounds_error=False,
                                                 fill_value=0.)

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
        """
        Method to get nvisits depend on combination (Nf_UD,Ns_UD)

        Returns
        -------
        restot : array
            the result

        """

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
        """
        Method to get the DD scenatios for the UD fields

        Returns
        -------
        scenario : dict
            scenario parameters.

        """

        scenario = {}

        for i in range(len(self.Nf_combi)):
            nv_UD = self.cad_UD*self.nvisits_zlim(self.zcomp[i])
            nv_UD = int(np.round(nv_UD, 0))
            name = self.scen_names[i]
            scenario[self.Nf_combi[i]] = [nv_UD, name, self.zcomp[i]]

        return scenario

    def get_zcomp_req(self):
        """
        Method to grab zcomp reauirements

        Returns
        -------
        zcomp_req : dict
            requirement values.

        """

        zc = '$z_{complete}^{UD}$'

        zcomp_req = {}
        for z in self.zcomp:
            nv = self.cad_UD*self.nvisits_zlim(z)
            key = '{}={:0.2f}'.format(zc, np.round(z, 2))
            zcomp_req[key] = int(np.round(nv, 2))

        return zcomp_req

    def get_zcomp_req_err(self, zcomp=[0.80, 0.75, 0.70], delta_z=0.01):
        """
        Method to grab zcomp requirements error


        Parameters
        ----------
        zcomp : list(float), optional
            zcomp values. The default is [0.80, 0.75, 0.70].
        delta_z: float, optional
            delta_z value for err req. The default is 0.01

        Returns
        -------
        zcomp_req_err : dict
            req error values.

        """

        zcomp_req_err = {}

        for z in zcomp:
            zmin = z-delta_z
            zmax = z+delta_z
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
        """
        Method to plot the results

        Parameters
        ----------
        restot : array
            results (with nvisits).
        varx : str, optional
            x-axis variable. The default is 'Nv_DD_night'.
        legx : str, optional
            x-axis label. The default is 'N$_{visits}^{DD}/obs. night}$'.
        vary : str, optional
            y-axis variable. The default is 'Nv_UD_night'.
        legy : str, optional
            y-axis legend. The default is 'N$_{visits}^{UD}/obs. night}$'.
        scenario : dict, optional
            scenarios for DDF survey. The default is {}.
        figtitle : str, optional
            figure title. The default is ''.
        zcomp_req : dict, optional
            zcomp requirements. The default is {}.
        pz_wl_req : dict, optional
            pz_wl requirements. The default is {}.
        pz_wl_req_err : dict, optional
            pz_wl req errors. The default is {}.
        zcomp_req_err : dict, optional
            zcomp requirement errors. The default is {}.

        Returns
        -------
        res : array
            DDF scenario (taking requirements into account).

        """

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
                tag = scenario[(Nf_UD, Ns_UD)]
                nv_UD = tag[0]
                name = tag[1]
                zcomp = tag[2]
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

                # print('scenario', name, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD))
                for_res.append(
                    (name, zcomp, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD)))
                ax.plot([nv_DD], [nv_UD], marker='o', ms=15,
                        color='b', mfc='None', markeredgewidth=3.)
                ax.plot([nv_DD], [nv_UD], marker='.', ms=5,
                        color='b', mfc='None', markeredgewidth=3.)
                # ax.text(1.05*nv_DD, 1.05*nv_UD, name, color='b', fontsize=12)
                if vx < 0:
                    vx = np.abs(nv_DD-0.8*nv_DD)
                if vy < 0:
                    vy = np.abs(nv_UD-0.95*nv_UD)
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

        ax.legend(bbox_to_anchor=(1.0, 0.55),
                  ncol=1, frameon=False, fontsize=13)
        ax.grid()

        res = None
        if for_res:
            res = np.rec.fromrecords(for_res, names=['name', 'zcomp', 'Nf_UD',
                                                     'Ns_UD',
                                                     'nvisits_UD_night',
                                                     'nvisits_DD_season'])
        return res

    def finish(self, res):

        df_res = pd.DataFrame.from_records(res)
        df_res['zcomp_new'] = self.zlim_nvisits(
            df_res['nvisits_UD_night']/self.cad_UD)

        df_res['delta_z'] = df_res['zcomp']-df_res['zcomp_new']

        bands = 'grizy'

        for b in bands:
            nv = self.nvisits_zlim_band[b](df_res['zcomp_new'])
            df_res[b] = nv*self.cad_UD
            df_res[b] = df_res[b].astype(int)

        df_res['nvisits_UD_night_recalc'] = df_res[list(bands)].sum(axis=1)

        # if mismatch between Nvisits and Nvisits_recalc-> diff on z-band
        df_res['z'] += df_res['nvisits_UD_night'] - \
            df_res['nvisits_UD_night_recalc']
        df_res['nvisits_UD_night_recalc'] = df_res[list(bands)].sum(axis=1)

        idx = df_res['name'] != 'DDF_SCOC'
        df_res = df_res[idx]
        df_res = df_res.round({'delta_z': 2})

        nights_UD = self.sl_UD/self.cad_UD
        n_UD = df_res['Nf_UD']*df_res['Ns_UD'] * \
            df_res['nvisits_UD_night']*nights_UD
        n_DD = (50-df_res['Nf_UD']*df_res['Ns_UD'])*df_res['nvisits_DD_season']
        df_res['Nvisits'] = n_UD+n_DD
        df_res['budget'] = df_res['Nvisits']/self.Nv_LSST

        return df_res

    def plot_budget_time(self, df):
        """
        Method to plot the budget vs time

        Parameters
        ----------
        sel : TYPE
            DESCRIPTION.
        df : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        years = range(1, 11)
        r = []
        for i, row in df.iterrows():
            Nf_UD = row['Nf_UD']
            Ns_UD = row['Ns_UD']
            budget = 0.
            for year in years:
                to = Ns_UD <= year
                nn = to*1
                n_UD = Nf_UD*nn*row['nvisits_UD_night']*self.sl_UD/self.cad_UD
                n_DD = (5-Nf_UD*nn)*row['nvisits_DD_season']
                budget += (n_UD+n_DD)
                r.append((row['name'], year, budget/self.Nv_LSST))

        print(r)


m5class = FiveSigmaDepth_Nvisits()

m5_summary = m5class.summary

m5_dict = m5_summary.to_dict()

corresp = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], ['PZ_y1', 'PZ_y2_y10']))
nseasons = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], [1, 9]))

pz_wl_req = {}
for key, vals in corresp.items():
    pz_wl_req[vals] = [85, int(m5_dict[key]/nseasons[key])]

pz_wl_req['WL_10xWFD'] = [85, 800]
pz_wl_req_err = {}
pz_wl_req_err['PZ_y2_y10'] = (m5_dict['Nvisits_y2_y10_m']/9.,
                              m5_dict['Nvisits_y2_y10_p']/9.)

myclass = DD_Scenario(Nf_combi=[(2, 2), (2, 3), (2, 4)],
                      zcomp=[0.80, 0.75, 0.70],
                      scen_names=['DDF_DESC_0.80',
                                  'DDF_DESC_0.75',
                                  'DDF_DESC_0.70'])
restot = myclass.get_combis()
zcomp_req = myclass.get_zcomp_req()
zcomp_req_err = myclass.get_zcomp_req_err()
scenario = myclass.get_scenario()
"""
N = 14852/9
Nb = 13545/9
Nc = 16285/9
pz_wl_req = dict(zip(['PZ_y1', 'PZ_y2_y10', 'WL_10xWFD'],
                     [[85, 1070],
                     [85, N],
                     [85, 800]]))
pz_wl_req_err = {}
pz_wl_req_err['PZ_y2_y10'] = (Nb, Nc)
"""
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
                   pz_wl_req=pz_wl_req, pz_wl_req_err=pz_wl_req_err,
                   figtitle=ffig)

df_res = myclass.finish(res)
myclass.plot_budget_time(df_res)

"""
df_res = pd.DataFrame.from_records(res)
df_res['zcomp_new'] = myclass.zlim_nvisits(
    df_res['nvisits_UD_night']/myclass.cad_UD)


df_res['delta_z'] = df_res['zcomp']-df_res['zcomp_new']

bands = 'grizy'

for b in bands:
    nv = myclass.nvisits_zlim_band[b](df_res['zcomp_new'])
    df_res[b] = nv*myclass.cad_UD
    df_res[b] = df_res[b].astype(int)

df_res['nvisits_UD_night_recalc'] = df_res[list(bands)].sum(axis=1)

# if mismatch between Nvisits and Nvisits_recalc-> diff on z-band
df_res['z'] += df_res['nvisits_UD_night']-df_res['nvisits_UD_night_recalc']
df_res['nvisits_UD_night_recalc'] = df_res[list(bands)].sum(axis=1)

idx = df_res['name'] != 'DDF_SCOC'
df_res = df_res[idx]
df_res = df_res.round({'delta_z': 2})

nights_UD = myclass.sl_UD/myclass.cad_UD
n_UD = df_res['Nf_UD']*df_res['Ns_UD']*df_res['nvisits_UD_night']*nights_UD
n_DD = (50-df_res['Nf_UD']*df_res['Ns_UD'])*df_res['nvisits_DD_season']
df_res['Nvisits'] = n_UD+n_DD
df_res['budget'] = df_res['Nvisits']/myclass.Nv_LSST
"""
toprint = ['name', 'Nf_UD', 'Ns_UD', 'nvisits_UD_night', 'nvisits_UD_night_recalc',
           'g', 'r', 'i', 'z', 'y', 'delta_z', 'nvisits_DD_season', 'budget']


print(df_res[toprint])

m5_resu = pd.DataFrame()
for vv in res:
    Nvisits = vv['nvisits_DD_season']*9
    res = m5class.m5_from_Nvisits(Nvisits=Nvisits)
    res['name'] = vv['name']
    m5_resu = pd.concat((m5_resu, res))

idx = m5_resu['name'] != 'DDF_SCOC'
m5_resu = m5_resu[idx]
m5_resu = m5_resu.round({'m5': 2, 'delta_m5': 2})
print(m5_resu[['name', 'band', 'Nvisits', 'm5', 'delta_m5']])


plt.show()
