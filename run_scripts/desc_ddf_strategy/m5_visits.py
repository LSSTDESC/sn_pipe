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
                 requirements='input/DESC_cohesive_strategy/pz_requirements.csv'):
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
        nseas = dict(zip(ccols, [1, 9, 9, 9]))

        for vv in ccols:
            diff = msingle['m5_{}'.format(vv)]-msingle['m5_med_single']
            Nv = 'Nvisits_{}'.format(vv)
            msingle[Nv] = 10**(0.8 * diff)
            msingle['nseason_{}'.format(vv)] = nseas[vv]
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
        # df['m5'] = df['m5_med_single']+1.25*np.log10(df['Nvisits'])
        # df['delta_m5'] = df['m5']-df['m5_y2_y10']

        return df

    def m5_band_from_Nvisits(self, m5_resu_orig, sl_DD=180., cad_DD=4):

        m5_resu = pd.DataFrame(m5_resu_orig)

        # m5_resu['Nvisits'] = m5_resu['Nvisits'].astype(int)

        m5_resu = self.visits_night_from_frac(m5_resu, sl=sl_DD,
                                              cad=cad_DD, col='Nvisits')

        topp = m5_resu[['name', 'band', 'Nvisits', 'Nvisits_night']]

        topp.to_csv('DD_res2.csv', index=False)

        # now estimate the number of visits per night for y1

        m5single = m5class.msingle_calc

        tt = m5single[['band', 'Nvisits_y1', 'm5_y1', 'm5_med_single']]
        tt['Nseasons'] = 1

        # tt['Nvisits_y1'] = tt['Nvisits_y1'].astype(int)

        """
        tt['Nvisits_y1_night'] = tt.apply(
            lambda x: x['Nvisits_y1']/(fracs[x['band']]*Nnights_DD_season), axis=1)

        tt['Nvisits_y1'] = tt['Nvisits_y1'].astype(int)
        tt['Nvisits_y1_night'] = tt['Nvisits_y1_night'].astype(int)
        """
        tt = self.visits_night_from_frac(tt, sl=sl_DD,
                                         cad=cad_DD, col='Nvisits_y1',
                                         colb='m5_y1')

        return m5_resu, tt

    def visits_night_from_frac(self, tab, sl, cad, col='Nvisits',
                               colb='m5_y2_y10', frac_moon=0.20):

        nights_season = int(sl/cad)
        bands = 'ugrizy'
        frac_night = [frac_moon, 1., 1., 1., 1.-frac_moon, 1.]
        fracs = dict(zip(bands, frac_night))
        tab['{}_night'.format(col)] = tab.apply(
            lambda x: x[col]/x['Nseasons'] /
            (fracs[x['band']]*nights_season),
            axis=1)

        # tab['{}_night'.format(col)] = tab['{}_night'.format(col)].astype(int)
        tab['cad'] = cad
        tab['sl'] = sl
        tab['nights_season'] = nights_season

        frac_df = pd.DataFrame(list(bands), columns=['band'])
        frac_df['frac_night'] = frac_night
        tab['Nseasons'] = tab['Nseasons']
        tab = tab.merge(frac_df, left_on=['band'], right_on=['band'])

        tab = tab.round({'{}'.format(col): 0, '{}_night'.format(col): 0})
        tab['{}_recalc'.format(col)] = tab['{}_night'.format(
            col)]*tab['nights_season']*tab['frac_night']*tab['Nseasons']

        tab['m5_recalc'] = 1.25 * \
            np.log10(tab['{}_recalc'.format(col)])+tab['m5_med_single']
        tab['diff_m5'] = tab['m5_recalc']-tab[colb]
        return tab


class DD_Scenario:
    def __init__(self, Nv_LSST=2100000,  # total number of visits
                 budget_DD=0.07,  # DD budget
                 NDD=5,  # 5 DDFs
                 Nseason=10,  # number of season of observation
                 sl_UD=180.,  # season length UD fields
                 cad_UD=2.,  # cadence of observation UD fields
                 cad_DD=4.,  # cadence of observation DD fields
                 sl_DD=180.,  # season length DD fields
                 Nf_DD_y1=3,  # number of DDF year 1
                 Nv_DD_y1=998,  # number of DD visits year 1
                 # nvisits vs zcomp
                 nvisits_zcomp_file='input/DESC_cohesive_strategy/Nvisits_zcomp_paper.csv',
                 Nf_combi=[(1, 3), (2, 2), (2, 3), (2, 4)],  # UD combi
                 zcomp=[0.66, 0.80, 0.75, 0.70],  # correesponding zcomp
                 scen_names=['DDF_SCOC', 'DDF_DESC_0.80',
                             'DDF_DESC_0.75',
                             'DDF_DESC_0.70'],  # scenario name
                 frac_moon=0.20):

        self.Nv_LSST = Nv_LSST
        self.budget_DD = budget_DD
        self.NDD = NDD
        self.Nseason = Nseason
        self.sl_UD = sl_UD
        self.cad_UD = cad_UD
        self.cad_DD = cad_DD
        self.sl_DD = sl_DD
        self.Nf_DD_y1 = Nf_DD_y1
        self.Nv_DD_y1 = Nv_DD_y1
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

    def get_Nv_DD(self, Nf_UD, Ns_UD, Nv_UD, Nf_DD, Ns_DD, Nv_DD, k):
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

        Nv_DD = self.budget_DD*self.Nv_LSST
        Nv_DD -= self.Nf_DD_y1*self.Nv_DD_y1
        Nv_DD -= UD.Nf*UD.Ns*UD.Nv
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
            Ns_DD = (self.NDD*self.Nseason-self.Nf_DD_y1-Nf_UD*Ns_UD)
            Ns_DD /= (self.NDD-Nf_UD)

            for k in np.arange(1., 22., 1.):
                res = self.get_Nv_DD(Nf_UD, Ns_UD, -1,
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
                    (name, zcomp, Nf_UD, Ns_UD, int(nv_UD), int(nv_DD), self.cad_UD, self.sl_UD))

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
                                                     'nvisits_DD_season',
                                                     'cad', 'sl'])
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
        df_res['nvisits_UD_season'] = df_res['nvisits_UD_night']*nights_UD
        for b in bands:
            df_res['{}_season'.format(b)] = df_res['{}'.format(b)]*nights_UD

        n_DD = self.NDD*self.Nseason
        n_DD -= self.Nf_DD_y1
        n_DD -= df_res['Nf_UD']*df_res['Ns_UD']
        n_DD *= df_res['nvisits_DD_season']
        df_res['Nvisits'] = n_UD+n_DD+self.Nf_DD_y1*self.Nv_DD_y1
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

        years = range(0, 11)
        r = []
        for i, row in df.iterrows():
            Nf_UD = row['Nf_UD']
            Ns_UD = row['Ns_UD']
            budget = 0.
            for year in years:
                to = year <= Ns_UD+1
                nn = to*1
                n_UD = Nf_UD*nn*row['nvisits_UD_night']*self.sl_UD/self.cad_UD
                n_DD = (self.NDD-Nf_UD*nn)*row['nvisits_DD_season']
                if year == 1:
                    n_DD = self.Nf_DD_y1*self.Nv_DD_y1
                    n_UD = 0
                if year > 0:
                    budget_y = (n_UD+n_DD)/self.Nv_LSST
                    budget += budget_y
                    bb = 100*budget
                    r.append((row['name'], year, budget, bb, budget_y))
                    print('helllo', year, n_DD, n_UD)
                else:
                    r.append((row['name'], year, 0.0, 0.0, 0.0))

        res = np.rec.fromrecords(
            r, names=['name', 'year', 'budget', 'budget_per', 'budget_yearly'])

        fig, ax = plt.subplots(figsize=(14, 8))

        names = np.unique(res['name'])
        ls = dict(zip([0, 1, 2], ['solid', 'dotted', 'dashed']))
        colors = dict(zip([0, 1, 2], ['blue', 'k', 'red']))
        for io, nn in enumerate(names):
            idx = res['name'] == nn
            sel = res[idx]
            ax.plot(sel['year'], sel['budget_per'], linestyle=ls[io],
                    color=colors[io], label=nn)

        ax.grid()
        ax.set_xlabel('Year')
        ax.set_ylabel('DDF budget [%]')
        ax.legend(frameon=False)
        ax.set_xlim(0, 10)
        ax.set_ylim(0., 7.)


def plot_budget_time(res):
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

    fig, ax = plt.subplots(figsize=(14, 8))

    names = np.unique(res['name'])
    ls = dict(zip([0, 1, 2], ['solid', 'dotted', 'dashed']))
    colors = dict(zip([0, 1, 2], ['blue', 'k', 'red']))
    for io, nn in enumerate(names):
        idx = res['name'] == nn
        sel = res[idx]
        ax.plot(sel['year'], sel['budget_per'], linestyle=ls[io],
                color=colors[io], label=nn)

    ax.grid()
    ax.set_xlabel('Year')
    ax.set_ylabel('DDF budget [%]')
    ax.legend(frameon=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0., 7.)


def check_m5(df_res, m5class):

    r = []
    for i, row in df_res.iterrows():
        for b in bands:
            r.append((b, row['{}_season'.format(b)], row['name']))

    df_new = pd.DataFrame(r, columns=['band', 'Nvisits', 'name'])
    print(df_new)

    df_new = df_new.merge(m5class.msingle, left_on='band', right_on='band')

    yrs = 'y2_y10'
    df_new['m5_UD_{}'.format(yrs)] = df_new['m5_med_single'] + \
        1.25*np.log10(df_new['Nvisits'])
    df_new = df_new.merge(m5class.req, left_on='band', right_on='band')
    df_new['diff_m5_{}'.format(yrs)] = df_new['m5_UD_{}'.format(
        yrs)]-df_new['m5_{}'.format(yrs)]+np.log10(9)

    print(df_new[['name', 'band', 'Nvisits', 'm5_med_single',
          'm5_{}'.format(yrs), 'diff_m5_{}'.format(yrs)]])

    df_new[['name', 'band', 'Nvisits', 'diff_m5_{}'.format(yrs)]].to_csv(
        'ddf_res3.csv', index=False)


def get_band_season(grp):

    res = {}

    for b in 'ugrizy':
        idx = grp['band'] == b
        selit = grp[idx]
        res[b] = selit['nvisits_band_season'].mean()

    return res


def get_budget(grp, NDD, m5_resu, m5_nvisits_y1, Nv_LSST):

    Nf_UD = grp['Nf_UD'].mean()
    Ns_UD = grp['Ns_UD'].mean()

    print(grp)
    grp['nvisits_band_season'] = grp['n_night_season']*grp['nvisits_night']
    n_UD_season = grp['nvisits_band_season'].sum()

    n_UD_band_season = get_band_season(grp)

    print('iiii', grp.name, n_UD_band_season)

    idx = m5_resu['name'] == grp.name
    sel_m5 = m5_resu[idx]

    print(sel_m5.columns)
    sel_m5['nvisits_band_season'] = sel_m5['Nvisits_night'] * \
        sel_m5['frac_night']*sel_m5['nights_season']

    n_DD_y2_y10 = sel_m5['nvisits_band_season'].sum()
    n_DD_band_y2_y10_season = get_band_season(sel_m5)

    print(m5_nvisits_y1)
    m5_nvisits_y1['nvisits_band_season'] = m5_nvisits_y1['Nvisits_y1_recalc']
    n_DD_y1 = m5_nvisits_y1['Nvisits_y1_recalc'].sum()
    n_DD_band_y1_season = get_band_season(m5_nvisits_y1)

    print('here', n_UD_season, n_DD_y2_y10, n_DD_y1)
    print('hhb', n_DD_band_y2_y10_season, n_DD_band_y1_season)

    years = range(0, 11)
    budget = 0.
    r = []
    bands = 'ugrizy'
    for year in years:
        to = year <= Ns_UD+1
        nn = to*1
        n_UD = Nf_UD*nn*n_UD_season
        n_DD = n_DD_y2_y10*(5-Nf_UD)
        nvisits_band = dict(zip(bands, [0.]*len(bands)))

        if year == 1:
            n_DD = n_DD_y1
            n_UD = 0
            for b in 'ugrizy':
                nvisits_band[b] = n_DD_band_y1_season[b]
        else:
            for b in 'ugrizy':
                nvisits_band[b] = Nf_UD*nn*n_UD_band_season[b]
                nvisits_band[b] += (NDD-Nf_UD*nn)*n_DD_band_y2_y10_season[b]
        if year > 0:
            budget_y = (n_UD+n_DD)/Nv_LSST
            budget += budget_y
            bb = 100*budget
            add = [nvisits_band[b] for b in bands]
            r.append([year, budget, bb, budget_y, Nf_UD, Ns_UD]+add)
            print('helllo', year, n_DD, n_UD, budget, budget_y, Nf_UD, Ns_UD)
        else:
            r.append([year, 0.0, 0.0, 0.0, 0.0, 0., 0., 0., 0., 0., 0., 0.0])

    res = np.rec.fromrecords(
        r, names=['year', 'budget', 'budget_per', 'budget_yearly', 'Nf_UD', 'Ns_UD']+list(bands))

    return pd.DataFrame.from_records(res)

################################################
##### get Nvisits from m5 req.          ########
## m5_dict: total number of visits (PZ req.) ##
###############################################


bands = 'ugrizy'

m5class = FiveSigmaDepth_Nvisits()

m5_summary = m5class.summary
m5_nvisits = m5class.msingle_calc
m5_dict = m5_summary.to_dict()

print(m5_dict)
print(m5_nvisits)

## get (Nvisits_UD vs N_visits_DD for (Kf_UD, Ns_UD) combinations ####

corresp = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], ['PZ_y1', 'PZ_y2_y10']))
nseasons = dict(zip(['Nvisits_y1', 'Nvisits_y2_y10'], [1, 9]))

pz_wl_req = {}
for key, vals in corresp.items():
    pz_wl_req[vals] = [85, int(m5_dict[key]/nseasons[key])]

pz_wl_req['WL_10xWFD'] = [85, 800]
pz_wl_req_err = {}
pz_wl_req_err['PZ_y2_y10'] = (m5_dict['Nvisits_y2_y10_m']/9.,
                              m5_dict['Nvisits_y2_y10_p']/9.)

# year 1 reqs are different -> take it into account
Nf_DD_y1 = 3  # UD the first year
f_DD_y1 = 5  # UD the second year
Nv_DD_y1 = int(m5_dict['Nvisits_y1'])
sl_UD = 180.
cad_UD = 2.
NDD = 5
Nv_LSST = 2.1e6
frac_moon = 0.20
myclass = DD_Scenario(Nf_combi=[(2, 2), (2, 3), (2, 4)],
                      zcomp=[0.80, 0.75, 0.70],
                      scen_names=['DDF_DESC_0.80',
                                  'DDF_DESC_0.75',
                                  'DDF_DESC_0.70'],
                      Nf_DD_y1=Nf_DD_y1,
                      Nv_DD_y1=Nv_DD_y1,
                      sl_UD=sl_UD, cad_UD=cad_UD, NDD=NDD, Nv_LSST=Nv_LSST,
                      frac_moon=frac_moon)

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

### plot the result ######
nvisits = '$N_{visits}^{LSST}$'
cadud = '$cad^{UD}$'
ftit = 'DD budget={}% - {}={} million'.format(int(100*myclass.budget_DD),
                                              nvisits, myclass.Nv_LSST/1.e6)
ffig = '{} \n'.format(ftit)
ffig += '{}={} days, season length={} days'.format(cadud, myclass.cad_UD,
                                                   int(myclass.sl_UD))

myclass.plot(restot, varx='Nv_DD',
             legx='N$_{visits}^{DD}/season}$', figtitle=ffig)

# zcomp_req = {}
# zcomp_req_err = {}
# pz_wl_req_err = {}
# scenario = {}

res = myclass.plot(restot, varx='Nv_DD',
                   legx='N$_{visits}^{DD}/season}$', scenario=scenario,
                   zcomp_req=zcomp_req, zcomp_req_err=zcomp_req_err,
                   pz_wl_req=pz_wl_req, pz_wl_req_err=pz_wl_req_err,
                   figtitle=ffig)

df_res = myclass.finish(res)
# myclass.plot_budget_time(df_res)

toprint = ['name', 'Nf_UD', 'Ns_UD', 'nvisits_UD_night',
           'g', 'r', 'i', 'z', 'y', 'delta_z',
           'nvisits_DD_season', 'budget']

df_res = df_res.round({'budget': 2})
print(df_res[toprint])
df_res[toprint].to_csv('ddf_res1.csv', index=False)


### m5_resu ###

m5_resu = pd.DataFrame()

Nseasons = 9
for vv in res:
    Nvisits = vv['nvisits_DD_season']*Nseasons
    res = m5class.m5_from_Nvisits(Nvisits=Nvisits)
    res['name'] = vv['name']
    res['Nseasons'] = Nseasons
    m5_resu = pd.concat((m5_resu, res))

print('iii', m5_resu)

"""
idx = m5_resu['name'] != 'DDF_SCOC'
m5_resu = m5_resu[idx]
m5_resu = m5_resu.round({'m5': 2, 'delta_m5': 2})
print(m5_resu.columns)
"""

# estimate the number of visits per obs night for DD
# check impact on m5
sl_DD = 180.
cad_DD = 3.
resa, resb = m5class.m5_band_from_Nvisits(m5_resu, sl_DD=sl_DD, cad_DD=cad_DD)

print(resa)
print(resa.columns)
print(resb)

# which m5 for UD fields? and comparison to y1 and y2_y10 requirements
# grab scenarios

print(df_res.columns)
# get u-visits from z-visits
df_res['u_season'] = frac_moon*df_res['z_season']
# df_res['z_season'] = (1.-frac_moon)*df_res['z_season']
df_res['n_night_season'] = df_res['sl']/df_res['cad']
df_res['u_recalc'] = df_res['u_season']/frac_moon/df_res['n_night_season']
df_res['z_recalc'] = df_res['z_season']/(1.-frac_moon)/df_res['n_night_season']
print(df_res[['g', 'r', 'i', 'z', 'y', 'u_recalc', 'z_recalc', 'z_season']])

check_m5(df_res, m5class)
# transform df_res

r = []

bbval = ['name', 'zcomp', 'Nf_UD', 'Ns_UD', 'nvisits_UD_night',
         'nvisits_DD_season', 'cad', 'sl', 'zcomp_new', 'Nvisits', 'delta_z',
         'nvisits_UD_night_recalc', 'nvisits_UD_season', 'n_night_season']

"""
'delta_z', 'g', 'r', 'i',
       'z', 'y', 'nvisits_UD_night_recalc', 'nvisits_UD_season', 'g_season',
       'r_season', 'i_season', 'z_season', 'y_season', 'Nvisits'
"""
for io, row in df_res.iterrows():
    ra = []
    for vv in bbval:
        ra.append(row[vv])
    for b in 'grizy':
        r.append(ra + [b, row[b]])
    # add a uband
    name = row['name']
    ipo = m5_resu['name'] == name
    ipo &= m5_resu['band'] == 'u'
    sel = m5_resu[ipo]
    nu_night = sel['Nvisits']/sel['Nseasons']
    n_nights = sl_UD/cad_UD
    nu_night /= (frac_moon*n_nights)
    r.append(ra+['u', np.round(nu_night[0], 0)])

print(r)
df_resb = pd.DataFrame(r, columns=bbval+['band', 'nvisits_night'])

print(df_resb.columns)

"""
vv = m5class.msingle_calc

print(vv.columns)

vv['Nvisits_y1'] = vv['Nvisits_y1'].astype(int)
vv['Nvisits_y2_y10'] = vv['Nvisits_y2_y10'].astype(int)
cad = 4
sl = 180
vv['Nvisits_y1_night'] = vv['Nvisits_y1']*cad/sl
vv['Nvisits_y2_y10_night'] = vv['Nvisits_y2_y10']*cad/sl
vv['Nvisits_y1_night'] = vv['Nvisits_y1_night'].astype(int)
vv['Nvisits_y2_y10_night'] = vv['Nvisits_y2_y10_night'].astype(int)

po = vv[['band', 'm5_y1', 'Nvisits_y1', 'Nvisits_y1_night',
         'm5_y2_y10', 'Nvisits_y2_y10', 'Nvisits_y2_y10_night']]

po.to_csv('testons.csv', index=False)
"""

# now re-estimate the budget vs time

dfres = df_resb.groupby('name').apply(
    lambda x: get_budget(x, NDD, resa, resb, Nv_LSST)).reset_index()

print(dfres)

plot_budget_time(dfres)
plt.show()
