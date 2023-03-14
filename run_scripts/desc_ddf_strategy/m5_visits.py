import numpy as np
import pandas as pd


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
                    (b, np.median(selb['fiveSigmaDepth']), field.split(':')[-1]))

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


myclass = FiveSigmaDepth_Nvisits()

res = myclass.m5_from_Nvisits(Nvisits=1664*9)

print(res[['band', 'm5', 'delta_m5']])
