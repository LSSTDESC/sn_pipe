import numpy as np
from optparse import OptionParser
import pandas as pd
import numpy.lib.recfunctions as rf


class newOS:
    def __init__(self, sum_night, data_OS, outName, lunar_phase=20, median_ref=dict(zip('grizy', [24.49, 24.04, 23.6, 22.98, 22.14]))):
        """
        class to make a "new" OS for an original one (data_OS)

        Parameters
        --------------
        sum_night: pandas df
           OS summary per night
        data_OS: numpy array
           original OS data
        outName: str
          file outputName
        lunar_phase: float, opt
          lunar phase threshold for u-band swapping
        """
        # grab original data types
        dtypes = data_OS.dtype

        # this is for obsid counting
        self.count = 1

        self.lunar_phase = lunar_phase
        self.median_ref = median_ref

        self.ref_config = {}
        self.ref_config['DD:COSMOS'] = {}
        self.ref_config['DD:COSMOS']['config'] = '0u-2g-9r-37i-52z-21y'
        self.ref_config['DD:COSMOS']['lunar_config'] = '1u-2g-9r-37i-0z-21y'

        self.ref_config['DD:XMM_LSS'] = {}
        self.ref_config['DD:XMM_LSS']['config'] = '0u-2g-9r-37i-52z-21y'
        self.ref_config['DD:XMM_LSS']['lunar_config'] = '1u-2g-9r-37i-0z-21y'

        self.ref_config['DD:EDFS_a'] = {}
        self.ref_config['DD:EDFS_a']['config'] = '0u-2g-9r-1i-1z-1y'
        self.ref_config['DD:EDFS_a']['lunar_config'] = '1u-2g-9r-1i-0z-1y'

        self.ref_config['DD:EDFS_b'] = {}
        self.ref_config['DD:EDFS_b']['config'] = '0u-2g-9r-1i-1z-1y'
        self.ref_config['DD:EDFS_b']['lunar_config'] = '1u-2g-9r-1i-0z-1y'

        # self.data = data_OS
        # extract non ddf data
        # idx = data_OS['note'].isin(sum_night['field'])
        idx = np.in1d(data_OS['note'], sum_night['field'].to_list())
        self.data_DD = pd.DataFrame(data_OS[idx])
        data_non_DD = data_OS[~idx]

        # idx = sum_night['field'] == 'DD:XMM_LSS'

        res = sum_night.groupby('field').apply(
            lambda x: self.make_new_OS(x)).reset_index()

        for cc in ['note', 'band']:
            res[cc] = res[cc].astype(pd.StringDtype())

        res = res.drop(columns=['field'])
        res = res.reset_index()

        # transform to record
        cols = list(res.columns)
        cols.remove('note')
        cols.remove('band')
        cols.remove('level_1')
        cols.remove('index')

        trans = np.copy(res[cols].to_records(index=False))

        """
        trans = rf.append_fields(
            trans, 'note', res['note'].to_list(), dtypes=['<U15'])
        trans = rf.append_fields(trans, 'band', res['band'].to_list())
        """
        # add non ddf data
        # print(trans.dtype, trans['note'])
        # print(data_non_DD.dtype, data_non_DD['note'])
        print(trans.dtype)
        print(data_non_DD[cols].dtype)
        """
        trans = np.concatenate((trans, data_non_DD[cols]))
        trans = rf.append_fields(
            trans, 'note', res['note'].to_list()+data_non_DD['note'].tolist())
        trans = rf.append_fields(
            trans, 'band', res['band'].to_list()+data_non_DD['band'].tolist())
        """
        trans = rf.append_fields(trans, 'note', res['note'].to_list())
        trans = rf.append_fields(trans, 'band', res['band'].to_list())

        np.save(outName, np.copy(trans))

    def grab_nvisits(self, config):

        vv = config.split('-')
        pos = dict(zip('ugrizy', [0, 1, 2, 3, 4, 5]))
        Nvisits = {}
        for key, vals in pos.items():
            Nvisits[key] = int(vv[vals].split(key)[0])

        return Nvisits

    def make_new_OS(self, sum_night):

        DDF = sum_night.name

        # select data corresponding to this DDF
        idx = self.data_DD['note'] == DDF
        data_DDF = self.data_DD[idx]

        # select configuration for cadence
        config_cadence = self.ref_config[DDF]

        print(len(np.unique(sum_night['night'])),
              len(np.unique(data_DDF['night'])))

        res = sum_night.groupby('season').apply(
            lambda x: self.make_new_OS_season(x, data_DDF, config_cadence)).reset_index(drop=True)

        res['note'] = DDF

        return res

    def make_new_OS_season(self, sum_night_season, data_DDF, config_cadence):

        season = sum_night_season.name

        # select data corresponding to this season
        idx = data_DDF['night'].isin(sum_night_season['night'])
        sel_DDF = data_DDF[idx]

        # grab median values per seasons
        colint = sel_DDF.select_dtypes(include=['int64']).columns
        # get the list of nights depending on the moon phase
        idx = sel_DDF['moonPhase'] < self.lunar_phase

        what = ['fiveSigmaDepth', 'band']
        meds_moon = sel_DDF[idx][what].groupby('band').median().reset_index()
        meds_nomoon = sel_DDF[~idx][what].groupby(
            'band').median().reset_index()

        for b in 'grizy':
            idxx = meds_moon['band'] == b
            if len(meds_moon[idxx]) == 0:
                dd = pd.DataFrame([[b, self.median_ref[b]]], columns=[
                                  'band', 'fiveSigmaDepth'])
                meds_moon = pd.concat((meds_moon, dd))
            idxx = meds_nomoon['band'] == b
            if len(meds_nomoon[idxx]) == 0:
                dd = pd.DataFrame([[b, self.median_ref[b]]], columns=[
                                  'band', 'fiveSigmaDepth'])
                meds_nomoon = pd.concat((meds_nomoon, dd))
        """
        # get the list of nights with the "good" and "bad" filter combination
        idf = sum_night_season['config'].isin(self.ref_config)
        good_nights = sum_night_season[idf]['night']
        bad_nights = sum_night_season[~idf]['night']
        """

        data_moon = sel_DDF[idx]
        data_nomoon = sel_DDF[~idx]
        list1 = np.unique(data_moon['night'])
        list2 = np.unique(data_nomoon['night'])
        intersect = list(set(list1).intersection(list2))
        idx = data_nomoon['night'].isin(intersect)
        data_nomoon = data_nomoon[~idx]

        new_data = pd.DataFrame()
        if len(data_moon) > 0:
            print('processing data moon')
            res = data_moon.groupby('night').apply(
                lambda x: self.make_new_OS_night(x, meds_moon, config_cadence['lunar_config'])).reset_index(level=0)
            new_data = pd.concat((new_data, res))
        if len(data_nomoon) > 0:
            print('processing data no moon')
            res = data_nomoon.groupby('night').apply(
                lambda x: self.make_new_OS_night(x, meds_nomoon, config_cadence['config'])).reset_index(level=0)
            new_data = pd.concat((new_data, res))

        """
        # good nights: leave data as it is
        idx_good = sel_DDF['night'].isin(good_nights)
        new_data = sel_DDF[idx_good]

        # bad nights: make new OS from ref
        bad_data = sel_DDF[~idx_good]
        if len(bad_data) > 0:
            res = bad_data.groupby('night').apply(
                lambda x: self.make_new_OS_night(x, meds)).reset_index(level=0)

            print('test', res.columns)
            print('baoum', res['night'])
            nanc = res['night'].isnull().sum()
            if nanc > 0:
                print('probleme here', nanc)
                print(res['night'])
            new_data = pd.concat((new_data, res))
        """
        return new_data

    def make_new_OS_night(self, grp, meds_season, config):

        night = grp.name

        # deltaT between two visits same band: 36 sec
        deltaT_visit_band = 36./(24.*3600.)
        # deltaT between two visits two bands: 2.56 sec
        deltaT_visit_band_seq = 2.56/(24.*60.)
        # grab some infos such as: deltaT between visits (same band)
        """
        import matplotlib.pyplot as plt
        plt.plot(grp['mjd'], grp['band'], 'ko')
        plt.show()
        
        visit_time = {}
        for b in 'ugrizy':
            idx = grp['band'] == b
            sel = grp[idx]
            if len(sel) > 0:
                visit_time[b] = sel['mjd']

        min_visit_time = {}
        max_visit_time = {}
        for b in visit_time.keys():
            diff = np.diff(visit_time[b])
            print(b, np.mean(diff)*24.*3600, np.std(diff)*24.*3600)
            min_visit_time[b] = np.min(visit_time[b])
            max_visit_time[b] = np.max(visit_time[b])

        for bb in ['gi', 'ir', 'ry', 'yz']:
            if bb[1] in min_visit_time.keys() and bb[0] in min_visit_time.keys():
                dd = min_visit_time[bb[1]]-max_visit_time[bb[0]]
                print(bb, dd*24.*60.)
        """
        # now build OS
        totdf = pd.DataFrame()

        mjd_min = grp['mjd'].min()
        meds = grp.groupby('band').median().reset_index()
        selmed = grp.drop(columns=['band', 'note', 'night'])
        meds_all = pd.DataFrame(
            [selmed.median().to_list()], columns=selmed.columns)
        colint = grp.select_dtypes(include=['int64']).columns

        for col in colint:
            meds[col] = meds[col].astype(int)
            if col in meds_all.columns:
                meds_all[col] = meds_all[col].astype(int)

        meds = meds.drop(columns=['night'])

        visits = self.grab_nvisits(config)

        for key, nv in visits.items():
            if nv > 0:
                idx = meds['band'] == key
                df = meds[idx]
                if df.empty:
                    df = meds_all
                    df['band'] = key
                    df['note'] = grp['note'].unique()
                    io = meds_season['band'] == key
                    selmeds = meds_season[io]
                    if len(selmeds) > 0:
                        df['fiveSigmaDepth'] = selmeds['fiveSigmaDepth']

                newdf = pd.concat([df]*nv, ignore_index=True)
                obsid = list(range(self.count, self.count+nv))
                newdf['observationId'] = obsid
                mjds = np.arange(
                    mjd_min, mjd_min+nv*deltaT_visit_band, deltaT_visit_band)
                newdf['mjd'] = mjds[:nv]
                self.count += nv
                mjd_min += nv*deltaT_visit_band+deltaT_visit_band_seq
                totdf = pd.concat((totdf, newdf))

        return totdf


parser = OptionParser()
parser.add_option("--dbName", type="str", default='ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs',
                  help="db to process[%default]")
parser.add_option("--dbDir", type="str", default='../DB_Files',
                  help="location dir of db to process [%default]")
parser.add_option("--sumNightName", type="str", default='Summary_night_ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs.hdf5',
                  help="summary file name[%default]")
parser.add_option("--outName", type=str, default='test_newOS.npy',
                  help="output file name [%default]")

opts, args = parser.parse_args()

dbName = opts.dbName
dbDir = opts.dbDir
sumNightName = opts.sumNightName
outName = opts.outName

summary_night = pd.read_hdf(sumNightName)

# grab dbName data
idx = summary_night['dbName'] == dbName
sum_night = summary_night[idx]

if len(sum_night) == 0:
    print('Problem here: night summary file could not be found. Stop.')

# load original OS
data_OS = np.load('{}/{}.npy'.format(dbDir, dbName), allow_pickle=True)

print(np.unique(data_OS['note']))

idd = sum_night['field'] == 'DD:COSMOS'
nOS = newOS(sum_night, data_OS, outName)
