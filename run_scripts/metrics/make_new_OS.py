import numpy as np
from optparse import OptionParser
import pandas as pd
import numpy.lib.recfunctions as rf


class newOS:
    def __init__(self, sum_night, data_OS, outName):
        """
        class to make a "new" OS for an original one (data_OS)

        Parameters
        --------------
        sum_night: pandas df
           OS summary per night
        data_OS: numpy array
           original OS data

        """
        # grab original data types
        dtypes = data_OS.dtype

        # rename DDF if necessary
        self.ref_config = ['0u-2g-9r-37i-52z-21y',
                           '1u-2g-9r-37i-0z-21y', '2u-2g-9r-37i-0z-21y']
        new_config = '0u-2g-9r-37i-52z-21y'
        # grap the number of visits required per band
        self.Nvisits = self.grab_nvisits(new_config)

        #self.data = data_OS
        # extract non ddf data
        #idx = data_OS['note'].isin(sum_night['field'])
        idx = np.in1d(data_OS['note'], sum_night['field'].to_list())
        self.data_DD = pd.DataFrame(data_OS[idx])
        data_non_DD = data_OS[~idx]

        #idx = sum_night['field'] == 'DD:XMM_LSS'

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
        #print(trans.dtype, trans['note'])
        #print(data_non_DD.dtype, data_non_DD['note'])
        trans = np.concatenate((trans, data_non_DD[cols]))
        trans = rf.append_fields(
            trans, 'note', res['note'].to_list()+data_non_DD['note'].tolist())
        trans = rf.append_fields(
            trans, 'band', res['band'].to_list()+data_non_DD['band'].tolist())

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

        print(len(np.unique(sum_night['night'])),
              len(np.unique(data_DDF['night'])))

        res = sum_night.groupby('season').apply(
            lambda x: self.make_new_OS_season(x, data_DDF)).reset_index(drop=True)

        res['note'] = DDF

        return res

    def make_new_OS_season(self, sum_night_season, data_DDF):

        season = sum_night_season.name

        # select data corresponding to this season
        idx = data_DDF['night'].isin(sum_night_season['night'])
        sel_DDF = data_DDF[idx]

        # grab median values per seasons
        colint = sel_DDF.select_dtypes(include=['int64']).columns
        meds = sel_DDF.groupby('band').median().reset_index()
        for col in colint:
            meds[col] = meds[col].astype(int)

        meds = meds.drop(columns=['night'])

        # get the list of nights with the "good" and "bad" filter combination
        idf = sum_night_season['config'].isin(self.ref_config)
        good_nights = sum_night_season[idf]['night']
        bad_nights = sum_night_season[~idf]['night']

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

        return new_data

    def make_new_OS_night(self, grp, meds):

        night = grp.name

        # now build OS

        totdf = pd.DataFrame()
        for key, nv in self.Nvisits.items():
            idx = meds['band'] == key
            df = meds[idx]
            if not df.empty:
                newdf = pd.DataFrame()

                if nv > 0:
                    newdf = pd.concat([df]*nv, ignore_index=True)
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

nOS = newOS(sum_night, data_OS, outName)
