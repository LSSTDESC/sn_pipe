import numpy as np
from optparse import OptionParser
import pandas as pd
import numpy.lib.recfunctions as rf
import yaml
from pathlib import Path
import operator


class newOS:
    def __init__(self, sum_night, data_OS, outDir, outName, ref_config, lunar_phase=20, no_dithering=False, medobs=False, add_nightly=False, median_ref=dict(zip('ugrizy', [23.38, 24.49, 24.04, 23.6, 22.98, 22.14]))):
        """
        class to make a "new" OS for an original one (data_OS)

        Parameters
        --------------
        sum_night: pandas df
           OS summary per night
        data_OS: numpy array
           original OS data
        outDir: str
          output directory
        outName: str
          file outputName
        ref_config: dict
          visits config for the fields considered
        lunar_phase: float, opt
          lunar phase threshold for u-band swapping
        median_ref: dict, opt
          median m5 values for grizy bands
        """
        # grab original data types
        dtypes = data_OS.dtype

        # this is for obsid counting
        self.count = 1

        self.lunar_phase = 100.*lunar_phase
        self.median_ref = median_ref
        self.ref_config = ref_config
        self.medobs = medobs

        # self.data = data_OS
        # extract non ddf data
        # idx = data_OS['note'].isin(sum_night['field'])
        idx = np.in1d(data_OS['note'], sum_night['field'].to_list())
        self.data_DD = pd.DataFrame(data_OS[idx])
        data_non_DD = data_OS[~idx]

        # idx = sum_night['field'] == 'DD:XMM_LSS'

        res = sum_night.groupby('field').apply(
            lambda x: self.make_new_OS(x)).reset_index()

        if add_nightly:
            print(res.columns)
            resb = res.merge(sum_night[['night', 'season']], left_on=[
                'night'], right_on=['night'])
            resb = resb.groupby(['note', 'season']).apply(
                lambda x: self.add_nights(x))
            print('hello', resb)

        if no_dithering:
            res = res.groupby('field').apply(
                lambda x: self.nodith(x)).reset_index()

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
        # print(data_non_DD[cols].dtype)
        """
        trans = np.concatenate((trans, data_non_DD[cols]))
        trans = rf.append_fields(
            trans, 'note', res['note'].to_list()+data_non_DD['note'].tolist())
        trans = rf.append_fields(
            trans, 'band', res['band'].to_list()+data_non_DD['band'].tolist())
        """
        trans = rf.append_fields(trans, 'note', res['note'].to_list())
        trans = rf.append_fields(trans, 'band', res['band'].to_list())

        # include lunar_phase value in outName
        dd = outName.split('_')
        if no_dithering:
            dd.insert(-2, 'nodith')
        if self.medobs:
            dd.insert(-2, 'medobs')
        dd.insert(-2, 'lp{}'.format(np.round(lunar_phase, 2)).ljust(6, '0'))

        outName = '_'.join(dd)
        np.save('{}/{}'.format(outDir, outName), np.copy(trans))

    def add_nights(self, grp):

        from scipy.interpolate import interp1d
        from astroplan import Observer
        from astropy.time import Time
        apo = Observer.at_site('APO')

        obs_night = grp.groupby(
            'night')['moonPhase', 'mjd'].median().reset_index()
        interp_night = interp1d(
            obs_night['night'], obs_night['mjd'], bounds_error=False, fill_value=0.)

        night_min = grp['night'].min()
        night_max = grp['night'].max()
        obs_allnight = pd.DataFrame(
            range(night_min, night_max), columns=['night'])
        obs_allnight['mjd'] = interp_night(obs_allnight['night'])
        idx = obs_allnight['night'].isin(np.unique(grp['night']))

        new_nights = obs_allnight[~idx]
        times = Time(new_nights['mjd'], format='mjd', scale='utc')
        new_nights['moonPhase'] = 100.*apo.moon_illumination(times)
        print(new_nights)

        idx = new_nights['moonPhase'] <= self.lunar_phase
        data_moon = new_nights[idx]
        data_nomoon = new_nights[~idx]

        new_data = pd.DataFrame()
        # select configuration for cadence
        config_cadence = self.ref_config[grp.name[0]]
        grpb = grp.drop(columns=['level_1', 'season'])
        if len(data_moon) > 0:
            meds, meds_all = self.get_meds(
                grpb, moon_cut=self.lunar_phase, op=operator.le)
            print('processing data moon - add night')
            res = data_moon.groupby('night').apply(
                lambda x: self.make_new_OS_night(x, meds_moon, config_cadence['lunar_config'], meds=meds, meds_all=meds_all)).reset_index(level=0)
            new_data = pd.concat((new_data, res))
        if len(data_nomoon) > 0:
            meds, meds_all = self.get_meds(
                grpb, moon_cut=self.lunar_phase, op=operator.gt)
            print('processing data no moon - add night')
            res = data_nomoon.groupby('night').apply(
                lambda x: self.make_new_OS_night(x, meds_nomoon, config_cadence['config'], meds=meds, meds_all=meds_all)).reset_index(level=0)
            new_data = pd.concat((new_data, res))

        return new_data

    def nodith(self, grp, fieldRA='RA', fieldDec='Dec'):
        """
        Method to remove the dithering

        Parameters
        ---------------
        grp: pandas group
          data to process
        fieldRA: str, opt
          field RA col (default: RA)
        fieldDec: str, opt
          field Dec col (default: Dec)

        Returns
        -----------
        pandas df with mean fieldRA and fieldDec; other columns unchanged

        """
        grp[fieldRA] = grp[fieldRA].mean()
        grp[fieldDec] = grp[fieldDec].mean()

        return grp

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

    def make_new_OS_night(self, grp, meds_season, config, meds=None, meds_all=None):

        night = grp.name

        # deltaT between two visits same band: 36 sec
        deltaT_visit_band = 36./(24.*3600.)
        # deltaT between two visits two bands: 2.56 sec
        deltaT_visit_band_seq = 2.56/(24.*60.)

        # now build OS
        totdf = pd.DataFrame()

        mjd_min = grp['mjd'].min()

        if meds is None:
            meds, meds_all = self.get_meds(grp)

        visits = self.grab_nvisits(config)

        for key, nv in visits.items():
            if nv > 0:
                idx = meds['band'] == key
                df = meds[idx]
                if df.empty:
                    df = meds_all
                    df['band'] = key
                    # df['note'] = grp['note'].unique()
                    io = meds_season['band'] == key
                    selmeds = meds_season[io]
                    if len(selmeds) > 0:
                        df['fiveSigmaDepth'] = selmeds['fiveSigmaDepth'].to_list()

                if self.medobs:
                    df['fiveSigmaDepth'] = self.median_ref[key]
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

    def get_meds(self, grpa, moonCol='moonPhase', moon_cut=-100., op=operator.ge):

        idx = op(grpa[moonCol], moon_cut)
        grp = grpa[idx]

        meds = grp.groupby('band').median().reset_index()
        selmed = grp.drop(columns=['band', 'note', 'night'])

        print('there man', len(meds.columns), len(selmed.columns))
        print(meds.columns)

        meds_all = pd.DataFrame(
            [selmed.median().to_list()], columns=selmed.columns)
        colint = grp.select_dtypes(include=['int64']).columns

        for col in colint:
            meds[col] = meds[col].astype(int)
            if col in meds_all.columns:
                meds_all[col] = meds_all[col].astype(int)

        meds = meds.drop(columns=['night'])

        return meds, meds_all


parser = OptionParser()
parser.add_option("--sumNightName", type="str", default='Summary_night_ddf_early_deep_slf0.20_f10.60_f20.80_v2.1_10yrs.hdf5',
                  help="summary file name[%default]")
parser.add_option("--lunar_phase", type=float, default=0.40,
                  help="lunar phase for night with Moon [%default]")
parser.add_option("--no_dithering", type=int, default=0,
                  help="OS with no dithering [%default]")
parser.add_option("--medobs", type=int, default=0,
                  help="median observing conditions (m5) [%default]")
parser.add_option("--add_nightly", type=int, default=0,
                  help="to add nightly visits if necessary [%default]")
parser.add_option("--config", type=str, default='config_newOS.yaml',
                  help="config file [%default]")

opts, args = parser.parse_args()
sumNightName = opts.sumNightName
lunar_phase = float(opts.lunar_phase)
no_dithering = opts.no_dithering
medobs = opts.medobs
add_nightly = opts.add_nightly
yaml_file = opts.config

summary_night = pd.read_hdf(sumNightName)

# load config parameters
yaml_dict = yaml.safe_load(Path(yaml_file).read_text())
dbName = yaml_dict['dbName']

# grab dbName data
idx = summary_night['dbName'] == dbName
sum_night = summary_night[idx]

if len(sum_night) == 0:
    print('Problem here: night summary file could not be found. Stop.')


# load original OS
dbDir = yaml_dict['dbDir']
data_OS = np.load('{}/{}.npy'.format(dbDir, dbName), allow_pickle=True)

print(np.unique(data_OS['note']))

idd = sum_night['field'] == 'DD:COSMOS'
ref_config = yaml_dict['fields_visits']
outName = yaml_dict['outName']
outDir = yaml_dict['outDir']

nOS = newOS(sum_night[idd],
            data_OS,
            outDir,
            outName,
            ref_config,
            lunar_phase=lunar_phase,
            no_dithering=no_dithering,
            medobs=medobs,
            add_nightly=add_nightly)
