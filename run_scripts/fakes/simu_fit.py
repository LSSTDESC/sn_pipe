from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config, make_dict_from_optparse
from sn_tools.sn_cadence_tools import GenerateFakeObservations
import numpy.lib.recfunctions as rf


class FakeObservations:
    """
    class to generate fake observations

    Parameters
    ----------------
    dict_config: dict
      configuration parameters

    """

    def __init__(self, dict_config):

        self.dd = dict_config

        # transform input conf dict
        self.transform_fakes()

        # generate fake observations

        self.obs = self.genData()

    def transform_fakes(self):
        """
        Method to transform the input dict
        to make it compatible with the fake observation generator

        """
        # few changes to be made here: transform some of the input to list
        for vv in ['seasons', 'seasonLength']:
            what = self.dd[vv]
            if '-' not in what or what[0] == '-':
                nn = list(map(int, what.split(',')))
                print('ici', nn)
            else:
                nn = list(map(int, what.split('-')))
                nn = range(np.min(nn), np.max(nn))
            self.dd[vv] = nn

        for vv in ['MJDmin']:
            what = self.dd[vv]
            if '-' not in what or what[0] == '-':
                nn = list(map(float, what.split(',')))
            else:
                nn = list(map(float, what.split('-')))
                nn = range(np.min(nn), np.max(nn))
            self.dd[vv] = nn

    def genData(self):
        """
        Method to generate fake observations

        Returns
        -----------
        numpy array with fake observations

        """

        mygen = GenerateFakeObservations(self.dd).Observations
        # add a night column

        mygen = rf.append_fields(mygen, 'night', list(range(1, len(mygen)+1)))

        # add Ra, dec columns
        mygen = rf.append_fields(mygen, 'Ra', mygen['fieldRA'])
        mygen = rf.append_fields(mygen, 'RA', mygen['fieldRA'])
        mygen = rf.append_fields(mygen, 'Dec', mygen['fieldRA'])
        # print(mygen.dtype)
        return mygen


# this is to load option for fake cadence
path = 'input/Fake_cadence'
confDict = make_dict_from_config(path, 'config_cadence.txt')

parser = OptionParser()

# add option for Fake data here
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0], vals[1]))
    parser.add_option('--{}'.format(key), help='{} [%default]'.format(
        vals[2]), default=vv, type=vals[0], metavar='')

parser.add_option(
    '--outputDir', help='main output directory [%default]', default='/sps/lsst/users/gris/config_zlim', type=str)

opts, args = parser.parse_args()


# make the fake config file here
newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key] = (vals[0], newval)

dd = make_dict_from_optparse(newDict)

fakeData = FakeObservations(dd).obs

print(fakeData)
