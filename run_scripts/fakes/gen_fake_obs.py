from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from fake_utils import FakeObservations, add_option, config
import os
import numpy as np
import yaml

# this is to load option for fake cadence
path = 'input/Fake_cadence'
confDict_fake = make_dict_from_config(path, 'config_cadence.txt')

parser = OptionParser()

# add option for Fake data here
add_option(parser, confDict_fake)

parser.add_option(
    '--outDir', help='output directory [%default]', default='Fake_Observations', type=str)
parser.add_option(
    '--outName', help='output file name [%default]', default='Fake_Obs', type=str)

opts, args = parser.parse_args()

outDir = opts.outDir
outName = opts.outName

# create outputDir
if not os.path.isdir(outDir):
    os.makedirs(outDir)

# make the config files here
config_fake = config(confDict_fake, opts)

configName = '{}/{}.yaml'.format(outDir, outName)

with open(configName, 'w') as fi:
    documents = yaml.dump(config_fake, fi)

fakeData = FakeObservations(config_fake).obs

fullName = '{}/{}.npy'.format(outDir, outName)
print(fakeData)

np.save(fullName, np.copy(fakeData))
