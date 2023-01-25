import numpy as np
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config
from sn_tools.sn_fake_utils import FakeObservations, add_option, config
import sn_script_input


# this is to load option for fake cadence
path = sn_script_input.__path__
confDict_fake = make_dict_from_config(path[0], 'config_cadence.txt')


parser = OptionParser(description='Script to generate fake observations.')
add_option(parser, confDict_fake)

opts, args = parser.parse_args()

config_fake = config(confDict_fake, opts)

# generate fake obs
fakeData = FakeObservations(config_fake).obs
print('fake Data', len(fakeData))

if opts.saveData:
    # save script parameters
    outDir = opts.outDir
    outName = opts.outName
    from sn_tools.sn_io import checkDir
    checkDir(outDir)
    import yaml
    outputyaml = '{}/{}.yaml'.format(outDir, outName)
    with open(outputyaml, 'w') as file:
        documents = yaml.dump(config_fake, file)
    # save fake obs
    path = '{}/{}.npy'.format(outDir, outName)
    np.save(path, np.copy(fakeData))

"""
parser = OptionParser()

parser.add_option("--config", type="str",
                  default='input/Fake_cadence/Fake_cadence_seqs.yaml',
                  help="config file name[%default]")
parser.add_option("--output", type="str", default='Fake_DESC',
                  help="output file name[%default]")
parser.add_option("--seqs", type="int", default='0',
                  help="sequences for generations[%default]")

opts, args = parser.parse_args()

configName = opts.config

outputName = opts.output
seqs = opts.seqs

config = yaml.load(open(configName), Loader=yaml.FullLoader)

print(config)
mygen = GenerateFakeObservations(config, sequences=seqs).Observations

# print(mygen)


# add a night column

mygen = rf.append_fields(mygen, 'night', list(range(1, len(mygen)+1)))

# add Ra, dec columns
mygen = rf.append_fields(mygen, 'Ra', mygen['fieldRA'])
mygen = rf.append_fields(mygen, 'RA', mygen['fieldRA'])
mygen = rf.append_fields(mygen, 'Dec', mygen['fieldRA'])
# print(mygen.dtype)

np.save('{}.npy'.format(outputName), np.copy(mygen))
"""
