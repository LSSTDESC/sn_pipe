import yaml
import numpy as np
from sn_tools.sn_cadence_tools import GenerateFakeObservations
from optparse import OptionParser
import numpy.lib.recfunctions as rf

parser = OptionParser()

parser.add_option("--config", type="str", default='input/Fake_cadence/Fake_cadence_seqs.yaml',
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

# print(config)
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
