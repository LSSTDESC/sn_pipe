import yaml
import numpy as np
from sn_tools.sn_cadence_tools import GenerateFakeObservations
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--config", type="str", default='',
                  help="config file name[%default]")
parser.add_option("--output", type="str", default='',
                  help="output file name[%default]")
parser.add_option("--seqs", type="int", default='0',
                  help="sequences for generations[%default]")

opts, args = parser.parse_args()

configName = opts.config
outputName = opts.output
seqs = opts.seqs

if configName == '':
    configName = 'input/Fake_cadence/Fake_cadence_seqs.yaml'
if outputName == '':
    outputName='Fake_DESC'

config = yaml.load(open(configName), Loader=yaml.FullLoader)

print(config)
mygen = GenerateFakeObservations(config,sequences=seqs).Observations

print(mygen)
print(mygen.dtype)

np.save('{}.npy'.format(outputName),np.copy(mygen))
