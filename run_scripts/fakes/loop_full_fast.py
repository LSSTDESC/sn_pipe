import numpy as np
import os
from optparse import OptionParser
from sn_tools.sn_io import make_dict_from_config,make_dict_from_optparse
import yaml

def run(x1,color,simus,ebv,bluecutoff,redcutoff,error_model,fake_config,snrmin,nbef,naft,zmax):
    """
    Method used to perform a set of runs (simulation+fit)

    Parameters
    ---------------
    x1: float
      SN x1
    color: float
      SN color
    simus: list(str)
     list of the simulators to use
    ebv: float
      ebvofMW value
    bluecutoff: float
      blue cutoff for SN
    redcutoff: float
     red cutoff for SN
    error_model: int
      to activate error model for LC points error
    fake_config: str
      reference config file to generate fakes
    snrmin: float
     SNR min value for LC points to be fitted
    nbef: int
      min number of points before max for LC points to be fitted
    naft: int
      min number of points after max for LC points to be fitted
    zmax: float
      max redshift value for fake generation

    """
    
    cutoff = '{}_{}'.format(bluecutoff,redcutoff)
    if error_model:
        cutoff = 'error_model'
    
    outDir_simu = 'Output_Simu_{}_ebvofMW_{}'.format(cutoff,ebv)
    outDir_fit = 'Output_Fit_{}_ebvofMW_{}_snrmin_{}'.format(cutoff,ebv,int(snrmin))
    
    cmd_comm = 'python run_scripts/fakes/full_simulation_fit.py  --ebvofMW {}'.format(ebv)
    cmd_comm += ' --outDir_simu {} --outDir_fit {}'.format(outDir_simu,outDir_fit)
    cmd_comm += ' --bluecutoff {} --redcutoff {}'.format(bluecutoff,redcutoff) 
    cmd_comm += ' --fake_config {} --x1 {} --color {} --error_model {}'.format(fake_config, x1, color,error_model)
    cmd_comm += ' --snrmin {} --nbef {} --naft {}'.format(snrmin,nbef,naft)
    cmd_comm += ' --zmax {}'.format(zmax)

    for simu in simus:
            cmd = cmd_comm
            cmd += ' --simulator {}'.format(simu)
            os.system(cmd)

# this is to load option for fake cadence
path = 'input/Fake_cadence'
confDict = make_dict_from_config(path,'config_cadence.txt')

parser = OptionParser()

parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1 [%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color [%default]")
parser.add_option("--zmax", type=float, default=0.8,
                  help="max redshift for simulated data [%default]")
parser.add_option("--ebv", type=float, default=0.0,
                  help="ebvofMW value[%default]")
parser.add_option("--bluecutoff", type=float, default=380.,
                  help="blue cutoff for SN spectrum[%default]")
parser.add_option("--redcutoff", type=float, default=800.,
                  help="red cutoff for SN spectrum[%default]")
parser.add_option("--simus", type=str, default='sn_fast,sn_cosmo',
                  help=" simulator to use[%default]")
parser.add_option("--snrmin", type=float, default=1.,
                  help="SNR min for LC points (fit)[%default]")
parser.add_option("--nbef", type=int, default=4,
                  help="min n LC points before max (fit)[%default]")
parser.add_option("--naft", type=int, default=10,
                  help="min n LC points after max (fit)[%default]")

#add option for Fake data here
for key, vals in confDict.items():
    vv = vals[1]
    if vals[0] != 'str':
        vv = eval('{}({})'.format(vals[0],vals[1]))
    parser.add_option('--{}'.format(key),help='{} [%default]'.format(vals[2]),default=vv,type=vals[0],metavar='')

parser.add_option('--fake_config',help='output file name [%default]',default='Fake_cadence.yaml',type='str')

opts, args = parser.parse_args()


# make the fake config file here
newDict = {}
for key, vals in confDict.items():
    newval = eval('opts.{}'.format(key))
    newDict[key]=(vals[0],newval)

dd = make_dict_from_optparse(newDict)
# few changes to be made here: transform some of the input to list
for vv in ['seasons','seasonLength']:
    what = dd[vv]
    if '-' not in what or what[0] == '-':
        nn = list(map(int,what.split(',')))
        print('ici',nn)
    else:
        nn = list(map(int,what.split('-')))
        nn = range(np.min(nn),np.max(nn))
    dd[vv] = nn

for vv in ['MJDmin']:
    what = dd[vv]
    if '-' not in what or what[0] == '-':
        nn = list(map(float,what.split(',')))
    else:
        nn = list(map(float,what.split('-')))
        nn = range(np.min(nn),np.max(nn))
    dd[vv] = nn


#print('boo',yaml.safe_dump(dd))

#print('config',dd)
with open(opts.fake_config, 'w') as f:
    data = yaml.safe_dump(dd, f)

#fake_config = 'input/Fake_cadence/Fake_cadence.yaml'

x1 = opts.x1
color = opts.color
zmax = opts.zmax
bluecutoff = opts.bluecutoff
redcutoff = opts.redcutoff
ebv = opts.ebv
snrmin = opts.snrmin
nbef = opts.nbef
naft = opts.naft

simus = list(map(str,opts.simus.split(',')))

# case error_model=1
run(x1,color,simus,ebv,bluecutoff,redcutoff,1,opts.fake_config,snrmin,nbef,naft,zmax)
# case error_model=0
#run(x1,color,simus,ebv,bluecutoff,redcutoff,0,opts.fake_config,snrmin,nbef,naft,zmax)
