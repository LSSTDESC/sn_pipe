import numpy as np
import yaml
import os
from optparse import OptionParser
import multiprocessing


class MakeYamlFake:
    """
    class to generate yaml files used as input to generate fake observations

    Parameters
    ---------------
    zval: float
      redshift values
    fake_orig: str
      original yaml file used to generate the new one
    fakeDir: str
      directory where the generated file will be placed

    """

    def __init__(self, zval, fake_orig, fakeDir):

        self.zval = zval
        self.fake_name = 'Fake_cadence_{}.yaml'.format(zval)
        self.fake_dir = fakeDir
        self.fake_orig = fake_orig
        self.gen_yaml()

    def gen_yaml(self):
        """
        Method to generate the yaml file

        """
        mjd_min = -21.*(1.+self.zval)
        mjd_max = 63.*(1.+self.zval)
        duration = (mjd_max-mjd_min)
        cad = 0.1*(1.+self.zval)

        with open(self.fake_orig, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('duration', str(duration))
            filedata = filedata.replace('mymin', str(mjd_min))
            filedata = filedata.replace('cadvalue', str(cad))
        with open('{}/{}'.format(self.fake_dir, self.fake_name), 'w') as file:
            file.write(filedata)


class MakeYamlSimulation:
    """
    class to generate yaml file used as input for LC simulation.

    Parameters
    ---------------
    x1: float
      SN x1
    color: float
      SN color
    z: float
      redshift
    bluecutoff: float
      blue cutoff for SN
    redcutoff: float
      red cutoff for SN
    config_orig: str
      original yaml file used to generate the new yaml file
    outDir: str
       dir where the generated yaml file will be placed
    datatoprocess: str
      data to be used by the simulation process (full path)
    outDirSimu: str
        dir where simulations will be copied.


    """

    def __init__(self, x1, color, z, ebvofMW, bluecutoff, redcutoff,error_model,
                 config_orig, outDir, datatoprocess, outDirSimu):
        # grab parameters
        self.x1 = x1
        self.color = color
        self.z = z
        self.ebvofMW = ebvofMW
        self.outDir = outDir
        self.config_orig = config_orig
        self.config_out = 'params_fakes_{}_{}_{}.yaml'.format(x1, color, z)
        self.datatoprocess = datatoprocess
        self.outDirSimu = outDirSimu
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        self.error_model = error_model

        self.makeYaml()

    def makeYaml(self):

        cutoff = '{}_{}'.format(self.bluecutoff, self.redcutoff)
        if self.error_model > 0:
            cutoff = self.error_model
        prodid = 'Fake_{}_{}_{}_{}_ebvofMW_{}'.format(
            self.x1, self.color, self.z, cutoff, self.ebvofMW)
        print('reading', self.config_orig)
        with open(self.config_orig, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('zval', str(self.z))
            filedata = filedata.replace('prodid', prodid)
            filedata = filedata.replace('datatoprocess', self.datatoprocess)
            filedata = filedata.replace('x1val', str(self.x1))
            filedata = filedata.replace('colorval', str(self.color))
            filedata = filedata.replace('outdirname', self.outDirSimu)
            filedata = filedata.replace('ebvofMWval', str(self.ebvofMW))
            filedata = filedata.replace('bluecutoffval', str(self.bluecutoff))
            filedata = filedata.replace('redcutoffval', str(self.redcutoff))
            filedata = filedata.replace('error_val', str(self.error_model))

        with open('{}/{}'.format(self.outDir, self.config_out), 'w') as file:
            file.write(filedata)


def SimuFakes(x1, color, z, ebvofMW, bluecutoff, redcutoff, error_model,outDir):
    """
    method to simulate LC from fake observations

    Parameters
    --------------
    x1: float
      SN x1
    color: float
       SN color
    z: float
       SN redshift
    ebvofMW: float
      ebv of MW
    bluecutoff: float
      blue cutoff for SN
    redcutoff: float
      red cutoff for SN
    outDir: str
      output directory
    """

    """
    # create output directory
    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    """

    # List of requested directories
    fake_obs_yaml = '{}/fake_obs_yaml'.format(outDir)
    fake_obs_data = '{}/fake_obs_data'.format(outDir)
    fake_simu_yaml = '{}/fake_simu_yaml'.format(outDir)
    fake_simu_data = '{}/fake_simu_data'.format(outDir)

    """
    # create these directory if necessary
    for vv in [fake_obs_yaml, fake_obs_data, fake_simu_yaml, fake_simu_data]:
        if not os.path.isdir(vv):
            os.makedirs(vv)
    """
    z = np.round(z, 2)
    x1 = np.round(x1, 1)
    color = np.round(color, 1)

    # first step: generate fake datas
    # make yaml file
    fake_orig = 'input/templates/Fake_cadence_template.yaml'
    fakes = MakeYamlFake(z, fake_orig, fake_obs_yaml)
    # generate fakes
    fake_config = '{}/{}'.format(fake_obs_yaml, fakes.fake_name)
    fake_npy = '.'.join(fakes.fake_name.split('.')[:-1])
    fake_output = '{}/{}'.format(fake_obs_data, fake_npy)
    cmd = 'python run_scripts/fakes/make_fake.py --config {} --output {}'.format(
        fake_config, fake_output)
    os.system(cmd)

    # second step: perform simulation
    # make input yaml file
    config_orig = 'input/templates/param_fakesimulation_template.yaml'
    genfakes = MakeYamlSimulation(
        x1, color, z, ebvofMW, bluecutoff, redcutoff, error_model,config_orig, fake_simu_yaml, '{}.npy'.format(fake_output), fake_simu_data)

    # run simulation on this
    cmd = 'python run_scripts/simulation/run_simulation_from_yaml.py'
    cmd += ' --config_yaml {}/{}'.format(genfakes.outDir, genfakes.config_out)
    cmd += ' --radius 0.01'
    cmd += ' --RAmin 0.'
    cmd += ' --RAmax 0.1'
    cmd += ' --npixels 1'
    cmd += ' --pixelmap_dir \'\''
    os.system(cmd)


class MultiSimuFakes:
    """
    class to multiprocess fake obs and make LC curves

    Parameters
    --------------
    x1: float
     SN x1
    color: float
      SN color
    zmin: float
      min redshift
    zmax: float
      max redshift
    zstep: float
      step for redshift
    nproc: int
      number of procs
    outDir: str
      output directory
    ebvofMW: float
      ebv of MW 
    bluecutoff: float
      blue cutoff for SN
    redcutoff: float
      red cutoff for SN
    """

    def __init__(self, x1, color, zmin, zmax, zstep, nproc, outDir, ebvofMW, bluecutoff, redcutoff,error_model):

        self.x1 = x1
        self.color = color
        self.outDir = outDir
        self.ebvofMW = ebvofMW
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        self.error_model = error_model

        zrange = np.arange(zmin, zmax, zstep)

        nz = len(zrange)

        batch = np.linspace(0, nz, nproc+1, dtype='int')
        result_queue = multiprocessing.Queue()

        for i in range(nproc):
            ida = batch[i]
            idb = batch[i+1]
            p = multiprocessing.Process(
                name='Subprocess-'+str(i), target=self.zLoop, args=(np.array([zrange[ida:idb]])))
            p.start()
            print('start', i)

    def zLoop(self, zval):
        """
        Method to process a set of redshift values

        Parameters
        ---------------
        zval: list(float)
          redshift values
        """
        if isinstance(zval, np.float):
            zval = [zval]
        for z in zval:
            SimuFakes(self.x1, self.color, z,
                      self.ebvofMW, self.bluecutoff, self.redcutoff,self.error_model, self.outDir)


parser = OptionParser()
parser.add_option("--x1", type="float", default=0.0, help="filter [%default]")
parser.add_option("--color", type="float", default=0.0,
                  help="filter [%default]")
parser.add_option("--nproc", type="int", default=1, help="filter [%default]")
parser.add_option("--zmin", type="float", default=0.01, help="zmin [%default]")
parser.add_option("--zmax", type="float", default=1.0, help="zmax [%default]")
parser.add_option("--zstep", type="float",
                  default=0.01, help="zstep [%default]")
parser.add_option("--outDir", type="str",
                  default='Test_fakes', help=" output directory [%default]")
parser.add_option("--ebvofMW", type=float,
                  default=0, help="ebvofMW to apply [%default]")
parser.add_option("--bluecutoff", type=float,
                  default=380, help="blue cutoff for SN[%default]")
parser.add_option("--redcutoff", type=float,
                  default=800, help="blue cutoff for SN[%default]")
parser.add_option("--error_model", type=int,
                  default=0, help="error model for SN[%default]")


opts, args = parser.parse_args()


  
# create output directory
    
if not os.path.isdir(opts.outDir):
        os.makedirs(opts.outDir)

# List of requested directories
fake_obs_yaml = '{}/fake_obs_yaml'.format(opts.outDir)
fake_obs_data = '{}/fake_obs_data'.format(opts.outDir)
fake_simu_yaml = '{}/fake_simu_yaml'.format(opts.outDir)
fake_simu_data = '{}/fake_simu_data'.format(opts.outDir)

# create these directory if necessary
for vv in [fake_obs_yaml, fake_obs_data, fake_simu_yaml, fake_simu_data]:
    if not os.path.isdir(vv):
        os.makedirs(vv)


MultiSimuFakes(opts.x1, opts.color, opts.zmin,
               opts.zmax, opts.zstep, opts.nproc,
               opts.outDir, opts.ebvofMW,
               opts.bluecutoff, opts.redcutoff,opts.error_model)
