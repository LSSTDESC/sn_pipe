#from sn_mafsim.sn_maf_simulation import SNMAFSimulation
from sn_wrapper.sn_simu import SNSimulation
import numpy as np
import yaml
import os


class SimuWrapper:
    """
    Wrapper class for simulation

    Parameters
    ---------------



    """

    def __init__(self, dbDir, dbName, nside, nproc, diffflux,
                 seasnum, outDir, fieldType,
                 x1Type, x1min, x1max, x1step,
                 colorType, colormin, colormax, colorstep,
                 zType, zmin, zmax, zstep, simu, daymaxType, coadd):

        self.dbDir = dbDir
        self.dbName = dbName
        self.nside = nside
        self.nproc = nproc
        self.diffflux = diffflux
        self.seasnum = seasnum
        self.outDir = outDir
        self.fieldType = fieldType
        self.x1Type = x1Type
        self.x1min = x1min
        self.x1max = x1max
        self.x1step = x1step
        self.colorType = colorType
        self.colormin = colormin
        self.colormax = colormax
        self.colorstep = colorstep
        self.zmin = zmin
        self.zmax = zmax
        self.simu = simu
        self.zType = zType
        self.daymaxType = daymaxType
        self.coadd = coadd

        self.name = 'simulation'

        # create the config file
        config = self.makeYaml('input/simulation/param_simulation_gen.yaml')

        # get X0 for SNIa normalization
        x0_tab = self.x0(config)

        # load references if simulator = sn_fast
        reference_lc = self.load_reference(config)

        # now define the metric instance
        # self.metric = SNMAFSimulation(config=config, x0_norm=x0_tab,
        #                              reference_lc=reference_lc, coadd=config['Observations']['coadd'])
        self.metric = SNSimulation(
            config=config, x0_norm=x0_tab)

    def x0(self, config):
        """
        Method to load x0 data

        Parameters
        ---------------
        config: dict
          parameters to load and (potentially) regenerate x0s

        Returns
        -----------

        """
        # check whether X0_norm file exist or not (and generate it if necessary)
        absMag = config['SN parameters']['absmag']
        salt2Dir = config['SN parameters']['salt2Dir']
        model = config['Simulator']['model']
        version = str(config['Simulator']['version'])

        x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)
        if not os.path.isfile(x0normFile):
            from sn_tools.sn_utils import X0_norm
            X0_norm(salt2Dir=salt2Dir, model=model, version=version,
                    absmag=absMag, outfile=x0normFile)

        return np.load(x0normFile)

    def load_reference(self, config):
        """
        Method to load reference LC for fast simulation

        Parameters
        ---------------
        config: dict
          parameters to load reference data if necessary

        Returns
        -----------
        reference data from class GetReference

        """

        ref_lc = None

        if 'sn_fast' in config['Simulator']['name']:
            print('Loading reference LCs from',
                  config['Simulator']['Reference File'])
            ref_lc = GetReference(
                config['Simulator']['Reference File'],
                config['Simulator']['Gamma File'],
                config['Instrument'])
            print('Reference LCs loaded')

        return ref_lc

    def makeYaml(self, input_file):
        """
        Method to generate a yaml file
        with parameters from generic input_file

        Parameters
        ---------------
        input_file: str
          input generic yaml file

        Returns
        -----------
        yaml file with parameters


        """
        with open(input_file, 'r') as file:
            filedata = file.read()

        prodid = '{}_{}_{}_seas_{}_{}_{}'.format(
            self.simu, self.fieldType, self.dbName, self.seasnum, self.x1min, self.colormin)
        fullDbName = '{}/{}.npy'.format(self.dbDir, self.dbName)
        filedata = filedata.replace('prodid', prodid)
        filedata = filedata.replace('fullDbName', fullDbName)
        filedata = filedata.replace('nnproc', str(self.nproc))
        filedata = filedata.replace('nnside', str(self.nside))
        filedata = filedata.replace('outputDir', self.outDir)
        filedata = filedata.replace('diffflux', str(self.diffflux))
        filedata = filedata.replace('seasval', str(self.seasnum))
        filedata = filedata.replace('ftype', self.fieldType)
        filedata = filedata.replace('x1Type', self.x1Type)
        filedata = filedata.replace('x1min', str(self.x1min))
        filedata = filedata.replace('x1max', str(self.x1max))
        filedata = filedata.replace('x1step', str(self.x1step))
        filedata = filedata.replace('colorType', self.colorType)
        filedata = filedata.replace('colormin', str(self.colormin))
        filedata = filedata.replace('colormax', str(self.colormax))
        filedata = filedata.replace('colorstep', str(self.colorstep))
        filedata = filedata.replace('zmin', str(self.zmin))
        filedata = filedata.replace('zmax', str(self.zmax))
        filedata = filedata.replace('zType', self.zType)
        filedata = filedata.replace('daymaxType', self.daymaxType)
        filedata = filedata.replace('fcoadd', str(self.coadd))

        return yaml.load(filedata, Loader=yaml.FullLoader)

    def run(self, obs):
        """
        Method to run the metric

        Parameters
        ---------------
        obs: array
          data to process

        """
        return self.metric.run(obs)

    def finish(self):
        """
        Method to save metadata to disk

        """
        self.metric.Finish()
