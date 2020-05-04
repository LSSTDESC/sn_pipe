from sn_mafsim.sn_maf_simulation import SNMAFSimulation
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
                 seasnum, outDir, fieldType, x1, color,
                 zmin, zmax, simu, x1colorType, zType, daymaxType):

        self.dbDir = dbDir
        self.dbName = dbName
        self.nside = nside
        self.nproc = nproc
        self.diffflux = diffflux
        self.seasnum = seasnum
        self.outDir = outDir
        self.fieldType = fieldType
        self.x1 = x1
        self.color = color
        self.zmin = zmin
        self.zmax = zmax
        self.simu = simu
        self.x1colorType = x1colorType
        self.zType = zType
        self.daymaxType = daymaxType

        self.name = 'simulation'

        # create the config file
        config = self.makeYaml('input/simulation/param_simulation_gen.yaml')

        # get X0 for SNIa normalization
        x0_tab = self.x0(config)

        # load references if simulator = sn_fast
        reference_lc = self.load_reference(config)

        # now define the metric instance
        self.metric = SNMAFSimulation(config=config, x0_norm=x0_tab,
                                      reference_lc=reference_lc, coadd=config['Observations']['coadd'])

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
            self.simu, self.fieldType, self.dbName, self.seasnum, self.x1, self.color)
        fullDbName = '{}/{}.npy'.format(self.dbDir, self.dbName)
        filedata = filedata.replace('prodid', prodid)
        filedata = filedata.replace('fullDbName', fullDbName)
        filedata = filedata.replace('nnproc', str(self.nproc))
        filedata = filedata.replace('nnside', str(self.nside))
        filedata = filedata.replace('outputDir', self.outDir)
        filedata = filedata.replace('diffflux', str(self.diffflux))
        filedata = filedata.replace('seasval', str(self.seasnum))
        filedata = filedata.replace('ftype', self.fieldType)
        filedata = filedata.replace('x1val', str(self.x1))
        filedata = filedata.replace('colorval', str(self.color))
        filedata = filedata.replace('zmin', str(self.zmin))
        filedata = filedata.replace('zmax', str(self.zmax))
        filedata = filedata.replace('x1colorType', self.x1colorType)
        filedata = filedata.replace('zType', self.zType)
        filedata = filedata.replace('daymaxType', self.daymaxType)
        filedata = filedata.replace('fcoadd', 'True')

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
