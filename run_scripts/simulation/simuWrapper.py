from sn_simu_wrapper.sn_simu import SNSimulation
import numpy as np
import yaml
import os
from sn_tools.sn_io import check_get_file


class MakeYaml:
    """
    class to generate a yaml file from a generic one

    Parameters
    ---------------
    dbDir: str
      location dir of the database
    dbName: str
      OS name
    db Extens: str
      db extension (npy or db)
    nside: int
      nside for healpix
    nproc: int
      number of proc for multiprocessing
    diffflux: bool
      to allow for simulation with differential params (ex: x1+epsilon_x1)
    seasnum: list(int)
      season numbers
    outDir: str
      output directory for the production (and also for this yaml file)
    fieldType: str
        type of the field to process (DD, WFD, fake)
     x1Type: str
       x1 type for simulation (unique, uniform, random)
     x1min: float
       x1 min value
     x1max: float
       x1 max value
     x1step: float
        x1 step value
    colorType: str
       color type for simulation (unique, uniform, random)
     colormin: float
       color min value
     colormax: float
       color max value
     colorstep: float
        color step value
     zType: str
       z type for simulation (unique, uniform, random)
     zmin: float
       z min value
     zmax: float
       z max value
     zstep: float
        z step value
     simu: str
       simulator type
     daymaxType: str
       daymax type for simulation (unique, uniform, random)
     daymaxstep: float
        daymax step value
     coadd: bool
       to coadd (True) or not (Fals) observations per night
    prodid: str
       production id ; the resulting yaml file is prodid.yaml
    ebvmw: float
      to specify an extinction value
    bluecutoff: float
       blue cutoff for SN
    redcutoff: float
       redcutoff for SN
    error_model: int
      error model for flux error estimation
    """

    def __init__(self, dbDir, dbName, dbExtens, nside, nproc, diffflux,
                 seasnum, outDir, fieldType,
                 x1Type, x1min, x1max, x1step,
                 colorType, colormin, colormax, colorstep,
                 zType, zmin, zmax, zstep,
                 simu, daymaxType, daymaxstep,
                 coadd, prodid,
                 ebvofMW, bluecutoff, redcutoff, error_model):

        self.dbDir = dbDir
        self.dbName = dbName
        self.dbExtens = dbExtens
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
        self.zstep = zstep
        self.simu = simu
        self.zType = zType
        self.daymaxType = daymaxType
        self.daymaxstep = daymaxstep
        self.coadd = coadd
        self.prodid = prodid
        self.ebvofMW = ebvofMW
        self.bluecutoff = bluecutoff
        self.redcutoff = redcutoff
        self.error_model = error_model

    def genYaml(self, input_file):
        """
        method to generate a yaml file
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

        fullDbName = '{}/{}.{}'.format(self.dbDir, self.dbName, self.dbExtens)
        filedata = filedata.replace('prodid', self.prodid)
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
        filedata = filedata.replace('zstep', str(self.zstep))
        filedata = filedata.replace('zType', self.zType)
        filedata = filedata.replace('daymaxType', self.daymaxType)
        filedata = filedata.replace('daymaxstep', str(self.daymaxstep))
        filedata = filedata.replace('fcoadd', str(self.coadd))
        filedata = filedata.replace('mysimu', self.simu)
        filedata = filedata.replace('ebvofMWval', str(self.ebvofMW))
        filedata = filedata.replace('bluecutoffval', str(self.bluecutoff))
        filedata = filedata.replace('redcutoffval', str(self.redcutoff))
        filedata = filedata.replace('errmod', str(self.error_model))

        return yaml.load(filedata, Loader=yaml.FullLoader)


class SimuWrapper:
    """
    Wrapper class for simulation

    Parameters
    ---------------
    yaml_config: str
      name of the yaml configuration file

    """

    def __init__(self, yaml_config):

        print('hh', type(yaml_config))
        fileName, fileExtension = os.path.splitext(yaml_config)
        if fileExtension == 'yaml':
            # load config file
            with open(yaml_config) as file:
                config = yaml.full_load(file)
        else:
            config = yaml_config

        print('oooooo', config)
        self.name = 'simulation'

        # get X0 for SNIa normalization
        x0_tab = self.x0(config)

        # load references if simulator = sn_fast
        # reference_lc = self.load_reference(config)

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
        absMag = config['SN']['absmag']
        x0normFile = 'reference_files/X0_norm_{}.npy'.format(absMag)
        if not os.path.isfile(x0normFile):
            # if this file does not exist, grab it from a web server
            check_get_file(config['WebPath'], 'reference_files',
                           'X0_norm_{}.npy'.format(absMag))

        if not os.path.isfile(x0normFile):
            # if the file could not be found, then have to generate it!
            salt2Dir = config['SN']['salt2Dir']
            model = config['Simulator']['model']
            version = str(config['Simulator']['version'])

            # need the SALT2 dir for this
            check_get_dir(config['Web path'], 'SALT2', salt2Dir)
            from sn_tools.sn_utils import X0_norm
            X0_norm(salt2Dir=salt2Dir, model=model, version=version,
                    absmag=absMag, outfile=x0normFile)

        return np.load(x0normFile)

    def run(self, obs, imulti=0):
        """
        Method to run the metric

        Parameters
        ---------------
        obs: array
          data to process

        """
        print('running here')
        return self.metric.run(obs, imulti=imulti)

    def finish(self):
        """
        Method to save metadata to disk

        """
        self.metric.save_metadata()
