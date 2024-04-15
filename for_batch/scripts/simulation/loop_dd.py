import os
from optparse import OptionParser
import pandas as pd
from sn_tools.sn_batchutils import BatchIt


def simulation(**dd):

    cmd = 'python for_batch/scripts/simulation/batch_dd_simu.py'
    for key, vals in dd.items():
        cmd += ' --{}={}'.format(key, vals)
    """
    cmd += ' --fieldName {}'.format(fieldName)
    cmd += ' --dbName {}'.format(dbName)
    cmd += ' --dbDir {}'.format(dbDir)
    cmd += ' --dbExtens {}'.format(dbExtens)
    cmd += ' --outDir {}'.format(outDir)
    # cmd += ' --mode {}'.format(mode)
    # cmd += ' --pixelmap_dir {}'.format(pixelmap_dir)
    cmd += ' --ebvofMW {}'.format(ebvofMW)
    cmd += ' --nproc {}'.format(nproc)
    """
    print(cmd)
    os.system(cmd)


def process_new(**params):
    """
    Method to process simulations

    Parameters
    ----------
    **params : dic
        Run parameters.

    Returns
    -------
    None.

    """

    script = 'run_scripts/simulation/run_simulation.py'
    script = 'run_scripts/simulation/run_simulation_wrapper.py'
    fields = params['fieldNames'].split(',')
    nprocSimu = params['nprocSimu']
    del params['fieldNames']
    del params['nprocSimu']

    for field in fields:
        processName = 'simu_{}_{}'.format(params['dbName'], field)
        mybatch = BatchIt(processName=processName)
        params['SN_color_type'] = 'random'
        params['SN_x1_type'] = 'random'
        params['SN_z_min'] = 0.01
        params['SN_z_max'] = 1.1
        params['SN_z_type'] = 'random'
        params['Simulator_model'] = 'salt3'
        params['Simulator_version'] = '2.0'
        params['SN_NSNfactor'] = 10
        params['Observations_fieldtype'] = params['fieldType']
        params['ProductionIDSimu'] = 'LC_{}_{}_spectroz'.format(params['fieldType'],
                                                                params['dbName'])
        params['nside'] = 128
        params['Pixelisation_nside'] = 128
        params['Observations_fieldname'] = field
        params['nproc'] = 1
        params['MultiprocessingSimu_nproc'] = nprocSimu
        params['OutputSimu_save'] = 0
        params['OutputSimu_savefromwrapper'] = 1
        params['OutputSimu_throwafterdump'] = 0
        mybatch.add_batch(script, params)
        mybatch.go_batch()


def fit(fieldName, dbName, simuDir, outDir, snrmin, nbands=0):

    cmd = 'python for_batch/scripts/simulation/batch_dd_fit.py'
    cmd += ' --fieldName {}'.format(fieldName)
    cmd += ' --dbName {}'.format(dbName)
    cmd += ' --simuDir {}'.format(simuDir)
    cmd += ' --outDir {}'.format(outDir)
    cmd += ' --snrmin {}'.format(snrmin)
    print(cmd)
    os.system(cmd)


parser = OptionParser()

parser.add_option('--dbList', type='str', default='DD_fbs_3.3.csv',
                  help='dbList to process [%default]')
parser.add_option('--OutputSimu_directory', type='str',
                  default='/sps/lsst/users/gris/DD/Simu',
                  help='simu dir [%default]')

parser.add_option('--nprocSimu', type=int, default=8,
                  help='number of proc for simu [%default]')
parser.add_option('--ebvofMW', type=float, default=-
                  1.0, help='E(B-V) [%default]')
parser.add_option('--fieldNames', type=str,
                  default='COSMOS,CDFS,ELAIS,XMM-LSS,ADFS1,ADFS2',
                  help='DD fields to process [%default]')
parser.add_option('--InstrumentSimu_airmassType', type=str, default='dep',
                  help='airmass for LCs const/dep[%default]')

opts, args = parser.parse_args()


dbList = pd.read_csv(opts.dbList)

dd = vars(opts)
del dd['dbList']
for i, row in dbList.iterrows():
    for col in dbList.columns:
        dd[col] = row[col]
    dd['Observations_coadd'] = dd.pop('coadd')
    dd['InstrumentSimu_telescope_tag'] = dd.pop('teltag')
    process_new(**dd)
