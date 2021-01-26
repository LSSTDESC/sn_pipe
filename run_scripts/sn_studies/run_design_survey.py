import os
from optparse import OptionParser

from sn_design_dd_survey.sequence import TemplateData, DD_SNR, OptiCombi
from sn_design_dd_survey.sequence import Nvisits_Cadence_z, Nvisits_Cadence_Fields


def chk_cr(dirname):
    """
    Function to create a dir if does not exist

    Parameters
    ---------------
    dirname: str
      name of the directory
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


parser = OptionParser()

parser.add_option('--x1', type=float, default=-2.0,
                  help='SN x1 [%default]')
parser.add_option('--color', type=float, default=0.2,
                  help='SN color [%default]')
parser.add_option('--error_model', type=int, default=1,
                  help='error model [%default]')
parser.add_option('--bluecutoff', type=float, default=380.,
                  help='blue cutoff if error_model=0 [%default]')
parser.add_option('--redcutoff', type=float, default=800.,
                  help='red cutoff if error_model=0 [%default]')
parser.add_option('--ebvofMW', type=float, default=0.,
                  help='E(B-V) [%default]')
parser.add_option('--sn_simulator', type=str, default='sn_fast',
                  help='simulator for templates SN [%default]')
parser.add_option('--bands', type=str, default='grizy',
                  help='bands to consider for this study [%default]')
parser.add_option('--zmin', type=float, default=0.1,
                  help='min redshift [%default]')
parser.add_option('--zmax', type=float, default=1.05,
                  help='max redshift [%default]')
parser.add_option('--zstep', type=float, default=0.05,
                  help='redshift step[%default]')
parser.add_option('--dirStudy', type=str, default='dd_design_test',
                  help='main dir output for the study [%default]')
# define subdir here
for ddir in ['Templates', 'SNR_m5', 'SNR_combi', 'SNR_opti', 'Nvisits_z']:
    parser.add_option('--dir{}'.format(ddir), type=str, default=ddir,
                      help='subdirectory {}[%default]'.format(ddir))
parser.add_option('--dirm5', type=str, default='m5_files',
                  help='sub dir with m5 single exp ref values [%default]')
parser.add_option('--m5File', type=str, default='medValues_flexddf_v1.4_10yrs_DD.npy',
                  help='m5 single exp. reference file[%default]')
parser.add_option('--web_server', type=str, default='https://me.lsst.eu/gris/DESC_SN_pipeline/m5_single_exposure',
                  help='web server where m5 reference file may be loaded from[%default]')
parser.add_option('--action', type=str, default='all',
                  help='what to do: all, Templates, SNR_combi, SNR_opti,Nvisits_z [%default]')
parser.add_option('--cadence_for_opti', type=int, default=3,
                  help='cadence used for optimisation [%default]')
parser.add_option('--cadence_min', type=int, default=2,
                  help='cadence max to consider in this study  [%default]')
parser.add_option('--cadence_max', type=int, default=4,
                  help='cadence max to consider in this study [%default]')
parser.add_option('--nproc', type=int, default=8,
                  help='number of procs when multiprocessing is to be used. [%default]')

opts, args = parser.parse_args()

# create output Directories here

chk_cr('{}/{}'.format(opts.dirStudy, opts.dirTemplates))
chk_cr('{}/{}'.format(opts.dirStudy, opts.dirSNR_m5))
chk_cr('{}/{}'.format(opts.dirStudy, opts.dirm5))
chk_cr('{}/{}'.format(opts.dirStudy, opts.dirSNR_combi))
chk_cr('{}/{}'.format(opts.dirStudy, opts.dirSNR_opti))
chk_cr('{}/{}'.format(opts.dirStudy, opts.dirNvisits_z))

# download m5 single exposure reference file if necessary

m5_web_name = '{}/{}'.format(opts.web_server, opts.m5File)
dir_disk_m5 = '{}/{}'.format(opts.dirStudy, opts.dirm5)
m5_disk_name = '{}/{}'.format(dir_disk_m5, opts.m5File)
if not os.path.isfile(m5_disk_name):
    cmd = 'wget --no-clobber --no-verbose {} --directory-prefix {}'.format(
        m5_web_name, dir_disk_m5)
    os.system(cmd)

all_actions = ['Templates', 'SNR_combi', 'SNR_opti', 'Nvisits_z']

actions = opts.action.split(',')
print(actions)
if 'all' in actions:
    actions = all_actions

if 'Templates' in actions:
    # class instance
    templ = TemplateData(x1=opts.x1, color=opts.color,
                         bands=opts.bands,
                         dirStudy=opts.dirStudy,
                         dirTemplates=opts.dirTemplates,
                         dirSNR_m5=opts.dirSNR_m5)
    # create template LC for cadence = 2 to 4 days
    for cadence in range(opts.cadence_min, opts.cadence_max+1):
        templ.templates(zmin=opts.zmin,
                        zmax=opts.zmax,
                        zstep=0.01,
                        error_model=opts.error_model,
                        bluecutoff=opts.bluecutoff,
                        redcutoff=opts.redcutoff,
                        ebvofMW=opts.ebvofMW,
                        simulator=opts.sn_simulator,
                        cadence=cadence)
    # estimate SNR vs m5 for the above generated templates
    templ.snr_m5(error_model=opts.error_model,
                 bluecutoff=opts.bluecutoff,
                 redcutoff=opts.redcutoff,)

if 'SNR_combi' or 'SNR_opti' or 'Nvisits_z' in actions:
    dd_snr = DD_SNR(x1=opts.x1, color=opts.color,
                    bands=opts.bands,
                    dirStudy=opts.dirStudy,
                    dirTemplates=opts.dirTemplates,
                    dirSNR_m5=opts.dirSNR_m5,
                    dirm5=opts.dirm5,
                    dirSNR_combi=opts.dirSNR_combi,
                    cadence=opts.cadence_for_opti,
                    error_model=opts.error_model,
                    bluecutoff=opts.bluecutoff,
                    redcutoff=opts.redcutoff,
                    ebvofMW=opts.ebvofMW,
                    sn_simulator=opts.sn_simulator,
                    m5_file=opts.m5File)

    # dd_snr.plot_data()

if 'SNR_combi' in actions:
    # estimate combis
    dd_snr.SNR_combi( SNR_par=dict(zip(['max', 'step', 'choice'], [70., 1., 'Nvisits'])),zmin=0.3)

opti_fileName = 'opti_{}_{}_{}_ebvofMW_{}_cad_{}.npy'.format(
    opts.x1, opts.color, dd_snr.cutoff, opts.ebvofMW, opts.cadence_for_opti)
if 'SNR_opti' in actions:
    # get 'best' combination

    OptiCombi(dd_snr.fracSignalBand,
              dirStudy=opts.dirStudy,
              dirSNR_combi=opts.dirSNR_combi,
              dirSNR_opti=opts.dirSNR_opti,
              snr_opti_file=opti_fileName,
              nproc=opts.nproc)

if 'Nvisits_z' in actions:
    file_Nvisits_z_med = 'Nvisits_z_{}_{}_{}_ebvofMW_{}.npy'.format(
        opts.x1, opts.color, dd_snr.cutoff, opts.ebvofMW)

    Nvisits_Cadence_z(dd_snr.data.m5_Band, opti_fileName,
                      dirStudy=opts.dirStudy,
                      dirSNR_m5=opts.dirSNR_m5,
                      dirSNR_opti=opts.dirSNR_opti,
                      dirNvisits=opts.dirNvisits_z,
                      outName=file_Nvisits_z_med)

    # get the number of visits per field
    file_Nvisits_z_fields = 'Nvisits_z_fields_{}_{}_{}_ebvofMW_{}.npy'.format(
        opts.x1, opts.color, dd_snr.cutoff, opts.ebvofMW)
    Nvisits_Cadence_Fields(x1=opts.x1, color=opts.color,
                           error_model=opts.error_model,
                           bluecutoff=opts.bluecutoff, redcutoff=opts.redcutoff,
                           ebvofMW=opts.ebvofMW,
                           sn_simulator=opts.sn_simulator,
                           dirStudy=opts.dirStudy,
                           dirTemplates=opts.dirTemplates,
                           dirNvisits=opts.dirNvisits_z,
                           Nvisits_z_med=file_Nvisits_z_med,
                           outName=file_Nvisits_z_fields)
