import os
from optparse import OptionParser


def addoption(cmd, name, val):
    cmd += ' --{} {}'.format(name, val)
    return cmd


parser = OptionParser()

parser.add_option(
    '--x1', help='SN x1 [%default]', default=-2.0, type=float)
parser.add_option(
    '--color', help='SN color [%default]', default=0.2, type=float)
parser.add_option(
    '--zmin', help='min redshift value  [%default]', default=0.01, type=float)
parser.add_option(
    '--zmax', help='max redshift value [%default]', default=1.1, type=float)
parser.add_option(
    '--zstep', help='redshift step value [%default]', default=0.01, type=float)
parser.add_option(
    '--sn_model', help='SN model [%default]', default='salt2-extended', type=str)
parser.add_option(
    '--sn_version', help='SN model version [%default]', default='1.0', type=str)
parser.add_option(
    '--bluecutoff', help='blue cutoff for SN [%default]', default=380.0, type=float)
parser.add_option(
    '--redcutoff', help='red cutoff for SN [%default]', default=700.0, type=float)
parser.add_option(
    '--ebvofMW', help='ebvofMW[%default]', default=0.0, type=float)
parser.add_option(
    '--nproc', help='nproc for multiproc [%default]', default=8, type=int)

opts, args = parser.parse_args()

x1 = opts.x1
color = opts.color
zmin = opts.zmin
zmax = opts.zmax
sn_model = opts.sn_model
sn_version = opts.sn_version
bluecutoff = opts.bluecutoff
redcutoff = opts.redcutoff
nproc = opts.nproc

x1_color = [(x1, color)]
outDirLC = 'fakes_for_templates_{}_{}_ebvofMW_{}'.format(
    bluecutoff, redcutoff, ebvofMW)
outDirTemplates = 'Template_LC_{}_{}_ebvofMW_{}'.format(
    bluecutoff, redcutoff, ebvofMW)

for (x1, color) in x1_color:
    # generate LCs
    cmd = 'python run_scripts/templates/run_template_LC.py'
    cmd = addoption(cmd, 'x1', x1)
    cmd = addoption(cmd, 'color', color)
    cmd = addoption(cmd, 'zmin', zmin)
    cmd = addoption(cmd, 'zmax', zmax)
    cmd = addoption(cmd, 'zstep', zstep)
    cmd = addoption(cmd, 'nproc', nproc)
    cmd = addoption(cmd, 'outDir', outDirLC)
    cmd = addoption(cmd, 'bluecutoff', bluecutoff)
    cmd = addoption(cmd, 'redcutoff', redcutoff)
    cmd = addoption(cmd, 'ebvofMW', ebvofMW)
    cmd = addoption(cmd, 'sn_model', sn_model)
    cmd = addoption(cmd, 'sn_version', sn_version)

    print(cmd)
    os.system(cmd)

    # stack produced LCs
    cmd = 'python run_scripts/templates/run_template_vstack.py'
    cmd = addoption(cmd, 'x1', x1)
    cmd = addoption(cmd, 'color', color)
    cmd = addoption(cmd, 'bluecutoff', bluecutoff)
    cmd = addoption(cmd, 'redcutoff', redcutoff)
    cmd = addoption(cmd, 'ebvofMW', ebvofMW)
    cmd = addoption(cmd, 'lcDir', '{}/fake_simu_data'.format(outDirLC))
    cmd = addoption(cmd, 'outDir', outDirTemplates)
    cmd = addoption(cmd, 'sn_model', sn_model)
    cmd = addoption(cmd, 'sn_version', sn_version)
    print(cmd)
    os.system(cmd)
