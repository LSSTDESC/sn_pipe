import os


def addoption(cmd, name, val):
    cmd += ' --{} {}'.format(name, val)
    return cmd


x1_color = [(-2.0, 0.2)]
zmin = 0.1
zmax = 1.1
zstep = 0.01
nproc = 8
#sn_model = 'salt2-extended'
sn_model = 'salt3'
sn_version = '1.0'
bluecutoff = 380
redcutoff = 700
ebvofMW = 0
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
