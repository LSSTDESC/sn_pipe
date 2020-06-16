import os


def addoption(cmd, name, val):
    cmd += ' --{} {}'.format(name, val)
    return cmd


x1_color = [(-2.0, 0.2)]
zmin = 0.01
zmax = 1.1
zstep = 0.01
nproc = 4
outDirLC = 'fakes_for_templates'
outDirTemplates = 'Template_LC'

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
    print(cmd)
    os.system(cmd)

    # stack produced LCs
    cmd = 'python run_scripts/templates/run_template_vstack.py'
    cmd = addoption(cmd, 'x1', x1)
    cmd = addoption(cmd, 'color', color)
    cmd = addoption(cmd, 'lcDir', '{}/fake_simu_data'.format(outDirLC))
    cmd = addoption(cmd, 'outDir', outDirTemplates)
    print(cmd)
    os.system(cmd)
