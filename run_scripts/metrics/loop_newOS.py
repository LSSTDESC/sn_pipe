import os

cmd_scr = 'python run_scripts/metrics/make_new_OS.py'

configs = ['config_newOS.yaml', 'config_newOS_double.yaml']

for b in 'riy':
    configs.append('config_newOS_moon{}.yaml'.format(b))
    configs.append('config_newOS_double_moon{}.yaml'.format(b))


for lp in [0.10, 0.20, 0.30, 0.40]:
    for conf in configs:
        cmd = '{} --config {} --lunar_phase {}'.format(cmd_scr, conf, lp)
        os.system(cmd)
