import os
from sn_tools.sn_utils import multiproc
from optparse import OptionParser


def func(data, params={}, j=0, output_q=None):
    """
    Function to process a set of data - required by multiproc

    Parameters
    --------------
    data: list(str)
      data to process
    params: dict, opt
      parameters (default: {})
    j: int, opt
      internal multiproc tag (default: 0)
    output_q:  multiprocessing queue, opt
      default: None

    Returns
    -----------
    1 (directly or in processing queue)

    """
    for dd in data:
        print('processing', dd)
        os.system(dd)

    if output_q is not None:
        return output_q.put({j: 1})
    else:
        return 1


parser = OptionParser()

parser.add_option("--nproc", type="int", default=3,
                  help="number of proc for multiproc [%default]")

opts, args = parser.parse_args()

nproc = opts.nproc

cmd_scr = 'python -W\"ignore\" run_scripts/metrics/make_new_OS.py'
configDir = 'input/metrics'

configs = ['config_newOS.yaml', 'config_newOS_double.yaml']

for b in 'riy':
    configs.append('config_newOS_moon{}.yaml'.format(b))
    configs.append('config_newOS_double_moon{}.yaml'.format(b))

processes = []
lps = [0.10, 0.20, 0.30, 0.40]
no_dither = [0, 1]
add_nightly = [0, 1]
medobs = [0, 1]
for lp in lps:
    for nod in no_dither:
        for addn in add_nightly:
            for med in medobs:
                for conf in configs:
                    cmd = '{} --config {}/{} --lunar_phase {} --add_nightly {} --no_dithering {} --medobs {}'.format(
                        cmd_scr, configDir, conf, lp, addn, nod, med)
                    # os.system(cmd)
                    processes.append(cmd)

print(processes, len(processes))

#multiproc(processes, {}, func, nproc)
