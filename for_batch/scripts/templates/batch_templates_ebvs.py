from optparse import OptionParser
import os
import numpy as np

parser = OptionParser()

parser.add_option("--action", type="str", default='simu',
                  help="what to do: simu or vstack[%default]")
parser.add_option("--mode", type="str", default='batch',
                  help="how to run: batch or interactive [%default]")
parser.add_option("--x1", type=float, default=-2.0,
                  help="SN x1[%default]")
parser.add_option("--color", type=float, default=0.2,
                  help="SN color[%default]")
parser.add_option("--zmax", type=float, default=1.01,
                  help="SN max redshift[%default]")
parser.add_option("--zmin", type=float, default=0.01,
                  help="SN min redshift[%default]")
parser.add_option("--bluecutoff", type=float, default=380.0,
                  help="blue cutoff[%default]")
parser.add_option("--redcutoff", type=float, default=800.0,
                  help="red cutoff[%default]")
parser.add_option("--error_model", type=int, default=1,
                  help="error model for SN LC estimation[%default]")


opts, args = parser.parse_args()

ebvs = np.arange(0.0, 0.31, 0.01)

cmd = 'python for_batch/scripts/batch_templates.py'
cmd += ' --x1 {}'.format(opts.x1)
cmd += ' --color {}'.format(opts.color)
cmd += ' --action {}'.format(opts.action)
cmd += ' --mode {}'.format(opts.mode)
cmd += ' --error_model {}'.format(opts.error_model)
cmd += ' --zmin {}'.format(opts.zmin)
cmd += ' --zmax {}'.format(opts.zmax)
cmd += ' --bluecutoff {}'.format(opts.bluecutoff)
cmd += ' --redcutoff {}'.format(opts.redcutoff)

for ebv in ebvs:
    cmd_ = cmd
    cmd_ +=' --ebv {}'.format(np.round(ebv,2))
    print(cmd_)
    os.system(cmd_)
