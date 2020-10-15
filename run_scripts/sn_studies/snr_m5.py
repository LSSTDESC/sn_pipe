from sn_design_dd_survey.snr_m5 import SNR_m5
from optparse import OptionParser

parser = OptionParser()

parser.add_option('--inputDir',help='input dir for reference data[%default]',default='input/sn_studies',type=str)
parser.add_option('--refLC',help='reference LC[%default]',
                  default='LC_sn_fast_Fake_Fake_DESC_seas_-1_-2.0_0.2_error_model_ebvofMW_0.0_0.hdf5',type=str)
parser.add_option('--outFile',help='output file name [%default]',default='SNR_m5_error_model_snrmin_1.npy',type=str)
parser.add_option('--error_model',help='bool for error model [%default]',default=1,type=int)
parser.add_option('--x1',help='SN x1 [%default]',default=-2.0,type=float)
parser.add_option('--color',help='SN color [%default]',default=0.2,type=float)
parser.add_option('--snrmin',help='min SNR LC points for SNR estimation[%default]',default=1.0,type=float)

opts, args = parser.parse_args()

snr = SNR_m5(opts.inputDir, opts.refLC,opts.outFile,
             opts.error_model,opts.x1,opts.color,opts.snrmin)
