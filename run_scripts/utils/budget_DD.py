from optparse import OptionParser
import numpy as np

parser = OptionParser()

parser.add_option("--Nvisits_DD", type=int, default=200,
                  help="number of DD visits [%default]")
parser.add_option("--season_length", type=int, default=180,
                  help="season length (days) [%default]")
parser.add_option("--cadence", type=int, default=1,
                  help="cadence of observation [%default]")
parser.add_option("--Nvisits_WFD", type=int, default=2122176,
                  help="number of WFD visits [%default]")
parser.add_option("--Nseasons", type=int, default=1,
                  help="number of seasons [%default]")
opts, args = parser.parse_args()

Nvisits_DD = opts.Nvisits_DD
season_length = opts.season_length
cadence = opts.cadence
Nvisits_WFD = opts.Nvisits_WFD
Nseasons = opts.Nseasons

Nv = Nvisits_DD*season_length*Nseasons/cadence
budget = Nv/(Nvisits_WFD+Nv)

print('budget', budget)
