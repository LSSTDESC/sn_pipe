import numpy as np
import os


def cmd(zStep, daymaxStep, zlim_coeff):

    outDir = 'Metric_check_100pixels/MetricOutput_{}_{}_{}'.format(
        zStep, daymaxStep, zlim_coeff)
    cmd_ = 'python run_scripts/metrics/run_metrics.py --dbName baseline_v2.0_10yrs --dbExtens npy --dbDir ../DB_Files --fieldType WFD --nproc 8 --metric NSNY --ebvofMW 0.0 --npixels -1 --saveData 1 --verbose 0 --pixelmap_dir ../ObsPixelized_fbs_2.0 --nside 64 --ploteffi 0 --RAmin 36.0 --RAmax 72.0 --coadd 1 --season -1 --Decmin -1. --Decmax -1. --season -1 --zStep {} --daymaxStep {} --zlim_coeff {} --outDir {}'.format(
        zStep, daymaxStep, zlim_coeff, outDir)

    cmd_ += ' --healpixID \'23273, 36175, 23398, 36744, 24232, 33656, 34807, 208, 33261, 17729, 22662, 22563, 792, 36847, 34608, 34466, 464, 21128, 23344, 35997, 33737, 23527, 22632, 36177, 398, 36054, 815, 1053, 23505, 22747, 36831, 33082, 626, 22782, 34755, 33589, 22729, 33233, 394, 23444, 22676, 22723, 594, 213, 532, 34322, 23166, 349, 36328, 36303, 23211, 467, 35898, 34717, 507, 22724, 59, 33688, 448, 43, 36057, 34745, 35847, 17778, 36785, 34756, 36127, 36339, 33551, 434, 17733, 33068, 36292, 23217, 332, 36648, 1087, 299, 23395, 1060, 36348, 34671, 34418, 22548, 1178, 51, 838, 22538, 33553, 23222, 36470, 22718, 34528, 36301, 120, 36768, 36235, 35188, 22756, 97\''
    return cmd_


zSteps = [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1]
daymaxSteps = [1, 2, 3, 5, 10, 15]
zlim_coeffs = [0.95, 0.99, 0.90]

for zStep in zSteps:
    for daymaxStep in daymaxSteps:
        for zlim_coeff in zlim_coeffs:
            ccmd = cmd(zStep, daymaxStep, zlim_coeff)
            print(ccmd)
            os.system(ccmd)
