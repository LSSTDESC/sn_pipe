#!bin/bash

source /pbs/throng/lsst/users/gris/anaconda3/etc/profile.d/conda.sh
conda activate myenv
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pbs/throng/lsst/users/gris/anaconda3/envs/myenv/lib
