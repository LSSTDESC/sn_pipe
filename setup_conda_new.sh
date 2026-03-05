source /usr/share/Modules/init/bash

module load Programming_Languages/anaconda/3.13

which anaconda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pbs/software/redhat-9-x86_64/anaconda/3.13/lib
