# Installation and Setup at NERSC

## Installation Steps

If you do not yet have a NERSC account, please follow [these instructions](https://confluence.slac.stanford.edu/display/LSSTDESC/Getting+a+NERSC+Computing+Account).

### Get the sn_pipe package

Follow the typical [sn_pipe instructions](https://github.com/LSSTDESC/sn_pipe#getting-the-package-from-github)

```
 git clone https://github.com/lsstdesc/sn_pipe (master)
or
git clone -b <tag_name> https://github.com/lsstdesc/sn_pipe (release tag_name)
```
### Environment Setup

CVMFS is accessible at NERSC.  Rather than using sn_pipe's release_setup.sh script, set up w_2020_15 version of lsst_sims:

`source /global/common/software/lsst/cori-haswell-gcc/stack/setup_any_sims.sh w_2020_15`

### Install sn_pipe

```
cd sn_pipe
python  pip_sn_pack.py --action install --package all
```

### Running Jupyter Notebooks

Rather than running jupyter at the command line, we can set up a python environment, called a jupyter kernel, which includes sn_pipe that will run in [jupyter.nersc.gov](jupyter.nersc.gov)

At the command line, from within the `sn_pipe` directory, run:

`./nersc/setup_nersc.sh`

you only need to do this step once when first installing sn_pipe.

