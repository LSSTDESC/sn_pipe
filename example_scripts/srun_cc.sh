#!/bin/bash
script=$1
srun -p htc_interactive --cpus-per-task 8 --time 04:00:00 --mem 60G ${script}
