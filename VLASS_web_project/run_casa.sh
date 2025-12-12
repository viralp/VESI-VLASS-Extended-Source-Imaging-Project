#!/bin/sh

#Don't put any commands before the #SBATCH options or they will not work
#SBATCH --mem=200G                        # Amount of memory needed by the whole job.
#SBATCH --time=14-00:00                 # Expected runtime of 2 hours and 30 minutes
#SBATCH --mail-type=END,FAIL             # Send email when Jobs end or fail
#SBATCH -n 10

# casa's python requires a DISPLAY for matplot, so create a virtual X server
xvfb-run -d /lustre/aoc/users/vparekh/CASA/casa-6.7.0-31-py3.10.el8/bin/casa --nologger --nogui -c VLASS_awp_joint_run.py

