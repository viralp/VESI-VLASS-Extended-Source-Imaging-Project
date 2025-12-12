#!/bin/sh

#Don't put any commands before the #SBATCH options or they will not work
#SBATCH --mem=200G                        # Amount of memory needed by the whole job.
#SBATCH --time=14-00:00                 # Expected runtime of 2 hours and 30 minutes
#SBATCH --mail-type=END,FAIL             # Send email when Jobs end or fail
#SBATCH -n 1

# casa's python requires a DISPLAY for matplot, so create a virtual X server
xvfb-run -d /lustre/aoc/users/vparekh/CASA/casa-6.6.6-17-pipeline-2025.1.0.19-py3.10.el8/bin/casa --pipeline --nogui -c casa_imaging_vlass.py

