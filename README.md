1, Find out multi epoch measurement sets for required source

Python3 VLASS_ms_info.py <Source_name> <RA_in_deg> <Dec_in_deg> 

This will print VLASS ms names which have covered the source pointings

Edit the "VLASS_awp_joint_V2.py" script, add ms names in it, select the options (True/False) and execute the script with CASA

casa --nologger -c VLASS_awp_joint_V2.py
