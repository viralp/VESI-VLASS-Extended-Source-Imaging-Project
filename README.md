# VESI-VLASS Extended Source Imaging Project

## Download VLASS multi epoch data from NRAO archive
1, Find out multi epoch measurement sets for required (extended) source

> Python3 VLASS_ms_info.py <Source_name> <RA_in_deg> <Dec_in_deg> 

This will print VLASS ms names which have covered the source pointings

2, Download data from NRAO archive "https://data.nrao.edu/portal/" by selecting the "Archive Filename" option.

Edit the "VLASS_awp_joint_run.py" script, add ms names in it, select the required options with "True" or "False" and execute the script with CASA (6.7)

> casa --nologger -c VLASS_awp_joint_run.py

## Processing Steps

1. **Per epoch**
   1. Split the VLASS pointings or fields from given phasecentre (default value is 0.1 deg around RA and DEC, if split option is True)  
   2. Initial imaging (mostly same imaging parameters as VLASS pipeline)  
   3. Masking with threshold (generate mask from (2) if masking is True)  
   4. Imaging with mask generated from (3)  
   5. Self-calibration (gaincal + applycal + clean, if `selfcal` is True)  
   6. Delay (G+K) self-calibration (with Quartical, if `delay selfcal` is True)  
   7. Final imaging  
   8. Polarisation imaging (if `polcleaning` is True)  

2. **Combined Epochs**  
   Combine VLASS epochs and image Stokes I and IQUV (total intensity and polarisation imaging)  
   (if `joint deconvolution` is True).  

3. **Spectral Cube Generation**  
   Generate spectral cube from each epoch (from `specmode="mvc"` in clean).  
   Convolve each plane to native resolution, and also convolve each plane to 4" beam size  
   (if `awp_image_cube` is True).  

4. **Spectral Index Mapping**  
   Generate spectral index map per epoch as well as combined epoch  
   using 4" convolved cube from step (3).  

5. **RMS-Weighted Maps**  
   Generate rms-weighted mean and median maps using selfcal maps of each epoch.  

6. **Flux Density Estimation**  
   Calculate statistics and estimate flux densities for extended source(s)  
   using contour method.  

7. **Export Results**  
   Export all results into an HTML page.  

