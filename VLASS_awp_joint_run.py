import sys,os,re,shutil
sys.path.append(os.getcwd())
from VLASS_awp_joint_workers import *
from VLASS_awp_joint_parameters import *

img_size=vlass_imagesize(R, pixel_scale=0.00016, min_size=512, round_to=32)
imsize= [img_size,img_size] 


###########################################################################################################################################################################################
for i, vis in enumerate(vis_list):
    flags = processing_flags[i]
    vis_base = vis[:-3]
    imagename = f"{imagename_base}_{vis_base}"
    vis_out = vis
    # Split by region if flag is True
    if flags['target_split']:
       vis_out = vis_base + f'_fields_seperation_{R}deg.ms'        
       if os.path.exists(vis_out):
         print(f"Output MS {vis_out} already exists, skipping mstransform.")
       else: 
        msmd.open(vis)
        field_names = msmd.fieldsforintent('OBSERVE_TARGET#UNSPECIFIED', True)
        msmd.done()
        fields = vishead(vis, mode='list', listitems=['field'])['field'][0]
        img_field = []
        for idx in range(len(fields)):
            val = vishead(vis=vis, mode='get', hdkey='ptcs', hdindex=str(idx))
            Ra = val[0][0][0] * 57.2958
            Dec = val[0][1][0] * 57.2958
            dist = seperation(Ra,Dec,Ra_1,Dec_1)  # Replace with accurate SkyCoord sep if needed
            if dist < R:
                img_field.append(fields[idx])
        img_fields = ','.join(img_field)
        mstransform(vis=vis, outputvis=vis_out, field=img_fields, combinespws=False,
                    datacolumn=colname, chanaverage=channelaverage, timeaverage=timeaverage)
    else:
        vis_out = vis_base + f'_fields_seperation_{R}deg.ms'

    # Initial imaging
    if flags['init_imaging']:
        imagename_out=imagename+'_init'
        cleaning_awp(vis=vis_out, imagename=imagename_out, imsize=imsize, phasecenter=phasecenter, datacolumn='data',savemodel='modelcolumn')
        imagename_out = imagename_out
    else:
        imagename_out = imagename+'_init'

    # Masking
    if flags['mask']:
        os.system(f"breizorro --restored-image {imagename_out}.image.tt0.fits --threshold {mask_threshold} --outfile {imagename_out}.sigma.mask.fits")
        importfits(fitsimage=imagename_out+".sigma.mask.fits", imagename=imagename_out+".sigma.mask")
        mask = imagename_out + ".sigma.mask"
        cleaning_awp(vis=vis_out,imagename=imagename_out+'_mask', imsize=imsize, phasecenter=phasecenter,datacolumn='data', mask=mask, savemodel='modelcolumn')
        imagename_out = imagename_out + '_mask'
        mask_maps.append(mask)
    else:
        mask = ''

    # Self-calibration
    if flags['selfcal']:
        list_ant = rank_refants(vis_out)
        gaincal(vis=vis_out, caltable=vis_out+'.G.selfcal', spw='', solint='inf',
                combine='', field='', selectdata=True, solnorm=False, refant=list_ant,
                minblperant=4, minsnr=5.0, gaintype='G', calmode='p', append=False, parang=False)
        applycal(vis=vis_out, spw='', selectdata=True, gaintable=[vis_out+'.G.selfcal'],
                 field='', interp=['linear'], calwt=False, parang=False, applymode='calonly')
        cleaning_awp(vis=vis_out, imagename=imagename_out+'.sc', datacolumn='corrected',imsize=imsize, 
                     phasecenter=phasecenter,mask=mask, savemodel='modelcolumn')
        imagename_out = imagename_out + '.sc'

    # Delay + phase self-cal using quartical
    if flags['delay_selfcal']:
        os.system(f"goquartical input_ms.path={vis_out} input_ms.data_column=CORRECTED_DATA "
                  f"input_ms.select_fields=[] input_ms.time_chunk='0' input_ms.freq_chunk='0' "
                  f"input_model.recipe=MODEL_DATA solver.terms='[G,K]' "
                  f"solver.iter_recipe='[50,50,50,50,50,50]' solver.propagate_flags=False solver.robust=True "
                  f"solver.threads=1 solver.convergence_fraction=0.99 solver.convergence_criteria=1e-06 "
                  f"output.log_directory=output/KG.outputs.qc/log output.gain_directory=output/KG.output.gain.qc "
                  f"output.overwrite=1 output.products=[corrected_data,corrected_residual] "
                  f"output.columns=[CORRECTED_DATA,CORRECTED_RESIDUAL] output.flags=False "
                  f"dask.threads=6 dask.workers=6 dask.scheduler=threads "
                  f"G.type=phase G.time_interval='20s' G.freq_interval='0' G.initial_estimate=False "
                  f"G.solve_per=antenna G.interp_mode=reim G.interp_method=2dlinear "
                  f"G.respect_scan_boundaries=True mad_flags.enable=True mad_flags.threshold_bl=6 "
                  f"mad_flags.threshold_global=8 mad_flags.max_deviation=1000 "
                  f"K.time_interval='1s' K.freq_interval='0' K.type=delay_and_offset K.initial_estimate=True "
                  f"K.interp_mode=reim K.interp_method=2dlinear K.respect_scan_boundaries=True")

        cleaning_awp(vis=vis_out, imagename=imagename_out+'.GK_SC', datacolumn='corrected',imsize=imsize, 
                     phasecenter=phasecenter,mask=mask, savemodel='modelcolumn')
        imagename_out = imagename_out + '.GK_SC'
        
    if flags['pol_cleaning']:
        imagename_pol = imagename_out + '.IQUV'
        pol_cleaning(vis=vis_out, imagename=imagename_pol, imsize=imsize, phasecenter=phasecenter, mask=mask,datacolumn='corrected')
        print(f"Polarization cleaning completed for {vis_out}")
        file_pol = glob.glob(imagename_out + '*.IQUV.image.tt0.fits')
        sin_pol_image_cube=fits.open(file_pol[0])
        sin_pol_data=sin_pol_image_cube[0].data
        sin_q_pol_data=sin_pol_data[1].data
        sin_u_pol_data=sin_pol_data[2].data
        sin_v_pol_data=sin_pol_data[3].data
        sin_pol_header=sin_pol_image_cube[0].header
        fits.writeto(imagename_out+'_single_Q_map.fits',sin_q_pol_data,sin_pol_header,overwrite=True)
        fits.writeto(imagename_out+'_single_U_map.fits',sin_u_pol_data,sin_pol_header,overwrite=True)
        fits.writeto(imagename_out+'_single_V_map.fits',sin_v_pol_data,sin_pol_header,overwrite=True)
   

    # Store final vis and image name for joint step
    processed_vis.append(vis_out)
    image_list.append(imagename_out)

# Final joint deconvolution
if mask:
    if len(mask_maps) > 1:
        # Build the expr dynamically, e.g. "IM0+IM1+IM2+..."
        expr = "+".join([f"IM{i}" for i in range(len(mask_maps))])
        immath(imagename=mask_maps,
               expr=expr,
               outfile=imagename_base+'sum_of_masks.mask')
        mask = imagename_base+'sum_of_masks.mask'
    else:
        # If only one mask, just use it directly
        mask = mask_maps[0]

if joint_deconvolution:
 cleaning_awp(vis=processed_vis, imagename=joint_image, datacolumn='corrected', imsize=imsize, phasecenter=phasecenter, mask=mask, savemodel='modelcolumn')
 

if joint_selfcal:
 for msfile in processed_vis:
   list_ant = rank_refants(msfile)
   gaincal(vis=msfile, caltable=msfile+'.G.selfcal2', spw='', solint='inf',
        combine='', field='', selectdata=True, solnorm=False, refant=list_ant,
        minblperant=4, minsnr=5.0, gaintype='G', calmode='p', append=False, parang=False)
   applycal(vis=msfile, spw='', selectdata=True, gaintable=[msfile+'.G.selfcal2'],
         field='', interp=['linear'], calwt=False, parang=False, applymode='calonly')
 cleaning_awp(vis=processed_vis, imagename=joint_image+'.sc', datacolumn='corrected',imsize=imsize, 
              phasecenter=phasecenter,mask=mask, savemodel='modelcolumn')
 imagename_out = imagename_out + '.joint.sc'

if joint_pol_deconvolution:
  imagename_pol = joint_image + '.IQUV'
  pol_cleaning(vis=vis_out, imagename=imagename_pol, mask=mask, phasecenter=phasecenter, imsize=imsize, datacolumn='corrected',savemodel='modelcolumn')
  print(f"Joint Polarization cleaning completed")
  pol_image_cube=fits.open(imagename_pol+'.image.tt0.fits')
  pol_data=pol_image_cube[0].data
  q_pol_data=pol_data[1].data
  u_pol_data=pol_data[2].data
  v_pol_data=pol_data[3].data
  pol_header=pol_image_cube[0].header
  fits.writeto(joint_image+'_Q_map.fits',q_pol_data,pol_header,overwrite=True)
  fits.writeto(joint_image+'_U_map.fits',u_pol_data,pol_header,overwrite=True)
  fits.writeto(joint_image+'_V_map.fits',v_pol_data,pol_header,overwrite=True)


if awp_image_cube:
    # Build file lists
    model_images    = sorted(glob.glob(imagename_base+'*.GK_SC.model')) + [joint_image + ".model"]
    residual_images = sorted(glob.glob(imagename_base+'*.GK_SC.residual')) + [joint_image + ".residual"]
    psf_images      = sorted(glob.glob(imagename_base+'*.GK_SC.psf')) + [joint_image + ".psf"]

    assert len(psf_images) == len(model_images) == len(residual_images), "Mismatch in image sets!"

    for idx, (model, resid, psf) in enumerate(zip(model_images, residual_images, psf_images)):
        name = os.path.basename(model).replace('.GK_SC.model', '').replace('.model', '')
        tmpdir = f"temp_planes_{name}"
        if os.path.exists(tmpdir):
          shutil.rmtree(tmpdir)
        os.makedirs(tmpdir)      
        outname = f"{name}_restored_cube.im"
        print(f"\n Processing: {name}")
        restore_per_channel(model, resid, psf, outname, tmpdir)

    print("\n All datasets processed and smoothed.")


# ========= GROUP FILES AND GENERATE =========
print("\n Grouping smoothed FITS files by VLASS epoch")
fits_all = sorted(glob.glob(f"temp_planes_{imagename_base}"+"*/*beam3arcsec.im.fits"))
groups = {'vlass1': [], 'vlass2': [], 'vlass3': [], 'combined': []}
for f in fits_all:
    fname = f#os.path.basename(f).lower()
    if 'VLASS1' in fname:
        groups['vlass1'].append(f)
    elif 'VLASS2' in fname:
        groups['vlass2'].append(f)
    elif 'VLASS3' in fname:
        groups['vlass3'].append(f)
    elif 'combined' in fname or 'comb' in fname:
        groups['combined'].append(f)

# Run based on user flags
if spx_map_VLASS1:      compute_spectral_index_map(groups['vlass1'], imagename_base+'_vlass1_spx.fits')
if spx_map_VLASS2:      compute_spectral_index_map(groups['vlass2'], imagename_base+'_vlass2_spx.fits')
if spx_map_VLASS3:      compute_spectral_index_map(groups['vlass3'], imagename_base+'_vlass3_spx.fits')
if spx_map_VLASS_combine:compute_spectral_index_map(groups['combined'], imagename_base+'_vlass_combined_spx.fits')



if weighted_combine_maps:

 directory = []
 directory1 = []
 directory2= []
 directory = glob.glob(imagename_base+'*init.sc.image.tt0.fits')

 for file in directory:
  hdulist=fits.open(file,mode='update')
  data=hdulist[0].data
  prihdr = hdulist[0].header
  if 'TIMESYS' in prihdr:
    prihdr['TIMESYS'] = 'utc'
    hdulist.flush()
  elif data.ndim == 2:
      add_axis(file)
  else:
      pass
    
          
 results = find_fits_extremes(directory)
    
 print(f"FITS file with the highest BMAJ: {results['max_bmaj_file']} (BMAJ: {results['max_bmaj_value']})")
 print(f"FITS file with the highest BMIN: {results['max_bmin_file']} (BMIN: {results['max_bmin_value']})")
 print(f"FITS file with the lowest pixel size: {results['min_pixel_file']} (Pixel Size: {results['min_pixel_size']})")

 hdu=fits.open(results['max_bmaj_file'])
 header=hdu[0].header
 BPA=header['BPA']
 data_cube=[]

 hdu1=fits.open(results['min_pixel_file'])[0]
 for file in directory:
  if file==results['min_pixel_file']:
   directory1.append(results['min_pixel_file'])
   pass
  else:
   hdu2=fits.open(file)[0]
   print(file)
   hdulist = fits.open(file,mode='update')
   prihdr = hdulist[0].header
   prihdr['CDELTA1']=-results['min_pixel_size']
   prihdr['CDELTA2']=results['min_pixel_size']
   hdulist.flush()
   new_rgd=reproject(hdu2,hdu1)
   fits.writeto(file[:-5]+'.rgd.fits', new_rgd,prihdr, overwrite=True)
   directory1.append(file[:-5]+'.rgd.fits')
   
 if len(directory1)>1:
  directory1.remove(directory1[0]) 
 for file in directory1:
  if file==results['max_bmaj_file']:
    directory2.append(results['max_bmaj_file'])
    pass
  if file==results['max_bmaj_file'][:-5]+'.rgd.fits':
    directory2.append(results['max_bmaj_file'][:-5]+'.rgd.fits')
    pass
  else:
    my_beam=Beam(results['max_bmaj_value']* u.deg,results['max_bmin_value']* u.deg,BPA* u.deg)
    pixel_scale=results['min_pixel_size'] * u.deg
    #kernel = max_beam.deconvolve(beam).as_kernel()
    kernal=my_beam.as_kernel(pixel_scale)
    Data=fits.open(file)[0]
    header = Data.header
    data = Data.data
    (a1,b1,xx,yy)=data.shape
    convolved_data=np.zeros((1,1,xx, yy))
    data = check_array(data)
    convolved_fft = convolve_fft(data, kernal,allow_huge=True)
    convolved_data= np.expand_dims(convolved_fft, axis=(0, 1))    
    header['BMAJ'] =  (results['max_bmaj_value']* u.deg).value                                              
    header['BMIN'] =  (results['max_bmin_value']* u.deg).value                                                
    header['BPA'] =   (BPA* u.deg).value                                               
    fits.writeto(file[:-5]+'.convolve.fits', convolved_data, header, overwrite=True)
    directory2.append(file[:-5]+'.convolve.fits')
  

 n=len(directory2)
 d=fits.getdata(directory2[0])
 (a,b,xx,yy)=d.shape
 cube = np.zeros((1,n,xx,yy)) 
 j=a-1       

 temp_header = fits.Header()
 for i in range(len(directory2)):
    header=fits.getheader(directory1[i])
    BMAJ=header['BMAJ']
    BMIN=header['BMIN']
    BPA=header['BPA']  
    keyword=f'Image{i}'
    value=directory[i]
    keyword2=f'BMAJ{i}'
    value2=BMAJ
    keyword3=f'BMIN{i}'
    value3=BMIN
    keyword4=f'BPA{i}'
    value4=BPA
    temp_header[keyword] = value
    temp_header[keyword2] = value2
    temp_header[keyword3] = value3
    temp_header[keyword4] = value4

 for i in range(len(directory2)):
    data = fits.getdata(directory2[i]) 
    data=np.nan_to_num(data)
    header=fits.getheader(directory2[i])
    cube[:,i,:,:] = data[j,:,:,:]
 for key, value in temp_header.items():
     header[key] = value   
    
 fits.writeto('VLASS_cube.fits',cube,header,overwrite=True)   


 input_file_path = "VLASS_cube.fits"
# get the data cube
 fits_cube = fits.open(input_file_path)
 data_cube = check_array2(fits_cube[0].data)
#data_cube=[]
#for images in directory:
#  Data=fits.open(images)[0]
#  data=check_array(Data.data)
#  header=Data.header
#  fits.writeto(images,data,header,overwrite=True)
#  data_cube.append(images)


# calculate the sigma image
 rms_image_list = [];rms_image_list2 = [];rms_image_list3 = []
 for image in data_cube:
    sigma_image = create_sigma_image(image, array_size)
    rms_image_list.append(sigma_image)
    sigma_image2 = create_sigma_image2(image, array_size)
    rms_image_list2.append(sigma_image2)
    sigma_image3 = create_sigma_image3(image, array_size)
    rms_image_list3.append(sigma_image3)

# divide each image by it's sigma image
 sigma_corrected_images = [];sigma_corrected_images2 = [];sigma_corrected_images3 = []
 for index, image in enumerate(data_cube):
    # sigma_corrected_image = image / (rms_image_list[index] ** 2)
    sigma_corrected_image = 1/ (rms_image_list[index] ** 2)
    sigma_corrected_images.append(sigma_corrected_image)
    sigma_corrected_image2 = 1/ (rms_image_list2[index] ** 2)
    sigma_corrected_images2.append(sigma_corrected_image2)
    sigma_corrected_image3 = 1/ (rms_image_list3[index] ** 2)
    sigma_corrected_images3.append(sigma_corrected_image3)

# calculate the weighted average of sigma corrected images
 weight_corrected_images = [];weight_corrected_images2 = [];weight_corrected_images3 = []
 for index, image in enumerate(data_cube):
    weight_corrected_image = image * sigma_corrected_images[index]
    weight_corrected_images.append(weight_corrected_image)
    weight_corrected_image2 = image * sigma_corrected_images2[index]
    weight_corrected_images2.append(weight_corrected_image2)
    weight_corrected_image3 = image * sigma_corrected_images3[index]
    weight_corrected_images3.append(weight_corrected_image3)

# sum weight corrected images
 final_avg_image = np.sum(weight_corrected_images,axis=0) / np.sum(sigma_corrected_images,axis=0)
 final_avg_image2 = np.sum(weight_corrected_images2,axis=0) / np.sum(sigma_corrected_images2,axis=0)
 final_avg_image3 = np.sum(weight_corrected_images3,axis=0) / np.sum(sigma_corrected_images3,axis=0)

# sum original_images
 original_images_avg = np.sum([image for image in data_cube],axis=0)
 convert_numpy_array_to_fits(original_images_avg, imagename_base+"_normal_average.fits",header)
 convert_numpy_array_to_fits(final_avg_image, imagename_base+"_weighted_average.fits",header)
 convert_numpy_array_to_fits(final_avg_image2, imagename_base+"_weighted_median.fits",header)
 convert_numpy_array_to_fits(final_avg_image3, imagename_base+"_weighted_sigma_clip.fits",header)
 os.remove("VLASS_cube.fits")





# Run based on user flags


if __name__ == "__main__":
  try:  
    fits_files = sorted(glob.glob("*init.sc.image.tt0.fits"))
    fits_files.append(joint_image+'.image.tt0.fits')
    common_output = f"{imagename_base}_all_flux_stats.csv"

    all_results = []

    for f in fits_files:
        print(f"\n--- Processing: {f} ---")
        stats = analyze_flux_in_fits(f)
        all_results.extend(stats)

    with open(common_output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image', 'Frequency_GHz', 'Flux_Jy', 'Error_Jy',
                         'RMS_Jy/beam', 'Beam_Maj_arcsec', 'Beam_Min_arcsec', 'Source_Area_arcsec2'])
        writer.writerows(all_results)

    print(f"\n All stats saved to {common_output}")
  except:
      pass

if __name__ == "__main__":
  try:
    input_init_files = sorted(glob.glob(imagename_base+"*init.image.tt0.fits"))
    input_sc_files = sorted(glob.glob(imagename_base+"*sc.image.tt0.fits"))
    input_sc_files = [f for f in input_sc_files if "combined" not in f]
    input_GK_files = sorted(glob.glob(imagename_base+"*GK_SC.image.tt0.fits"))
    input_pol_files = sorted(glob.glob(imagename_base+'*_single_*_map.fits')) 
    input_comb_files = sorted(glob.glob(joint_image+'.image.tt0.fits'))
    input_comb_sc_files= sorted(glob.glob(joint_image+".sc.image.tt0.fits"))
    input_comb_pol_files=[joint_image+'_Q_map.fits',
                     joint_image+'_U_map.fits',
                     joint_image+'_V_map.fits']
    input_weighted_files=[imagename_base+"_normal_average.fits",
                          imagename_base+"_weighted_average.fits",
                          imagename_base+"_weighted_median.fits",
                          imagename_base+"_weighted_sigma_clip.fits"]
    input_files = sorted(glob.glob(imagename_base+"*init.sc.image.tt0.fits"))
    input_files.append(joint_image+'.image.tt0.fits')
    input_files.append(joint_image+'.sc.image.tt0.fits')
    all_html_sections = []
    all_html_sections.append(analyze_and_plot_group(input_init_files, "Initial Images"))
    all_html_sections.append(analyze_and_plot_group(input_sc_files, "Self-Calibrated Images"))
    all_html_sections.append(analyze_and_plot_group(input_GK_files, "Delay cal Images"))
    all_html_sections.append(analyze_and_plot_group(input_pol_files, "Polarization Images"))
    all_html_sections.append(analyze_and_plot_group(input_comb_files, "Combined Image"))
    all_html_sections.append(analyze_and_plot_group(input_comb_sc_files, "Combined Selfcal Image"))
    all_html_sections.append(analyze_and_plot_group(input_comb_pol_files, "Polarisation Combined Images"))
    all_html_sections.append(analyze_and_plot_group(input_weighted_files, "RMS Weighted Images"))
    if not input_files:
        print("Please provide one or more FITS files as input.")
        sys.exit(1)

    for f in input_files:
        print(f"Processing {f}")
        section = process_fits_file(f,"Selfcal map flux measurements")  # Assuming you have this function defined
        all_html_sections.append(section)


    if not input_pol_files:
        print("Please provide one or more FITS files as input.")
        sys.exit(1)

    for f1 in input_pol_files:
        print(f"Processing {f1}")
        section = process_pol_fits_file(f1,"Polarization map flux measurements")  # Assuming you have this function defined
        all_html_sections.append(section)

    if not input_comb_pol_files:
        print("Please provide one or more FITS files as input.")
        sys.exit(1)

    for f2 in input_comb_pol_files:
        print(f"Processing {f2}")
        section = process_pol_fits_file(f2,"Combined polarization map flux measurements")  # Assuming you have this function defined
        all_html_sections.append(section)

    # Spectral index plots
    spx_files = glob.glob(imagename_base+"*spx*.fits")
    spx_files = sorted(spx_files, key=extract_version_number)
    spx_html_section = "<h2>Spectral Index Maps</h2>\n"
    if not spx_files:
        spx_html_section += "<p>No spectral index maps found.</p>"
    else:
        for spx_file in spx_files:
            print(f"Adding spectral index map: {spx_file}")
            img_tag = plot_fits_image(spx_file, title=spx_file)
            spx_html_section += img_tag

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Multi-FITS Flux Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; }}
            th, td {{ text-align: center; }}
            img {{ margin-top: 10px; margin-bottom: 10px; }}
            hr {{ margin-top: 40px; }}
        </style>
    </head>
    <body>
        <h1>Multi-FITS Flux Report</h1>
        {''.join(all_html_sections)}
        <hr>
        {spx_html_section}
    </body>
    </html>
    """

    outname = imagename_base+"_multi_flux_report.html"
    with open(outname, 'w') as f:
        f.write(html_content)
    print(f"\nCombined HTML report saved to: {outname}")
  except:
      pass



