import sys,os,re,shutil
sys.path.append(os.getcwd())
from VLASS_awp_joint_workers import *
from VLASS_awp_joint_parameters import *

img_size=vlass_imagesize(R, pixel_scale=0.00016, min_size=512, round_to=32)
imsize= [img_size,img_size] 


###########################################################################################################################################################################################
for i, vis in enumerate(vis_list):
    # NEW: allow "report-only" epochs – skip empty or missing MS
    if not str(vis).strip():
        print(f"Skipping VLASS {i+1}: no MS selected in vis_list (report-only for this epoch).")
        continue
    if not os.path.exists(vis):
        print(f"Skipping VLASS {i+1}: MS '{vis}' not found on disk.")
        continue

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
                combine='spw,field', field='', selectdata=True, solnorm=False, refant=list_ant,
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
      for spw in range(2, n_spw): 
        try:   
           imagename_pol = imagename_out +'_spw'+str(spw)+'.IQUV'
           pol_cleaning(vis=vis_out, spw=str(spw),imagename=imagename_pol, imsize=imsize, phasecenter=phasecenter, mask=mask,datacolumn='corrected')
           print(f"Polarization cleaning completed for {vis_out}")
           file_pol = glob.glob(imagename_out +'_spw'+str(spw)+'.IQUV.image.tt0.fits')
           sin_pol_image_cube=fits.open(file_pol[0])
           sin_pol_data=sin_pol_image_cube[0].data
           sin_q_pol_data=sin_pol_data[1].data
           sin_u_pol_data=sin_pol_data[2].data
           sin_v_pol_data=sin_pol_data[3].data
           sin_pol_header=sin_pol_image_cube[0].header
           fits.writeto(imagename_out+'_spw'+str(spw)+'_single_Q_map.fits',sin_q_pol_data,sin_pol_header,overwrite=True)
           fits.writeto(imagename_out+'_spw'+str(spw)+'_single_U_map.fits',sin_u_pol_data,sin_pol_header,overwrite=True)
           fits.writeto(imagename_out+'_spw'+str(spw)+'_single_V_map.fits',sin_v_pol_data,sin_pol_header,overwrite=True)
        except:
           pass

    # Store final vis and image name for joint step
    processed_vis.append(vis_out)
    image_list.append(imagename_out)

# Final joint deconvolution
if processed_vis and mask:
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

if processed_vis and joint_deconvolution:
 cleaning_awp(vis=processed_vis, imagename=joint_image, datacolumn='corrected', imsize=imsize, phasecenter=phasecenter, mask=mask, savemodel='modelcolumn')
 

if processed_vis and joint_selfcal:
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

if processed_vis and joint_pol_deconvolution:
 for spw in range(2,n_spw):
   try:   
     imagename_pol = joint_image+'_spw'+str(spw)+'.IQUV'
     pol_cleaning(vis=vis_out, spw=str(spw),imagename=imagename_pol, imsize=imsize, phasecenter=phasecenter, mask=mask,datacolumn='corrected')
     print(f"Joint Polarization cleaning completed")
     pol_image_cube=fits.open(imagename_pol+'.image.tt0.fits')
     pol_data=pol_image_cube[0].data
     q_pol_data=pol_data[1].data
     u_pol_data=pol_data[2].data
     v_pol_data=pol_data[3].data
     pol_header=pol_image_cube[0].header
     fits.writeto(joint_image+'_spw'+str(spw)+'_Q_map.fits',q_pol_data,pol_header,overwrite=True)
     fits.writeto(joint_image+'_spw'+str(spw)+'_U_map.fits',u_pol_data,pol_header,overwrite=True)
     fits.writeto(joint_image+'_spw'+str(spw)+'_V_map.fits',v_pol_data,pol_header,overwrite=True)
   except:
     pass 

if processed_vis and awp_image_cube:
    delay_sc_flag=any(d.get('delay_selfcal', False) for d in processing_flags)
    # Build file lists
    if delay_sc_flag:
       model_images    = sorted(glob.glob(imagename_base+'*.GK_SC.model')) + [joint_image + ".model"]
       residual_images = sorted(glob.glob(imagename_base+'*.GK_SC.residual')) + [joint_image + ".residual"]
       psf_images      = sorted(glob.glob(imagename_base+'*.GK_SC.psf')) + [joint_image + ".psf"]
    else:
       model_images    = sorted(glob.glob(imagename_base+'*.sc.model')) + [joint_image + ".model"]
       residual_images = sorted(glob.glob(imagename_base+'*.sc.residual')) + [joint_image + ".residual"]
       psf_images      = sorted(glob.glob(imagename_base+'*.sc.psf')) + [joint_image + ".psf"]

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
if spx_map_VLASS1 and groups['vlass1']:      compute_spectral_index_map(groups['vlass1'], imagename_base+'_vlass1_spx.fits')
if spx_map_VLASS2 and groups['vlass2']:      compute_spectral_index_map(groups['vlass2'], imagename_base+'_vlass2_spx.fits')
if spx_map_VLASS3 and groups['vlass3']:      compute_spectral_index_map(groups['vlass3'], imagename_base+'_vlass3_spx.fits')
if spx_map_VLASS_combine and groups['combined']:compute_spectral_index_map(groups['combined'], imagename_base+'_vlass_combined_spx.fits')



if weighted_combine_maps:

 directory = []
 directory1 = []
 directory2= []
 mask_flag=any(d.get('mask', False) for d in processing_flags)
 if mask:
   directory = glob.glob(imagename_base+'*init_mask.sc.image.tt0.fits')
 else:
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
  #try:
    input_init_files = sorted(glob.glob(imagename_base+"*init.image.tt0.fits"))
    input_sc_files = sorted(glob.glob(imagename_base+"*sc.image.tt0.fits"))
    input_sc_files = [f for f in input_sc_files if "combined" not in f]
    input_GK_files = sorted(glob.glob(imagename_base+"*GK_SC.image.tt0.fits"))
    input_pol_files = sorted(glob.glob(imagename_base+'*_single_*_map.fits')) 
    input_comb_files = sorted(glob.glob(joint_image+'.image.tt0.fits'))
    input_comb_sc_files= sorted(glob.glob(joint_image+".sc.image.tt0.fits"))
    #input_comb_pol_files=[joint_image+'_Q_map.fits',
    #                 joint_image+'_U_map.fits',
    #                 joint_image+'_V_map.fits']
    input_comb_pol_files=sorted(glob.glob(joint_image+'_spw*'+'_*_map.fits'))
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
  #except:
  #    pass





# ==================== Added: combine selfcal models + residuals and restore ====================
def _parse_beam_strings(bmaj_str, bmin_str, bpa_str):
    if not (bmaj_str and bmin_str):
        return None, None, None
    bmaj = str(bmaj_str)
    bmin = str(bmin_str)
    bpa  = str(bpa_str) if bpa_str else "0deg"
    return bmaj, bmin, bpa

def combine_selfcal_models_and_residuals(
    imagename_base,
    epochs=("VLASS1", "VLASS2", "VLASS3"),
    weighting="mean",
    bmaj_str=None,
    bmin_str=None,
    bpa_str=None,
):
    import os, glob
    try:
        from casatools import image as ia_tool
        from casatasks import imsmooth, immath, imstat
    except Exception as e:
        print("CASA tools not available for model combination:", e)
        return None

    ia = ia_tool()
    bmaj, bmin, bpa = _parse_beam_strings(bmaj_str, bmin_str, bpa_str)

    model_images = []
    weights = []

    for ep in epochs:
        pattern_model = f"{imagename_base}_{ep}*init.sc.model.tt0"
        pattern_resid = f"{imagename_base}_{ep}*init.sc.residual.tt0"
        matches_model = sorted(glob.glob(pattern_model))
        matches_resid = sorted(glob.glob(pattern_resid))
        if not matches_model or not matches_resid:
            continue
        model = matches_model[0]
        resid = matches_resid[0]

        if bmaj is not None:
            sm_model = model + ".smooth"
            sm_resid = resid + ".smooth"
            try:
                for out_im in (sm_model, sm_resid):
                    if os.path.exists(out_im):
                        os.system("rm -rf " + out_im)
                imsmooth(imagename=model, outfile=sm_model, kernel="gauss", major=bmaj, minor=bmin, pa=bpa)
                imsmooth(imagename=resid, outfile=sm_resid, kernel="gauss", major=bmaj, minor=bmin, pa=bpa)
            except Exception as e:
                print("imsmooth failed; falling back to native beam:", e)
                sm_model = model
                sm_resid = resid
        else:
            sm_model = model
            sm_resid = resid

        restored = sm_model.replace(".model.tt0", ".restored_model.tt0")
        try:
            if os.path.exists(restored):
                os.system("rm -rf " + restored)
            immath(imagename=[sm_model, sm_resid], expr="IM0+IM1", outfile=restored)
            model_images.append(restored)
        except Exception as e:
            print("immath failed to make restored image:", e)

        try:
            st = imstat(sm_resid)
            rms = float(st.get("rms", [None])[0]) if st else None
        except Exception:
            rms = None

        if weighting == "rms" and rms and rms > 0:
            weights.append(1.0/(rms*rms))
        else:
            weights.append(1.0)

    if not model_images:
        print("No per-epoch restored model images were created; skipping combined model restored image.")
        return None

    if weighting == "rms":
        wsum = sum(weights)
        if wsum > 0:
            weights = [w/wsum for w in weights]
        else:
            weights = [1.0/len(weights)]*len(weights)
    else:
        weights = [1.0/len(weights)]*len(weights)

    expr = " + ".join([f"{w:.8g}*IM{i}" for i, w in enumerate(weights)])
    out_image = f"{imagename_base}_combined_model_restored.tt0"
    try:
        if os.path.exists(out_image):
            os.system("rm -rf " + out_image)
        immath(imagename=model_images, expr=expr, outfile=out_image)
        return out_image
    except Exception as e:
        print("Failed to make combined model restored image:", e)
        return None
# =================================================================================================


# ==================== Added: optional execution of model-based combination ====================
try:
    if 'combine_models_restore' in globals() and combine_models_restore and processed_vis:
        weighting = globals().get('model_combine_method', 'mean')
        bmaj = globals().get('model_restore_beam_maj', None)
        bmin = globals().get('model_restore_beam_min', None)
        bpa  = globals().get('model_restore_beam_pa',  None)
        base = globals().get('imagename_base', None)
        if base:
            _out = combine_selfcal_models_and_residuals(base, weighting=weighting, bmaj_str=bmaj, bmin_str=bmin, bpa_str=bpa)
            if _out:
                try:
                    from casatasks import exportfits
                    fitsname = _out + ".fits"
                    if os.path.exists(fitsname):
                        os.remove(fitsname)
                    exportfits(imagename=_out, fitsimage=fitsname, overwrite=True, history=False)
                    print("Exported combined model restored FITS:", fitsname)
                except Exception as e:
                    print("exportfits failed:", e)
except Exception as _e:
    print("Model-based combination block failed:", _e)
# ===============================================================================================


# ==================== NEW: write VLASS_processing_results/index.html and section pages =========
if __name__ == "__main__":
    # Directory to hold all web outputs
    results_dir = os.path.join(os.getcwd(), "VLASS_processing_results")
    os.makedirs(results_dir, exist_ok=True)

    # Collect FITS products (same groupings as above)
    input_init_files = sorted(glob.glob(imagename_base + "*init.image.tt0.fits"))
    input_sc_files   = sorted(glob.glob(imagename_base + "*sc.image.tt0.fits"))
    input_sc_files   = [f for f in input_sc_files if "combined" not in f]
    input_GK_files   = sorted(glob.glob(imagename_base + "*GK_SC.image.tt0.fits"))
    input_pol_files  = sorted(glob.glob(imagename_base + "*_single_*_map.fits"))

    input_comb_files     = sorted(glob.glob(joint_image + ".image.tt0.fits"))
    input_comb_sc_files  = sorted(glob.glob(joint_image + ".sc.image.tt0.fits"))
    input_comb_pol_files = sorted(glob.glob(joint_image + "_spw*" + "_*_map.fits"))

    input_weighted_files = [
        imagename_base + "_normal_average.fits",
        imagename_base + "_weighted_average.fits",
        imagename_base + "_weighted_median.fits",
        imagename_base + "_weighted_sigma_clip.fits",
    ]

    input_files = sorted(glob.glob(imagename_base + "*init.sc.image.tt0.fits"))
    if joint_image + ".image.tt0.fits" in glob.glob(joint_image + ".image.tt0.fits"):
        input_files.append(joint_image + ".image.tt0.fits")
    if joint_image + ".sc.image.tt0.fits" in glob.glob(joint_image + ".sc.image.tt0.fits"):
        input_files.append(joint_image + ".sc.image.tt0.fits")

    section_pages = []  # list of (filename, title, body_html)

    # Section 1: Initial Images
    if input_init_files:
        body = analyze_and_plot_group(input_init_files, "Initial Images")
        section_pages.append(("initial_images.html", "Initial Images", body))

    # Section 2: Self-Calibrated Images
    if input_sc_files:
        body = analyze_and_plot_group(input_sc_files, "Self-Calibrated Images")
        section_pages.append(("selfcal_images.html", "Self-Calibrated Images", body))

    # Section 3: Delay cal Images
    if input_GK_files:
        body = analyze_and_plot_group(input_GK_files, "Delay cal Images")
        section_pages.append(("delay_images.html", "Delay cal Images", body))

    # Section 4: Polarization Images
    if input_pol_files:
        body = analyze_and_plot_group(input_pol_files, "Polarization Images")
        section_pages.append(("pol_images.html", "Polarization Images", body))

    # Section 5: Combined Image
    if input_comb_files:
        body = analyze_and_plot_group(input_comb_files, "Combined Image")
        section_pages.append(("combined_image.html", "Combined Image", body))

    # Section 6: Combined Selfcal Image
    if input_comb_sc_files:
        body = analyze_and_plot_group(input_comb_sc_files, "Combined Selfcal Image")
        section_pages.append(("combined_selfcal_image.html", "Combined Selfcal Image", body))

    # Section 7: Polarisation Combined Images
    if input_comb_pol_files:
        body = analyze_and_plot_group(input_comb_pol_files, "Polarisation Combined Images")
        section_pages.append(("combined_pol_images.html", "Polarisation Combined Images", body))

    # Section 8: RMS Weighted Images
    existing_weighted = [f for f in input_weighted_files if os.path.exists(f)]

    # Also include the model-based combined image, if it exists
    combined_model_fits = imagename_base + "_combined_model_restored.tt0.fits"
    if os.path.exists(combined_model_fits):
        existing_weighted.append(combined_model_fits)

    if existing_weighted:
        body = analyze_and_plot_group(existing_weighted, "RMS Weighted Images")
        section_pages.append(("rms_weighted_images.html", "RMS Weighted Images", body))

    # Section 9: Selfcal map flux measurements
    if input_files:
        multi_html = ""
        for f in input_files:
            print(f"Processing {f} (selfcal flux)")
            multi_html += process_fits_file(f, "Selfcal map flux measurements")
        section_pages.append(("selfcal_flux.html", "Selfcal map flux measurements", multi_html))
    else:
        print("No self-cal FITS files found for VLASS_processing_results flux-measurement section.")

    # Section 10: Polarization map flux measurements
    if input_pol_files:
        pol_html = ""
        for f1 in input_pol_files:
            print(f"Processing {f1} (pol flux)")
            pol_html += process_pol_fits_file(f1, "Polarization map flux measurements")
        section_pages.append(("pol_flux.html", "Polarization map flux measurements", pol_html))

    # Section 11: Combined polarization map flux measurements
    if input_comb_pol_files:
        comb_pol_html = ""
        for f2 in input_comb_pol_files:
            print(f"Processing {f2} (combined pol flux)")
            comb_pol_html += process_pol_fits_file(
                f2, "Combined polarization map flux measurements"
            )
        section_pages.append(
            (
                "combined_pol_flux.html",
                "Combined polarization map flux measurements",
                comb_pol_html,
            )
        )

    # Section 12: Spectral Index Maps
    spx_files = glob.glob(imagename_base + "*spx*.fits")
    spx_files = sorted(spx_files, key=extract_version_number)
    spx_html_body = "<h2>Spectral Index Maps</h2>\n"
    if not spx_files:
        spx_html_body += "<p>No spectral index maps found.</p>"
    else:
        for spx_file in spx_files:
            print(f"Adding spectral index map: {spx_file}")
            img_tag = plot_fits_image(spx_file, title=os.path.basename(spx_file))
            spx_html_body += img_tag
    section_pages.append(("spectral_index_maps.html", "Spectral Index Maps", spx_html_body))

    # Write per-section pages under VLASS_processing_results/
    for fname, title, body in section_pages:
        page_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} – VLASS processing</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 20px; }}
      img  {{ margin-top: 10px; margin-bottom: 10px; }}
      hr   {{ margin-top: 40px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p><a href="index.html">&larr; Back to index</a></p>
    {body}
</body>
</html>
"""
        with open(os.path.join(results_dir, fname), "w") as f:
            f.write(page_html)

    # Main index.html linking all sections
    index_items = "\n".join(
        f'<li><a href="{fname}">{title}</a></li>' for fname, title, _ in section_pages
    )
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VLASS processing results</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 20px; }}
      li   {{ margin-bottom: 0.4em; }}
    </style>
</head>
<body>
    <h1>VLASS processing results</h1>
    <ul>
      {index_items}
    </ul>
</body>
</html>
"""
    with open(os.path.join(results_dir, "index.html"), "w") as f:
        f.write(index_html)

    print(f"\nHTML report pages written under: {results_dir}")
# ===============================================================================================


