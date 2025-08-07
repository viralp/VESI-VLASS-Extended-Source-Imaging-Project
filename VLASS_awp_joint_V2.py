import math
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Column
from astropy.table import Table
from numpy import *
import os,glob,sys,numpy,csv
import numpy as np
from astropy import units as u
from casatools import image as ia
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from skimage import measure, draw
from radio_beam import Beam
from scipy.ndimage import gaussian_filter
import os
import glob
import numpy as np
from casatools import image as ia_tool
from casatools import regionmanager
from casatasks import imsmooth, immath, exportfits, imhead
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter, binary_closing
import base64
from io import BytesIO
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


rg = regionmanager()


# Configuration per MS
vis_list = ['VLASS1.1.sb34446513.eb34497773.58018.931944780095.ms',
           'VLASS3.1.sb43597007.eb43637953.59988.53174554398.ms'
            ]
processing_flags = [
    {'target_split': False, 'init_imaging': False, 'mask': False, 'selfcal': False, 'delay_selfcal': False, 'pol_cleaning': False},
    {'target_split': False, 'init_imaging': False, 'mask': False, 'selfcal': False, 'delay_selfcal': False, 'pol_cleaning': False},
    {'target_split': False, 'init_imaging': False, 'mask': False, 'selfcal': False, 'delay_selfcal': False, 'pol_cleaning': False},
]

joint_deconvolution = False
awp_image_cube = False

# ========= CONTROL FLAGS FOR SPX MAPS =========
spx_map_VLASS1 = False
spx_map_VLASS2 = False
spx_map_VLASS3 = False
spx_map_VLASScombine = False


# For storing final visibility and image names
processed_vis = []
image_list = []


#split parameters
colname='corrected'
channelaverage=False
timeaverage=False
Ra_1=299.8681250;Dec_1=40.7339158
R=0.1

#masking parameter
mask_threshold=10

#imaging parameters
imsize= [2000,2000] 
cell= '0.6arcsec'
niter=20000
parallel=False
imagename_base='SgrA_star'
joint_image = imagename_base+'_combined_VLASS'
phasecenter='J2000 '+str(Ra_1)+'deg '+str(Dec_1)+'deg'
field=''
spw=''
uvrange=''
mask=''


def rank_refants(vis, caltable=None):
     # Get the antenna names and offsets.

     msmd = casatools.msmetadata()
     tb = casatools.table()

     msmd.open(vis)
     ids = msmd.antennasforscan(msmd.scansforintent("*OBSERVE_TARGET*")[0])
     names = msmd.antennanames(ids)
     offset = [msmd.antennaoffset(name) for name in names]
     msmd.close()

     # Calculate the mean longitude and latitude.

     mean_longitude = numpy.mean([offset[i]["longitude offset"]\
             ['value'] for i in range(len(names))])
     mean_latitude = numpy.mean([offset[i]["latitude offset"]\
             ['value'] for i in range(len(names))])

     # Calculate the offsets from the center.

     offsets = [numpy.sqrt((offset[i]["longitude offset"]['value'] -\
             mean_longitude)**2 + (offset[i]["latitude offset"]\
             ['value'] - mean_latitude)**2) for i in \
             range(len(names))]

     # Calculate the number of flags for each antenna.

     nflags = [tb.calc('[select from '+vis+' where ANTENNA1=='+\
             str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in ids]

     # Calculate the median SNR for each antenna.

     if caltable != None:
         total_snr = [tb.calc('[select from '+caltable+' where ANTENNA1=='+\
                 str(i)+' giving  [sum(SNR)]]')['0'].sum() for i in ids]

     # Calculate a score based on those two.

     score = [offsets[i] / max(offsets) + nflags[i] / max(nflags) \
             for i in range(len(names))]
     if caltable != None:
         score = [score[i] + (1 - total_snr[i] / max(total_snr)) for i in range(len(names))]

     # Print out the antenna scores.

     print("Refant list for "+vis)
     #for i in numpy.argsort(score):
     #    print(names[i], score[i])
     print(','.join(numpy.array(ids)[numpy.argsort(score)].astype(str)))
     # Return the antenna names sorted by score.

     return ','.join(numpy.array(ids)[numpy.argsort(score)].astype(str))

def cleaning(vis,field=field,spw=['2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17'], uvrange='',
       antenna=['0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25&'],
       scan=[''],
       intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected',
       imagename=imagename_base, imsize=imsize,
       cell='0.6arcsec', phasecenter=phasecenter,
       stokes='I', specmode='mfs', reffreq='3.0GHz', nchan=-1, outframe='LSRK',
       perchanweightdensity=False, gridder='mosaic', wprojplanes=32,
       mosweight=False, conjbeams=False, usepointing=False, rotatepastep=5.0,
       pointingoffsetsigdev=[300, 30], pblimit=0.2, deconvolver='mtmfs',
       scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True,
       restoringbeam='common', pbcor=False, weighting='briggs', robust=0.0,
       npixels=0, niter=20000, threshold='1e-06', nsigma=10, cycleniter=500,
       cyclefactor=3.0, interactive=False, fullsummary=True,mask=mask,
       pbmask=0.4, fastnoise=True, restart=True, savemodel='modelcolumn',
       calcres=True, calcpsf=True, parallel=False):
  if os.path.exists(imagename+'.mask'):
        os.system("rm -rf "+str(imagename+'.mask')) 
  tclean(vis=vis,field=field,spw=spw,uvrange=uvrange,antenna=antenna,
       scan=scan,intent=intent,datacolumn=datacolumn,imagename=imagename,imsize=imsize,cell=cell,
       phasecenter=phasecenter,stokes=stokes,specmode=specmode,reffreq=reffreq,
       nchan=nchan,outframe=outframe,perchanweightdensity=perchanweightdensity,
       gridder=gridder,wprojplanes=wprojplanes,mosweight=mosweight,
       conjbeams=conjbeams,usepointing=usepointing,rotatepastep=rotatepastep,
       pointingoffsetsigdev=pointingoffsetsigdev,pblimit=pblimit,
       deconvolver=deconvolver,scales=scales,nterms=nterms,smallscalebias=smallscalebias,
       restoration=restoration,restoringbeam=restoringbeam,
       pbcor=pbcor, weighting=weighting,robust=robust,npixels=npixels,
       niter=niter,threshold=threshold,nsigma=nsigma,cycleniter=cycleniter,
       cyclefactor=cyclefactor,interactive=interactive,fullsummary=fullsummary,
       mask=mask,pbmask=pbmask,fastnoise=fastnoise,
       restart=restart,savemodel=savemodel,calcres=calcres,calcpsf=calcpsf,parallel=parallel)
  exportfits(imagename=str(imagename)+'.image.tt0',fitsimage=str(imagename)+'.image.tt0.fits',overwrite=True)
  
def pol_cleaning(vis,field=field,spw=['2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17'], uvrange='',
       antenna=['0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25&'],
       scan=[''],
       intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected',
       imagename=imagename_base, imsize=imsize,
       cell='0.6arcsec', phasecenter=phasecenter,
       stokes='IQUV', specmode='mfs', reffreq='3.0GHz', nchan=-1, outframe='LSRK',
       perchanweightdensity=False, gridder='mosaic', wprojplanes=32,
       mosweight=False, conjbeams=False, usepointing=False, rotatepastep=5.0,
       pointingoffsetsigdev=[300, 30], pblimit=0.2, deconvolver='mtmfs',
       scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True,
       restoringbeam='common', pbcor=False, weighting='briggs', robust=0.0,
       npixels=0, niter=20000, threshold='1e-06', nsigma=10, cycleniter=500,
       cyclefactor=3.0, interactive=False, fullsummary=True,mask=mask,
       pbmask=0.4, fastnoise=True, restart=True, savemodel='modelcolumn',
       calcres=True, calcpsf=True, parallel=False):
  if os.path.exists(imagename+'.mask'):
        os.system("rm -rf "+str(imagename+'.mask')) 
  tclean(vis=vis,field=field,spw=spw,uvrange=uvrange,antenna=antenna,
       scan=scan,intent=intent,datacolumn=datacolumn,imagename=imagename,imsize=imsize,cell=cell,
       phasecenter=phasecenter,stokes=stokes,specmode=specmode,reffreq=reffreq,
       nchan=nchan,outframe=outframe,perchanweightdensity=perchanweightdensity,
       gridder=gridder,wprojplanes=wprojplanes,mosweight=mosweight,
       conjbeams=conjbeams,usepointing=usepointing,rotatepastep=rotatepastep,
       pointingoffsetsigdev=pointingoffsetsigdev,pblimit=pblimit,
       deconvolver=deconvolver,scales=scales,nterms=nterms,smallscalebias=smallscalebias,
       restoration=restoration,restoringbeam=restoringbeam,
       pbcor=pbcor, weighting=weighting,robust=robust,npixels=npixels,
       niter=niter,threshold=threshold,nsigma=nsigma,cycleniter=cycleniter,
       cyclefactor=cyclefactor,interactive=interactive,fullsummary=fullsummary,
       mask=mask,pbmask=pbmask,fastnoise=fastnoise,
       restart=restart,savemodel=savemodel,calcres=calcres,calcpsf=calcpsf,parallel=parallel)
  exportfits(imagename=str(imagename)+'.image.tt0',fitsimage=str(imagename)+'.image.tt0.fits',overwrite=True)
  

def cleaning_awp(vis,field='',spw=['2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17'], uvrange='',
       antenna=['0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25&'],
       scan=[''],
       intent='OBSERVE_TARGET#UNSPECIFIED', datacolumn='corrected',
       imagename=imagename_base, imsize=imsize,
       cell='0.6arcsec', phasecenter=phasecenter,
       stokes='I', specmode='mvc', reffreq='3.0GHz', nchan=-1, outframe='LSRK',
       perchanweightdensity=False, gridder='awp2', wprojplanes=32,mosweight=False,
       usepointing=False,computepastep=360.0, normtype='flatnoise',
       aterm=True,psterm=True,conjbeams=False, rotatepastep=5.0,
       pointingoffsetsigdev=[300, 30], pblimit=0.15, deconvolver='mtmfs',
       scales=[0, 5, 12], nterms=2, smallscalebias=0.4, restoration=True,
       restoringbeam='common', pbcor=False, weighting='briggs', robust=0.0,
       npixels=0, niter=20000, threshold='1e-06', nsigma=10, cycleniter=500,
       cyclefactor=3.0, interactive=False, fullsummary=True,mask=mask,
       pbmask=0.4, fastnoise=True, restart=True, savemodel='modelcolumn',
       calcres=True, calcpsf=True, parallel=False):
  if os.path.exists(imagename+'.mask'):
        os.system("rm -rf "+str(imagename+'.mask')) 
  tclean(vis=vis,field=field,spw=spw,uvrange=uvrange,antenna=antenna,
       scan=scan,intent=intent,datacolumn=datacolumn,imagename=imagename,imsize=imsize,cell=cell,
       phasecenter=phasecenter,stokes=stokes,specmode=specmode,reffreq=reffreq,
       nchan=nchan,outframe=outframe,perchanweightdensity=perchanweightdensity,
       gridder=gridder,wprojplanes=wprojplanes,mosweight=mosweight,
       conjbeams=conjbeams,usepointing=usepointing,rotatepastep=rotatepastep,
       pointingoffsetsigdev=pointingoffsetsigdev,pblimit=pblimit,
       deconvolver=deconvolver,scales=scales,nterms=nterms,smallscalebias=smallscalebias,
       restoration=restoration,restoringbeam=restoringbeam,
       pbcor=pbcor, weighting=weighting,robust=robust,npixels=npixels,
       niter=niter,threshold=threshold,nsigma=nsigma,cycleniter=cycleniter,
       cyclefactor=cyclefactor,interactive=interactive,fullsummary=fullsummary,
       mask=mask,pbmask=pbmask,fastnoise=fastnoise,
       restart=restart,savemodel=savemodel,calcres=calcres,calcpsf=calcpsf,parallel=parallel)
  exportfits(imagename=str(imagename)+'.image.tt0',fitsimage=str(imagename)+'.image.tt0.fits',overwrite=True)

  
def seperation(a1,d1,a2,d2):
 c1=SkyCoord(a1*u.degree,d1*u.degree)
 c2=SkyCoord(a2*u.degree,d2*u.degree)
 sep = c1.separation(c2)
 return sep.deg

def trim_coordsys(csys_record, num_axes=2):
    new_rec = {}
    for key, value in csys_record.items():
        if hasattr(value, '__getitem__') and not isinstance(value, str):
            try:
                new_rec[key] = value[:num_axes]
            except Exception:
                new_rec[key] = value
        else:
            new_rec[key] = value
    return new_rec


def restore_per_channel(model_image, residual_image, psf_image, output_image, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)

    ia = ia_tool()
    ia.open(model_image)
    shape = ia.shape()
    nchan = shape[3]
    unit = ia.brightnessunit()
    coordsys = ia.coordsys().torecord()
    freq_ref = ia.coordsys().referencevalue()['numeric'][3]
    freq_incr = ia.coordsys().increment()['numeric'][3]
    ia.close()
    frequencies = freq_ref + freq_incr * np.arange(nchan)

    beam_table = imhead(imagename=psf_image, mode='list')['perplanebeams']
    output_planes = []

    for ch in range(nchan):
        print(f"Processing channel {ch+1}/{nchan}")
        beam = beam_table[f'*{ch}']
        major, minor, pa = beam['major']['value'], beam['minor']['value'], beam['positionangle']['value']

        ia.open(model_image)
        model_data = ia.getchunk()[:, :, 0, ch]
        region = rg.box(blc=[0, 0, 0, ch], trc=[1, 1, 0, ch])
        ia_sub = ia.subimage(region=region, dropdeg=True)
        coords2d = ia_sub.coordsys().torecord()
        ia_sub.close()
        ia.close()

        model_plane = f'{tmpdir}/model_plane_{ch}.im'
        ia.fromarray(outfile=model_plane, pixels=model_data, overwrite=True)
        ia.setcoordsys(coords2d)
        ia.setbrightnessunit(unit)
        ia.close()

        ia.open(residual_image)
        resid_data = ia.getchunk()[:, :, 0, ch]
        region = rg.box(blc=[0, 0, 0, ch], trc=[1, 1, 0, ch])
        ia_sub = ia.subimage(region=region, dropdeg=True)
        coords2d = ia_sub.coordsys().torecord()
        ia_sub.close()
        ia.close()

        resid_plane = f'{tmpdir}/residual_plane_{ch}.im'
        ia.fromarray(outfile=resid_plane, pixels=resid_data, overwrite=True)
        ia.setcoordsys(coords2d)
        ia.setbrightnessunit(unit)
        ia.close()

        smoothed_model = f'{tmpdir}/model_conv_{ch}.im'
        imsmooth(imagename=model_plane,
                 outfile=smoothed_model,
                 kernel='gauss',
                 major=f'{major}arcsec',
                 minor=f'{minor}arcsec',
                 pa=f'{pa}deg',
                 overwrite=True)

        restored_plane = f'{tmpdir}/restored_plane_{ch}.im'
        immath(imagename=[smoothed_model, resid_plane],
               expr='IM0 + IM1',
               outfile=restored_plane)
        output_planes.append(restored_plane)

        # Export restored image
        fitsfile = restored_plane + '.fits'
        exportfits(imagename=restored_plane, fitsimage=fitsfile)
        with fits.open(fitsfile, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['BMAJ'] = major / 3600.0
            hdr['BMIN'] = minor / 3600.0
            hdr['BPA'] = pa
            hdr['CTYPE3'] = 'FREQ'
            hdr['CUNIT3'] = 'Hz'
            hdr['CRVAL3'] = frequencies[ch]
            hdr['RESTFRQ'] = frequencies[ch]
            hdul.flush()

        # === Now convolve restored image to 3″ beam and save FITS ===
        smoothed_3arcsec = restored_plane.replace('.im', '_beam3arcsec.im')
        imsmooth(imagename=restored_plane,
                 outfile=smoothed_3arcsec,
                 kernel='gauss',
                 major='3.0arcsec',
                 minor='3.0arcsec',
                 pa='0deg',
                 overwrite=True)
        fits_3arcsec = smoothed_3arcsec + '.fits'
        exportfits(imagename=smoothed_3arcsec, fitsimage=fits_3arcsec)
        print(f"Exported 3″ convolved FITS: {fits_3arcsec}")
        with fits.open(fits_3arcsec, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['BMAJ'] = major / 3600.0
            hdr['BMIN'] = minor / 3600.0
            hdr['BPA'] = pa
            hdr['CTYPE3'] = 'FREQ'
            hdr['CUNIT3'] = 'Hz'
            hdr['CRVAL3'] = frequencies[ch]
            hdr['RESTFRQ'] = frequencies[ch]
            hdul.flush()

    if os.path.exists(output_image):
        os.system(f"rm -rf {output_image}")
    #ia.imageconcat(outfile=output_image, infiles=output_planes, relax=True, axis=3)
    print(f"Restored cube saved: {output_image}")
    
# ========= FUNCTION: GENERATE SPX MAP =========
def fit_line(index_data):
        idx, oned_array = index_data
        valid = mask_stack_flat[idx]
        if np.count_nonzero(valid) < 2:
            return np.nan
        try:
            slope, _, _, _, _ = linregress(freqs[valid], oned_array[valid])
            return slope
        except Exception:
            return np.nan


def generate_spectral_index_map(filelist, tag, ncores=8):
    from astropy.io import fits
    import numpy as np
    from scipy.stats import linregress
    from multiprocessing import Pool
    from datetime import datetime
    if len(filelist) < 2:
        print(f" Skipping {tag}: need at least 2 images.")
        return
    print(f"\n Generating spectral index map for {tag.upper()} with {len(filelist)} planes")
    freqs = []
    data_stack = []
    mask_stack = []
    for f in sorted(filelist):
        with fits.open(f) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
            freq = hdr.get('CRVAL3')
            if freq is None:
                raise ValueError(f"No CRVAL3 in {f}")
            freqs.append(np.log10(freq))
            arr = np.squeeze(data)
            arr[arr <= 0] = np.nan
            arr_log = np.log10(arr)
            rms = np.nanstd(arr)
            mask = arr > 3 * rms
            data_stack.append(arr_log)
            mask_stack.append(mask)
    freqs = np.array(freqs)
    data_stack = np.stack(data_stack)
    mask_stack = np.stack(mask_stack).astype(bool)
    nchan, ny, nx = data_stack.shape
    # Prepare for fitting
    transposed_array = np.transpose(data_stack, (1, 2, 0)).reshape(ny * nx, nchan)
    mask_stack_flat = np.transpose(mask_stack, (1, 2, 0)).reshape(ny * nx, nchan)
    fit_input = list(enumerate(transposed_array))
    del data_stack  # free memory
    print("Fitting...")
    start = datetime.now()
    with Pool(ncores) as p:
        slopes = p.map(fit_line, fit_input)
    alpha_map = np.array(slopes).reshape(ny, nx)
    print(f"Done. Shape: {alpha_map.shape}. Time: {datetime.now() - start}")
    # Save FITS
    header = fits.getheader(filelist[0]).copy()
    for k in ['CRVAL3', 'CDELT3', 'CTYPE3', 'CUNIT3']:
        header.pop(k, None)
    header['BUNIT'] = 'Spectral Index'
    fits.writeto(f"spx_map_{tag}.fits", alpha_map, header, overwrite=True)
    print(f"Saved: spx_map_{tag}.fits")
    

def analyze_flux_in_fits(file):
    hdu = fits.open(file)
    header = hdu[0].header
    data = hdu[0].data
    (a, b, xx, yy) = data.shape
    spws = b
    pixsize = abs(header['CDELT1']) * 3600  # arcsec
    freq = header['CRVAL3']
    freq_delta = header['CDELT3']

    output_stats = []

    for i in range(spws):
        data_2d = data[0][i]
        data_2d = np.nan_to_num(data_2d, nan=0.0, posinf=0.0, neginf=0.0)
        data_smoothed = gaussian_filter(data_2d, sigma=1.0)
        mean, median, std = sigma_clipped_stats(data_smoothed, sigma=3.0)
        threshold = median + 3 * std
        binary_map = data_smoothed > threshold
        contours = measure.find_contours(binary_map, level=0.5)

        if contours:
            largest_contour = max(contours, key=len)
            rr, cc = draw.polygon(largest_contour[:, 0], largest_contour[:, 1], data_2d.shape)
            mask = np.zeros(data_2d.shape, dtype=bool)
            mask[rr, cc] = True
            masked_data = data_2d * mask
            raw_flux = np.sum(masked_data)

            major = header['BMAJ'] * 3600
            minor = header['BMIN'] * 3600
            beam = Beam(major=u.arcsec * major, minor=u.arcsec * minor)
            beam_area_arcsec2 = beam.sr.to(u.arcsec**2).value
            beam_area_pix = beam_area_arcsec2 / (pixsize ** 2)
            flux_density = raw_flux / beam_area_pix

            # Error calculation
            sigma1 = 0.10 * flux_density
            sigma_rms = std
            src_area_pix = np.count_nonzero(mask)
            src_area_arcsec2 = src_area_pix * (pixsize ** 2)
            syn_beam_area = major * minor
            sigma2 = sigma_rms * np.sqrt(src_area_arcsec2 / syn_beam_area)
            flux_error = np.sqrt(sigma1**2 + sigma2**2)

            print(f"✔ SPW {i+1} | {freq/1e9:.2f} GHz | Flux: {flux_density:.2f} ± {flux_error:.2f} Jy")

            # Contrast enhancement
            contrast_factor = 10.0
            center_value = np.median(data_2d)
            data_std = np.std(data_2d)
            vmin = center_value - contrast_factor * data_std
            vmax = center_value + contrast_factor * data_std

            plt.figure(figsize=(10, 8))
            plt.imshow(data_2d, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
            plt.colorbar(label='Intensity')
            plt.plot(largest_contour[:, 1], largest_contour[:, 0], color='red', linewidth=1.5)
            plt.title(f"{os.path.basename(file)} | SPW {i+1}: {flux_density:.2f} ± {flux_error:.2f} Jy")
            fig_name = file.replace('.fits', f'_spw{i+1:03d}_fluxmap.png')
            plt.savefig(fig_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ↳ Saved: {fig_name}")

            # Save to list
            output_stats.append([
                os.path.basename(file), round(freq / 1e9, 4), round(flux_density, 4),
                round(flux_error, 4), round(std, 4), round(major, 2),
                round(minor, 2), round(src_area_arcsec2, 2)
            ])
        else:
            print(f" No contour found at {freq / 1e9:.2f} GHz")

        freq += freq_delta

    return output_stats

def analyze_and_plot_group(files, group_title, zoom_fraction=0.25):
    html_section = f"<h2>{group_title}</h2>\n"
    for file in files:
        if not os.path.exists(file):
            continue
        print(f"Analyzing {file}")

        with fits.open(file) as hdu:
            header = hdu[0].header
            data = hdu[0].data.squeeze()

        if data.ndim != 2:
            print(f"Skipping {file}, not a 2D image.")
            continue

        # Image properties
        ny, nx = data.shape
        pixsize = abs(header['CDELT1']) * 3600  # arcsec
        major = header.get('BMAJ', 0) * 3600
        minor = header.get('BMIN', 0) * 3600

        # Stats
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        smoothed = gaussian_filter(data, sigma=3.0)
        mean, median, std = sigma_clipped_stats(smoothed, sigma=3.0)
        peak = np.max(data)
        min_pixel = np.min(data)
        dr1 = peak / std
        dr2 = abs(peak / min_pixel)

        # Display scaling
        contrast = 10.0
        vmin = np.median(data) - contrast * std
        vmax = np.median(data) + contrast * std

        # Full image
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        ax1.set_title("Full Image")
        ax1.set_xticks([])
        ax1.set_yticks([])
        buf1 = BytesIO()
        fig1.tight_layout()
        fig1.savefig(buf1, format='png', dpi=150)
        plt.close(fig1)
        img_full = base64.b64encode(buf1.getvalue()).decode('utf-8')
        buf1.close()

        # Zoom image
        cx, cy = nx // 2, ny // 2
        half = int(nx * zoom_fraction // 2)
        zoom_data = data[cy - half:cy + half, cx - half:cx + half]

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.imshow(zoom_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        ax2.set_title("Zoomed Center")
        ax2.set_xticks([])
        ax2.set_yticks([])
        buf2 = BytesIO()
        fig2.tight_layout()
        fig2.savefig(buf2, format='png', dpi=150)
        plt.close(fig2)
        img_zoom = base64.b64encode(buf2.getvalue()).decode('utf-8')
        buf2.close()

        # Add to HTML
        html_section += f"""
        <h3>{os.path.basename(file)}</h3>
        <p><b>RMS</b>: {std*1e6:.2f} µJy/beam | <b>Beam</b>: {major:.2f}" × {minor:.2f}" |
           <b>Peak</b>: {peak:.4f} Jy/beam | <b>DR1</b>: {dr1:.1f} | <b>DR2</b>: {dr2:.1f}</p>
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
          <div><img src="data:image/png;base64,{img_full}" style="border:1px solid #ccc; max-width:100%;"/></div>
          <div><img src="data:image/png;base64,{img_zoom}" style="border:1px solid #ccc; max-width:100%;"/></div>
        </div>
        <hr>
        """

    return html_section


def process_fits_file(file):
    hdu = fits.open(file)
    header = hdu[0].header
    data = hdu[0].data
    (a, b, xx, yy) = data.shape
    spws = b
    pixsize = abs(header['CDELT1']) * 3600  # arcsec
    freq = header['CRVAL3']
    freq_delta = header['CDELT3']
    html_sections = []

    for i in range(spws):
        data_2d = data[0][i]
        data_2d = np.nan_to_num(data_2d, nan=0.0, posinf=0.0, neginf=0.0)
        data_smoothed = gaussian_filter(data_2d, sigma=3.0)
        mean, median, std = sigma_clipped_stats(data_smoothed, sigma=3.0)
        rms_ujy = std * 1e6
        threshold = median + 5 * std
        binary_map = data_smoothed > threshold
        binary_map = binary_closing(binary_map, structure=np.ones((5, 5)))

        contours = measure.find_contours(binary_map, level=0.1)
        if not contours:
            print(f"No source detected in {file} at freq {freq / 1e9:.2f} GHz")
            freq += freq_delta
            continue

        # Beam info
        major = header['BMAJ'] * 3600
        minor = header['BMIN'] * 3600
        beam = Beam(major=u.arcsec * major, minor=u.arcsec * minor)
        beam_area_arcsec2 = beam.sr.to(u.arcsec**2).value
        beam_area_pix = beam_area_arcsec2 / (pixsize ** 2)
        syn_beam_area = major * minor

        peak = np.max(data_2d)
        min_pixel = np.min(data_2d)
        dr1 = peak / std
        dr2 = abs(peak / min_pixel)

        source_table_rows = ""
        label_coords = []

        for s, contour in enumerate(contours):
            rr, cc = draw.polygon(contour[:, 0], contour[:, 1], data_2d.shape)
            mask = np.zeros_like(data_2d, dtype=bool)
            mask[rr, cc] = True

            masked_data = data_2d * mask
            raw_flux = np.sum(masked_data)
            flux_density = raw_flux / beam_area_pix

            sigma1 = 0.10 * flux_density
            src_area_pix = np.count_nonzero(mask)
            src_area_arcsec2 = src_area_pix * (pixsize ** 2)
            sigma2 = std * np.sqrt(src_area_arcsec2 / syn_beam_area)
            flux_error = np.sqrt(sigma1**2 + sigma2**2)

            y_coords, x_coords = contour[:, 0], contour[:, 1]
            max_dist_pix = np.max([
                np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                for x1, y1 in zip(x_coords, y_coords)
                for x2, y2 in zip(x_coords, y_coords)
            ])
            max_dist_arcsec = max_dist_pix * pixsize

            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            label_coords.append((s+1, centroid_x, centroid_y))

            source_table_rows += f"""
            <tr>
              <td>{s+1}</td><td>{freq/1e9:.2f}</td><td>{flux_density:.4f}</td><td>{flux_error:.4f}</td>
              <td>{src_area_arcsec2:.2f}</td><td>{max_dist_arcsec:.2f}</td>
            </tr>
            """

        # Plot full image
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        contrast_factor = 10.0
        vmin = np.median(data_2d) - contrast_factor * np.std(data_2d)
        vmax = np.median(data_2d) + contrast_factor * np.std(data_2d)
        ax1.imshow(data_2d, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

        for s, contour in enumerate(contours):
            y_coords, x_coords = contour[:, 0], contour[:, 1]
            ax1.plot(x_coords, y_coords, color='red', linewidth=1.5)
        for sid, cx, cy in label_coords:
            ax1.text(cx, cy, f'{sid}', color='yellow', fontsize=8,
                     ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=1))

        ax1.set_title(f"{os.path.basename(file)} | SPW {i+1}")
        buf1 = BytesIO()
        fig1.tight_layout()
        fig1.savefig(buf1, format='png', dpi=150)
        plt.close(fig1)
        img_full_b64 = base64.b64encode(buf1.getvalue()).decode('utf-8')
        buf1.close()

        # Plot zoomed center
        zoom_fraction = 0.25
        x_center = xx // 2
        y_center = yy // 2
        half_size = int(xx * zoom_fraction // 2)
        x1, x2 = x_center - half_size, x_center + half_size
        y1, y2 = y_center - half_size, y_center + half_size
        zoomed_data = data_2d[y1:y2, x1:x2]

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.imshow(zoomed_data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)

        for s, contour in enumerate(contours):
            y_coords, x_coords = contour[:, 0], contour[:, 1]
            inside = (x_coords >= x1) & (x_coords < x2) & (y_coords >= y1) & (y_coords < y2)
            if np.any(inside):
                ax2.plot(x_coords[inside] - x1, y_coords[inside] - y1, color='red', linewidth=1.5)

        for sid, cx, cy in label_coords:
            if x1 <= cx < x2 and y1 <= cy < y2:
                ax2.text(cx - x1, cy - y1, f'{sid}', color='yellow', fontsize=8,
                         ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, pad=1))

        ax2.set_title("Zoomed Center")
        buf2 = BytesIO()
        fig2.tight_layout()
        fig2.savefig(buf2, format='png', dpi=150)
        plt.close(fig2)
        img_zoom_b64 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        buf2.close()

        html_section = f"""
        <h2>{os.path.basename(file)} | SPW {i+1}: {freq/1e9:.2f} GHz</h2>
        <p><b>RMS</b>: {rms_ujy:.2f} µJy/beam | <b>Beam</b>: {major:.2f}" × {minor:.2f}" |
           <b>Peak</b>: {peak:.4f} Jy/beam | <b>DR1</b>: {dr1:.1f} | <b>DR2</b>: {dr2:.1f}</p>
        <div style="display:flex; gap:20px; flex-wrap:wrap;">
          <div>
            <h4>Full Image</h4>
            <img src="data:image/png;base64,{img_full_b64}" style="border:1px solid #ccc; max-width:100%;"/>
          </div>
          <div>
            <h4>Zoomed Center</h4>
            <img src="data:image/png;base64,{img_zoom_b64}" style="border:1px solid #ccc; max-width:100%;"/>
          </div>
        </div>
        <br>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr>
            <th>Source ID</th><th>Frequency (GHz)</th><th>Flux (Jy)</th><th>Error (Jy)</th>
            <th>Area (arcsec²)</th><th>Size (arcsec)</th>
          </tr>
          {source_table_rows}
        </table>
        <hr>
        """
        html_sections.append(html_section)
        freq += freq_delta

    return '\n'.join(html_sections)
    

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_fits_image(fits_file, cmap='coolwarm', title=None, zoom_size=100):
    with fits.open(fits_file) as hdul:
        data = hdul[0].data.squeeze()
        if data.ndim != 2:
            raise ValueError(f"{fits_file} is not a 2D FITS image.")

        vmin, vmax = np.nanpercentile(data, [5, 95])
        ny, nx = data.shape
        cx, cy = nx // 2, ny // 2

        # Extract zoom region
        zoom_data = data[cy - zoom_size:cy + zoom_size, cx - zoom_size:cx + zoom_size]

        # Create side-by-side plots
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Full image
        im0 = axs[0].imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axs[0].set_title('Full Image', fontsize=10)
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='Spectral Index')

        # Zoomed-in image
        im1 = axs[1].imshow(zoom_data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title('Central Zoom', fontsize=10)
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(title or fits_file, fontsize=12)
        plt.tight_layout()

        # Convert to base64 image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_base64}" width="800"><br><br>'

def extract_version_number(filename):
    match = re.search(r'V(\d+)', filename)
    return int(match.group(1)) if match else float('inf')    

for i, vis in enumerate(vis_list):
    flags = processing_flags[i]
    vis_base = vis[:-3]
    imagename = f"{imagename_base}_{vis_base}"
    vis_out = vis

    # Split by region if flag is True
    if flags['target_split']:
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
        vis_out = vis_base + f'_fields_seperation_{R}deg.ms'
        mstransform(vis=vis, outputvis=vis_out, field=img_fields, combinespws=False,
                    datacolumn=colname, chanaverage=channelaverage, timeaverage=timeaverage)
    else:
        vis_out = vis_base + f'_fields_seperation_{R}deg.ms'

    # Initial imaging
    if flags['init_imaging']:
        imagename_out=imagename+'_init'
        cleaning_awp(vis=vis_out, imagename=imagename_out, datacolumn='data',savemodel='modelcolumn')
        imagename_out = imagename_out
    else:
        imagename_out = imagename+'_init'

    # Masking
    if flags['mask']:
        os.system(f"breizorro --restored-image {imagename_out}.image.tt0.fits --threshold {mask_threshold} --outfile {imagename_out}.sigma.mask.fits")
        importfits(fitsimage=imagename_out+".sigma.mask.fits", imagename=imagename_out+".sigma.mask")
        mask = imagename_out + ".sigma.mask"
        cleaning_awp(vis=vis_out,imagename=imagename_out+'_mask', datacolumn='data', mask=mask, savemodel='modelcolumn')
        imagename_out = imagename_out + '_mask'
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
        cleaning_awp(vis=vis_out, imagename=imagename_out+'.sc', datacolumn='corrected',
                     mask=mask, savemodel='modelcolumn')
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

        cleaning_awp(vis=vis_out, imagename=imagename_out+'.GK_SC', datacolumn='corrected',
                     mask=mask, savemodel='modelcolumn')
        imagename_out = imagename_out + '.GK_SC'
        
    if flags['pol_cleaning']:
        imagename_pol = imagename_out + '.IQUV'
        pol_cleaning(vis=vis_out, imagename=imagename_pol, mask=mask,datacolumn='corrected')
        print(f"Polarization cleaning completed for {vis_out}")
   

    # Store final vis and image name for joint step
    processed_vis.append(vis_out)
    image_list.append(imagename_out)

# Final joint deconvolution
if joint_deconvolution:
 cleaning_awp(vis=processed_vis, imagename=joint_image, datacolumn='corrected', mask=mask, savemodel='modelcolumn')
 imagename_pol = joint_image + '.IQUV'
 pol_cleaning(vis=vis_out, imagename=imagename_pol, mask=mask,datacolumn='corrected',savemodel='modelcolumn')
 print(f"Joint AWP and Polarization cleaning completed")


if awp_image_cube:
    # Build file lists
    model_images    = sorted(glob.glob('*.GK_SC.model')) + [joint_image + ".model"]
    residual_images = sorted(glob.glob('*.GK_SC.residual')) + [joint_image + ".residual"]
    psf_images      = sorted(glob.glob('*.GK_SC.psf')) + [joint_image + ".psf"]

    assert len(psf_images) == len(model_images) == len(residual_images), "Mismatch in image sets!"

    for idx, (model, resid, psf) in enumerate(zip(model_images, residual_images, psf_images)):
        name = os.path.basename(model).replace('.GK_SC.model', '').replace('.model', '')
        tmpdir = f"temp_planes_{name}"
        os.system('mkdir -p ' + tmpdir)
        outname = f"{name}_restored_cube.im"
        print(f"\n Processing: {name}")
        restore_per_channel(model, resid, psf, outname, tmpdir)

    print("\n All datasets processed and smoothed.")





# ========= GROUP FILES AND GENERATE =========
from glob import glob
import os

print("\n Grouping smoothed FITS files by VLASS epoch")
fits_all = sorted(glob("temp_planes*/*beam3arcsec.im.fits"))
groups = {'VLASS1': [], 'VLASS2': [], 'VLASS3': [], 'combined': []}
for f in fits_all:
    fname = f#os.path.basename(f).lower()
    if 'VLASS1' in fname:
        groups['VLASS1'].append(f)
    elif 'vlass2' in fname:
        groups['vlass2'].append(f)
    elif 'vlass3' in fname:
        groups['vlass3'].append(f)
    elif 'joint' in fname or 'comb' in fname:
        groups['combined'].append(f)

# Run based on user flags
if spx_map_VLASS1:      generate_spectral_index_map(groups['vlass1'], 'vlass1')
if spx_map_VLASS2:      generate_spectral_index_map(groups['vlass2'], 'vlass2')
if spx_map_VLASS3:      generate_spectral_index_map(groups['vlass3'], 'vlass3')
if spx_map_VLASScombine:generate_spectral_index_map(groups['combined'], 'combined')

import glob
if __name__ == "__main__":
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
    
if __name__ == "__main__":

    input_init_files = sorted(glob.glob("*init.image.tt0.fits"))
    input_sc_files = sorted(glob.glob("*sc.image.tt0.fits"))
    input_GK_files = sorted(glob.glob("*GK_SC.image.tt0.fits"))
    input_comb_files = sorted(glob.glob(joint_image+'.image.tt0.fits'))
    input_files = sorted(glob.glob("*init.sc.image.tt0.fits"))
    input_files.append(joint_image+'.image.tt0.fits')
    all_html_sections = []
    all_html_sections.append(analyze_and_plot_group(input_init_files, "Initial Images"))
    all_html_sections.append(analyze_and_plot_group(input_sc_files, "Self-Calibrated Images"))
    all_html_sections.append(analyze_and_plot_group(input_GK_files, "Delay cal Images"))
    all_html_sections.append(analyze_and_plot_group(input_comb_files, "Combined Images"))
    if not input_files:
        print("Please provide one or more FITS files as input.")
        sys.exit(1)

    for f in input_files:
        print(f"Processing {f}")
        section = process_fits_file(f)  # Assuming you have this function defined
        all_html_sections.append(section)

    # Spectral index plots
    spx_files = (glob.glob("*spx*.fits"))
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

    outname = "multi_flux_report.html"
    with open(outname, 'w') as f:
        f.write(html_content)
    print(f"\nCombined HTML report saved to: {outname}")

