# Configuration per MS
vis_list = ['VLASS1.1.sb34523741.eb34557765.58027.86045872685_target.ms',
           'VLASS2.1.sb38607339.eb38658209.59109.972473263886_target_split.ms',
           'VLASS3.1.sb43271439.eb43441449.59966.291735266204_target_split.ms'
            ]

processing_flags = [
    {'target_split': True, 'init_imaging': True, 'mask': False, 'selfcal': True, 'delay_selfcal': True, 'pol_cleaning': True},
    {'target_split': True, 'init_imaging': True, 'mask': False, 'selfcal': True, 'delay_selfcal': True, 'pol_cleaning': True},
    {'target_split': True, 'init_imaging': True, 'mask': False, 'selfcal': True, 'delay_selfcal': True, 'pol_cleaning': True},
]

joint_deconvolution = True
joint_selfcal = True
joint_pol_deconvolution = True
awp_image_cube = True


# ========= CONTROL FLAGS FOR SPX MAPS =========
spx_map_VLASS1 = True
spx_map_VLASS2 = True
spx_map_VLASS3 = True
spx_map_VLASS_combine = True
weighted_combine_maps = True


# For storing final visibility and image names
processed_vis = []
image_list = []


#split parameters
colname='corrected'
channelaverage=False
timeaverage=False
Ra_1=202.7844767;Dec_1=30.5091463
R=0.1

#masking parameter
mask_threshold=10

#imaging parameters
cell= '0.6arcsec'
niter=20000
parallel=False
imagename_base='3C286_QSO'
joint_image = imagename_base+'_combined_VLASS'
phasecenter='J2000 '+str(Ra_1)+'deg '+str(Dec_1)+'deg'
field=''
spw=''
uvrange=''
mask='';mask_maps=[]
usepointing=True
pointingoff=[300,30]

#weighted combined maps
array_size=5


