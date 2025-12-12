import os
import json
import casatools

# -------------------- Added: model-based combined restore options --------------------
combine_models_restore = False        # default: off
model_combine_method = "mean"         # "mean" or "rms"
model_restore_beam_maj = None         # e.g. "2.5arcsec"
model_restore_beam_min = None         # e.g. "2.5arcsec"
model_restore_beam_pa  = None         # e.g. "0deg"
# ------------------------------------------------------------------------------------

# ----------------------------------------------------------------------
# DEFAULT PARAMETERS (used when no VLASS_web_config.json is present)
# ----------------------------------------------------------------------

# Default measurement sets (fallback; will be overridden by JSON when used via web)
vis_list = [
    'VLASS1.1.sb34899305.eb35070592.58164.31599563657.ms',
    'VLASS2.1.sb38757737.eb38880054.59154.83674578703.ms',
    'VLASS3.1.sb43970504.eb44126827.60110.254475497684.ms',
]

processing_flags = [
    {
        'target_split': False,
        'init_imaging': True,
        'mask': False,
        'selfcal': True,
        'delay_selfcal': False,
        'pol_cleaning': True,
    },
    {
        'target_split': False,
        'init_imaging': True,
        'mask': False,
        'selfcal': True,
        'delay_selfcal': False,
        'pol_cleaning': True,
    },
    {
        'target_split': False,
        'init_imaging': True,
        'mask': False,
        'selfcal': True,
        'delay_selfcal': False,
        'pol_cleaning': True,
    },
]

joint_deconvolution = True
joint_selfcal = True
joint_pol_deconvolution = True
awp_image_cube = True

# ========= CONTROL FLAGS FOR SPX MAPS =========
# ========= CONTROL FLAGS FOR SPX MAPS =========
spx_map_VLASS1 = True
spx_map_VLASS2 = True
spx_map_VLASS3 = True
spx_map_VLASS_combine = True
weighted_combine_maps = True

# Additional control for model-combined restored image from self-cal maps
combine_models_restore   = False
model_combine_weighting  = 'mean'   # 'mean' or 'rms'
model_restore_beam_maj   = 3.0      # arcsec
model_restore_beam_min   = 3.0      # arcsec
model_restore_beam_pa    = 0.0      # deg

# For storing final visibility and image names
processed_vis = []
image_list = []

# Split parameters
colname        = 'corrected'
channelaverage = False
timeaverage    = False
Ra_1           = 266.4194
Dec_1          = -29.0042
R              = 0.1  # deg, half-size

# Masking parameter
mask_threshold = 3.0

# Imaging parameters
cell           = '0.6arcsec'
niter          = 20000
parallel       = False
imagename_base = 'Sgr_pol'
joint_image    = imagename_base + '_combined_VLASS'
phasecenter    = f'J2000 {Ra_1}deg {Dec_1}deg'
field          = ''
spw            = ''
uvrange        = ''
mask           = ''
usepointing    = True
pointingoff    = [300, 30]

# Weighted combined maps
array_size = 5
mask_maps  = []

# Pol-cube / spectral defaults (will be refined once we inspect an MS)
freq_0         = None          # e.g. "2.999GHz"
channel_width  = None          # e.g. "2.0MHz"
total_channels = 64            # VLASS default
specmode       = 'mfs'         # for tclean


# ----------------------------------------------------------------------
# Optional override from web interface (VLASS_web_config.json)
# ----------------------------------------------------------------------
_cfg_path = "VLASS_web_config.json"
if os.path.exists(_cfg_path):
    try:
        with open(_cfg_path) as _f:
            _cfg = json.load(_f)

        # vis_list + per-epoch flags
        vis_list        = _cfg.get("vis_list",        vis_list)
        processing_flags = _cfg.get("processing_flags", processing_flags)

        # joint_* and AWP / SPX flags
        joint_deconvolution     = _cfg.get("joint_deconvolution",     joint_deconvolution)
        joint_selfcal           = _cfg.get("joint_selfcal",           joint_selfcal)
        joint_pol_deconvolution = _cfg.get("joint_pol_deconvolution", joint_pol_deconvolution)
        awp_image_cube          = _cfg.get("awp_image_cube",          awp_image_cube)

        spx_map_VLASS1        = _cfg.get("spx_map_VLASS1",        spx_map_VLASS1)
        spx_map_VLASS2        = _cfg.get("spx_map_VLASS2",        spx_map_VLASS2)
        spx_map_VLASS3        = _cfg.get("spx_map_VLASS3",        spx_map_VLASS3)
        spx_map_VLASS_combine = _cfg.get("spx_map_VLASS_combine", spx_map_VLASS_combine)
        weighted_combine_maps = _cfg.get("weighted_combine_maps", weighted_combine_maps)

        # split parameters
        colname        = _cfg.get("colname",        colname)
        channelaverage = _cfg.get("channelaverage", channelaverage)
        timeaverage    = _cfg.get("timeaverage",    timeaverage)
        Ra_1           = _cfg.get("Ra_1",           Ra_1)
        Dec_1          = _cfg.get("Dec_1",          Dec_1)
        R              = _cfg.get("R",              R)

        # masking parameter
        mask_threshold = _cfg.get("mask_threshold", mask_threshold)

        # imaging parameters
        cell           = _cfg.get("cell",           cell)
        niter          = _cfg.get("niter",          niter)
        parallel       = _cfg.get("parallel",       parallel)
        imagename_base = _cfg.get("imagename_base", imagename_base)
        joint_image    = imagename_base + '_combined_VLASS'
        phasecenter    = f'J2000 {Ra_1}deg {Dec_1}deg'
        field          = _cfg.get("field",          field)
        spw            = _cfg.get("spw",            spw)
        uvrange        = _cfg.get("uvrange",        uvrange)
        mask           = _cfg.get("mask",           mask)
        usepointing    = _cfg.get("usepointing",    usepointing)
        pointingoff    = _cfg.get("pointingoff",    pointingoff)

        # weighted combined maps
        array_size     = _cfg.get("array_size",     array_size)

        print(f"Loaded VLASS_web_config.json, vis_list = {vis_list}")
    except Exception as _exc:
        print("Warning: could not read VLASS_web_config.json:", _exc)
        print("         Falling back to hard-coded defaults.")
else:
    print("No VLASS_web_config.json found; using hard-coded defaults.")


# ----------------------------------------------------------------------
# Pol-cube / spectral setup AFTER vis_list is final
# ----------------------------------------------------------------------

msmd = casatools.msmetadata()

# Find the first non-empty, existing MS to use as a spectral reference
_ref_ms = None
for _candidate in vis_list:
    if _candidate and isinstance(_candidate, str) and os.path.exists(_candidate):
        _ref_ms = _candidate
        break

if _ref_ms:
    print(f"Using {_ref_ms} as reference MS for spectral setup.")
    msmd.open(_ref_ms)
    # First SPW (id=0), first channel (index 0)
    start_freq_Hz  = msmd.chanfreqs(0)[0]   # center frequency of chan 0
    chan_width_Hz  = msmd.chanwidths(0)[0]  # width of chan 0 (may be negative)
    n_spw          = msmd.nspw()
    msmd.done()

    freq_0        = f"{start_freq_Hz / 1e9}GHz"
    channel_width = f"{abs(chan_width_Hz) / 1e6}MHz"
    total_channels = 64    # you can adjust if you want to infer this from MS
    specmode       = 'mfs'
else:
    # This will happen if:
    #  - All vis_list entries are "" (report-only/HTML-only run), or
    #  - The paths don't exist in the current working directory.
    print("Warning: no valid MS found in vis_list for spectral setup.")
    print("         Using generic defaults for freq_0 and channel_width.")
    if freq_0 is None:
        freq_0 = "3.0GHz"       # safe generic value
    if channel_width is None:
        channel_width = "2.0MHz"
    total_channels = 64
    specmode       = 'mfs'

# At this point:
#  - vis_list, processing_flags, etc. reflect the web interface when used there
#  - freq_0, channel_width, total_channels, specmode are consistent with
#    the first valid MS, or fall back to generic defaults if we had none.



# -------------------- Added: read model-combine options from web config --------------------
try:
    _web_cfg_path = "VLASS_web_config.json"
    if os.path.exists(_web_cfg_path):
        with open(_web_cfg_path) as _wf:
            _wcfg = json.load(_wf)
        try:
            combine_models_restore = _wcfg.get("combine_models_restore", combine_models_restore)
            model_combine_method   = _wcfg.get("model_combine_method", model_combine_method)
            model_target_beam      = _wcfg.get("model_target_beam", "")
            if model_target_beam:
                _parts = [p.strip() for p in re.split(r"[,\s]+", model_target_beam) if p.strip()]
                if len(_parts) >= 2:
                    model_restore_beam_maj = _parts[0]
                    model_restore_beam_min = _parts[1]
                if len(_parts) >= 3:
                    model_restore_beam_pa = _parts[2]
        except Exception as _e:
            pass
except Exception:
    pass
# ------------------------------------------------------------------------------------------
