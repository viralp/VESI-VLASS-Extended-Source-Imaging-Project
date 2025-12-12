__rethrow_casa_exceptions = True
context = h_init()
context.set_state('ProjectSummary', 'proposal_code', 'VLASS')
context.set_state('ProjectSummary', 'proposal_title', 'unknown')
context.set_state('ProjectSummary', 'piname', 'unknown')
context.set_state('ProjectSummary', 'observatory', 'Karl G. Jansky Very Large Array')
context.set_state('ProjectSummary', 'telescope', 'EVLA')
context.set_state('ProjectStructure', 'ppr_file', 'PPR.xml')
context.set_state('ProjectStructure', 'recipe_name', 'hifv_vlassSEIP')
vis='VLASS1.1.sb34899305.eb35070592.58164.31599563657.ms'


try:
    hifv_importdata(vis=vis,session=['session_1'],nocopy=True)
    hif_editimlist(parameter_file='SEIP_parameter.list')
    hif_transformimagedata(datacolumn='corrected', clear_pointing=False, modify_weights=True, wtmode='nyq')
    hifv_vlassmasking(maskingmode='vlass-se-tier-1', vlass_ql_database='/lustre/aoc/users/mlacy/VLASS12Q_CIRADA.fits')
    hif_makeimages(hm_masking='manual')
    hifv_checkflag(checkflagmode='vlass-imaging')
    hifv_statwt(statwtmode='VLASS-SE', datacolumn='residual_data')
    hifv_selfcal(selfcalmode='VLASS-SE')
    hif_editimlist(parameter_file='SEIP_parameter.list')
    hif_makeimages(hm_masking='manual')
    hif_editimlist(parameter_file='SEIP_parameter.list')
    hifv_vlassmasking(maskingmode='vlass-se-tier-2')
    hif_makeimages(hm_masking='manual')
    hifv_pbcor()
    hif_makermsimages()
    hif_makecutoutimages()
    hif_analyzealpha()
    hifv_exportvlassdata()
finally:
    h_save()

