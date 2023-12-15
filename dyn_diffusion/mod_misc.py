import dyn_divdamp.problem_size as ps
import mod_precision as mp

#++ Public parameters & variables

EX_CSTEP_diffusion = 0
EX_TSTEP_diffusion = 60
EX_CSTEP_divdamp3d = 0
EX_TSTEP_divdamp3d = 140
EX_fid = 0
EX_err = 0
EX_fname = ''
EX_item = ''
EX_max = 0.0
EX_min = 0.0
EX_sum = 0.0

#++ Private parameters & variables

DEBUG_rapnlimit = 100
DEBUG_rapnmax   = 0
DEBUG_rapname = ''
DEBUG_raptstr = 0.0
DEBUG_rapttot = 0.0
DEBUG_rapnstr = 0
DEBUG_rapnend = 0

# ifdef _FIXEDINDEX_

# function DEBUG_rapid

# subroutine DEBUG_rapstart

# subroutine DEBUG_rapend

# subroutine DEUG_rapreport

# subroutine DEBUG_valuecheck_1D

# subroutine DEBUG_valuecheck_2D

# subroutine DEBUG_valuecheck_3D

# subroutine DEBUG_valuecheck_4D

# subroutine DEBUG_valuecheck_5D

# subroutine DEBUG_valuecheck_6D

# subroutine MISC_make_idstr

# function MISC_get_available_fid()

# subroutine ADM_proc_stop

# subroutine CONST_setup

# subroutine GRD_setup

# subroutine GRD_input_vgrid

