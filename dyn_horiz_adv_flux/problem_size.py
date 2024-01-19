ADM_NSYS     = 32
ADM_MAXFNAME = 1024
ADM_LOG_FID  = 6

IO_FREAD = 0

#--- Identifier of triangle element (i-axis-side or j-axis-side)
TI = 0
TJ = 1
#--- Identifier of triangle element (i-axis-side or j-axis-side)
AI  = 0
AIJ = 1
AJ  = 2

#--- Identifier of 1 variable
K0 = 1

ADM_nxyz = 3 # dimension of the spacial vector

#--- Region
ADM_lall    = 1 # number of regular region per process
ADM_lall_pl = 2 # number of polex    region per process

#--- horizontal grid
ADM_gall      = 16900 # number of horizontal grid per regular region
ADM_gall_1d   = 130   # number of horizontal grid (1D)
ADM_gmin      = 2     # start index of 1D horizontal grid
ADM_gmax      = 129   # end   index of 1D horizontal grid

ADM_gall_pl   = 6     # number of horizontal grid for pole region
ADM_gslf_pl   = 1     # index for pole point
ADM_gmin_pl   = 2     # start index of grid around the pole point
ADM_gmax_pl   = 6     # end   index of grid around the pole point
ADM_vlink     = 5     # number of grid around the pole point

#--- vertical grid
ADM_vlayer    = 40    # number of vertical layer
ADM_kall      = 42    # number of vertical grid
ADM_kmin      = 2     # start index of vertical grid
ADM_kmax      = 41    
ADM_have_pl   = True  # this ID manages pole region?

def ADM_have_sgp(n):
    return n <= 1

#--- constant parameters
CONST_PI    = 3.14159265358979    #< pi
CONST_EPS   = 1.E-16              #< small number
CONST_HUGE  = 1.E+30              #< huge  number
CONST_GRAV  = 9.80616             #< Gravitational accerlaration of the Earth [m/s2]
CONST_Rdry  =   287.0             #< Gas constant of air
CONST_CVdry =   717.5             #< Specific heat of air (consant volume)
NON_HYDRO_ALPHA = 1               #< Nonhydrostatic/hydrostatic flag

#--- mod_grd
XDIR = 0
YDIR = 1
ZDIR = 2

GRD_LAT = 1
GRD_LON = 2

GRD_rscale = 6.37122E+6 #< radius of the planet [m]

GRD_grid_type_on_sphere = 1
GRD_grid_type_on_plane  = 2
GRD_grid_type           = GRD_grid_type_on_sphere

vgrid_fname = './vgrid40_600m_24km.dat'

#--- mod_gmtr
GMTR_polygon_type = 'ON_SPHERE'

GMTR_p_nmax = 8

GMTR_p_AREA  = 1
GMTR_p_RAREA = 2
GMTR_p_IX    = 3
GMTR_p_IY    = 4
GMTR_p_IZ    = 5
GMTR_p_JX    = 6
GMTR_p_JY    = 7
GMTR_p_JZ    = 8

GMTR_t_nmax = 5

GMTR_t_AREA  = 1
GMTR_t_RAREA = 2
GMTR_t_W1    = 3
GMTR_t_W2    = 4
GMTR_t_W3    = 5

GMTR_a_nmax    = 12
GMTR_a_nmax_pl = 18

GMTR_a_HNX  = 1
GMTR_a_HNY  = 2
GMTR_a_HNZ  = 3
GMTR_a_HTX  = 4
GMTR_a_HTY  = 5
GMTR_a_HTZ  = 6
GMTR_a_TNX  = 7
GMTR_a_TNY  = 8
GMTR_a_TNZ  = 9
GMTR_a_TTX  = 10
GMTR_a_TTY  = 11
GMTR_a_TTZ  = 12

GMTR_a_TN2X = 13
GMTR_a_TN2Y = 14
GMTR_a_TN2Z = 15
GMTR_a_TT2X = 16
GMTR_a_TT2Y = 17
GMTR_a_TT2Z = 18

P_RAREA = GMTR_p_RAREA
T_RAREA = GMTR_t_RAREA
W1      = GMTR_t_W1
W2      = GMTR_t_W2
W3      = GMTR_t_W3
HNX     = GMTR_a_HNX
HNY     = GMTR_a_HNY
HNZ     = GMTR_a_HNZ
HTX     = GMTR_a_HTX
TNX     = GMTR_a_TNX
TN2X    = GMTR_a_TN2X

SET_iteration = 1
SET_prc_me    = 1
SET_dt_large  = 360.0
SET_dt_small  =  60.0 


