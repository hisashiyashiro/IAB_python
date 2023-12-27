import problem_size as ps
import numpy as np
import pandas as pd

EX_CSTEP_diffusion = 0
EX_TSTEP_diffusion = 60
EX_CSTEP_divdamp3d = 0
EX_TSTEP_divdamp3d = 140

GRD_gz = np.zeros(ps.ADM_kall)
GRD_gzh = np.zeros(ps.ADM_kall)
GRD_dgz = np.zeros(ps.ADM_kall)
GRD_dgzh = np.zeros(ps.ADM_kall)
GRD_rdgz = np.zeros(ps.ADM_kall)
GRD_rdgzh = np.zeros(ps.ADM_kall)
GRD_afact = np.zeros(ps.ADM_kall)
GRD_bfact = np.zeros(ps.ADM_kall)
GRD_cfact = np.zeros(ps.ADM_kall)
GRD_dfact = np.zeros(ps.ADM_kall)

data = np.fromfile("dyn_vi_rhow_solver/vgrid40_600m_24km.dat", dtype=np.dtype('>f8'))

# for i in range(len(data)):
#     print(data[i])

def GRD_input_vgrid():
    gz = np.reshape(data[2:ps.ADM_kall+2], (ps.ADM_kall))
    gzh = np.reshape(data[ps.ADM_kall+3:], (ps.ADM_kall))
    GRD_gz[:] = gz
    GRD_gzh[:] = gzh
    # print(GRD_gz)
    # print(GRD_gzh)
    return #GRD_gz, GRD_gzh

def GRD_setup():
    GRD_input_vgrid()
    for k in range(ps.ADM_kmin-2, ps.ADM_kmax):
        GRD_dgz[k] = GRD_gzh[k+1] - GRD_gzh[k]
    GRD_dgz[ps.ADM_kmax] = GRD_dgz[ps.ADM_kmax-1]

    for k in range(ps.ADM_kmin - 1, ps.ADM_kmax+1):
        GRD_dgzh[k] = GRD_gz[k] - GRD_gz[k-1]
    
    GRD_dgzh[ps.ADM_kmin-2] = GRD_dgzh[ps.ADM_kmin-1]

    for k in range(ps.ADM_kall):
        GRD_rdgz[k] = 1.0 / GRD_dgz[k]
        GRD_rdgzh[k] = 1.0 / GRD_dgzh[k]

    for k in range(ps.ADM_kmin-1, ps.ADM_kmax+1):
        GRD_afact[k] = (GRD_gzh[k] - GRD_gz[k-1]) / (GRD_gz[k] - GRD_gz[k-1])
    
    GRD_afact[ps.ADM_kmin - 2] = 1.0
    GRD_bfact[:] = 1.0 - GRD_afact[:]

    for k in range(ps.ADM_kmin-1, ps.ADM_kmax):
        GRD_cfact[k] = (GRD_gz[k] - GRD_gzh[k]) / (GRD_gzh[k+1] - GRD_gzh[k])

    GRD_cfact[ps.ADM_kmin-2] = 1.0
    GRD_cfact[ps.ADM_kmax] = 0.0

    GRD_dfact[:] = 1.0 - GRD_cfact[:]

    return


# GRD_gz, GRD_gzh = GRD_input_vgrid()

# print(GRD_gz)
# print(GRD_gzh)

# GRD_input_vgrid()


# print(GRD_gz)
# print(GRD_gzh)
# print(GRD_dgz)
# print(GRD_dgzh)
# print(GRD_rdgz)
# print(GRD_rdgzh)
# print(GRD_afact)
# print(GRD_bfact)