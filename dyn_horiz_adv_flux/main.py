import numpy as np
import problem_size as ps
import mod_src_tracer as mst
import mod_misc as mm
from timeit import default_timer as timer

data = np.fromfile("dyn_horiz_adv_flux/snapshot.dyn_horiz_adv_flux.pe000000", dtype=np.dtype('>f8'))

size_all = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
size_pl = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
size_k0 = ps.ADM_gall * ps.K0 * ps.ADM_lall
size_k0_pl = ps.ADM_gall_pl * ps.K0 * ps.ADM_lall_pl

shape_all = (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall)
shape_pl = (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl)

size1 = size_all * 6
size2 = size_pl
size3 = size_all * 3 * 3
size4 = size_pl * 3
size5 = size_k0 * 3 * 3
size6 = size_k0_pl * 3
size7 = size_k0 * ps.GMTR_p_nmax
size8 = size_k0_pl * ps.GMTR_p_nmax
size9 = size_k0 * 2 * ps.GMTR_t_nmax
size10 = size_k0_pl * ps.GMTR_t_nmax
size11 = size_k0 * 3 * ps.GMTR_a_nmax
size12 = size_k0_pl * ps.GMTR_a_nmax_pl


flx_h = np.zeros((ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 6))
flx_h_pl = np.zeros(shape_pl)

GRD_xc = np.zeros((ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 3, 3))
GRD_xc_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl, 3))

check_flx_h = np.reshape(data[:size1], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 6), order='F')
data = data[size1:]

check_flx_h_pl = np.reshape(data[:size2], shape_pl, order='F')
data = data[size2:]

check_GRD_xc = np.reshape(data[:size3], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 3, 3), order='F')
data = data[size3:]

check_GRD_xc_pl = np.reshape(data[:size4], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl, 3), order='F')
data = data[size4:]

rhog_mean = np.reshape(data[:size_all], shape_all, order='F')
data = data[size_all:]

rhog_mean_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

rhog_vx = np.reshape(data[:size_all], shape_all, order='F')
data = data[size_all:]

rhog_vx_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

rhog_vy = np.reshape(data[:size_all], shape_all, order='F')
data = data[size_all:]

rhog_vy_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

rhog_vz = np.reshape(data[:size_all], shape_all, order='F')
data = data[size_all:]

rhog_vz_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

GRD_xr = np.reshape(data[:size5], (ps.ADM_gall, ps.K0, ps.ADM_lall, 3, 3), order='F')
data = data[size5:]

GRD_xr_pl = np.reshape(data[:size6], (ps.ADM_gall_pl, ps.K0, ps.ADM_lall_pl, 3), order='F')
data = data[size6:]

GMTR_p = np.reshape(data[:size7], (ps.ADM_gall, ps.K0, ps.ADM_lall, ps.GMTR_p_nmax), order='F')
data = data[size7:]

GMTR_p_pl = np.reshape(data[:size8], (ps.ADM_gall_pl, ps.K0, ps.ADM_lall_pl, ps.GMTR_p_nmax), order='F')
data = data[size8:]

GMTR_t = np.reshape(data[:size9], (ps.ADM_gall, ps.K0, ps.ADM_lall, 2, ps.GMTR_t_nmax), order='F')
data = data[size9:]

GMTR_t_pl = np.reshape(data[:size10], (ps.ADM_gall_pl, ps.K0, ps.ADM_lall_pl, ps.GMTR_t_nmax), order='F')
data = data[size10:]

GMTR_a = np.reshape(data[:size11], (ps.ADM_gall, ps.K0, ps.ADM_lall, 3, ps.GMTR_a_nmax), order='F')
data = data[size11:]

GMTR_a_pl = np.reshape(data[:size12], (ps.ADM_gall_pl, ps.K0, ps.ADM_lall_pl, ps.GMTR_a_nmax_pl), order='F')
data = data[size12:]

# flx_h[:,:,:,:] = check_flx_h[:,:,:,:]
# GRD_xc[:,:,:,:,:] = check_GRD_xc[:,:,:,:,:]
def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

# print_res(flx_h, 'pre')

# print('min', '%.1f' % np.sum(GRD_xr_pl))
start = timer()
for iteration in range(ps.SET_iteration):
    print('in')
    mst.horizontal_flux(flx_h, flx_h_pl, GRD_xc, GRD_xc_pl, rhog_mean, rhog_mean_pl,
                        rhog_vx, rhog_vx_pl, rhog_vy, rhog_vy_pl, rhog_vz, rhog_vz_pl, ps.SET_dt_large,
                        GRD_xr, GRD_xr_pl, GMTR_p, GMTR_p_pl, GMTR_t, GMTR_t_pl, GMTR_a, GMTR_a_pl)
end = timer()
# print_res(flx_h, 'flx_h')
# print_res(check_flx_h, 'check')
# print_res(GRD_xc,    'GRD_xc')
# print_res(check_GRD_xc,    'check ')

# print_res(flx_h_pl, 'flx_h_pl')
# print_res(check_flx_h_pl, 'check   ')
# print_res(GRD_xc_pl, 'GRD_xc_pl')
# print_res(check_GRD_xc_pl, 'check    ')


print('Input')
print_res(check_flx_h, 'flx_h')
print_res(check_flx_h_pl, 'flx_h_pl')

print_res(check_GRD_xc,    'GRD_xc')
print_res(check_GRD_xc_pl, 'GRD_xc_pl')

print_res(rhog_mean,    'rhog_mean')
print_res(rhog_mean_pl, 'rhog_mean_pl')

print_res(rhog_vx,    'rhog_vx')
print_res(rhog_vx_pl, 'rhog_vx_pl')

print_res(rhog_vy,    'rhog_vy')
print_res(rhog_vy_pl, 'rhog_vy_pl')

print_res(rhog_vz,    'rhog_vz')
print_res(rhog_vz_pl, 'rhog_vz_pl')

print_res(GRD_xr,    'GRD_xr')
print_res(GRD_xr_pl, 'GRD_xr_pl')

print_res(GMTR_p,    'GMTR_p')
print_res(GMTR_p_pl, 'GMTR_p_pl')

print_res(GMTR_t,    'GMTR_t')
print_res(GMTR_t_pl, 'GMTR_t_pl')

print_res(GMTR_a,    'GMTR_a')
print_res(GMTR_a_pl, 'GMTR_a_pl')

print('### Output ###')
print_res(flx_h, 'flx_h')
print_res(flx_h_pl, 'flx_h_pl')
print_res(GRD_xc,    'GRD_xc')
print_res(GRD_xc_pl, 'GRD_xc_pl')

res_flx = np.subtract(check_flx_h, flx_h)
res_flx_pl = np.subtract(check_flx_h_pl, flx_h_pl)

res_GRD = np.subtract(check_GRD_xc, GRD_xc)
res_GRD_pl = np.subtract(check_GRD_xc_pl, GRD_xc_pl)

print('### Validation : point-by-point diff ###')
print_res(res_flx, 'check_flx_h')
print_res(res_flx_pl, 'check_flx_h_pl')
print_res(res_GRD, 'check_GRD_xc')
print_res(res_GRD_pl, "check_GRD_xc_pl")

print('Elapsed time=', round(end - start, 2), 'sec')