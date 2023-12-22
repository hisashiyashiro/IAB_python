import numpy as np
import problem_size as ps
import mod_oprt3d as mo
from timeit import default_timer as timer

data = np.fromfile("dyn_divdamp/snapshot.dyn_divdamp.pe000000", dtype=np.dtype('>f8'))

ddivdx_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
ddivdx_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
ddivdy_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
ddivdy_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
ddivdz_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
ddivdz_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
rhogvx_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
rhogvx_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
rhogvy_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
rhogvy_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
rhogvz_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
rhogvz_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
rhogw_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
rhogw_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl

coef_intp_f    = ps.ADM_gall*3*ps.ADM_nxyz*2*ps.ADM_lall
coef_intp_pl_f = ps.ADM_gall_pl*3*ps.ADM_nxyz*ps.ADM_lall_pl
coef_diff_f    = ps.ADM_gall*6*ps.ADM_nxyz*ps.ADM_lall
coef_diff_pl_f = ps.ADM_vlink*ps.ADM_nxyz*ps.ADM_lall_pl

RGSQRTH_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
RGSQRTH_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl

RGAM_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
RGAM_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl

RGAMH_f = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
RGAMH_pl_f = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl

C2WfactGz_f = ps.ADM_gall * ps.ADM_kall * 6 * ps.ADM_lall
C2WfactGz_pl_f = ps.ADM_gall_pl * ps.ADM_kall * 6 * ps.ADM_lall_pl

f_list = [ddivdx_f, ddivdx_pl_f, ddivdy_f, ddivdy_pl_f, ddivdz_f, ddivdz_pl_f, rhogvx_f, rhogvx_pl_f,
          rhogvy_f, rhogvy_pl_f, rhogvz_f, rhogvz_pl_f, rhogw_f, rhogw_pl_f,
          coef_intp_f, coef_intp_pl_f, coef_diff_f, coef_diff_pl_f, RGSQRTH_f,
          RGSQRTH_pl_f, RGAM_f, RGAM_pl_f, RGAMH_f, RGAMH_pl_f, C2WfactGz_f, C2WfactGz_pl_f]

ddivdx    = np.zeros((ps.ADM_gall   , ps.ADM_kall, ps.ADM_lall), order='F')
ddivdx_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
ddivdy    = np.zeros((ps.ADM_gall   , ps.ADM_kall, ps.ADM_lall), order='F')
ddivdy_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
ddivdz    = np.zeros((ps.ADM_gall   , ps.ADM_kall, ps.ADM_lall), order='F')
ddivdz_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')

check_ddivdx_np = np.reshape(data[:f_list[0]], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
check_ddivdx_pl_np = np.reshape(data[f_list[0]:sum(f_list[:2])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
check_ddivdy_np = np.reshape(data[sum(f_list[:2]):sum(f_list[:3])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
check_ddivdy_pl_np = np.reshape(data[sum(f_list[:3]):sum(f_list[:4])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
check_ddivdz_np = np.reshape(data[sum(f_list[:4]):sum(f_list[:5])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
check_ddivdz_pl_np = np.reshape(data[sum(f_list[:5]):sum(f_list[:6])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
rhogvx_np = np.reshape(data[sum(f_list[:6]):sum(f_list[:7])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
rhogvx_pl_np = np.reshape(data[sum(f_list[:7]):sum(f_list[:8])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
rhogvy_np = np.reshape(data[sum(f_list[:8]):sum(f_list[:9])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
rhogvy_pl_np = np.reshape(data[sum(f_list[:9]):sum(f_list[:10])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
rhogvz_np = np.reshape(data[sum(f_list[:10]):sum(f_list[:11])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
rhogvz_pl_np = np.reshape(data[sum(f_list[:11]):sum(f_list[:12])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')
rhogw_np = np.reshape(data[sum(f_list[:12]):sum(f_list[:13])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
rhogw_pl_np = np.reshape(data[sum(f_list[:13]):sum(f_list[:14])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')

coef_intp_np = np.reshape(data[sum(f_list[:14]):sum(f_list[:15])], (ps.ADM_gall,3,ps.ADM_nxyz,2,ps.ADM_lall), order='F')
coef_intp_pl_np = np.reshape(data[sum(f_list[:15]):sum(f_list[:16])], (ps.ADM_gall_pl,3,ps.ADM_nxyz,ps.ADM_lall_pl), order='F')
coef_diff_np = np.reshape(data[sum(f_list[:16]):sum(f_list[:17])], (ps.ADM_gall, 6, ps.ADM_nxyz, ps.ADM_lall), order='F')
coef_diff_pl_np = np.reshape(data[sum(f_list[:17]):sum(f_list[:18])], (ps.ADM_vlink, ps.ADM_nxyz, ps.ADM_lall_pl), order='F')

RGSQRTH_np = np.reshape(data[sum(f_list[:18]):sum(f_list[:19])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
RGSQRTH_pl_np = np.reshape(data[sum(f_list[:19]):sum(f_list[:20])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')

RGAM_np = np.reshape(data[sum(f_list[:20]):sum(f_list[:21])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
RGAM_pl_np = np.reshape(data[sum(f_list[:21]):sum(f_list[:22])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl) , order='F')

RGAMH_np = np.reshape(data[sum(f_list[:22]):sum(f_list[:23])], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall), order='F')
RGAMH_pl_np = np.reshape(data[sum(f_list[:23]):sum(f_list[:24])], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl), order='F')

C2WfactGz_np = np.reshape(data[sum(f_list[:24]):sum(f_list[:25])], (ps.ADM_gall, ps.ADM_kall, 6, ps.ADM_lall), order='F')
C2WfactGz_pl_np = np.reshape(data[sum(f_list[:25]):sum(f_list[:26])], (ps.ADM_gall_pl, ps.ADM_kall, 6, ps.ADM_lall_pl), order='F')

def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

start = timer()
for iteration in range(ps.SET_iteration):
    mo.OPRT3D_divdamp(ddivdx, ddivdx_pl, ddivdy, ddivdy_pl, ddivdz, ddivdz_pl, 
                      rhogvx_np, rhogvx_pl_np, rhogvy_np, rhogvy_pl_np, rhogvz_np, rhogvz_pl_np, 
                      rhogw_np, rhogw_pl_np, coef_intp_np, coef_intp_pl_np, 
                      coef_diff_np, coef_diff_pl_np, RGSQRTH_np, RGSQRTH_pl_np, RGAM_np, RGAM_pl_np,
                      RGAMH_np, RGAMH_pl_np, C2WfactGz_np, C2WfactGz_pl_np)
end = timer()  


print('### Input ###')

print_res(check_ddivdx_np, 'x_check\t\t')
print_res(check_ddivdx_pl_np, 'x_check_pl\t')
print_res(check_ddivdy_np, 'y_check\t\t')
print_res(check_ddivdy_pl_np, 'y_check_pl\t')
print_res(check_ddivdz_np, 'z_check\t\t')
print_res(check_ddivdz_pl_np, 'z_check_pl\t')

print_res(rhogvx_np, 'rhogvx\t\t')
print_res(rhogvx_pl_np, 'rhogvx_pl\t')
print_res(rhogvy_np, 'rhogvy\t\t')
print_res(rhogvy_pl_np, 'rhogvy_pl\t')
print_res(rhogvz_np, 'rhogvz\t\t')
print_res(rhogvz_pl_np, 'rhogvz_pl\t')
print_res(rhogw_np, 'rhogw\t\t')
print_res(rhogw_pl_np, 'rhogw_pl\t')

print('### Output ###')
print_res(ddivdx, 'x_post\t\t')
print_res(ddivdx_pl, 'x_post_pl\t')
print_res(ddivdy, 'y_post\t\t')
print_res(ddivdy_pl, 'y_post_pl\t')
print_res(ddivdz, 'z_post\t\t')
print_res(ddivdz_pl, 'z_post_pl\t')

print('### Validation : point-by-point diff ###')
res_x = np.subtract(check_ddivdx_np, ddivdx)
res_x_pl = np.subtract(check_ddivdx_pl_np, ddivdx_pl)

res_y = np.subtract(check_ddivdy_np, ddivdy)
res_y_pl = np.subtract(check_ddivdy_pl_np, ddivdy_pl)

res_z = np.subtract(check_ddivdz_np, ddivdz)
res_z_pl = np.subtract(check_ddivdz_pl_np, ddivdz_pl)

print_res(res_x, 'res_x\t\t')
print_res(res_x_pl, 'res_x_pl\t')
print_res(res_y, 'res_y\t\t')
print_res(res_y_pl, 'res_y_pl\t')
print_res(res_z, 'res_z\t\t')
print_res(res_z_pl, 'res_z_pl\t')

print('Elapsed time=', round(end - start, 2), 'sec')