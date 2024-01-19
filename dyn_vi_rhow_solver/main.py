import numpy as np
import problem_size as ps
import mod_vi as mv
import mod_misc as mm
from timeit import default_timer as timer

data = np.fromfile("dyn_vi_rhow_solver/snapshot.dyn_vi_rhow_solver.pe000000", dtype=np.dtype('>f8'))

size_all = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
size_pl = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl

shape_all = (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall)
shape_pl = (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl)

rhogw = np.zeros(shape_all)
rhogw_pl = np.zeros(shape_pl)

rhogw_prev = np.reshape(data[:size_all], shape_all, order='F')
rhogw_prev_pl = np.reshape(data[size_all:size_all+size_pl], shape_pl, order='F')

check_rhogw = np.reshape(data[size_all+size_pl:2*size_all+size_pl], shape_all, order='F')
check_rhogw_pl = np.reshape(data[2*size_all+size_pl:2*(size_all+size_pl)], shape_pl, order='F')

rhogw0 = np.reshape(data[2*(size_all+size_pl):3*size_all+2*size_pl], shape_all, order='F')
rhogw0_pl = np.reshape(data[3*size_all+2*size_pl:3*(size_all+size_pl)], shape_pl, order='F')

preg0 = np.reshape(data[3*(size_all+size_pl):4*size_all+3*size_pl], shape_all, order='F')
preg0_pl = np.reshape(data[4*size_all+3*size_pl:4*(size_all+size_pl)], shape_pl, order='F')

rhog0 = np.reshape(data[4*(size_all+size_pl):5*size_all+4*size_pl], shape_all, order='F')
rhog0_pl = np.reshape(data[5*size_all+4*size_pl:5*(size_all+size_pl)], shape_pl, order='F')

Srho = np.reshape(data[5*(size_all+size_pl):6*size_all+5*size_pl], shape_all, order='F')
Srho_pl = np.reshape(data[6*size_all+5*size_pl:6*(size_all+size_pl)], shape_pl, order='F')

Sw = np.reshape(data[6*(size_all+size_pl):7*size_all+6*size_pl], shape_all, order='F')
Sw_pl = np.reshape(data[7*size_all+6*size_pl:7*(size_all+size_pl)], shape_pl, order='F')

Spre = np.reshape(data[7*(size_all+size_pl):8*size_all+7*size_pl], shape_all, order='F')
Spre_pl = np.reshape(data[8*size_all+7*size_pl:8*(size_all+size_pl)], shape_pl, order='F')

Mc = np.reshape(data[8*(size_all+size_pl):9*size_all+8*size_pl], shape_all, order='F')
Mc_pl = np.reshape(data[9*size_all+8*size_pl:9*(size_all+size_pl)], shape_pl, order='F')

Ml = np.reshape(data[9*(size_all+size_pl):10*size_all+9*size_pl], shape_all, order='F')
Ml_pl = np.reshape(data[10*size_all+9*size_pl:10*(size_all+size_pl)], shape_pl, order='F')

Mu = np.reshape(data[10*(size_all+size_pl):11*size_all+10*size_pl], shape_all, order='F')
Mu_pl = np.reshape(data[11*size_all+10*size_pl:11*(size_all+size_pl)], shape_pl, order='F')

RGSGAM2 = np.reshape(data[11*(size_all+size_pl):size_all+11*(size_all+size_pl)], shape_all, order='F')
RGSGAM2_pl = np.reshape(data[size_all+11*(size_all+size_pl):12*(size_all+size_pl)], shape_pl, order='F')

RGSGAM2H = np.reshape(data[12*(size_all+size_pl):size_all+12*(size_all+size_pl)], shape_all, order='F')
RGSGAM2H_pl = np.reshape(data[size_all+12*(size_all+size_pl):13*(size_all+size_pl)], shape_pl, order='F')

RGAMH = np.reshape(data[13*(size_all+size_pl):size_all+13*(size_all+size_pl)], shape_all, order='F')
RGAMH_pl = np.reshape(data[size_all+13*(size_all+size_pl):14*(size_all+size_pl)], shape_pl, order='F')

RGAM = np.reshape(data[14*(size_all+size_pl):size_all+14*(size_all+size_pl)], shape_all, order='F')
RGAM_pl = np.reshape(data[size_all+14*(size_all+size_pl):15*(size_all+size_pl)], shape_pl, order='F')

GSGAM2H = np.reshape(data[15*(size_all+size_pl):size_all+15*(size_all+size_pl)], shape_all, order='F')
GSGAM2H_pl = np.reshape(data[size_all+15*(size_all+size_pl):16*(size_all+size_pl)], shape_pl, order='F')

# arr = GSGAM2H
# f = open('test.txt', 'w+')
# for l in range(ps.ADM_lall):
#     for k in range(ps.ADM_kall):
#         for g in range(ps.ADM_gall):
#             f.write(str(arr[g][k][l]))
#             f.write('\n')

# print(arr, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

# arr_pl = GSGAM2H_pl
# f2 = open('test2.txt', 'w+')
# for l in range(ps.ADM_lall_pl):
#     for k in range(ps.ADM_kall):
#         for g in range(ps.ADM_gall_pl):
#             f2.write(str(arr_pl[g][k][l]))
#             f2.write('\n')

# print(arr_pl, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

print_res(rhogw_pl, 'pre\t')
mm.GRD_setup()
start = timer()
for i in range(ps.SET_iteration):
    rhogw = rhogw_prev
    rhogw_pl = rhogw_prev_pl

    mv.vi_rhow_solver(rhogw, rhogw_pl, rhogw0, rhogw0_pl, preg0, preg0_pl, rhog0, rhog0_pl,
                      Srho, Srho_pl, Sw, Sw_pl, Spre, Spre_pl, Mu, Mu_pl, Ml, Ml_pl, Mc, Mc_pl, ps.SET_dt_small, 
                      RGAMH, RGAMH_pl, RGSGAM2, RGSGAM2_pl,
                      RGAM, RGAM_pl, RGSGAM2H, RGSGAM2H_pl, GSGAM2H, GSGAM2H_pl)
end = timer()


print('### Input ###')
print_res(rhogw_prev, 'rhogw_prev\t')
print_res(rhogw_prev_pl, 'rhogw_prev_pl\t')
print_res(check_rhogw,    'check_rhogw\t')
print_res(check_rhogw_pl, 'check_rhogw_pl\t')
print_res(rhogw0,    'rhogw0\t')
print_res(rhogw0_pl, 'rhogw0_pl\t')
print_res(preg0,    'preg0\t')
print_res(preg0_pl, 'preg0_pl\t')
print_res(Srho,    'Srho\t')
print_res(Srho_pl, 'Srho_pl\t')
print_res(Sw,    'Sw\t')
print_res(Sw_pl, 'Sw_pl\t')
print_res(Spre,    'Spre\t')
print_res(Spre_pl, 'Spre_pl\t')
print_res(Mc,    'Mc\t')
print_res(Mc_pl, 'Mc_pl\t')
print_res(Ml,    'Ml\t')
print_res(Ml_pl, 'Ml_pl\t')
print_res(Mu,    'Mu\t')
print_res(Mu_pl, 'Mu_pl\t')
print('### Output ###')
print_res(rhogw,    'rhogw\t')
print_res(rhogw_pl, 'rhogw_pl\t')

print('### Validation : point-by-point diff ###')
res_rhogw    = np.subtract(check_rhogw,    rhogw)
res_rhogw_pl = np.subtract(check_rhogw_pl, rhogw_pl)
print_res(res_rhogw,    'res_rhogw\t')
print_res(res_rhogw_pl, 'res_rhogw_pl\t')

print('Elapsed time=', round(end - start, 2), 'sec')