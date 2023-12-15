# dyn_diffusion

# import pefile
import codecs
import base64
import struct
import numpy as np
import problem_size as ps
from mod_oprt import OPRT_diffusion
from timeit import default_timer as timer

data = np.fromfile("dyn_diffusion/snapshot.dyn_diffusion.pe000000", dtype=np.dtype('>f8'))


# for i in data:
#     data_file.write(str(i))
#     data_file.write('\n')

# data_file.close()

# Create variables

dscl    = np.zeros((ps.ADM_gall   , ps.ADM_kall, ps.ADM_lall))
dscl_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl))
scl     = np.zeros((ps.ADM_gall   , ps.ADM_kall, ps.ADM_lall))
scl_pl  = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl))
kh      = np.zeros((ps.ADM_gall   , ps.ADM_kall, ps.ADM_lall))
kh_pl   = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl))

# coef_intp    = np.zeros((ps.ADM_gall,    3, ps.ADM_nxyz,  2,  ps.ADM_lall))
coef_intp_pl = np.zeros((ps.ADM_gall_pl, 3, ps.ADM_nxyz,      ps.ADM_lall_pl))
coef_diff    = np.zeros((ps.ADM_gall,    6, ps.ADM_nxyz,      ps.ADM_lall))
coef_diff_pl = np.zeros((     ps.ADM_vlink, ps.ADM_nxyz,      ps.ADM_lall_pl))

check_dscl    = np.zeros((ps.ADM_gall   , ps.ADM_kall, ps.ADM_lall))
check_dscl_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl))

dscl_f    =   ps.ADM_gall*ps.ADM_kall*ps.ADM_lall
dscl_pl_f =   ps.ADM_gall*ps.ADM_kall*ps.ADM_lall + ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl 
scl_f     = 2*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall + ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl 
scl_pl_f  = 2*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall + 2*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl 
kh_f      = 3*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall + 2*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl 
kh_pl_f   = 3*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall + 3*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl

coef_intp_f    = ( 3*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall 
                 + 3*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl
                 + ps.ADM_gall*3*ps.ADM_nxyz*2*ps.ADM_lall )
coef_intp_pl_f = ( 3*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall 
                 + 3*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl
                 + ps.ADM_gall*3*ps.ADM_nxyz*2*ps.ADM_lall 
                 + ps.ADM_gall_pl*3*ps.ADM_nxyz*ps.ADM_lall_pl)

coef_diff_f    = ( 3*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall 
                 + 3*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl
                 + ps.ADM_gall*3*ps.ADM_nxyz*2*ps.ADM_lall 
                 + ps.ADM_gall_pl*3*ps.ADM_nxyz*ps.ADM_lall_pl
                 + ps.ADM_gall*6*ps.ADM_nxyz*ps.ADM_lall)

coef_diff_pl_f = ( 3*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall 
                 + 3*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl
                 + ps.ADM_gall*3*ps.ADM_nxyz*2*ps.ADM_lall 
                 + ps.ADM_gall_pl*3*ps.ADM_nxyz*ps.ADM_lall_pl
                 + ps.ADM_gall*6*ps.ADM_nxyz*ps.ADM_lall
                 + ps.ADM_vlink*ps.ADM_nxyz*ps.ADM_lall_pl)

check_dsc1_f   = ( 4*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall 
                 + 3*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl
                 + ps.ADM_gall*3*ps.ADM_nxyz*2*ps.ADM_lall 
                 + ps.ADM_gall_pl*3*ps.ADM_nxyz*ps.ADM_lall_pl
                 + ps.ADM_gall*6*ps.ADM_nxyz*ps.ADM_lall
                 + ps.ADM_vlink*ps.ADM_nxyz*ps.ADM_lall_pl)

check_dsc1_pl_f = ( 4*ps.ADM_gall*ps.ADM_kall*ps.ADM_lall 
                  + 4*ps.ADM_gall_pl*ps.ADM_kall*ps.ADM_lall_pl
                  + ps.ADM_gall*3*ps.ADM_nxyz*2*ps.ADM_lall 
                  + ps.ADM_gall_pl*3*ps.ADM_nxyz*ps.ADM_lall_pl
                  + ps.ADM_gall*6*ps.ADM_nxyz*ps.ADM_lall
                  + ps.ADM_vlink*ps.ADM_nxyz*ps.ADM_lall_pl)




# dscl_np = np.reshape(data[:dscl_f], (ps.ADM_lall, ps.ADM_kall, ps.ADM_gall))
# dscl_pl_np = np.reshape(data[dscl_f:dscl_pl_f], (ps.ADM_lall_pl, ps.ADM_kall, ps.ADM_gall_pl))
check_dscl_np = np.reshape(data[:dscl_f], (ps.ADM_lall, ps.ADM_kall, ps.ADM_gall))
check_dscl_pl_np = np.reshape(data[dscl_f:dscl_pl_f], (ps.ADM_lall_pl, ps.ADM_kall, ps.ADM_gall_pl))
scl_np = np.reshape(data[dscl_pl_f:scl_f], (ps.ADM_lall, ps.ADM_kall, ps.ADM_gall))
scl_pl_np = np.reshape(data[scl_f:scl_pl_f], (ps.ADM_lall_pl, ps.ADM_kall, ps.ADM_gall_pl))
kh_np = np.reshape(data[scl_pl_f:kh_f], (ps.ADM_lall, ps.ADM_kall, ps.ADM_gall))
kh_pl_np = np.reshape(data[kh_f:kh_pl_f], (ps.ADM_lall_pl, ps.ADM_kall, ps.ADM_gall_pl))
coef_intp_np = np.reshape(data[kh_pl_f:coef_intp_f], (ps.ADM_lall, 2, ps.ADM_nxyz, 3, ps.ADM_gall))
coef_intp_pl_np = np.reshape(data[coef_intp_f:coef_intp_pl_f], (ps.ADM_lall_pl, ps.ADM_nxyz, 3, ps.ADM_gall_pl))
coef_diff_np = np.reshape(data[coef_intp_pl_f:coef_diff_f], (ps.ADM_lall, ps.ADM_nxyz, 6, ps.ADM_gall))
coef_diff_pl_np = np.reshape(data[coef_diff_f:coef_diff_pl_f], (ps.ADM_lall_pl, ps.ADM_nxyz, ps.ADM_vlink))


# dscl = np.transpose(dscl)
# dscl_pl = np.transpose(dscl_pl)
scl = np.transpose(scl_np)
scl_pl = np.transpose(scl_pl_np)
kh = np.transpose(kh_np)
kh_pl = np.transpose(kh_pl_np)
coef_intp = np.transpose(coef_intp_np)
coef_intp_pl = np.transpose(coef_intp_pl_np)
coef_diff = np.transpose(coef_diff_np)
coef_diff_pl = np.transpose(coef_diff_pl_np)
check_dscl = np.transpose(check_dscl_np)
check_dscl_pl = np.transpose(check_dscl_pl_np)

# f = open('coef_intp_pl.txt', 'w+')
# for l in range(ps.ADM_lall_pl):
#     for n in range(ps.ADM_nxyz):
#         for i in range(3):
#             for g in range(ps.ADM_gall_pl):
#                 f.write(str(coef_intp_pl[g][i][n][l]))
#                 f.write('\n')

# f2 = open('coef_diff_pl.txt', 'w+')
# for l in range(ps.ADM_lall_pl):
#     for n in range(ps.ADM_nxyz):
#         for v in range(ps.ADM_vlink):
#             f2.write(str(coef_diff_pl[v][n][l]))
#             f2.write('\n')


# for l in range(ps.ADM_lall):
#     for j in range(2):
#         for n in range(ps.ADM_nxyz):
#             for i in range(3):
#                 for g in range(ps.ADM_gall):
#                     f.write(str(coef_intp_np[l][j][n][i][g]))
#                     f.write('\n')


# f.close()
 
# for g in range(5):
#     print('\n')
#     print(coef_intp[g, 0, 0, 0, 0])
#     print(coef_intp[g, 1, 0, 0, 0])
#     print(coef_intp[g, 2, 0, 0, 0])
#     print(coef_intp[g, 0, 0, 0, 0])
#     print(coef_intp[g, 1, 0, 0, 0])
#     print(coef_intp[g, 2, 0, 0, 0])
#     print(coef_intp[g, 0, 0, 0, 0])
#     print(coef_intp[g, 1, 0, 0, 0])
#     print(coef_intp[g, 2, 0, 0, 0])


# f = open('test.txt', 'w+')
# for l in range(ps.ADM_lall):
#     for k in range(ps.ADM_kall):
#         for g in range(ps.ADM_gall):
#             f.write(str(dscl_np[l][k][g]))
#             f.write('\n')

# f2 = open('kh_pl.txt', 'w+')
# for l in range(ps.ADM_lall_pl):
#     for k in range(ps.ADM_kall):
#         for g in range(ps.ADM_gall_pl):
#             f2.write(str(kh_pl[g][k][l]))
#             f2.write('\n')

# f3 = open('scl_pl.txt', 'w+')
# for l in range(ps.ADM_lall_pl):
#     for k in range(ps.ADM_kall):
#         for g in range(ps.ADM_gall_pl):
#             f3.write(str(scl_pl[g][k][l]))
#             f3.write('\n')

# f4 = open('dscl_pl.txt', 'w+')
# for l in range(ps.ADM_lall_pl):
#     for k in range(ps.ADM_kall):
#         for g in range(ps.ADM_gall_pl):
#             f4.write(str(dscl_pl[g][k][l]))
#             f4.write('\n')

# for l in range(ps.ADM_lall_pl):
#     for k in range(ps.ADM_kall):
#         for g in range(ps.ADM_gall_pl):
#             f2.write(str(kh_pl[g][k][l]))
#             f2.write('\n')




def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

print_res(dscl_pl, 'dscl_pl pre')

# Start kernel
start = timer()
for i in range(0, ps.SET_iteration):
#     # new = OPRT_diffusion(dscl_np, dscl_pl_np, scl_np, scl_pl_np,
#     #                kh_np, kh_pl_np, coef_intp_np, coef_intp_pl_np, 
#     #                coef_diff_np, coef_diff_pl_np)
    new = OPRT_diffusion(dscl, dscl_pl, scl, scl_pl,
                   kh, kh_pl, coef_intp, coef_intp_pl, 
                   coef_diff, coef_diff_pl)
end = timer()

print('### Input ###')
print_res(check_dscl, 'check_dscl')
print_res(check_dscl_pl, 'check_dscl_pl')
print_res(scl, 'scl')
print_res(scl_pl, 'scl_pl')
print_res(kh, 'kh')
print_res(kh_pl, 'kh_pl')

print('### Output ###')
print_res(dscl, 'dscl')
print_res(dscl_pl, 'dscl_pl')

print('### Validation : point-by-point diff ###')
res_dscl = np.subtract(check_dscl, dscl)
res_dscl_pl = np.subtract(check_dscl_pl, dscl_pl)
print_res(res_dscl, 'res_dscl')
print_res(res_dscl_pl, 'res_dscl_pl')

print('Elapsed time=', round(end - start, 2), 'sec')

