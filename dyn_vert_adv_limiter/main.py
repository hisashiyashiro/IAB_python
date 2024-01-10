import numpy as np
import problem_size as ps
import mod_misc as mm
import mod_src_tracer as mst
from timeit import default_timer as timer

data = np.fromfile("dyn_vert_adv_limiter/snapshot.dyn_vert_adv_limiter.pe000000", dtype=np.dtype('>f8'))

size = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
size_pl = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
size_sum = size + size_pl

shape = (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall)
shape_pl = (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl)

q_h = np.zeros(shape)
q_h_pl = np.zeros(shape_pl)

q_h_prev    = np.reshape(data[:size], shape, order='F')
q_h_prev_pl = np.reshape(data[size:size_sum], shape_pl, order='F')

check_q_h    = np.reshape(data[size_sum:size+size_sum], shape, order='F')
check_q_h_pl = np.reshape(data[size+size_sum:2*size_sum], shape_pl, order='F')

q    = np.reshape(data[2*size_sum:size+2*size_sum], shape, order='F')
q_pl = np.reshape(data[size+2*size_sum:3*size_sum], shape_pl, order='F')

d    = np.reshape(data[3*size_sum:size+3*size_sum], shape, order='F')
d_pl = np.reshape(data[size+3*size_sum:4*size_sum], shape_pl, order='F')

ck    = np.reshape(data[4*size_sum:2*size+4*size_sum], (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 2), order='F')
ck_pl = np.reshape(data[2*size+4*size_sum:6*size_sum], (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl, 2), order='F')

def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))


start = timer()
for iteration in range(ps.SET_iteration):
    q_h = q_h_prev
    q_h_pl = q_h_prev_pl

    mst.vertical_limiter_thuburn(q_h, q_h_pl, q, q_pl, d, d_pl, ck, ck_pl, check=check_q_h)
end = timer()


print('### Input ###')
print_res(q_h_prev,    'q_h_prev\t')
print_res(q_h_prev_pl, 'q_h_prev_pl\t')
print_res(check_q_h,    'check_q_h\t')
print_res(check_q_h_pl, 'check_q_h_pl\t')
print_res(q,    'q\t\t')
print_res(q_pl, 'q_pl\t\t')
print_res(d,    'd\t\t')
print_res(d_pl, 'd_pl\t\t')
print_res(ck,    'ck\t\t')
print_res(ck_pl, 'ck_pl\t\t')
print_res(ck_pl, 'ck_pl\t\t')

print('### Output ###')
print_res(q_h,    'q_h\t\t')
print_res(q_h_pl, 'q_h_pl\t\t')

print('### Validation : point-by-point diff ###')
res = np.subtract(check_q_h, q_h)
res_pl = np.subtract(check_q_h_pl, q_h_pl)
print_res(res,    'res_q_h\t\t')
print_res(res_pl, 'res_q_h_pl\t')

print('Elapsed time=', round(end - start, 2), 'sec')