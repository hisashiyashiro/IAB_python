import numpy as np
import problem_size as ps
import mod_src_tracer as mst
from timeit import default_timer as timer

data = np.fromfile("dyn_horiz_adv_limiter/snapshot.dyn_horiz_adv_limiter.pe000000", dtype=np.dtype('>f8'))

size = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall
size_pl = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl
size_sum = size + size_pl

shape = (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall)
shape_pl = (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl)

size2 = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall * 6
size3 = ps.ADM_gall * ps.ADM_kall * ps.ADM_lall * 2

size_pl2 = ps.ADM_gall_pl * ps.ADM_kall * ps.ADM_lall_pl * 2

shape2 = (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 6)
shape3 = (ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 2)
shape_pl2 = (ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl, 2)

q_a = np.zeros(shape2)
q_a_pl = np.zeros(shape_pl)

Qout_prev = np.zeros(shape3)
Qout_prev_pl = np.zeros(shape_pl2)

q_a_prev = np.reshape(data[:size2], shape2, order='F')
data = data[size2:]

q_a_prev_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

check_q_a = np.reshape(data[:size2], shape2, order='F')
data = data[size2:]

check_q_a_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

q = np.reshape(data[:size], shape, order='F')
data = data[size:]

q_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

d = np.reshape(data[:size], shape, order='F')
data = data[size:]

d_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

ch = np.reshape(data[:size2], shape2, order='F')
data = data[size2:]

ch_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

cmask = np.reshape(data[:size2], shape2, order='F')
data = data[size2:]

cmask_pl = np.reshape(data[:size_pl], shape_pl, order='F')
data = data[size_pl:]

check_Qout_prev = np.reshape(data[:size3], shape3, order='F')
data = data[size3:]

check_Qout_prev_pl = np.reshape(data[:size_pl2], shape_pl2, order='F')
data = data[size_pl2:]

Qout_post = np.reshape(data[:size3], shape3, order='F')
data = data[size3:]

Qout_post_pl = np.reshape(data[:size_pl2], shape_pl2, order='F')
data = data[size_pl2:]




def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

start = timer()
for iteration in range(ps.SET_iteration):
    print('in')
    q_a[:,:,:,:] = q_a_prev [:,:,:,:]
    q_a_pl[:,:,:] = q_a_prev_pl [:,:,:]
    mst.horizontal_limiter_thuburn(q_a, q_a_pl, q, q_pl, d, d_pl, ch, ch_pl, cmask, cmask_pl,
                                   Qout_prev, Qout_prev_pl, Qout_post, Qout_post_pl)
end = timer()

# print_res(check_q_a, 'check_q_a')
# print_res(check_q_a_pl, 'check_q_a_pl')
# print_res(check_Qout_prev_pl, 'check_Qout_prev_pl')

print('### Input ###')
print_res(q_a_prev, 'q_a_prev')
print_res(q_a_prev_pl, 'q_a_prev_pl')
print_res(check_q_a, 'check_q_a')
print_res(check_q_a_pl, 'check_q_a_pl')
print_res(q, 'q')
print_res(q_pl, 'q_pl')
print_res(d, 'd')
print_res(d_pl, 'd_pl')
print_res(ch,    'ch')
print_res(ch_pl, 'ch_pl')
print_res(cmask,    'cmask')
print_res(cmask_pl, 'cmask_pl')
print_res(check_Qout_prev,    'check_Qout_prev')
print_res(check_Qout_prev_pl, 'check_Qout_prev_pl')
print_res(Qout_post,    'Qout_post')
print_res(Qout_post_pl, 'Qout_post_pl')

print('### Output ###')
print_res(q_a, 'q_a')
print_res(q_a_pl, 'q_a_pl')
print_res(Qout_prev, 'Qout_prev')
print_res(Qout_prev_pl, 'Qout_prev_pl')

print('### Validation : point-by-point diff ###')
res1 = np.subtract(check_q_a,    q_a)
res2 = np.subtract(check_q_a_pl, q_a_pl)
res3 = np.subtract(check_Qout_prev,    Qout_prev)
res4 = np.subtract(check_Qout_prev_pl, Qout_prev_pl)

print_res(res1, "check_q_a")
print_res(res2, "check_q_a_pl")

print_res(res3, "check_Qout_prev")
print_res(res4, "check_Qout_prev_pl")

print('Elapsed time=', round(end - start, 2), 'sec')