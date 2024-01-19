import numpy as np
import problem_size as ps

def sign(a, b):
    s = 1 - 2 * (b < 0)
    return s * abs(a)

def horizontal_limiter_thuburn(q_a, q_a_pl, q, q_pl, d, d_pl, 
                               ch, ch_pl, cmask, cmask_pl, Qout_prev,
                               Qout_prev_pl, Qout_post, Qout_post_pl):

    I_min = 0
    I_max = 1
    Qin = np.zeros((ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 2, 6))
    Qin_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl, 2, 2))
    Qout = np.zeros((ps.ADM_gall, ps.ADM_kall, ps.ADM_lall, 2))
    Qout_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl, 2))

    gmin = ps.ADM_gmin-1
    gmax = ps.ADM_gmax
    kall = ps.ADM_kall
    iall = ps.ADM_gall_1d

    EPS = ps.CONST_EPS
    BIG = ps.CONST_HUGE

    for l in range(ps.ADM_lall):
        for k in range(kall):
            for j in range(gmin-1, gmax):
                for i in range(gmin-1, gmax):
                    ij     = (j)*iall + i
                    ip1j   = ij + 1
                    ip1jp1 = ij + iall + 1
                    ijp1   = ij + iall
                    im1j   = ij - 1
                    ijm1   = ij - iall

                    im1j   = max( im1j  , 0 )
                    ijm1   = max( ijm1  , 0 )

                    q_min_AI = min(q[ij, k, l], q[ijm1, k, l], q[ip1j, k, l], q[ip1jp1, k, l])
                    q_max_AI = max(q[ij, k, l], q[ijm1, k, l], q[ip1j, k, l], q[ip1jp1, k, l])

                    q_min_AIJ = min(q[ij, k, l], q[ip1j, k, l], q[ip1jp1, k, l], q[ijp1, k, l])
                    q_max_AIJ = max(q[ij, k, l], q[ip1j, k, l], q[ip1jp1, k, l], q[ijp1, k, l])

                    q_min_AJ = min(q[ij, k, l], q[ip1jp1, k, l], q[ijp1, k, l], q[im1j, k, l])
                    q_max_AJ = max(q[ij, k, l], q[ip1jp1, k, l], q[ijp1, k, l], q[im1j, k, l])

                    Qin[ij, k, l, I_min, 0] = (cmask[ij, k, l, 0]) * q_min_AI + (1.0 - cmask[ij, k, l, 0]) * BIG
                    Qin[ip1j, k, l, I_min, 3] = (cmask[ij, k, l, 0]) * BIG + (1.0 - cmask[ij, k, l, 0]) * q_min_AI

                    Qin[ij, k, l, I_max, 0] = (cmask[ij, k, l, 0]) * q_max_AI + (1.0 - cmask[ij, k, l, 0]) * (-BIG)
                    Qin[ip1j, k, l, I_max, 3] = (cmask[ij, k, l, 0]) * (-BIG) + (1.0 - cmask[ij, k, l, 0]) * q_max_AI

                    Qin[ij, k, l, I_min, 1] = (cmask[ij, k, l, 1]) * q_min_AIJ + (1.0 - cmask[ij, k, l, 1]) * BIG
                    Qin[ip1jp1, k, l, I_min, 4] = (cmask[ij, k, l, 1]) * BIG + (1.0 - cmask[ij, k, l, 1]) * q_min_AIJ

                    Qin[ij, k, l, I_max, 1] = (cmask[ij, k, l, 1]) * q_max_AIJ + (1.0 - cmask[ij, k, l, 1]) * (-BIG)
                    Qin[ip1jp1, k, l, I_max, 4] = (cmask[ij, k, l, 1]) * (-BIG) + (1.0 - cmask[ij, k, l, 1]) * q_max_AIJ

                    Qin[ij, k, l, I_min, 2] = (cmask[ij, k, l, 2]) * q_min_AJ + (1.0 - cmask[ij, k, l, 2]) * BIG
                    Qin[ijp1, k, l, I_min, 5] = (cmask[ij, k, l, 2]) * BIG + (1.0 - cmask[ij, k, l, 2]) * q_min_AJ

                    Qin[ij, k, l, I_max, 2] = (cmask[ij, k, l, 2]) * q_max_AJ + (1.0 - cmask[ij, k, l, 2]) * (-BIG)
                    Qin[ijp1, k, l, I_max, 5] = (cmask[ij, k, l, 2]) * (-BIG) + (1.0 - cmask[ij, k, l, 2]) * q_max_AJ
                
            if (ps.ADM_have_sgp(l)):
                j = gmin - 1
                i = gmin - 1

                ij     = (j)*iall + i
                ijp1   = ij + iall
                ip1jp1 = ij + iall + 1
                ip2jp1 = ij + iall + 2

                q_min_AIJ = min(q[ij, k, l], q[ip1jp1, k, l], q[ip2jp1, k, l], q[ijp1, k, l])
                q_max_AIJ = max(q[ij, k, l], q[ip1jp1, k, l], q[ip2jp1, k, l], q[ijp1, k, l])

                Qin[ij, k, l, I_min, 1] = (cmask[ij, k, l, 1]) * q_min_AIJ + (1.0 - cmask[ij, k, l, 1]) * BIG
                Qin[ip1jp1, k, l, I_min, 4] = (cmask[ij, k, l, 1]) * BIG + (1.0 - cmask[ij, k, l, 1]) * q_min_AIJ

                Qin[ij, k, l, I_max, 1] = (cmask[ij, k, l, 1]) * q_max_AIJ + (1.0 - cmask[ij, k, l, 1]) * (-BIG)
                Qin[ip1jp1, k, l, I_max, 4] = (cmask[ij, k, l, 1]) * (-BIG) + (1.0 - cmask[ij, k, l, 1]) * q_max_AIJ

            for j in range(gmin, gmax):
                for i in range(gmin, gmax):
                    ij = (j)*iall + i

                    qnext_min = min(q[ij, k, l],
                                    Qin[ij, k, l, I_min, 0],
                                    Qin[ij, k, l, I_min, 1],
                                    Qin[ij, k, l, I_min, 2],
                                    Qin[ij, k, l, I_min, 3],
                                    Qin[ij, k, l, I_min, 4],
                                    Qin[ij, k, l, I_min, 5])
                    
                    qnext_max = max(q[ij, k, l],
                                    Qin[ij, k, l, I_max, 0],
                                    Qin[ij, k, l, I_max, 1],
                                    Qin[ij, k, l, I_max, 2],
                                    Qin[ij, k, l, I_max, 3],
                                    Qin[ij, k, l, I_max, 4],
                                    Qin[ij, k, l, I_max, 5])
                    
                    ch_masked1 = min(ch[ij, k, l, 0], 0.0)
                    ch_masked2 = min(ch[ij, k, l, 1], 0.0)
                    ch_masked3 = min(ch[ij, k, l, 2], 0.0)
                    ch_masked4 = min(ch[ij, k, l, 3], 0.0)
                    ch_masked5 = min(ch[ij, k, l, 4], 0.0)
                    ch_masked6 = min(ch[ij, k, l, 5], 0.0)

                    Cin_sum = ch_masked1 + ch_masked2 + ch_masked3 + ch_masked4 + ch_masked5 + ch_masked6

                    Cout_sum = (ch[ij, k, l, 0] - ch_masked1 +
                                ch[ij, k, l, 1] - ch_masked2 +
                                ch[ij, k, l, 2] - ch_masked3 +
                                ch[ij, k, l, 3] - ch_masked4 +
                                ch[ij, k, l, 4] - ch_masked5 +
                                ch[ij, k, l, 5] - ch_masked6)
                    
                    CQin_min_sum = (ch_masked1 * Qin[ij, k, l, I_min, 0] +
                                    ch_masked2 * Qin[ij, k, l, I_min, 1] +
                                    ch_masked3 * Qin[ij, k, l, I_min, 2] +
                                    ch_masked4 * Qin[ij, k, l, I_min, 3] +
                                    ch_masked5 * Qin[ij, k, l, I_min, 4] +
                                    ch_masked6 * Qin[ij, k, l, I_min, 5])

                    CQin_max_sum = (ch_masked1 * Qin[ij, k, l, I_max, 0] +
                                    ch_masked2 * Qin[ij, k, l, I_max, 1] +
                                    ch_masked3 * Qin[ij, k, l, I_max, 2] +
                                    ch_masked4 * Qin[ij, k, l, I_max, 3] +
                                    ch_masked5 * Qin[ij, k, l, I_max, 4] +
                                    ch_masked6 * Qin[ij, k, l, I_max, 5])

                    zerosw = 0.5 - sign(0.5,np.abs(Cout_sum)-EPS)

                    Qout[ij, k, l, I_min] = (q[ij, k, l] - CQin_max_sum - qnext_max * (1.0 - Cin_sum - Cout_sum + d[ij, k, l])) \
                                            / (Cout_sum + zerosw) * (1.0 - zerosw) + q[ij, k, l] * zerosw
                    
                    Qout[ij, k, l, I_max] = (q[ij, k, l] - CQin_min_sum - qnext_min * (1.0 - Cin_sum - Cout_sum + d[ij, k, l])) \
                                            / (Cout_sum + zerosw) * (1.0 - zerosw) + q[ij, k, l] * zerosw
                    
            for j in range(iall):
                for i in range(iall):
                    if (i < gmin) or (i > gmax-1) or (j < gmin) or (j > gmax-1):
                        ij = (j)*iall + i

                        Qout[ij, k, l, I_min] = q[ij, k, l]
                        Qout[ij, k, l, I_max] = q[ij, k, l]

    if ps.ADM_have_pl:
        n = ps.ADM_gslf_pl - 1

        for l in range(ps.ADM_lall_pl):
            for k in range(ps.ADM_kall):
                for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                    ij = v
                    ijp1 = v + 1
                    ijm1 = v - 1
                    if (ijp1 == ps.ADM_gmax_pl):
                        ijp1 = ps.ADM_gmin_pl - 1
                    if (ijm1 == ps.ADM_gmin_pl-2):
                        ijm1 = ps.ADM_gmax_pl - 1

                    q_min_pl = min(q_pl[n, k, l], q_pl[ij, k, l], q_pl[ijm1, k, l], q_pl[ijp1, k, l])
                    q_max_pl = max(q_pl[n, k, l], q_pl[ij, k, l], q_pl[ijm1, k, l], q_pl[ijp1, k, l])
        
                    Qin_pl[ij, k, l, I_min, 0] = (cmask_pl[ij, k, l]) * q_min_pl + (1.0 - cmask_pl[ij, k, l]) * BIG
                    Qin_pl[ij, k, l, I_min, 1] = (cmask_pl[ij, k, l]) * BIG + (1.0 - cmask_pl[ij, k, l]) * q_min_pl
                    Qin_pl[ij, k, l, I_max, 0] = (cmask_pl[ij, k, l]) * q_max_pl + (1.0 - cmask_pl[ij, k, l]) * (-BIG)
                    Qin_pl[ij, k, l, I_max, 1] = (cmask_pl[ij, k, l]) * (-BIG) + (1.0 - cmask_pl[ij, k, l]) * q_max_pl

                qnext_min_pl = q_pl[n, k, l]
                qnext_max_pl = q_pl[n, k, l]

                for v in range(ps.ADM_gmin_pl - 1, ps.ADM_gmax_pl):
                    qnext_min_pl = min(qnext_min_pl, Qin_pl[v, k, l, I_min, 0])
                    qnext_max_pl = max(qnext_max_pl, Qin_pl[v, k, l, I_max, 0])

                Cin_sum_pl      = 0.0
                Cout_sum_pl     = 0.0
                CQin_max_sum_pl = 0.0
                CQin_min_sum_pl = 0.0

                for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                    ch_masked = cmask_pl[v, k, l] * ch_pl[v, k, l]
                    Cin_sum_pl = Cin_sum_pl + ch_masked
                    Cout_sum_pl = Cout_sum_pl - ch_masked + ch_pl[v, k, l]
                    CQin_min_sum_pl = CQin_min_sum_pl + ch_masked * Qin_pl[v, k, l, I_min, 0]
                    CQin_max_sum_pl = CQin_max_sum_pl + ch_masked * Qin_pl[v, k, l, I_max, 0]
                
                zerosw = 0.5 - sign(0.5, abs(Cout_sum_pl) - EPS)

                Qout_pl[n, k, l, I_min] = (q_pl[n, k, l] - CQin_max_sum_pl - qnext_max_pl * (1.0 - Cin_sum_pl - Cout_sum_pl + d_pl[n, k, l])) \
                                        / (Cout_sum_pl + zerosw) * (1.0 - zerosw) + q_pl[n, k, l] * zerosw
                
                Qout_pl[n, k, l, I_max] = (q_pl[n, k, l] - CQin_min_sum_pl - qnext_min_pl * (1.0 - Cin_sum_pl - Cout_sum_pl + d_pl[n, k, l])) \
                                        / (Cout_sum_pl + zerosw) * (1.0 - zerosw) + q_pl[n, k, l] * zerosw

    Qout_pl[ps.ADM_gmin_pl-1:ps.ADM_gmax_pl, :, :, :] = 0.0

    Qout_prev[:,:,:,:] = Qout[:,:,:,:]
    Qout_prev_pl[:,:,:,:] = Qout_pl[:,:,:,:]

    Qout[:, :, :, :] = Qout_post[:, :, :, :]
    Qout_pl[:, :, :, :] = Qout_post_pl[:, :, :, :]

    for l in range(ps.ADM_lall):
        for k in range(kall):
            for j in range(gmin-1, gmax):
                for i in range(gmin-1, gmax):
                    ij     = (j)*iall + i
                    ip1j   = ij + 1
                    ip1jp1 = ij + iall + 1
                    ijp1   = ij + iall

                    q_a[ij, k, l, 0] = (cmask[ij, k, l, 0]) * min(max(q_a[ij, k, l, 0], Qin[ij, k, l, I_min, 0]), Qin[ij, k, l, I_max, 0]) \
                                     + (1.0 - cmask[ij, k, l, 0]) * min(max(q_a[ij, k, l, 0], Qin[ip1j, k, l, I_min, 3]), Qin[ip1j, k, l, I_max, 3])

                    q_a[ij, k, l, 0] = (cmask[ij, k, l, 0]) * max(min(q_a[ij, k, l, 0], Qout[ip1j, k, l, I_max]), Qout[ip1j, k, l, I_min]) \
                                     + (1.0 - cmask[ij, k, l, 0]) * max(min(q_a[ij, k, l, 0], Qout[ij, k, l, I_max]), Qout[ij, k, l, I_min])
                    
                    q_a[ip1j, k, l, 3] = q_a[ij, k, l, 0]

                    q_a[ij, k, l, 1] = (cmask[ij, k, l, 1]) * min(max(q_a[ij, k, l, 1], Qin[ij, k, l, I_min, 1]), Qin[ij, k, l, I_max, 1]) \
                                     + (1.0 - cmask[ij, k, l, 1]) * min(max(q_a[ij, k, l, 1], Qin[ip1jp1, k, l, I_min, 4]), Qin[ip1jp1, k, l, I_max, 4])
                    
                    q_a[ij, k, l, 1] = (cmask[ij, k, l, 1]) * max(min(q_a[ij, k, l, 1], Qout[ip1jp1, k, l, I_max]), Qout[ip1jp1, k, l, I_min]) \
                                     + (1.0 - cmask[ij, k, l, 1]) * max(min(q_a[ij, k, l, 1], Qout[ij, k, l, I_max]), Qout[ij, k, l, I_min])

                    q_a[ip1jp1, k, l, 4] = q_a[ij, k, l, 1]

                    q_a[ij, k, l, 2] = (cmask[ij, k, l, 2]) * min(max(q_a[ij, k, l, 2], Qin[ij, k, l, I_min, 2]), Qin[ij, k, l, I_max, 2]) \
                                     + (1.0 - cmask[ij, k, l, 2]) * min(max(q_a[ij, k, l, 2], Qin[ijp1, k, l, I_min, 5]), Qin[ijp1, k, l, I_max, 5])
                    
                    q_a[ij, k, l, 2] = (cmask[ij, k, l, 2]) * max(min(q_a[ij, k, l, 2], Qout[ijp1, k, l, I_max]), Qout[ijp1, k, l, I_min]) \
                                     + (1.0 - cmask[ij, k, l, 2]) * max(min(q_a[ij, k, l, 2], Qout[ij, k, l, I_max]), Qout[ij, k, l, I_min])
                    
                    q_a[ijp1, k, l, 5] = q_a[ij, k, l, 2]

    if ps.ADM_have_pl:
        n = ps.ADM_gslf_pl - 1

        for l in range(ps.ADM_lall_pl):
            for k in range(ps.ADM_kall):
                for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):

                    q_a_pl[v, k, l] = (cmask_pl[v, k, l]) * min(max(q_a_pl[v, k, l], Qin_pl[v, k, l, I_min, 0]), Qin_pl[v, k, l, I_max, 0]) \
                                    + (1.0 - cmask_pl[v, k, l]) * min(max(q_a_pl[v, k, l], Qin_pl[v, k, l, I_min, 1]), Qin_pl[v, k, l, I_max, 1])
                    
                    q_a_pl[v, k, l] = (cmask_pl[v, k, l]) * max(min(q_a_pl[v, k, l], Qout_pl[v, k, l, I_max]), Qout_pl[v, k, l, I_min]) \
                                    + (1.0 - cmask_pl[v, k, l]) * max(min(q_a_pl[v, k, l], Qout_pl[n, k, l, I_max]), Qout_pl[n, k, l, I_min])






                    




                


                












                    









                                        









                    
    return