import problem_size as ps
import mod_misc as mm
import numpy as np

def sign(a, b):
    s = 1 - 2 * (b < 0)
    return s * abs(a)

def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

def vertical_limiter_thuburn(q_h, q_h_pl, q, q_pl, d, d_pl, ck, ck_pl, check=None, check_pl=None):
    Qout_min_km1 = np.zeros((ps.ADM_gall))
    Qout_max_km1 = np.zeros((ps.ADM_gall))
    Qout_min_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall))
    Qout_max_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall))
    
    gall = ps.ADM_gall
    kmin = ps.ADM_kmin
    kmax = ps.ADM_kmax

    EPS = ps.CONST_EPS
    BIG = ps.CONST_HUGE

    count = 0

    for l in range(ps.ADM_lall):
        for g in range(gall):
            k = kmin-1
            inflagL = 0.5 - sign(0.5, ck[g, k, l, 0])
            inflagU = 0.5 + sign(0.5, ck[g, k+1, l, 0])

            Qin_minL = min( q[g, k, l], q[g, k-1, l]) + (1.0-inflagL) * BIG
            Qin_minU = min( q[g, k, l], q[g, k+1, l]) + (1.0-inflagU) * BIG
            Qin_maxL = max( q[g, k, l], q[g, k-1, l]) - (1.0-inflagL) * BIG
            Qin_maxU = max( q[g, k, l], q[g, k+1, l]) - (1.0-inflagU) * BIG

            qnext_min = min(Qin_minL, Qin_minU, q[g, k, l])
            qnext_max = max(Qin_maxL, Qin_maxU, q[g, k, l])

            Cin  = inflagL * ck[g, k, l, 0] + inflagU * ck[g, k, l, 1]
            Cout = (1.0 - inflagL) * ck[g, k, l, 0] + (1.0 - inflagU) * ck[g, k, l, 1]

            CQin_min = inflagL * ck[g, k, l, 0] * Qin_minL + inflagU * ck[g, k, l, 1] * Qin_minU
            CQin_max = inflagL * ck[g, k, l, 0] * Qin_maxL + inflagU * ck[g, k, l, 1] * Qin_maxU

            zerosw = 0.5 - sign(0.5, abs(Cout)-EPS)

            Qout_min_k = ( ( ( q[g, k, l] - qnext_max) + qnext_max*(Cin + Cout - d[g,k,l]) - CQin_max)
                         / (Cout + zerosw) * (1.0 - zerosw) + q[g, k, l] * zerosw
                          )
            
            Qout_max_k = ( ( ( q[g, k, l] - qnext_min) + qnext_min*(Cin + Cout - d[g,k,l]) - CQin_min)
                         / (Cout + zerosw) * (1.0 - zerosw) + q[g, k, l] * zerosw
                          )
            
            Qout_min_km1[g] = Qout_min_k
            Qout_max_km1[g] = Qout_max_k
        

        for k in range(kmin, kmax):
            for g in range(gall):
                inflagL = 0.5 - sign(0.5, ck[g, k, l, 0])
                inflagU = 0.5 + sign(0.5, ck[g, k+1, l, 0])

                Qin_minL = min(q[g, k, l], q[g, k-1, l]) + (1.0 - inflagL) * BIG
                Qin_minU = min(q[g, k, l], q[g, k+1, l]) + (1.0 - inflagU) * BIG
                Qin_maxL = max(q[g, k, l], q[g, k-1, l]) - (1.0 - inflagL) * BIG
                Qin_maxU = max(q[g, k, l], q[g, k+1, l]) - (1.0 - inflagU) * BIG

                qnext_min = min(Qin_minL, Qin_minU, q[g, k, l])
                qnext_max = max(Qin_maxL, Qin_maxU, q[g, k, l])

                Cin = inflagL * ck[g, k, l, 0] + inflagU * ck[g, k, l, 1]
                Cout = (1.0 - inflagL) * ck[g, k, l, 0] + (1.0 - inflagU) * ck[g, k, l, 1]

                CQin_min = inflagL * ck[g, k, l, 0] * Qin_minL + inflagU * ck[g, k, l, 1] * Qin_minU
                CQin_max = inflagL * ck[g, k, l, 0] * Qin_maxL + inflagU * ck[g, k, l, 1] * Qin_maxU

                zerosw = 0.5 - sign(0.5, abs(Cout) - EPS)

                Qout_min_k =( 
                            ((q[g, k, l] - qnext_max) + qnext_max * (Cin + Cout - d[g, k, l]) - CQin_max) 
                            / (Cout + zerosw) * (1.0 - zerosw) 
                            + q[g, k, l] * zerosw
                            )
                
                Qout_max_k = (
                             ((q[g, k, l] - qnext_min) + qnext_min * (Cin + Cout - d[g, k, l]) - CQin_min) 
                             / (Cout + zerosw) * (1.0 - zerosw) 
                             + q[g, k, l] * zerosw
                             )

                q_h[g, k, l] = (
                                      inflagL  * max( min( q_h[g, k, l], Qout_max_km1[g] ), Qout_min_km1[g]) 
                             + (1.0 - inflagL) * max( min( q_h[g, k, l], Qout_max_k      ), Qout_min_k)
                               )
               
                Qout_min_km1[g] = Qout_min_k
                Qout_max_km1[g] = Qout_max_k
            
    if ps.ADM_have_pl:
        for l in range(ps.ADM_lall_pl):
            for k in range(ps.ADM_kmin-1, ps.ADM_kmax):
                for g in range(ps.ADM_gall_pl):
                    inflagL = 0.5 - sign(0.5, ck_pl[g,   k, l, 0])  # incoming flux: flag=1
                    inflagU = 0.5 + sign(0.5, ck_pl[g, k+1, l, 0])  # incoming flux: flag=1

                    Qin_minL = min(q_pl[g, k, l], q_pl[g, k-1, l]) + (1.0 - inflagL) * BIG
                    Qin_minU = min(q_pl[g, k, l], q_pl[g, k+1, l]) + (1.0 - inflagU) * BIG
                    Qin_maxL = max(q_pl[g, k, l], q_pl[g, k-1, l]) - (1.0 - inflagL) * BIG
                    Qin_maxU = max(q_pl[g, k, l], q_pl[g, k+1, l]) - (1.0 - inflagU) * BIG

                    qnext_min = min(Qin_minL, Qin_minU, q_pl[g, k, l])
                    qnext_max = max(Qin_maxL, Qin_maxU, q_pl[g, k, l])

                    Cin = inflagL * ck_pl[g, k, l, 0] + inflagU * ck_pl[g, k, l, 1]
                    Cout = (1.0 - inflagL) * ck_pl[g, k, l, 0] + (1.0 - inflagU) * ck_pl[g, k, l, 1]

                    CQin_max = inflagL * ck_pl[g, k, l, 0] * Qin_maxL + inflagU * ck_pl[g, k, l, 1] * Qin_maxU
                    CQin_min = inflagL * ck_pl[g, k, l, 0] * Qin_minL + inflagU * ck_pl[g, k, l, 1] * Qin_minU

                    zerosw = 0.5 - sign(0.5, abs(Cout) - EPS)  # if Cout = 0, sw = 1

                    Qout_min_pl[g, k] = ((q_pl[g, k, l] - qnext_max) + qnext_max * (Cin + Cout - d_pl[g, k, l]) - CQin_max) / (Cout + zerosw) * (1.0 - zerosw) + q_pl[g, k, l] * zerosw
                    Qout_max_pl[g, k] = ((q_pl[g, k, l] - qnext_min) + qnext_min * (Cin + Cout - d_pl[g, k, l]) - CQin_min) / (Cout + zerosw) * (1.0 - zerosw) + q_pl[g, k, l] * zerosw
            
            for k in range(ps.ADM_kmin, ps.ADM_kmax):
                for g in range(ps.ADM_gall_pl):
                    inflagL = 0.5 - sign(0.5, ck_pl[g, k, l, 0])  # incoming flux: flag=1

                    q_h_pl[g, k, l] = inflagL * max(min(q_h_pl[g, k, l], Qout_max_pl[g, k-1]), Qout_min_pl[g, k-1]) \
                                    + (1.0 - inflagL) * max(min(q_h_pl[g, k, l], Qout_max_pl[g, k]), Qout_min_pl[g, k])

    return