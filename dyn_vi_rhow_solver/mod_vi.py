import problem_size as ps
import mod_misc as mm
import numpy as np

def vi_rhow_solver(rhogw, rhogw_pl, rhogw0, rhogw0_pl, preg0, preg0_pl, rhog0, rhog0_pl,
                   Srho, Srho_pl, Sw, Sw_pl, Spre, Spre_pl, Mu, Mu_pl, Ml, Ml_pl, Mc, Mc_pl, dt, RGAMH, RGAMH_pl, RGSGAM2, RGSGAM2_pl,
                   RGAM, RGAM_pl, RGSGAM2H, RGSGAM2H_pl, GSGAM2H, GSGAM2H_pl):
    
    Sall = np.zeros((ps.ADM_gall, ps.ADM_kall))
    Sall_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall))

    beta = np.zeros((ps.ADM_gall))
    beta_pl = np.zeros((ps.ADM_gall_pl))
    gamma = np.zeros((ps.ADM_gall, ps.ADM_kall))
    gamma_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall))
    gall = ps.ADM_gall
    kmin = ps.ADM_kmin
    kmax = ps.ADM_kmax
    lall = ps.ADM_lall

    grav = ps.CONST_GRAV
    CVovRt2 = ps.CONST_CVdry / ps.CONST_Rdry / (dt * dt)
    alpha = ps.NON_HYDRO_ALPHA

    gstr = 0
    gend = gall

    def print_res(arr, name):
        print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

    for l in range(lall):
        for k in range(kmin, kmax):
            for g in range(gstr, gend):
                Sall[g, k] = (  (   ( rhogw0[g, k, l] * alpha + dt * Sw  [g, k, l] ) * RGAMH[g, k, l]**2
                                - ( ( preg0 [g, k, l]         + dt * Spre[g, k, l] ) * RGSGAM2[g, k, l]
                                  - ( preg0 [g, k-1, l]       + dt * Spre[g, k-1, l]) * RGSGAM2[g, k-1, l]
                                  ) * dt * mm.GRD_rdgzh[k]
                                - ( ( rhog0[g, k ,l]          + dt * Srho[g, k , l])  * RGAM[g, k, l]**2 * mm.GRD_afact[k]
                                  + ( rhog0[g, k-1, l]        + dt * Srho[g, k-1, l]) * RGAM[g, k-1, l]**2 * mm.GRD_bfact[k]
                                  ) * dt * grav) * CVovRt2 )
        


        for g in range(gstr, gend):
            rhogw[g, kmin-1, l] = rhogw[g, kmin-1, l] * RGSGAM2H[g, kmin-1, l]
            rhogw[g, kmax, l]   = rhogw[g, kmax, l]   * RGSGAM2H[g, kmax, l]
            Sall [g, kmin]      = Sall[g, kmin]   - Ml[g, kmin, l]   * rhogw[g, kmin-1, l]
            Sall [g, kmax-1]    = Sall[g, kmax-1] - Mu[g, kmax-1, l] * rhogw[g, kmax, l]

        k = kmin + 1
        for g in range(gstr, gend):
            beta[g] = Mc[g, k-1, l]
            # print(rhogw[g, k, l], Sall[g, k], beta[g])
            rhogw[g, k-1, l] = Sall[g, k-1] / beta[g]
        
        # print_res(Mc, 'Mc 1')
        # print_res(beta, 'beta 1')
        # print_res(rhogw, 'rhogw 1')
        
        for k in range(kmin+1, kmax):
            for g in range(gstr, gend):
                gamma[g, k] = Mu[g, k-1, l] / beta[g]
                beta[g] = Mc[g, k, l] - Ml[g, k, l] * gamma[g, k]
                rhogw[g, k, l] = (Sall[g, k] - Ml[g, k, l] * rhogw[g, k-1, l]) / beta[g]
        

        for k in range(kmax-2, kmin-1, -1):
            for g in range(gstr, gend):
                rhogw[g, k, l] = rhogw[g, k, l] - gamma[g, k+1] * rhogw[g, k+1, l]
                rhogw[g, k+1, l] = rhogw[g, k+1, l] * GSGAM2H[g, k+1, l]

        for g in range(gstr, gend):
            rhogw[g, kmin-1, l] = rhogw[g, kmin-1, l] * GSGAM2H[g, kmin-1, l]
            rhogw[g, kmin, l] = rhogw[g, kmin, l] * GSGAM2H[g, kmin, l]
            rhogw[g, kmax, l] = rhogw[g, kmax, l] * GSGAM2H[g, kmax, l]

    if (ps.ADM_have_pl):
        for l in range(ps.ADM_lall_pl):
            for k in range(ps.ADM_kmin, ps.ADM_kmax):
                for g in range(ps.ADM_gall_pl):
                    Sall_pl[g, k] = ((rhogw0_pl[g, k, l] * alpha + dt * Sw_pl[g, k, l]) * RGAMH_pl[g, k, l]**2 -
                                    ((preg0_pl[g, k, l] + dt * Spre_pl[g, k, l]) * RGSGAM2_pl[g, k, l] -
                                    (preg0_pl[g, k-1, l] + dt * Spre_pl[g, k-1, l]) * RGSGAM2_pl[g, k-1, l]) * dt * mm.GRD_rdgzh[k] -
                                    ((rhog0_pl[g, k, l] + dt * Srho_pl[g, k, l]) * RGAM_pl[g, k, l]**2 * mm.GRD_afact[k] +
                                    (rhog0_pl[g, k-1, l] + dt * Srho_pl[g, k-1, l]) * RGAM_pl[g, k-1, l]**2 * mm.GRD_bfact[k]) * dt * grav) * CVovRt2
                    
            
            for g in range(ps.ADM_gall_pl):
                rhogw_pl[g, ps.ADM_kmin, l] = rhogw_pl[g, ps.ADM_kmin, l] * RGSGAM2H_pl[g, ps.ADM_kmin, l]
                rhogw_pl[g, ps.ADM_kmax, l] = rhogw_pl[g, ps.ADM_kmax, l] * RGSGAM2H_pl[g, ps.ADM_kmax, l]
                Sall_pl[g, ps.ADM_kmin+1] = Sall_pl[g, ps.ADM_kmin+1] - Ml_pl[g, ps.ADM_kmin+1, l] * rhogw_pl[g, ps.ADM_kmin, l]
                Sall_pl[g, ps.ADM_kmax-1] = Sall_pl[g, ps.ADM_kmax-1] - Mu_pl[g, ps.ADM_kmax-1, l] * rhogw_pl[g, ps.ADM_kmax, l]

            k = ps.ADM_kmin

            for g in range(ps.ADM_gall_pl):
                beta_pl[g] = Mc_pl[g, k, l]
                rhogw_pl[g, k, l] = Sall_pl[g, k] / beta_pl[g]
            
            for k in range(ps.ADM_kmin+1, ps.ADM_kmax):
                for g in range(ps.ADM_gall_pl):
                    gamma_pl[g, k] = Mu_pl[g, k-1, l] / beta_pl[g]
                    beta_pl[g] = Mc_pl[g, k, l] - Ml_pl[g, k, l] * gamma_pl[g, k]
                    rhogw_pl[g, k, l] = (Sall_pl[g, k] - Ml_pl[g, k, l] * rhogw_pl[g, k-1, l]) / beta_pl[g]

            for k in range(ps.ADM_kmax-2, ps.ADM_kmin-1, -1):
                for g in range(ps.ADM_gall_pl):
                    rhogw_pl[g, k, l] = rhogw_pl[g, k, l] - gamma_pl[g, k+1] * rhogw_pl[g, k+1, l]
                    rhogw_pl[g, k+1, l] = rhogw_pl[g, k+1, l] * GSGAM2H_pl[g, k+1, l] 

            for g in range(ps.ADM_gall_pl):
                rhogw_pl[g, ps.ADM_kmin, l] = rhogw_pl[g, ps.ADM_kmin, l] * GSGAM2H_pl[g, ps.ADM_kmin, l]
                rhogw_pl[g, ps.ADM_kmin+1, l] = rhogw_pl[g, ps.ADM_kmin+1, l] * GSGAM2H_pl[g, ps.ADM_kmin+1, l]
                rhogw_pl[g, ps.ADM_kmax, l] = rhogw_pl[g, ps.ADM_kmax, l] * GSGAM2H_pl[g, ps.ADM_kmax, l]










    return