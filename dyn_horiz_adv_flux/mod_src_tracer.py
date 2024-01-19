import problem_size as ps
import mod_misc as mm
import numpy as np

def print_res(arr, name):
    print(name, 'max=', np.max(arr), ', min=', np.min(arr), ', sum=', np.sum(arr))

def horizontal_flux(flx_h, flx_h_pl, GRD_xc, GRD_xc_pl, rho, rho_pl, 
                    rhovx, rhovx_pl, rhovy, rhovy_pl, rhovz, rhovz_pl, dt,
                    GRD_xr, GRD_xr_pl, GMTR_p, GMTR_p_pl, GMTR_t, GMTR_t_pl,
                    GMTR_a, GMTR_a_pl):
    
    rhot_TI = np.zeros(ps.ADM_gall)
    rhot_TJ = np.zeros(ps.ADM_gall)
    rhot_pl = np.zeros(ps.ADM_gall_pl)
    rhovxt_TI = np.zeros(ps.ADM_gall   )
    rhovxt_TJ = np.zeros(ps.ADM_gall   )
    rhovxt_pl = np.zeros(ps.ADM_gall_pl)
    rhovyt_TI = np.zeros(ps.ADM_gall   )
    rhovyt_TJ = np.zeros(ps.ADM_gall   )
    rhovyt_pl = np.zeros(ps.ADM_gall_pl)
    rhovzt_TI = np.zeros(ps.ADM_gall   )
    rhovzt_TJ = np.zeros(ps.ADM_gall   )
    rhovzt_pl = np.zeros(ps.ADM_gall_pl)


    gmin = ps.ADM_gmin
    gmax = ps.ADM_gmax
    kall = ps.ADM_kall
    iall = ps.ADM_gall_1d

    EPS = ps.CONST_EPS

    for l in range(ps.ADM_lall):
        for k in range(kall):
            for j in range(gmin-2, gmax):
                for i in range(gmin-2, gmax):
                    ij = (j) * iall + i
                    ip1j = ij + 1
                    rhot_TI[ij]   = rho[ij, k, l]   * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W1 - 1] + rho[ip1j, k, l]   * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W2 - 1]
                    rhovxt_TI[ij] = rhovx[ij, k, l] * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W1 - 1] + rhovx[ip1j, k, l] * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W2 - 1]
                    rhovyt_TI[ij] = rhovy[ij, k, l] * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W1 - 1] + rhovy[ip1j, k, l] * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W2 - 1]
                    rhovzt_TI[ij] = rhovz[ij, k, l] * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W1 - 1] + rhovz[ip1j, k, l] * GMTR_t[ij, ps.K0 - 1, l, ps.TI, ps.W2 - 1]
                    
                    rhot_TJ[ij]   =   rho[ij, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W1-1]
                    rhovxt_TJ[ij] = rhovx[ij, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W1-1]
                    rhovyt_TJ[ij] = rhovy[ij, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W1-1]
                    rhovzt_TJ[ij] = rhovz[ij, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W1-1]
            
            
            for j in range(gmin-2, gmax):
                for i in range(gmin-2, gmax):
                    ij     = (j)*iall + i
                    ijp1   = ij + iall
                    ip1jp1 = ij + iall + 1

                    rhot_TI[ij]   =   rhot_TI[ij] +   rho[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TI, ps.W3-1]
                    rhovxt_TI[ij] = rhovxt_TI[ij] + rhovx[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TI, ps.W3-1]
                    rhovyt_TI[ij] = rhovyt_TI[ij] + rhovy[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TI, ps.W3-1]
                    rhovzt_TI[ij] = rhovzt_TI[ij] + rhovz[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TI, ps.W3-1]


                    rhot_TJ[ij]   =   rhot_TJ[ij] +   rho[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W2-1] +   rho[ijp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W3-1]
                    rhovxt_TJ[ij] = rhovxt_TJ[ij] + rhovx[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W2-1] + rhovx[ijp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W3-1]
                    rhovyt_TJ[ij] = rhovyt_TJ[ij] + rhovy[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W2-1] + rhovy[ijp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W3-1]
                    rhovzt_TJ[ij] = rhovzt_TJ[ij] + rhovz[ip1jp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W2-1] + rhovz[ijp1, k, l] * GMTR_t[ij, ps.K0-1, l, ps.TJ, ps.W3-1]


            if ps.ADM_have_sgp(l):
                j = gmin-2
                i = gmin-2

                ij   = (j)*iall + i
                ip1j = ij + 1

                rhot_TI[ij] = rhot_TJ[ip1j]
                rhovxt_TI[ij] = rhovxt_TJ[ip1j]
                rhovyt_TI[ij] = rhovyt_TJ[ip1j]
                rhovzt_TI[ij] = rhovzt_TJ[ip1j]
            
            for j in range(iall):
                for i in range(iall):
                    if i < gmin-1 or i > gmax-1 or j < gmin-1 or j > gmax-1:
                        ij = (j) * iall + i

                        flx_h[ij, k, l, 0] = 0.0  
                        flx_h[ij, k, l, 1] = 0.0
                        flx_h[ij, k, l, 2] = 0.0
                        flx_h[ij, k, l, 3] = 0.0
                        flx_h[ij, k, l, 4] = 0.0
                        flx_h[ij, k, l, 5] = 0.0

                        GRD_xc[ij, k, l, ps.AI, ps.XDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AI, ps.YDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AI, ps.ZDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AIJ, ps.XDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AIJ, ps.YDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AIJ, ps.ZDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AJ, ps.XDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AJ, ps.YDIR] = 0.0
                        GRD_xc[ij, k, l, ps.AJ, ps.ZDIR] = 0.0


            for j in range(gmin - 1, gmax):
                for i in range(gmin - 2, gmax):
                    ij     = (j)*iall + i
                    ip1j   = ij + 1
                    ijm1   = ij - iall

                    rrhoa2 = 1.0 / max(rhot_TJ[ijm1] + rhot_TI[ij], EPS) 
                    rhovxt2 = rhovxt_TJ[ijm1] + rhovxt_TI[ij]
                    rhovyt2 = rhovyt_TJ[ijm1] + rhovyt_TI[ij]
                    rhovzt2 = rhovzt_TJ[ijm1] + rhovzt_TI[ij]

                    flux = 0.5 * (rhovxt2 * GMTR_a[ij, ps.K0-1, l, ps.AI, ps.HNX-1] +
                                  rhovyt2 * GMTR_a[ij, ps.K0-1, l, ps.AI, ps.HNY-1] +
                                  rhovzt2 * GMTR_a[ij, ps.K0-1, l, ps.AI, ps.HNZ-1])

                    flx_h[ij, k, l, 0]   =  flux * GMTR_p[ij,   ps.K0-1, l, ps.P_RAREA-1] * dt  
                    flx_h[ip1j, k, l, 3] = -flux * GMTR_p[ip1j, ps.K0-1, l, ps.P_RAREA-1] * dt

                    GRD_xc[ij, k, l, ps.AI, ps.XDIR] = GRD_xr[ij, ps.K0-1, l, ps.AI, ps.XDIR] - rhovxt2 * rrhoa2 * dt * 0.5  
                    GRD_xc[ij, k, l, ps.AI, ps.YDIR] = GRD_xr[ij, ps.K0-1, l, ps.AI, ps.YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                    GRD_xc[ij, k, l, ps.AI, ps.ZDIR] = GRD_xr[ij, ps.K0-1, l, ps.AI, ps.ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5


            for j in range(gmin-2, gmax):
                for i in range(gmin-2, gmax):
                    ij     = (j)*iall + i
                    ip1jp1 = ij + iall + 1

                    rrhoa2 = 1.0 / max(rhot_TI[ij] + rhot_TJ[ij], EPS) 
                    rhovxt2 = rhovxt_TI[ij] + rhovxt_TJ[ij]
                    rhovyt2 = rhovyt_TI[ij] + rhovyt_TJ[ij]
                    rhovzt2 = rhovzt_TI[ij] + rhovzt_TJ[ij]

                    flux = 0.5 * (rhovxt2 * GMTR_a[ij, ps.K0-1, l, ps.AIJ, ps.HNX-1] +
                                  rhovyt2 * GMTR_a[ij, ps.K0-1, l, ps.AIJ, ps.HNY-1] +
                                  rhovzt2 * GMTR_a[ij, ps.K0-1, l, ps.AIJ, ps.HNZ-1])

                    flx_h[ij, k, l, 1] =      flux * GMTR_p[ij,     ps.K0-1, l, ps.P_RAREA-1] * dt  
                    flx_h[ip1jp1, k, l, 4] = -flux * GMTR_p[ip1jp1, ps.K0-1, l, ps.P_RAREA-1] * dt

                    GRD_xc[ij, k, l, ps.AIJ, ps.XDIR] = GRD_xr[ij, ps.K0-1, l, ps.AIJ, ps.XDIR] - rhovxt2 * rrhoa2 * dt * 0.5  
                    GRD_xc[ij, k, l, ps.AIJ, ps.YDIR] = GRD_xr[ij, ps.K0-1, l, ps.AIJ, ps.YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                    GRD_xc[ij, k, l, ps.AIJ, ps.ZDIR] = GRD_xr[ij, ps.K0-1, l, ps.AIJ, ps.ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5


            for j in range(gmin-2, gmax):
                for i in range(gmin-1, gmax):
                    ij     = (j)*iall + i
                    ijp1   = ij + iall
                    im1j   = ij - 1

                    rrhoa2 = 1.0 / max(rhot_TJ[ij] + rhot_TI[im1j], EPS)  
                    rhovxt2 = rhovxt_TJ[ij] + rhovxt_TI[im1j]
                    rhovyt2 = rhovyt_TJ[ij] + rhovyt_TI[im1j]
                    rhovzt2 = rhovzt_TJ[ij] + rhovzt_TI[im1j]

                    flux = 0.5 * (rhovxt2 * GMTR_a[ij, ps.K0-1, l, ps.AJ, ps.HNX-1] +
                                  rhovyt2 * GMTR_a[ij, ps.K0-1, l, ps.AJ, ps.HNY-1] +
                                  rhovzt2 * GMTR_a[ij, ps.K0-1, l, ps.AJ, ps.HNZ-1])

                    flx_h[ij, k, l, 2] =    flux * GMTR_p[ij,   ps.K0-1, l, ps.P_RAREA-1] * dt  
                    flx_h[ijp1, k, l, 5] = -flux * GMTR_p[ijp1, ps.K0-1, l, ps.P_RAREA-1] * dt

                    GRD_xc[ij, k, l, ps.AJ, ps.XDIR] = GRD_xr[ij, ps.K0-1, l, ps.AJ, ps.XDIR] - rhovxt2 * rrhoa2 * dt * 0.5  
                    GRD_xc[ij, k, l, ps.AJ, ps.YDIR] = GRD_xr[ij, ps.K0-1, l, ps.AJ, ps.YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                    GRD_xc[ij, k, l, ps.AJ, ps.ZDIR] = GRD_xr[ij, ps.K0-1, l, ps.AJ, ps.ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5

            # print(k)
            # print_res(flx_h, 'flx_h')
            # # print_res(GMTR_a, 'gmtr_a')
            # # print_res(GMTR_p, 'gmtr_p')
            # # print_res(rhovzt_TJ[:], 'z tj')
            # # print_res(rhovzt_TI[:], 'z ti')
            # print_res(GRD_xc, 'GRD_xc')

            if ps.ADM_have_sgp(l):
                j = gmin - 1
                i = gmin - 1

                ij = (j)*iall + i

                flx_h[ij, k, l, 5] = 0.0
           
    if ps.ADM_have_pl:
        n = ps.ADM_gslf_pl - 1

        for l in range(ps.ADM_lall_pl):
            for k in range(ps.ADM_kall):
                for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                    ij = v
                    ijp1 = v + 1

                    if (ijp1 == ps.ADM_gmax_pl):
                        ijp1 = ps.ADM_gmin_pl -1
                    
                    rhot_pl[v] = (rho_pl[n, k, l]    * GMTR_t_pl[ij, ps.K0-1, l, ps.W1-1] +
                                  rho_pl[ij, k, l]   * GMTR_t_pl[ij, ps.K0-1, l, ps.W2-1] +
                                  rho_pl[ijp1, k, l] * GMTR_t_pl[ij, ps.K0-1, l, ps.W3-1])

                    rhovxt_pl[v] = (rhovx_pl[n, k, l]    * GMTR_t_pl[ij, ps.K0-1, l, ps.W1-1] +
                                    rhovx_pl[ij, k, l]   * GMTR_t_pl[ij, ps.K0-1, l, ps.W2-1] +
                                    rhovx_pl[ijp1, k, l] * GMTR_t_pl[ij, ps.K0-1, l, ps.W3-1])

                    rhovyt_pl[v] = (rhovy_pl[n, k, l]    * GMTR_t_pl[ij, ps.K0-1, l, ps.W1-1] +
                                    rhovy_pl[ij, k, l]   * GMTR_t_pl[ij, ps.K0-1, l, ps.W2-1] +
                                    rhovy_pl[ijp1, k, l] * GMTR_t_pl[ij, ps.K0-1, l, ps.W3-1])

                    rhovzt_pl[v] = (rhovz_pl[n, k, l]    * GMTR_t_pl[ij, ps.K0-1, l, ps.W1-1] +
                                    rhovz_pl[ij, k, l]   * GMTR_t_pl[ij, ps.K0-1, l, ps.W2-1] +
                                    rhovz_pl[ijp1, k, l] * GMTR_t_pl[ij, ps.K0-1, l, ps.W3-1])
                
                for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                    # print(True)
                    ij = v
                    ijm1 = v-1

                    if (ijm1 == ps.ADM_gmin_pl - 2):
                        ijm1 = ps.ADM_gmax_pl - 1 
                    
                    rrhoa2 = 1.0 / max(rhot_pl[ijm1] + rhot_pl[ij], EPS)  
                    rhovxt2 = rhovxt_pl[ijm1] + rhovxt_pl[ij]
                    rhovyt2 = rhovyt_pl[ijm1] + rhovyt_pl[ij]
                    rhovzt2 = rhovzt_pl[ijm1] + rhovzt_pl[ij]

                    flux = 0.5 * (rhovxt2 * GMTR_a_pl[ij, ps.K0-1, l, ps.HNX-1] +
                                  rhovyt2 * GMTR_a_pl[ij, ps.K0-1, l, ps.HNY-1] +
                                  rhovzt2 * GMTR_a_pl[ij, ps.K0-1, l, ps.HNZ-1])
                    
                    flx_h_pl[v, k, l] = flux * GMTR_p_pl[n, ps.K0-1, l, ps.P_RAREA-1] * dt  

                    GRD_xc_pl[v, k, l, ps.XDIR] = GRD_xr_pl[v, ps.K0-1, l, ps.XDIR] - rhovxt2 * rrhoa2 * dt * 0.5
                    GRD_xc_pl[v, k, l, ps.YDIR] = GRD_xr_pl[v, ps.K0-1, l, ps.YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                    GRD_xc_pl[v, k, l, ps.ZDIR] = GRD_xr_pl[v, ps.K0-1, l, ps.ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5





    return