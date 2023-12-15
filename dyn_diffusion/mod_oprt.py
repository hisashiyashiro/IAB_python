import mod_precision as mp
import mod_misc as mm
import dyn_divdamp.problem_size as ps 
import numpy as np

####### might have an issue with loop index (should start from 0)#######

def OPRT_diffusion (dscl, dscl_pl, scl, scl_pl, 
                    kh, kh_pl, coef_intp, coef_intp_pl, 
                    coef_diff, coef_diff_pl):
    # variables 
    vt = np.zeros((ps.ADM_gall, ps.ADM_nxyz, 2))
    vt_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_nxyz))
    kf = np.zeros(6)

    # call DEBUG_rapstart
    gmin = (ps.ADM_gmin - 1) * ps.ADM_gall_1d + ps.ADM_gmin
    gmax = (ps.ADM_gmax - 1) * ps.ADM_gall_1d + ps.ADM_gmax
    iall = ps.ADM_gall_1d
    gall = ps.ADM_gall
    kall = ps.ADM_kall
    lall = ps.ADM_lall
    nxyz = ps.ADM_nxyz

    XDIR = 0
    YDIR = 1
    ZDIR = 2

    TI = 0
    TJ = 1

    gminm1 = (ps.ADM_gmin - 1 - 1) * ps.ADM_gall_1d + ps.ADM_gmin - 1

    # loop
    for l in range(0, lall):
        for k in range(0, kall):
            for d in range(0, nxyz):
                for g in range(gminm1-1, gmax):
                    ij = g
                    ip1j = g + 1
                    ip1jp1 = g + iall + 1
                    ijp1 = g + iall

                    # parallelization start
                    vt[g, d, TI] = ( (  2.0 * coef_intp[g, 0, d, TI, l] 
                                      - 1.0 * coef_intp[g, 1, d, TI, l] 
                                      - 1.0 * coef_intp[g, 2, d, TI, l]) * scl[ij, k, l]
                                  + ( - 1.0 * coef_intp[g, 0, d, TI, l]
                                      + 2.0 * coef_intp[g, 1, d, TI, l]
                                      - 1.0 * coef_intp[g, 2, d, TI, l]) * scl[ip1j, k, l]
                                  + ( - 1.0 * coef_intp[g, 0, d, TI, l]
                                      - 1.0 * coef_intp[g, 1, d, TI, l]
                                      + 2.0 * coef_intp[g, 2, d, TI, l]) * scl[ip1jp1, k, l] 
                                  ) / 3.0
                    
                    # print(vt[g, d, TI])

                for g in range(gminm1-1, gmax):
                    ij = g
                    ip1j = g + 1
                    ip1jp1 = g + iall + 1
                    ijp1 = g + iall
                    vt[g, d, TJ] = ( (  2.0 * coef_intp[g, 0, d, TJ, l] 
                                      - 1.0 * coef_intp[g, 1, d, TJ, l] 
                                      - 1.0 * coef_intp[g, 2, d, TJ, l]) * scl[ij, k, l]
                                  + ( - 1.0 * coef_intp[g, 0, d, TJ, l]
                                      + 2.0 * coef_intp[g, 1, d, TJ, l]
                                      - 1.0 * coef_intp[g, 2, d, TJ, l]) * scl[ip1jp1, k, l]
                                  + ( - 1.0 * coef_intp[g, 0, d, TJ, l]
                                      - 1.0 * coef_intp[g, 1, d, TJ, l]
                                      + 2.0 * coef_intp[g, 2, d, TJ, l]) * scl[ijp1, k, l] 
                                  ) / 3.0
                    # print(vt[g, d, TI])
                    # parallelization end

            if (ps.ADM_have_sgp(l)):
                vt[gminm1-1, XDIR, TI] = vt[gminm1, XDIR, TJ] 
                vt[gminm1-1, YDIR, TI] = vt[gminm1, YDIR, TJ] 
                vt[gminm1-1, ZDIR, TI] = vt[gminm1, ZDIR, TJ]

            for g in range(0, gmin-1):
                dscl[g, k, l] = 0.0

            for g in range(gmin-1, gmax):
                ij = g
                ip1j = g + 1
                ip1jp1 = g + iall + 1
                ijp1 = g + iall
                im1j = g - 1
                im1jm1 = g - iall - 1
                ijm1 = g - iall

                kf[0] = 0.5 * (kh[ij    , k, l] + kh[ip1jp1, k, l])
                kf[1] = 0.5 * (kh[ij    , k, l] + kh[ijp1  , k, l])
                kf[2] = 0.5 * (kh[im1j  , k, l] + kh[ij    , k, l])
                kf[3] = 0.5 * (kh[im1jm1, k, l] + kh[ij    , k, l])
                kf[4] = 0.5 * (kh[ijm1  , k, l] + kh[ij    , k, l])
                kf[5] = 0.5 * (kh[ij    , k, l] + kh[ip1j  , k, l])

                dscl[g, k, l] = ( kf[0] * coef_diff[g, 0, XDIR, l] * ( vt[ij    , XDIR, TI] + vt[ij    , XDIR, TJ] )
                                + kf[1] * coef_diff[g, 1, XDIR, l] * ( vt[ij    , XDIR, TJ] + vt[im1j  , XDIR, TI] )
                                + kf[2] * coef_diff[g, 2, XDIR, l] * ( vt[im1j  , XDIR, TI] + vt[im1jm1, XDIR, TJ] )
                                + kf[3] * coef_diff[g, 3, XDIR, l] * ( vt[im1jm1, XDIR, TJ] + vt[im1jm1, XDIR, TI] )
                                + kf[4] * coef_diff[g, 4, XDIR, l] * ( vt[im1jm1, XDIR, TI] + vt[ijm1  , XDIR, TJ] )
                                + kf[5] * coef_diff[g, 5, XDIR, l] * ( vt[ijm1  , XDIR, TJ] + vt[ij    , XDIR, TI] ))
                
            for g in range(gmin-1, gmax):
                ij = g
                ip1j = g + 1
                ip1jp1 = g + iall + 1
                ijp1 = g + iall
                im1j = g - 1
                im1jm1 = g - iall - 1
                ijm1 = g - iall

                kf[0] = 0.5 * (kh[ij    , k, l] + kh[ip1jp1, k, l])
                kf[1] = 0.5 * (kh[ij    , k, l] + kh[ijp1  , k, l])
                kf[2] = 0.5 * (kh[im1j  , k, l] + kh[ij    , k, l])
                kf[3] = 0.5 * (kh[im1jm1, k, l] + kh[ij    , k, l])
                kf[4] = 0.5 * (kh[ijm1  , k, l] + kh[ij    , k, l])
                kf[5] = 0.5 * (kh[ij    , k, l] + kh[ip1j  , k, l])

                dscl[g, k, l] = dscl[g, k, l] + ( kf[0] * coef_diff[g, 0, YDIR, l] * ( vt[ij    , XDIR, TI] + vt[ij    , YDIR, TJ] )
                                                + kf[1] * coef_diff[g, 1, YDIR, l] * ( vt[ij    , XDIR, TJ] + vt[im1j  , YDIR, TI] )
                                                + kf[2] * coef_diff[g, 2, YDIR, l] * ( vt[im1j  , XDIR, TI] + vt[im1jm1, YDIR, TJ] )
                                                + kf[3] * coef_diff[g, 3, YDIR, l] * ( vt[im1jm1, XDIR, TJ] + vt[im1jm1, YDIR, TI] )
                                                + kf[4] * coef_diff[g, 4, YDIR, l] * ( vt[im1jm1, XDIR, TI] + vt[ijm1  , YDIR, TJ] )
                                                + kf[5] * coef_diff[g, 5, YDIR, l] * ( vt[ijm1  , XDIR, TJ] + vt[ij    , YDIR, TI] ))
            for g in range(gmin-1, gmax):
                ij     = g
                ip1j   = g + 1
                ip1jp1 = g + iall + 1
                ijp1   = g + iall
                im1j   = g - 1
                im1jm1 = g - iall - 1
                ijm1   = g - iall

                kf[0] = 0.5 * (kh[ij    , k, l] + kh[ip1jp1, k, l])
                kf[1] = 0.5 * (kh[ij    , k, l] + kh[ijp1  , k, l])
                kf[2] = 0.5 * (kh[im1j  , k, l] + kh[ij    , k, l])
                kf[3] = 0.5 * (kh[im1jm1, k, l] + kh[ij    , k, l])
                kf[4] = 0.5 * (kh[ijm1  , k, l] + kh[ij    , k, l])
                kf[5] = 0.5 * (kh[ij    , k, l] + kh[ip1j  , k, l])
                
                dscl[g, k, l] = dscl[g, k, l] + ( kf[0] * coef_diff[g, 0, ZDIR, l] * ( vt[ij    , XDIR, TI] + vt[ij    , ZDIR, TJ] )
                                                + kf[1] * coef_diff[g, 1, ZDIR, l] * ( vt[ij    , XDIR, TJ] + vt[im1j  , ZDIR, TI] )
                                                + kf[2] * coef_diff[g, 2, ZDIR, l] * ( vt[im1j  , XDIR, TI] + vt[im1jm1, ZDIR, TJ] )
                                                + kf[3] * coef_diff[g, 3, ZDIR, l] * ( vt[im1jm1, XDIR, TJ] + vt[im1jm1, ZDIR, TI] )
                                                + kf[4] * coef_diff[g, 4, ZDIR, l] * ( vt[im1jm1, XDIR, TI] + vt[ijm1  , ZDIR, TJ] )
                                                + kf[5] * coef_diff[g, 5, ZDIR, l] * ( vt[ijm1  , XDIR, TJ] + vt[ij    , ZDIR, TI] ))
            
            for g in range(gmax, gall):
                dscl[g, k, l] = 0.0
    
            # print(k, np.max(dscl[:, k, l]), np.min(dscl[:, k, l]), np.sum(dscl[:, k, l]))
    
        


    if ps.ADM_have_pl:
        n = ps.ADM_gslf_pl - 1

        for l in range(0, ps.ADM_lall_pl):
            for k in range(0, ps.ADM_kall):
                for d in range(0, ps.ADM_nxyz):
                    for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                        ij = v
                        ijp1 = v + 1
                        if ijp1 == (ps.ADM_gmax_pl):
                            ijp1 = ps.ADM_gmin_pl - 1
                        
                        vt_pl[ij, d] = ( ( 2.0 * coef_intp_pl[v, 0, d, l]
                                          -1.0 * coef_intp_pl[v, 1, d, l]
                                          -1.0 * coef_intp_pl[v, 2, d, l]) * scl_pl[n, k, l]
                                       + (-1.0 * coef_intp_pl[v, 0, d, l]
                                          +2.0 * coef_intp_pl[v, 1, d, l]
                                          -1.0 * coef_intp_pl[v, 2, d, l]) * scl_pl[ij, k, l]
                                       + (-1.0 * coef_intp_pl[v, 0, d, l]
                                          -1.0 * coef_intp_pl[v, 1, d, l]
                                          +2.0 * coef_intp_pl[v, 2, d, l]) * scl_pl[ijp1, k, l]
                                    ) / 3.0
            
                dscl_pl[:, k, l] = 0.0

                for v in range (ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                    ij = v
                    ijm1 = v - 1
                    if ijm1 == (ps.ADM_gmin_pl - 2):
                        ijm1 = ps.ADM_gmax_pl - 1
                    
                    dscl_pl[n, k, l] = (dscl_pl[n, k, l] 
                                     + (coef_diff_pl[v-1, XDIR, l] * ( vt_pl[ijm1, XDIR] + vt_pl[ij, XDIR])
                                     + coef_diff_pl[v-1, YDIR, l] * ( vt_pl[ijm1, YDIR] + vt_pl[ij, YDIR])
                                     + coef_diff_pl[v-1, ZDIR, l] * ( vt_pl[ijm1, ZDIR] + vt_pl[ij, ZDIR])
                                     ) * 0.5 * (kh_pl[n, k, l] + kh_pl[ij, k, l])
                                     )
                
                # print(l, k, np.max(dscl_pl[:, k, l]), np.min(dscl_pl[:, k, l]), np.sum(dscl_pl[:, k, l]))

    else:
        dscl_pl[:, :, :] = 0.0
    

    return

    # call DEBUG_rapend('OPRT_diffusion')
 




