import problem_size as ps
import mod_misc as mm
import numpy as np



def OPRT3D_divdamp(ddivdx, ddivdx_pl, ddivdy, ddivdy_pl, ddivdz, ddivdz_pl,
                   rhogvx, rhogvx_pl, rhogvy, rhogvy_pl, rhogvz, rhogvz_pl,
                   rhogw, rhogw_pl, coef_intp, coef_intp_pl, coef_diff,
                   coef_diff_pl, RGSQRTH, RGSQRTH_pl, RGAM, RGAM_pl, RGAMH, RGAMH_pl,
                   C2WfactGz, C2WfactGz_pl):
    
    sclt = np.zeros((ps.ADM_gall, 2))
    sclt_pl = np.zeros((ps.ADM_gall_pl))
    sclt_rhogw = 0.0
    sclt_rhogw_pl = 0.0

    rhogvx_vm = np.zeros((ps.ADM_gall))
    rhogvx_vm_pl = np.zeros((ps.ADM_gall_pl))
    rhogvy_vm = np.zeros((ps.ADM_gall))
    rhogvy_vm_pl = np.zeros((ps.ADM_gall_pl))
    rhogvz_vm = np.zeros((ps.ADM_gall))
    rhogvz_vm_pl = np.zeros((ps.ADM_gall_pl))
    rhogw_vm = np.zeros((ps.ADM_gall, ps.ADM_kall, ps.ADM_lall))
    rhogw_vm_pl = np.zeros((ps.ADM_gall_pl, ps.ADM_kall, ps.ADM_lall_pl))


    gmin = (ps.ADM_gmin - 1) * ps.ADM_gall_1d + ps.ADM_gmin
    gmax = (ps.ADM_gmax - 1) * ps.ADM_gall_1d + ps.ADM_gmax
    iall = ps.ADM_gall_1d
    gall = ps.ADM_gall
    kall = ps.ADM_kall
    kmin = ps.ADM_kmin
    kmax = ps.ADM_kmax
    lall = ps.ADM_lall

    gmin1 = (ps.ADM_gmin-1-1)*ps.ADM_gall_1d + ps.ADM_gmin-1

    GRD_rdgz = mm.GRD_rdgz

    for l in range(lall):
        for k in range(kmin, kmax):
            for g in range(gall):
                rhogw_vm[g, k, l] = ( C2WfactGz[g, k, 0, l] * rhogvx[g, k, l]
                                    + C2WfactGz[g, k, 1, l] * rhogvx[g, k-1, l]
                                    + C2WfactGz[g, k, 2, l] * rhogvy[g, k, l]
                                    + C2WfactGz[g, k, 3, l] * rhogvy[g, k-1, l]
                                    + C2WfactGz[g, k, 4, l] * rhogvz[g, k, l]
                                    + C2WfactGz[g, k, 5, l] * rhogvz[g, k-1, l]
                                    ) * RGAMH[g, k, l] + rhogw[g, k, l] * RGSQRTH[g, k, l]
                
        
        for g in range(gall):
            rhogw_vm[g, kmin-1, l] = 0.0
            rhogw_vm[g, kmax, l] = 0.0
        
    for l in range(lall):
        for k in range(kmin-1, kmax):
            for g in range(gall):
                rhogvx_vm[g] = rhogvx[g, k, l] * RGAM[g, k, l]
                rhogvy_vm[g] = rhogvy[g, k, l] * RGAM[g, k, l]
                rhogvz_vm[g] = rhogvz[g, k, l] * RGAM[g, k, l]
            
            for g in range(gmin1-1, gmax):
                ij = g
                ip1j = g + 1
                ip1jp1 = g + iall + 1
                ijp1 = g + iall
            
                sclt_rhogw = ( (rhogw_vm[ij, k+1,l] + rhogw_vm[ip1j, k+1, l] + rhogw_vm[ip1jp1, k+1, l])
                             - (rhogw_vm[ij, k, l] + rhogw_vm[ip1j, k, l] + rhogw_vm[ip1jp1, k, l])
                             ) / 3.0 * GRD_rdgz[k]
                
                sclt[g, ps.TI] = ( coef_intp[g, 0, ps.XDIR, ps.TI, l] * rhogvx_vm[ij]
                                 + coef_intp[g, 1, ps.XDIR, ps.TI, l] * rhogvx_vm[ip1j]
                                 + coef_intp[g, 2, ps.XDIR, ps.TI, l] * rhogvx_vm[ip1jp1]
                                 + coef_intp[g, 0, ps.YDIR, ps.TI, l] * rhogvy_vm[ij]
                                 + coef_intp[g, 1, ps.YDIR, ps.TI, l] * rhogvy_vm[ip1j]
                                 + coef_intp[g, 2, ps.YDIR, ps.TI, l] * rhogvy_vm[ip1jp1]
                                 + coef_intp[g, 0, ps.ZDIR, ps.TI, l] * rhogvz_vm[ij]
                                 + coef_intp[g, 1, ps.ZDIR, ps.TI, l] * rhogvz_vm[ip1j]
                                 + coef_intp[g, 2, ps.ZDIR, ps.TI, l] * rhogvz_vm[ip1jp1]
                                 + sclt_rhogw)
                
            for g in range(gmin1-1, gmax):
                ij = g
                ip1j = g + 1
                ip1jp1 = g + iall + 1
                ijp1 = g + iall
                
                sclt_rhogw = ( (rhogw_vm[ij, k+1,l] + rhogw_vm[ip1jp1, k+1, l] + rhogw_vm[ijp1, k+1, l])
                             - (rhogw_vm[ij, k, l] + rhogw_vm[ip1jp1, k, l] + rhogw_vm[ijp1, k, l])
                             ) / 3.0 * GRD_rdgz[k]
                
                sclt[g, ps.TJ] = ( coef_intp[g, 0, ps.XDIR, ps.TJ, l] * rhogvx_vm[ij]
                                 + coef_intp[g, 1, ps.XDIR, ps.TJ, l] * rhogvx_vm[ip1jp1]
                                 + coef_intp[g, 2, ps.XDIR, ps.TJ, l] * rhogvx_vm[ijp1]
                                 + coef_intp[g, 0, ps.YDIR, ps.TJ, l] * rhogvy_vm[ij]
                                 + coef_intp[g, 1, ps.YDIR, ps.TJ, l] * rhogvy_vm[ip1jp1]
                                 + coef_intp[g, 2, ps.YDIR, ps.TJ, l] * rhogvy_vm[ijp1]
                                 + coef_intp[g, 0, ps.ZDIR, ps.TJ, l] * rhogvz_vm[ij]
                                 + coef_intp[g, 1, ps.ZDIR, ps.TJ, l] * rhogvz_vm[ip1jp1]
                                 + coef_intp[g, 2, ps.ZDIR, ps.TJ, l] * rhogvz_vm[ijp1]
                                 + sclt_rhogw)
                
            if (ps.ADM_have_sgp(l)):
                sclt[gmin1-1, ps.TI] = sclt[gmin1, ps.TJ]
            
            for g in range(gmin-1):
                ddivdx[g, k, l] = 0.0
                ddivdy[g, k, l] = 0.0
                ddivdz[g, k, l] = 0.0

            for g in range(gmin-1, gmax):
                ij = g
                im1j = g - 1
                im1jm1 = g - iall - 1
                ijm1 = g - iall

                ddivdx[g, k, l] = ( coef_diff[g, 0, ps.XDIR, l] * (sclt[ij, ps.TI]     + sclt[ij, ps.TJ])
                                  + coef_diff[g, 1, ps.XDIR, l] * (sclt[ij, ps.TJ]     + sclt[im1j, ps.TI])
                                  + coef_diff[g, 2, ps.XDIR, l] * (sclt[im1j, ps.TI]   + sclt[im1jm1, ps.TJ])
                                  + coef_diff[g, 3, ps.XDIR, l] * (sclt[im1jm1, ps.TJ] + sclt[im1jm1, ps.TI])
                                  + coef_diff[g, 4, ps.XDIR, l] * (sclt[im1jm1, ps.TI] + sclt[ijm1, ps.TJ])
                                  + coef_diff[g, 5, ps.XDIR, l] * (sclt[ijm1, ps.TJ]   + sclt[ij, ps.TI]) )
            

            for g in range(gmin-1, gmax):
                ij = g
                im1j = g - 1
                im1jm1 = g - iall - 1
                ijm1 = g - iall

                ddivdy[g, k, l] = ( coef_diff[g, 0, ps.YDIR, l] * (sclt[ij, ps.TI]     + sclt[ij, ps.TJ])
                                  + coef_diff[g, 1, ps.YDIR, l] * (sclt[ij, ps.TJ]     + sclt[im1j, ps.TI])
                                  + coef_diff[g, 2, ps.YDIR, l] * (sclt[im1j, ps.TI]   + sclt[im1jm1, ps.TJ])
                                  + coef_diff[g, 3, ps.YDIR, l] * (sclt[im1jm1, ps.TJ] + sclt[im1jm1, ps.TI])
                                  + coef_diff[g, 4, ps.YDIR, l] * (sclt[im1jm1, ps.TI] + sclt[ijm1, ps.TJ])
                                  + coef_diff[g, 5, ps.YDIR, l] * (sclt[ijm1, ps.TJ]   + sclt[ij, ps.TI]) )
                            
            for g in range(gmin-1, gmax):
                ij = g
                im1j = g - 1
                im1jm1 = g - iall - 1
                ijm1 = g - iall

                ddivdz[g, k, l] = ( coef_diff[g, 0, ps.ZDIR, l] * (sclt[ij, ps.TI]     + sclt[ij, ps.TJ])
                                  + coef_diff[g, 1, ps.ZDIR, l] * (sclt[ij, ps.TJ]     + sclt[im1j, ps.TI])
                                  + coef_diff[g, 2, ps.ZDIR, l] * (sclt[im1j, ps.TI]   + sclt[im1jm1, ps.TJ])
                                  + coef_diff[g, 3, ps.ZDIR, l] * (sclt[im1jm1, ps.TJ] + sclt[im1jm1, ps.TI])
                                  + coef_diff[g, 4, ps.ZDIR, l] * (sclt[im1jm1, ps.TI] + sclt[ijm1, ps.TJ])
                                  + coef_diff[g, 5, ps.ZDIR, l] * (sclt[ijm1, ps.TJ]   + sclt[ij, ps.TI]) )
                            
            for g in range(gmax, gall):
                ddivdx[g, k, l] = 0.0
                ddivdy[g, k, l] = 0.0
                ddivdz[g, k, l] = 0.0

        for g in range(gall):
            ddivdx[g, kmin-2, l] = 0.0    
            ddivdy[g, kmin-2, l] = 0.0    
            ddivdz[g, kmin-2, l] = 0.0    
            ddivdx[g, kmax, l] = 0.0    
            ddivdy[g, kmax, l] = 0.0    
            ddivdz[g, kmax, l] = 0.0    

    if ps.ADM_have_pl:
        n = ps.ADM_gslf_pl - 1

        for l in range(ps.ADM_lall_pl):
            for k in range(ps.ADM_kmin, ps.ADM_kmax):
                for g in range(ps.ADM_gall_pl):
                        rhogw_vm_pl[g, k, l] = ( C2WfactGz_pl[g, k, 0, l] * rhogvx_pl[g, k, l]
                                               + C2WfactGz_pl[g, k, 1, l] * rhogvx_pl[g, k-1, l]
                                               + C2WfactGz_pl[g, k, 2, l] * rhogvy_pl[g, k, l]
                                               + C2WfactGz_pl[g, k, 3, l] * rhogvy_pl[g, k-1, l]
                                               + C2WfactGz_pl[g, k, 4, l] * rhogvz_pl[g, k, l]
                                               + C2WfactGz_pl[g, k, 5, l] * rhogvz_pl[g, k-1, l]
                                               ) * RGAMH_pl[g, k, l] + rhogw_pl[g, k, l] * RGSQRTH_pl[g, k, l]
                
            for g in range(ps.ADM_gall_pl):
                rhogw_vm_pl[g, ps.ADM_kmin-1, l] = 0.0
                rhogw_vm_pl[g, ps.ADM_kmax, l] = 0.0
        
        for l in range(ps.ADM_lall_pl):
            for k in range(ps.ADM_kmin-1, ps.ADM_kmax):
                for v in range(ps.ADM_gall_pl):
                    rhogvx_vm_pl[v] = rhogvx_pl[v, k, l] * RGAM_pl[v, k, l]
                    rhogvy_vm_pl[v] = rhogvy_pl[v, k, l] * RGAM_pl[v, k, l]
                    rhogvz_vm_pl[v] = rhogvz_pl[v, k, l] * RGAM_pl[v, k, l]
                
                for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                    ij = v
                    ijp1 = v + 1
                    if ijp1 == ps.ADM_gmax_pl:
                        ijp1 = ps.ADM_gmin_pl - 1

                    sclt_rhogw_pl = ( ( rhogw_vm_pl[n, k+1, l] + rhogw_vm_pl[ij, k+1, l] + rhogw_vm_pl[ijp1, k+1, l])
                                    - ( rhogw_vm_pl[n, k, l] + rhogw_vm_pl[ij, k, l] + rhogw_vm_pl[ijp1, k, l])
                                    ) / 3.0 * GRD_rdgz[k]
                    
                    sclt_pl[ij] = ( 
                                    coef_intp_pl[v, 0, ps.XDIR, l] * rhogvx_vm_pl[n]
                                  + coef_intp_pl[v, 1, ps.XDIR, l] * rhogvx_vm_pl[ij]
                                  + coef_intp_pl[v, 2, ps.XDIR, l] * rhogvx_vm_pl[ijp1]
                                  + coef_intp_pl[v, 0, ps.YDIR, l] * rhogvy_vm_pl[n]
                                  + coef_intp_pl[v, 1, ps.YDIR, l] * rhogvy_vm_pl[ij]
                                  + coef_intp_pl[v, 2, ps.YDIR, l] * rhogvy_vm_pl[ijp1]
                                  + coef_intp_pl[v, 0, ps.ZDIR, l] * rhogvz_vm_pl[n]
                                  + coef_intp_pl[v, 1, ps.ZDIR, l] * rhogvz_vm_pl[ij]
                                  + coef_intp_pl[v, 2, ps.ZDIR, l] * rhogvz_vm_pl[ijp1]
                                  + sclt_rhogw_pl
                                  )
                
                ddivdx_pl[:, k, l] = 0.0
                ddivdy_pl[:, k, l] = 0.0
                ddivdz_pl[:, k, l] = 0.0

                for v in range(ps.ADM_gmin_pl-1, ps.ADM_gmax_pl):
                    ij = v
                    ijm1 = v - 1
                    if ijm1 == ps.ADM_gmin_pl - 2:
                        ijm1 = ps.ADM_gmax_pl -1
                    
                    ddivdx_pl[n, k, l] = ddivdx_pl[n, k, l] + coef_diff_pl[v-1, ps.XDIR, l] * (sclt_pl[ijm1] + sclt_pl[ij])
                    ddivdy_pl[n, k, l] = ddivdy_pl[n, k, l] + coef_diff_pl[v-1, ps.YDIR, l] * (sclt_pl[ijm1] + sclt_pl[ij])
                    ddivdz_pl[n, k, l] = ddivdz_pl[n, k, l] + coef_diff_pl[v-1, ps.ZDIR, l] * (sclt_pl[ijm1] + sclt_pl[ij])
            
            ddivdx_pl[:, ps.ADM_kmin-2, l] = 0.0
            ddivdx_pl[:, ps.ADM_kmax, l] = 0.0
            ddivdy_pl[:, ps.ADM_kmin-2, l] = 0.0
            ddivdy_pl[:, ps.ADM_kmax, l] = 0.0
            ddivdz_pl[:, ps.ADM_kmin-2, l] = 0.0
            ddivdz_pl[:, ps.ADM_kmax, l] = 0.0

    else:
        ddivdx_pl[:, :, :] = 0.0
        ddivdy_pl[:, :, :] = 0.0
        ddivdz_pl[:, :, :] = 0.0

    return
        
            
        
    
            

            



