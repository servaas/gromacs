/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.
 * Copyright (c) 2012,2013, by the GROMACS development team, led by
 * David van der Spoel, Berk Hess, Erik Lindahl, and including many
 * others, as listed in the AUTHORS file in the top-level source
 * directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>

#include "vec.h"
#include "typedefs.h"
#include "nonbonded.h"
#include "nb_kernel.h"
#include "nrnb.h"

void
gmx_nb_free_energy_kernel_gapsys(t_nblist *                nlist,
                          rvec *                    xx,
                          rvec *                    ff,
                          t_forcerec *              fr,
                          t_mdatoms *               mdatoms,
                          nb_kernel_data_t *        kernel_data,
                          t_nrnb *                  nrnb)
{
#define  STATE_A  0
#define  STATE_B  1
#define  NSTATES  2
    int           i, j, n, ii, is3, ii3, k, nj0, nj1, jnr, j3, ggid;
    real          shX, shY, shZ;
    real          Fscal, FscalC[NSTATES], FscalV[NSTATES], tx, ty, tz;
    real          Vcoul[NSTATES], Vvdw[NSTATES];
    real          rinv6, r, rt, rtC, rtV;
    real          iqA, iqB;
    real          qq[NSTATES], vctot, krsq;
    int           ntiA, ntiB, tj[NSTATES];
    real          Vvdw6, Vvdw12, vvtot;
    real          ix, iy, iz, fix, fiy, fiz;
    real          dx, dy, dz, rsq, rinv, rinvsq;
    real          c6[NSTATES], c12[NSTATES];
    real          LFC[NSTATES], LFV[NSTATES], LFC_inv[NSTATES], LFV_inv[NSTATES], DLF[NSTATES];
    double        dvdl_coul, dvdl_vdw;
    real          lfac_coul[NSTATES], dlfac_coul[NSTATES], lfac_vdw[NSTATES], dlfac_vdw[NSTATES];
    real          sigma6[NSTATES], alpha_vdw_eff, alpha_coul_eff, sigma2_def, sigma2_min;
    real          rp, rpm2, rC, rV, rinvC, rinv4C, rpinvC, rinvV, rpinvV;
    real          sigma2[NSTATES], sigma_pow[NSTATES], sigma_powm2[NSTATES], rs, rs2;
    int           do_coultab, do_vdwtab, do_tab, tab_elemsize;
    int           n0, n1C, n1V, nnn;
    real          Y, F, G, H, Fp, Geps, Heps2, epsC, eps2C, epsV, eps2V, VV, FF;
    int           icoul, ivdw;
    int           nri;
    int *         iinr;
    int *         jindex;
    int *         jjnr;
    int *         shift;
    int *         gid;
    int *         typeA;
    int *         typeB;
    int           ntype;
    real *        shiftvec;
    real          dvdl_part;
    real *        fshift;
    real          tabscale;
    real *        VFtab;
    real *        x;
    real *        f;
    real          facel, krf, crf;
    real *        chargeA;
    real *        chargeB;
    real          sigma6_min, sigma6_def, lam_power, sc_power, sc_r_power;
    real          alpha_coul, alpha_vdw, lambda_coul, lambda_vdw, ewc;
    real *        nbfp;
    real *        dvdl;
    real *        Vv;
    real *        Vc;
    gmx_bool      bDoForces;
    real          rcoulomb, rvdw, factor_coul, factor_vdw, sh_invrc6;
    gmx_bool      bExactElecCutoff, bExactVdwCutoff;
    real          rcutoff, rcutoff2, rswitch, d, d2, swV3, swV4, swV5, swF2, swF3, swF4, sw, dsw, rinvcorr;

    x                   = xx[0];
    f                   = ff[0];

    fshift              = fr->fshift[0];
    Vc                  = kernel_data->energygrp_elec;
    Vv                  = kernel_data->energygrp_vdw;
    tabscale            = kernel_data->table_elec_vdw->scale;
    VFtab               = kernel_data->table_elec_vdw->data;

    nri                 = nlist->nri;
    iinr                = nlist->iinr;
    jindex              = nlist->jindex;
    jjnr                = nlist->jjnr;
    icoul               = nlist->ielec;
    ivdw                = nlist->ivdw;
    shift               = nlist->shift;
    gid                 = nlist->gid;

    shiftvec            = fr->shift_vec[0];
    chargeA             = mdatoms->chargeA;
    chargeB             = mdatoms->chargeB;
    facel               = fr->epsfac;
    krf                 = fr->k_rf;
    crf                 = fr->c_rf;
    ewc                 = fr->ewaldcoeff;
    Vc                  = kernel_data->energygrp_elec;
    typeA               = mdatoms->typeA;
    typeB               = mdatoms->typeB;
    ntype               = fr->ntype;
    nbfp                = fr->nbfp;
    Vv                  = kernel_data->energygrp_vdw;
    tabscale            = kernel_data->table_elec_vdw->scale;
    VFtab               = kernel_data->table_elec_vdw->data;
    lambda_coul         = kernel_data->lambda[efptCOUL];
    lambda_vdw          = kernel_data->lambda[efptVDW];
    dvdl                = kernel_data->dvdl;
    alpha_coul          = fr->sc_alphacoul;
    alpha_vdw           = fr->sc_alphavdw;
    lam_power           = fr->sc_power;
    sc_r_power          = fr->sc_r_power;
    sigma6_def          = fr->sc_sigma6_def;
    sigma6_min          = fr->sc_sigma6_min;
    bDoForces           = kernel_data->flags & GMX_NONBONDED_DO_FORCE;

    rcoulomb            = fr->rcoulomb;
    rvdw                = fr->rvdw;
    sh_invrc6           = fr->ic->sh_invrc6;

    if (fr->coulomb_modifier == eintmodPOTSWITCH || fr->vdw_modifier == eintmodPOTSWITCH)
    {
        rcutoff         = (fr->coulomb_modifier == eintmodPOTSWITCH) ? fr->rcoulomb : fr->rvdw;
        rcutoff2        = rcutoff*rcutoff;
        rswitch         = (fr->coulomb_modifier == eintmodPOTSWITCH) ? fr->rcoulomb_switch : fr->rvdw_switch;
        d               = rcutoff-rswitch;
        swV3            = -10.0/(d*d*d);
        swV4            =  15.0/(d*d*d*d);
        swV5            =  -6.0/(d*d*d*d*d);
        swF2            = -30.0/(d*d*d);
        swF3            =  60.0/(d*d*d*d);
        swF4            = -30.0/(d*d*d*d*d);
    }
    else
    {
        /* Stupid compilers dont realize these variables will not be used */
        rswitch         = 0.0;
        swV3            = 0.0;
        swV4            = 0.0;
        swV5            = 0.0;
        swF2            = 0.0;
        swF3            = 0.0;
        swF4            = 0.0;
    }

    bExactElecCutoff    = (fr->coulomb_modifier != eintmodNONE) || fr->eeltype == eelRF_ZERO;
    bExactVdwCutoff     = (fr->vdw_modifier != eintmodNONE);

    /* fix compiler warnings */
    nj1   = 0;
    n1C   = n1V   = 0;
    epsC  = epsV  = 0;
    eps2C = eps2V = 0;

    dvdl_coul  = 0;
    dvdl_vdw   = 0;

    /* Lambda factor for state A, 1-lambda*/
    LFC[STATE_A] = 1.0 - lambda_coul;
    LFV[STATE_A] = 1.0 - lambda_vdw;
    LFC_inv[STATE_A] = lambda_coul;
    LFV_inv[STATE_A] = lambda_vdw;

    /* Lambda factor for state B, lambda*/
    LFC[STATE_B] = lambda_coul;
    LFV[STATE_B] = lambda_vdw;
    LFC_inv[STATE_B] = 1.0 - lambda_coul;
    LFV_inv[STATE_B] = 1.0 - lambda_vdw;

    /*derivative of the lambda factor for state A and B */
    DLF[STATE_A] = -1;
    DLF[STATE_B] = 1;

    for (i = 0; i < NSTATES; i++)
    {
        lfac_coul[i]  = (lam_power == 2 ? (1-LFC[i])*(1-LFC[i]) : (1-LFC[i]));
        dlfac_coul[i] = DLF[i]*lam_power/sc_r_power*(lam_power == 2 ? (1-LFC[i]) : 1);
        lfac_vdw[i]   = (lam_power == 2 ? (1-LFV[i])*(1-LFV[i]) : (1-LFV[i]));
        dlfac_vdw[i]  = DLF[i]*lam_power/sc_r_power*(lam_power == 2 ? (1-LFV[i]) : 1);
    }
    /* precalculate */
    sigma2_def = pow(sigma6_def, 1.0/3.0);
    sigma2_min = pow(sigma6_min, 1.0/3.0);

    /* Ewald (not PME) table is special (icoul==enbcoulFEWALD) */

    do_coultab = (icoul == GMX_NBKERNEL_ELEC_CUBICSPLINETABLE);
    do_vdwtab  = (ivdw == GMX_NBKERNEL_VDW_CUBICSPLINETABLE);

    do_tab = do_coultab || do_vdwtab;

    /* we always use the combined table here */
    tab_elemsize = 12;

    // VG //
    // VG // starting modifying from here, the above variables are left for later
    // VG //
    real r0Q=0.0;
    real r0LJ=0.0;
    real b_q, a_q, a_lj, b_lj;
    real rinvQ, rinv2Q, rinv3Q;
    real rinv14LJ, rinv13LJ, rinv12LJ;
    real rinv8LJ, rinv7LJ, rinv6LJ;
    real twentysix_seventh = 26.0/7.0;
    real one_sixth = 1.0/6.0;
    real charge1[NSTATES], charge2[NSTATES];
    real state_ratioC[NSTATES], state_ratioV[NSTATES], state_sign[NSTATES];
    real c12_scaled, c6_scaled;

    state_ratioC[STATE_A] = LFC[STATE_A]/LFC[STATE_B];
    state_ratioC[STATE_B] = LFC[STATE_B]/LFC[STATE_A];
    state_ratioV[STATE_A] = LFV[STATE_A]/LFV[STATE_B];
    state_ratioV[STATE_B] = LFV[STATE_B]/LFV[STATE_A];
    state_sign[STATE_A] = -1.0;
    state_sign[STATE_B] = 1.0;

    // for now the parameters are hard-coded
    alpha_vdw = 0.85;//0.85
    real alpha_Q = 0.3;//0.3
    real sigma_Q = 1.0;

    for (n = 0; (n < nri); n++)
    {
        is3              = 3*shift[n];
        shX              = shiftvec[is3];
        shY              = shiftvec[is3+1];
        shZ              = shiftvec[is3+2];
        nj0              = jindex[n];
        nj1              = jindex[n+1];
        ii               = iinr[n];
        ii3              = 3*ii;
        ix               = shX + x[ii3+0];
        iy               = shY + x[ii3+1];
        iz               = shZ + x[ii3+2];
        iqA              = facel*chargeA[ii];
        iqB              = facel*chargeB[ii];
        ntiA             = 2*ntype*typeA[ii];
        ntiB             = 2*ntype*typeB[ii];
        vctot            = 0;
        vvtot            = 0;
        fix              = 0;
        fiy              = 0;
        fiz              = 0;
        charge1[STATE_A] = chargeA[ii];
        charge1[STATE_B] = chargeB[ii];

        for (k = nj0; (k < nj1); k++)
        {
            jnr              = jjnr[k];
            j3               = 3*jnr;
            dx               = ix - x[j3];
            dy               = iy - x[j3+1];
            dz               = iz - x[j3+2];
            rsq              = dx*dx+dy*dy+dz*dz;
            rinv             = gmx_invsqrt(rsq);
            rinvsq			 = rinv*rinv;
            r                = rsq*rinv;
            tj[STATE_A]      = ntiA+2*typeA[jnr];
            tj[STATE_B]      = ntiB+2*typeB[jnr];
            qq[STATE_A]      = iqA*chargeA[jnr];
            qq[STATE_B]      = iqB*chargeB[jnr];
            charge2[STATE_A] = chargeA[jnr];
            charge2[STATE_B] = chargeB[jnr];
            rpm2             = rsq*rsq;  /* r4 */
            rp               = rpm2*rsq; /* r6 */


            r0Q=0.0; b_q=0.0; a_q=0.0; a_lj=0.0; b_lj=0.0;
            rinvQ=0.0; rinv2Q=0.0; rinv3Q=0.0;
            rinv14LJ=0.0; rinv13LJ=0.0; rinv12LJ=0.0;
            rinv8LJ=0.0; rinv7LJ=0.0; rinv6LJ=0.0;

            for (i = 0; i < NSTATES; i++)
            {
                FscalC[i]    = 0;
                FscalV[i]    = 0;
                Vcoul[i]     = 0;
                Vvdw[i]      = 0;

            	c6[i] = nbfp[tj[i]];
            	c12[i] = nbfp[tj[i]+1];

            	c6_scaled = c6[i]/6.0;
            	c12_scaled = c12[i]/12.0;

                if (do_tab)
                {
                    rtC        = r*tabscale;
                    n0         = rtC;
                    epsC       = rtC-n0;
                    eps2C      = epsC*epsC;
                    n1C        = tab_elemsize*n0;

                    rtV        = r*tabscale;
                    n0         = rtV;
                    epsV       = rtV-n0;
                    eps2V      = epsV*epsV;
                    n1V        = tab_elemsize*n0;
                }

            	if((c6[i] > 0) && (c12[i] > 0))
            	{
                	r0LJ = alpha_vdw*pow(twentysix_seventh*0.5*c12[i]/c6[i]*LFV_inv[i],one_sixth);

	                if(r0LJ>0)
       		        {
                	        rinv8LJ = 1.0/r0LJ;
                	        rinv8LJ = rinv8LJ*rinv8LJ;
                            rinv8LJ = rinv8LJ*rinv8LJ;
                            rinv8LJ = rinv8LJ*rinv8LJ;
                            rinv7LJ = rinv8LJ*r0LJ;
                            rinv6LJ = rinv7LJ*r0LJ;
                            rinv14LJ = rinv7LJ*rinv7LJ;
                            rinv13LJ = rinv7LJ*rinv6LJ;
                            rinv12LJ = rinv6LJ*rinv6LJ;
               		}
            	}

               	if( qq[i]!=0 )
                {
                	r0Q = alpha_Q*pow(LFC_inv[i],one_sixth)*(1+sigma_Q*fabs(charge2[i]*charge1[i]));

                    if(r0Q>0)
                    {
                    	rinvQ = 1.0/r0Q;
                        rinv2Q = rinvQ*rinvQ;
                        rinv3Q = rinv2Q*rinvQ;
                    }
                }


				// from here the soft-core starts
				FscalC[i] = 0.0;
				FscalV[i] = 0.0;
				Vcoul[i] = 0.0;
                Vvdw[i] = 0.0;
				rinv4C = 0.0;
				rinvC = 0.0;

				if( qq[i] != 0 ) //Coulomb
				{
					rpinvC = rinvsq;
					rinvC = rinv;
					rC = r;

					if( r<r0Q ) //soft-core
					{
						b_q = -2.0*qq[i]*rinv3Q;
						a_q = 3.0*qq[i]*rinv2Q;
						FscalC[i] = b_q + a_q*rinv;//multiplication by r in the very end
						Vcoul[i] = -0.5*b_q*r*r - a_q*r + 3.0*qq[i]*rinvQ;
						dvdl_coul += state_sign[i] * ( 0.5*state_ratioC[i]*(qq[i]*rinv3Q*r*r - 2.0*qq[i]*rinv2Q*r + qq[i]*rinvQ) );
					}
					else //hard-core
					{
						if (do_tab)
						{
							rtC        = rC*tabscale;
							n0         = rtC;
							epsC       = rtC-n0;
							eps2C      = epsC*epsC;
							n1C        = tab_elemsize*n0;
						}
						switch (icoul)
						{
							case GMX_NBKERNEL_ELEC_COULOMB:
							case GMX_NBKERNEL_ELEC_EWALD:
								/* simple cutoff (yes, ewald is done all on direct space for free energy) */
								Vcoul[i]   = qq[i]*rinvC;
								FscalC[i]  = Vcoul[i]*rpinvC;
								break;

							case GMX_NBKERNEL_ELEC_REACTIONFIELD:
								/* reaction-field */
								Vcoul[i]   = qq[i]*(rinvC+krf*rC*rC-crf);
								FscalC[i]  = qq[i]*(rinvC*rpinvC-2.0*krf);
								break;

							case GMX_NBKERNEL_ELEC_CUBICSPLINETABLE:
								/* non-Ewald tabulated coulomb */
								nnn        = n1C;
								Y          = VFtab[nnn];
								F          = VFtab[nnn+1];
								Geps       = epsC*VFtab[nnn+2];
								Heps2      = eps2C*VFtab[nnn+3];
								Fp         = F+Geps+Heps2;
								VV         = Y+epsC*Fp;
								FF         = Fp+Geps+2.0*Heps2;
								Vcoul[i]   = qq[i]*VV;
								FscalC[i]  = -qq[i]*tabscale*FF*rinvC;//rC*rpinvC;
								break;

							default:
								FscalC[i]  = 0.0;
								Vcoul[i]   = 0.0;
								break;
						}
					}
					//not sure what this means for the new soft-core
					if (fr->coulomb_modifier == eintmodPOTSWITCH)
					{
						d                = rC-rswitch;
						d                = (d > 0.0) ? d : 0.0;
						d2               = d*d;
						sw               = 1.0+d2*d*(swV3+d*(swV4+d*swV5));
						dsw              = d2*(swF2+d*(swF3+d*swF4));

						Vcoul[i]        *= sw;
						FscalC[i]        = FscalC[i]*sw + Vcoul[i]*dsw;
					}
				} //end Coulomb


				if(c12[i]!=0) // vdw
				{
					 rpinvV = 1.0/rp; //rp = r^6
					 rinvV = rinv;
					 rV = r;

					if( (r<r0LJ) ) //soft-core
					{
						//stateA
						b_lj = -156.0*c12_scaled*rinv14LJ + 42.0*c6_scaled*rinv8LJ;
						a_lj = 168.0*c12_scaled*rinv13LJ - 48.0*c6_scaled*rinv7LJ;
						FscalV[i] = b_lj + a_lj*rinv;//multiplication by r in the very end
						Vvdw[i] = -0.5*b_lj*r*r - a_lj*r + 91.0*c12_scaled*rinv12LJ - 28.0*c6_scaled*rinv6LJ;
						dvdl_vdw += state_sign[i] * ( 28.0*state_ratioV[i]*((6.5*c12_scaled*rinv14LJ-c6_scaled*rinv8LJ)*r*r-
								(13.0*c12_scaled*rinv13LJ-2.0*c6_scaled*rinv7LJ)*r+6.5*c12_scaled*rinv12LJ-c6_scaled*rinv6LJ) );
					 }
					 else //hard-core
					 {
						 switch (ivdw)
						 {
							 case GMX_NBKERNEL_VDW_LENNARDJONES:
								 /* cutoff LJ */
								 rinv6            = rpinvV;
								 Vvdw6            = c6[i]*rinv6;
								 Vvdw12           = c12[i]*rinv6*rinv6;
								 if (fr->vdw_modifier == eintmodPOTSHIFT)
								 {
									 Vvdw[i]          = ( (Vvdw12-c12[i]*sh_invrc6*sh_invrc6)*(1.0/12.0)
														  -(Vvdw6-c6[i]*sh_invrc6)*(1.0/6.0));
								 }
								 else
								 {
									 Vvdw[i]          = Vvdw12*(1.0/12.0)-Vvdw6*(1.0/6.0);
								 }
								 FscalV[i]        = (Vvdw12-Vvdw6)*rinvV*rinvV;
								 break;

							 case GMX_NBKERNEL_VDW_BUCKINGHAM:
								 gmx_fatal(FARGS, "Buckingham free energy not supported.");
								 break;

							 case GMX_NBKERNEL_VDW_CUBICSPLINETABLE:
								 /* Table LJ */
								 nnn = n1V+4;
								 /* dispersion */
								 Y          = VFtab[nnn];
								 F          = VFtab[nnn+1];
								 Geps       = epsV*VFtab[nnn+2];
								 Heps2      = eps2V*VFtab[nnn+3];
								 Fp         = F+Geps+Heps2;
								 VV         = Y+epsV*Fp;
								 FF         = Fp+Geps+2.0*Heps2;
								 Vvdw[i]   += c6[i]*VV;
								 FscalV[i] -= c6[i]*tabscale*FF*rinvV;//rV*rpinvV;

								 /* repulsion */
								 Y          = VFtab[nnn+4];
								 F          = VFtab[nnn+5];
								 Geps       = epsV*VFtab[nnn+6];
								 Heps2      = eps2V*VFtab[nnn+7];
								 Fp         = F+Geps+Heps2;
								 VV         = Y+epsV*Fp;
								 FF         = Fp+Geps+2.0*Heps2;
								 Vvdw[i]   += c12[i]*VV;
								 FscalV[i] -= c12[i]*tabscale*FF*rinvV;//rV*rpinvV;
								 break;

							 default:
								 Vvdw[i]    = 0.0;
								 FscalV[i]  = 0.0;
								 break;
						 }
					 }
					 //not sure what this means for the new soft-core
					 if (fr->vdw_modifier == eintmodPOTSWITCH)
					 {
						 d                = rV-rswitch;
						 d                = (d > 0.0) ? d : 0.0;
						 d2               = d*d;
						 sw               = 1.0+d2*d*(swV3+d*(swV4+d*swV5));
						 dsw              = d2*(swF2+d*(swF3+d*swF4));

						 Vvdw[i]         *= sw;
						 FscalV[i]        = FscalV[i]*sw + Vvdw[i]*dsw;

						 FscalV[i]        = (rV < rvdw) ? FscalV[i] : 0.0;
						 Vvdw[i]          = (rV < rvdw) ? Vvdw[i] : 0.0;
					 }

				} //end vdw
            } //end loop over NSTATES


			Fscal = 0.0;

			// Ewald
			if (icoul == GMX_NBKERNEL_ELEC_EWALD && !(bExactElecCutoff && r >= rcoulomb))
			{
				/* because we compute the softcore normally,
				   we have to remove the ewald short range portion. Done outside of
				   the states loop because this part doesn't depend on the scaled R */

#ifdef GMX_DOUBLE
				/* Relative accuracy at R_ERF_R_INACC of 3e-10 */
#define         R_ERF_R_INACC 0.006
#else
				/* Relative accuracy at R_ERF_R_INACC of 2e-5 */
#define         R_ERF_R_INACC 0.1
#endif
				if (ewc*r > R_ERF_R_INACC)
				{
					VV    = gmx_erf(ewc*r)*rinv;
					FF    = rinv*rinv*(VV - ewc*M_2_SQRTPI*exp(-ewc*ewc*rsq));
				}
				else
				{
					VV    = ewc*M_2_SQRTPI;
					FF    = ewc*ewc*ewc*M_2_SQRTPI*(2.0/3.0 - 0.4*ewc*ewc*rsq);
				}

				for (i = 0; i < NSTATES; i++)
				{
					vctot      -= LFC[i]*qq[i]*VV;
					Fscal      -= LFC[i]*qq[i]*FF;
					dvdl_coul  -= qq[i]*state_sign[i]*VV;	//(DLF[i]*qq[i])*VV;
				}
			}


			/* Assemble A and B states */
			for (i = 0; i < NSTATES; i++)
			{
				vctot         += LFC[i]*Vcoul[i];
				vvtot         += LFV[i]*Vvdw[i];

				Fscal         += LFC[i]*FscalC[i];
				Fscal         += LFV[i]*FscalV[i];

				dvdl_coul	  += state_sign[i]*Vcoul[i];
				dvdl_vdw	  += state_sign[i]*Vvdw[i];
			}

			// do forces
            if (bDoForces)
            {
                tx         = Fscal*dx;
                ty         = Fscal*dy;
                tz         = Fscal*dz;
                fix        = fix + tx;
                fiy        = fiy + ty;
                fiz        = fiz + tz;
                f[j3]      = f[j3]   - tx;
                f[j3+1]    = f[j3+1] - ty;
                f[j3+2]    = f[j3+2] - tz;
            }

        } //end the inner loop

        if (bDoForces)
        {
            f[ii3]         = f[ii3]        + fix;
            f[ii3+1]       = f[ii3+1]      + fiy;
            f[ii3+2]       = f[ii3+2]      + fiz;
            fshift[is3]    = fshift[is3]   + fix;
            fshift[is3+1]  = fshift[is3+1] + fiy;
            fshift[is3+2]  = fshift[is3+2] + fiz;
        }
        ggid               = gid[n];
        Vc[ggid]           = Vc[ggid] + vctot;
        Vv[ggid]           = Vv[ggid] + vvtot;

    } //end the outter loop

    dvdl[efptCOUL]     += dvdl_coul;
    dvdl[efptVDW]      += dvdl_vdw;

    /* Estimate flops, average for free energy stuff:
     * 12  flops per outer iteration
     * 150 flops per inner iteration
     */
    inc_nrnb(nrnb, eNR_NBKERNEL_FREE_ENERGY, nlist->nri*12 + nlist->jindex[n]*150);

}

real
nb_free_energy_evaluate_single_gapsys(real r2, real sc_r_power, real alpha_coul, real alpha_vdw,
                               real tabscale, real *vftab,
                               real qqA, real c6A, real c12A, real qqB, real c6B, real c12B,
                               real LFC[2], real LFV[2], real DLF[2],
                               real lfac_coul[2], real lfac_vdw[2], real dlfac_coul[2], real dlfac_vdw[2],
                               real sigma6_def, real sigma6_min, real sigma2_def, real sigma2_min,
                               real *velectot, real *vvdwtot, real *dvdl)
{
    real       r, rinv, rp, rpm2, rtab, eps, eps2, Y, F, Geps, Heps2, Fp, VV, FF, fscal;
    real       qq[2], c6[2], c12[2], sigma6[2], sigma2[2], sigma_pow[2], sigma_powm2[2];
    real       alpha_coul_eff, alpha_vdw_eff, dvdl_coul, dvdl_vdw;
    real       rpinv, r_coul, r_vdw, velecsum, vvdwsum;
    real       fscal_vdw[2], fscal_elec[2];
    real       velec[2], vvdw[2];
    int        i, ntab;

    qq[0]    = qqA;
    qq[1]    = qqB;
    c6[0]    = c6A;
    c6[1]    = c6B;
    c12[0]   = c12A;
    c12[1]   = c12B;

    r = sqrt(r2);
    rinv = 1.0/r;
    rpm2             = r2*r2;   /* r4 */
    rp               = rpm2*r2; /* r6 */

    // VG //
    // VG // starting modifying from here, the above variables are left for later
    // VG //
    real LFC_inv[2], LFV_inv[2];

    /* Lambda factor for state A, 1-lambda*/
    LFC_inv[0] = LFC[1]; //lambda_coul;
    LFV_inv[0] = LFV[1]; //lambda_vdw;

    /* Lambda factor for state B, lambda*/
    LFC_inv[1] = LFC[0]; //1.0 - lambda_coul;
    LFV_inv[1] = LFV[0]; //1.0 - lambda_vdw;

    real r0Q, r0LJ, b_q, a_q, a_lj, b_lj;
    real rinvQ, rinv2Q, rinv3Q;
    real rinv14LJ, rinv13LJ, rinv12LJ;
    real rinv8LJ, rinv7LJ, rinv6LJ;
    real twentysix_seventh = 26.0/7.0;
    real one_sixth = 1.0/6.0;
    real charge1[NSTATES], charge2[NSTATES];
    real state_ratioC[NSTATES], state_ratioV[NSTATES], state_sign[NSTATES];
    real c12_scaled, c6_scaled;

    state_ratioC[STATE_A] = LFC[STATE_A]/LFC[STATE_B];
    state_ratioC[STATE_B] = LFC[STATE_B]/LFC[STATE_A];
    state_ratioV[STATE_A] = LFV[STATE_A]/LFV[STATE_B];
    state_ratioV[STATE_B] = LFV[STATE_B]/LFV[STATE_A];
    state_sign[STATE_A] = -1.0;
    state_sign[STATE_B] = 1.0;

    r0Q=0.0; b_q=0.0; a_q=0.0; a_lj=0.0; b_lj=0.0;
    rinvQ=0.0; rinv2Q=0.0; rinv3Q=0.0;
    rinv14LJ=0.0; rinv13LJ=0.0; rinv12LJ=0.0;
    rinv8LJ=0.0; rinv7LJ=0.0; rinv6LJ=0.0;

    // for now the parameters are hard-coded
    alpha_vdw = 0.85; //0.85
    real alpha_Q = 0.3; //0.3
    real sigma_Q = 1.0;

    // old sc parameters
    alpha_coul_eff = 0.0;
    alpha_vdw_eff = 0.0;

    // VG //
    // VG //
    // VG //


    /* Loop over A(i=0) and B(i=1) states */
    for (i = 0; i < 2; i++)
    {
        fscal_elec[i] = 0;
        fscal_vdw[i]  = 0;
        velec[i]      = 0;
        vvdw[i]       = 0;

        c6_scaled = c6[i]/6.0;
        c12_scaled = c12[i]/12.0;

        if((c6[i] > 0) && (c12[i] > 0))
        {
                r0LJ = alpha_vdw*pow(twentysix_seventh*0.5*c12[i]/c6[i]*LFV_inv[i],one_sixth);

                if(r0LJ>0)
                {
                	rinv8LJ = 1.0/r0LJ;
                    rinv8LJ = rinv8LJ*rinv8LJ;
                    rinv8LJ = rinv8LJ*rinv8LJ;
                    rinv8LJ = rinv8LJ*rinv8LJ;
                    rinv7LJ = rinv8LJ*r0LJ;
                    rinv6LJ = rinv7LJ*r0LJ;
                    rinv14LJ = rinv7LJ*rinv7LJ;
                    rinv13LJ = rinv7LJ*rinv6LJ;
                    rinv12LJ = rinv6LJ*rinv6LJ;
                }
        }

        if( qq[i]!=0 )
        {
                r0Q = alpha_Q*pow(LFC_inv[i],one_sixth)*(1+sigma_Q*fabs(charge2[i]*charge1[i]));

            if(r0Q>0)
            {
                rinvQ = 1.0/r0Q;
                rinv2Q = rinvQ*rinvQ;
                rinv3Q = rinv2Q*rinvQ;
            }
        }


        //Coulomb
        if( qq[i] != 0 )
        {
                if( r<r0Q ) //soft-core
                {
                        b_q = -2.0*qq[i]*rinv3Q;
                        a_q = 3.0*qq[i]*rinv2Q;
                        fscal_elec[i] = b_q + a_q*rinv;//multiplication by r in the very end
                        velec[i] = -0.5*b_q*r*r - a_q*r + 3.0*qq[i]*rinvQ;
                        dvdl_coul += state_sign[i] * ( 0.5*state_ratioC[i]*(qq[i]*rinv3Q*r*r - 2.0*qq[i]*rinv2Q*r + qq[i]*rinvQ) );
                }
                else //hard-core
                {
                    /* Coulomb */
                    rpinv            = 1.0/(alpha_coul_eff*lfac_coul[i]*sigma_pow[i]+rp);
                    r_coul           = pow(rpinv, -1.0/sc_r_power);

                    /* Electrostatics table lookup data */
                    rtab             = r_coul*tabscale;
                    ntab             = rtab;
                    eps              = rtab-ntab;
                    eps2             = eps*eps;
                    ntab             = 12*ntab;
                    /* Electrostatics */
                    Y                = vftab[ntab];
                    F                = vftab[ntab+1];
                    Geps             = eps*vftab[ntab+2];
                    Heps2            = eps2*vftab[ntab+3];
                    Fp               = F+Geps+Heps2;
                    VV               = Y+eps*Fp;
                    FF               = Fp+Geps+2.0*Heps2;
                    velec[i]         = qq[i]*VV;
                    fscal_elec[i]    = -qq[i]*FF*tabscale*rinv;//*r_coul*rpinv*tabscale;
                }

        } // end Coulomb

        // vdw
        if(c12[i]!=0)
        {
                if( (r<r0LJ) ) //soft-core
                {
                        //stateA
                        b_lj = -156.0*c12_scaled*rinv14LJ + 42.0*c6_scaled*rinv8LJ;
                        a_lj = 168.0*c12_scaled*rinv13LJ - 48.0*c6_scaled*rinv7LJ;
                        fscal_vdw[i] = b_lj + a_lj*rinv;//multiplication by r in the very end
                        vvdw[i] = -0.5*b_lj*r*r - a_lj*r + 91.0*c12_scaled*rinv12LJ - 28.0*c6_scaled*rinv6LJ;
                        dvdl_vdw += state_sign[i] * ( 28.0*state_ratioV[i]*((6.5*c12_scaled*rinv14LJ-c6_scaled*rinv8LJ)*r*r-
                                        (13.0*c12_scaled*rinv13LJ-2.0*c6_scaled*rinv7LJ)*r+6.5*c12_scaled*rinv12LJ-c6_scaled*rinv6LJ) );
                 }
                 else //hard-core
                 {
                     //rpinv            = 1.0/(alpha_vdw_eff*lfac_vdw[i]*sigma_pow[i]+rp);
                     r_vdw            = r; //pow(rpinv, -1.0/sc_r_power);
                     /* Vdw table lookup data */
                     rtab             = r_vdw*tabscale;
                     ntab             = rtab;
                     eps              = rtab-ntab;
                     eps2             = eps*eps;
                     ntab             = 12*ntab;
                     /* Dispersion */
                     Y                = vftab[ntab+4];
                     F                = vftab[ntab+5];
                     Geps             = eps*vftab[ntab+6];
                     Heps2            = eps2*vftab[ntab+7];
                     Fp               = F+Geps+Heps2;
                     VV               = Y+eps*Fp;
                     FF               = Fp+Geps+2.0*Heps2;
                     vvdw[i]          = c6[i]*VV;
                     fscal_vdw[i]     = -c6[i]*FF;

                     /* Repulsion */
                     Y                = vftab[ntab+8];
                     F                = vftab[ntab+9];
                     Geps             = eps*vftab[ntab+10];
                     Heps2            = eps2*vftab[ntab+11];
                     Fp               = F+Geps+Heps2;
                     VV               = Y+eps*Fp;
                     FF               = Fp+Geps+2.0*Heps2;
                     vvdw[i]         += c12[i]*VV;
                     fscal_vdw[i]    -= c12[i]*FF;
                     fscal_vdw[i]    *= rinv;//r_vdw*rpinv*tabscale;
                 }
        } // end vdw

    } // end loop over states


    /* Now we have velec[i], vvdw[i], and fscal[i] for both states */
    /* Assemble A and B states */
    velecsum  = 0;
    vvdwsum   = 0;
    dvdl_coul = 0;
    dvdl_vdw  = 0;
    fscal     = 0;
    for (i = 0; i < 2; i++)
    {
        velecsum      += LFC[i]*velec[i];
        vvdwsum       += LFV[i]*vvdw[i];

        fscal		  += LFC[i]*fscal_elec[i];
        fscal		  += LFC[i]*fscal_vdw[i];

		dvdl_coul	  += state_sign[i]*velec[i];
		dvdl_vdw	  += state_sign[i]*vvdw[i];
    }

    dvdl[efptCOUL]     += dvdl_coul;
    dvdl[efptVDW]      += dvdl_vdw;

    *velectot           = velecsum;
    *vvdwtot            = vvdwsum;

    return fscal;
}
