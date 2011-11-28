/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
 *
 *                This source code is part of
 *
 *                 G   R   O   M   A   C   S
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2009, The GROMACS Development Team
 *
 * Gromacs is a library for molecular simulation and trajectory analysis,
 * written by Erik Lindahl, David van der Spoel, Berk Hess, and others - for
 * a full list of developers and information, check out http://www.gromacs.org
 *
 * This program is free software; you can redistribute it and/or modify it under 
 * the terms of the GNU Lesser General Public License as published by the Free 
 * Software Foundation; either version 2 of the License, or (at your option) any 
 * later version.
 * As a special exception, you may use this file as part of a free software
 * library without restriction.  Specifically, if other files instantiate
 * templates or use macros or inline functions from this file, or you compile
 * this file and link it with other files to produce an executable, this
 * file does not by itself cause the resulting executable to be covered by
 * the GNU Lesser General Public License.  
 *
 * In plain-speak: do not worry about classes/macros/templates either - only
 * changes to the library have to be LGPL, not an application linking with it.
 *
 * To help fund GROMACS development, we humbly ask that you cite
 * the papers people have written on it - you can find them on the website!
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>

#include "typedefs.h"
#include "vec.h"
#include "smalloc.h"
#include "force.h"
#include "nbnxn_kernel_sse.h"

/* Analytical reaction-field kernels */
#define CALC_COUL_RF

/* Include the force+energy kernels */
#define CALC_ENERGIES
#define LJ_COMB_GEOM
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_GEOM
#define LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef CALC_ENERGIES

/* Include the force+energygroups kernels */
#define CALC_ENERGIES
#define ENERGY_GROUPS
#define LJ_COMB_GEOM
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_GEOM
#define LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef ENERGY_GROUPS
#undef CALC_ENERGIES

/* Include the force only kernels */
#define LJ_COMB_GEOM
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_GEOM
#define LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"

#undef CALC_COUL_RF


/* Tabulated Coulomb kernels */

/* Include the force+energy kernels */
#define CALC_ENERGIES
#define LJ_COMB_GEOM
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_GEOM
#define LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef CALC_ENERGIES

/* Include the force+energygroups kernels */
#define CALC_ENERGIES
#define ENERGY_GROUPS
#define LJ_COMB_GEOM
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_GEOM
#define LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef ENERGY_GROUPS
#undef CALC_ENERGIES

/* Include the force only kernels */
#define LJ_COMB_GEOM
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_GEOM
#define LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"
#undef LJ_COMB_LB
#include "nbnxn_kernel_sse_single_outer.h"


typedef void (*p_nbk_func_ener)(const nbnxn_pairlist_t     *nbl,
                                const nbnxn_atomdata_t     *nbat,
                                const interaction_const_t  *ic,
                                rvec                       *shift_vec,
                                real                       *f,
                                real                       *fshift,
                                real                       *Vvdw,
                                real                       *Vc);

typedef void (*p_nbk_func_noener)(const nbnxn_pairlist_t     *nbl,
                                  const nbnxn_atomdata_t     *nbat,
                                  const interaction_const_t  *ic,
                                  rvec                       *shift_vec,
                                  real                       *f,
                                  real                       *fshift);

enum { coultRF, coultTAB, coultNR };

p_nbk_func_ener p_nbk_ener[coultNR][ljcrNR] =
{ { nbnxn_kernel_sse_single_rf_comb_geom_ener,
    nbnxn_kernel_sse_single_rf_comb_lb_ener,
    nbnxn_kernel_sse_single_rf_comb_none_ener },
  { nbnxn_kernel_sse_single_tab_comb_geom_ener,
    nbnxn_kernel_sse_single_tab_comb_lb_ener,
    nbnxn_kernel_sse_single_tab_comb_none_ener } };

p_nbk_func_ener p_nbk_energrp[coultNR][ljcrNR] =
{ { nbnxn_kernel_sse_single_rf_comb_geom_energrp,
    nbnxn_kernel_sse_single_rf_comb_lb_energrp,
    nbnxn_kernel_sse_single_rf_comb_none_energrp },
  { nbnxn_kernel_sse_single_tab_comb_geom_energrp,
    nbnxn_kernel_sse_single_tab_comb_lb_energrp,
    nbnxn_kernel_sse_single_tab_comb_none_energrp } };

p_nbk_func_noener p_nbk_noener[coultNR][ljcrNR] =
{ { nbnxn_kernel_sse_single_rf_comb_geom_noener,
    nbnxn_kernel_sse_single_rf_comb_lb_noener,
    nbnxn_kernel_sse_single_rf_comb_none_noener },
  { nbnxn_kernel_sse_single_tab_comb_geom_noener,
    nbnxn_kernel_sse_single_tab_comb_lb_noener,
    nbnxn_kernel_sse_single_tab_comb_none_noener } };

static void clear_f(const nbnxn_atomdata_t *nbat,
                    real *f)
{
    int i;

    for(i=0; i<nbat->natoms*nbat->xstride; i++)
    {
        f[i] = 0;
    }
}

static void clear_fshift(real *fshift)
{
    int i;

    for(i=0; i<SHIFTS*DIM; i++)
    {
        fshift[i] = 0;
    }
}

static void reduce_group_energies(int ng,
                                  const real *VSvdw,const real *VSc,
                                  real *Vvdw,real *Vc)
{
    int c,i,j,j0,j1,j2,j3;

    /* The size of the SSE energy group buffer array is ng^5 */
    c = 0;
    for(i=0; i<ng; i++)
    {
        for(j=0; j<ng; j++)
        {
            Vvdw[i*ng+j] = 0;
            Vc[i*ng+j]   = 0;
        }

        for(j3=0; j3<ng; j3++)
        {
            for(j2=0; j2<ng; j2++)
            {
                for(j1=0; j1<ng; j1++)
                {
                    for(j0=0; j0<ng; j0++)
                    {
                        Vvdw[i*ng+j0] += VSvdw[c+0];
                        Vvdw[i*ng+j1] += VSvdw[c+1];
                        Vvdw[i*ng+j2] += VSvdw[c+2];
                        Vvdw[i*ng+j3] += VSvdw[c+3];
                        Vc  [i*ng+j0] += VSc  [c+0];
                        Vc  [i*ng+j1] += VSc  [c+1];
                        Vc  [i*ng+j2] += VSc  [c+2];
                        Vc  [i*ng+j3] += VSc  [c+3];
                        c += 4;
                    }
                }
            }
        }
    }
}

void
nbnxn_kernel_sse(nbnxn_pairlist_set_t       *nbl_list,
                 const nbnxn_atomdata_t     *nbat,
                 const interaction_const_t  *ic,
                 rvec                       *shift_vec, 
                 int                        force_flags,
                 gmx_bool                   clearF,
                 real                       *fshift,
                 real                       *Vc,
                 real                       *Vvdw)
{
    int              nnbl;
    nbnxn_pairlist_t **nbl;
    int coult;
    int nb;

    nnbl = nbl_list->nnbl;
    nbl  = nbl_list->nbl;

    if (EEL_RF(ic->eeltype) || ic->eeltype == eelCUT)
    {
        coult = coultRF;
    }
    else
    {
        coult = coultTAB;
    }

#pragma omp parallel for schedule(static)
    for(nb=0; nb<nnbl; nb++)
    {
        nbnxn_atomdata_output_t *out;
        real *fshift_p;

        out = &nbat->out[nb];

        if (clearF)
        {
            clear_f(nbat,out->f);
        }

        if ((force_flags & GMX_FORCE_VIRIAL) && nnbl == 1)
        {
            fshift_p = fshift;
        }
        else
        {
            fshift_p = out->fshift;

            if (clearF)
            {
                clear_fshift(fshift_p);
            }
        }

        if (!(force_flags & GMX_FORCE_ENERGY))
        {
            /* Don't calculate energies */
            p_nbk_noener[coult][nbat->comb_rule](nbl[nb],nbat,
                                                 ic,
                                                 shift_vec,
                                                 out->f,
                                                 fshift_p);
        }
        else if (out->nV == 1)
        {
            /* No energy groups */
            out->Vvdw[0] = 0;
            out->Vc[0]   = 0;

            p_nbk_ener[coult][nbat->comb_rule](nbl[nb],nbat,
                                               ic,
                                               shift_vec,
                                               out->f,
                                               fshift_p,
                                               out->Vvdw,
                                               out->Vc);
        }
        else
        {
            /* Calculate energy group contributions */
            int i;

            for(i=0; i<out->nVS; i++)
            {
                out->VSvdw[i] = 0;
            }
            for(i=0; i<out->nVS; i++)
            {
                out->VSc[i] = 0;
            }

            p_nbk_energrp[coult][nbat->comb_rule](nbl[nb],nbat,
                                                  ic,
                                                  shift_vec,
                                                  out->f,
                                                  fshift_p,
                                                  out->VSvdw,
                                                  out->VSc);

            reduce_group_energies(nbat->nenergrp,out->VSvdw,out->VSc,
                                  out->Vvdw,out->Vc);
        }
    }

    if (force_flags & GMX_FORCE_ENERGY)
    {
        /* Reduce the energies */
        for(nb=0; nb<nnbl; nb++)
        {
            int i;

            for(i=0; i<nbat->out[nb].nV; i++)
            {
                Vvdw[i] += nbat->out[nb].Vvdw[i];
                Vc[i]   += nbat->out[nb].Vc[i];
            }
        }
    }
}