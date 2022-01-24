#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 18:50:42 2022

@author: albtorval
"""
from instancias import *
from results import *
from omt_heuristics import *
from omt_relax import *
from omt_exact import *
from omt_exact_covering import *
from omt_benders import *

import math

criterion     = 1
criteria      = [1,2,3]
server        = 3
list_N        = [100]
density       = 1
ctype         = 1 #cost type
ninstances    = 5 
tlimit        = 3600
heuristics    = [OMT_heuristic_PMEDTOM]
init_sol_pos  = [1,1,1,1,1] #[1]*ninstances
fix_cov_bool  = [False]*5 + [True]*3
models        = [OMT_mtz,
                 OMT_flow_1,
                 OMT_flow_2,
                 OMT_subelim_1,
                 OMT_subelim_2,
                 OMT_mtz_covering,
                 OMT_flow_1_covering,
                 OMT_flow_2_covering]
# Loop:
for N in list_N:
    if criterion == 1: #MEDIAN CRITERION
        lamb = [1]*N  
    if criterion == 2: #K-CENTRUM CRITERION
        lamb = [0]*math.floor(2/3*N)+[1]*(N-math.floor(2/3*N))
    if criterion == 3: #K-TRIMMED MEAN CRITERION
        lamb = [0]*math.floor(1/3*N)+[1]*(N-math.floor(2/3*N))+[0]*math.floor(1/3*N)
    list_p  = [math.floor(N/4), math.floor(N/3), math.floor(N/2)]
    for p in list_p:
#        instances_generator_txt(criteria,N,p,density,ctype,ninstances)
        instancias = instances_recuperator_txt(criterion,N,p,density,ctype)
#        lanzadera_initsol_txt(criterion,N,p,density,ctype,lamb,heuristics,instancias)
        init_sol   = initsol_recuperator_txt(criterion,N,p,density,ctype)
#        lanzadera_fix_covering_txt(criterion,N,p,density,ctype,lamb,instancias)
        fixing_sol = fix_covering_recuperator_txt(criterion,N,p,density,ctype)
        lanzadera_txt(server, N, p, density, ctype, lamb, tlimit, models, instancias, 
                      heuristicos   = heuristics,
                      init_sols     = init_sol, 
                      init_sols_pos = init_sol_pos, 
                      init_bool     = True,
                      fixing_sols   = fixing_sol, 
                      fixing_bool   = fix_cov_bool)
        bounds_actualization(criterion,server,N,p,density,ctype)
        print("\n")
        print_results(criterion,server,N,p,density,ctype,tlimit,models)
