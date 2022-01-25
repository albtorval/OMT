# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:41:17 2021

@author: atorrejon
"""
import multiprocessing

from instancias import *
from results import *
from omt import *
from omt_heuristics import *

###############
### Testing ###
###############

### Loop for results

## Initial data: 
##   + list_N (list): node sizes N to use
##   + list_p (list): of server sizes p to use
##   + density (int): density of the graphs to use
##   + ninstances (int): if instances have to be build, number of instances to build
##   + tlimit (int): limit of time for model solving instances
##   + heuristics (list): list of the heuristic functions for computing initial solution
##   + init_sol_pos (list): list of integers with len number of instances indicating which init_sol for each instance
##   + models (list): list of model functions for computing final results

list_N     = [10,20] 
# list_p   = [3,5,8]
density    = 1
ninstances = 5        #careful, should match instances len already spec
tlimit     = 800      #only for model computations
heuristics    = [OMT_heuristic_PMEDTOM]  #[OMT_heuristic_OMMST, OMT_heuristic_PMEDTOM]
init_sols_pos = [1,1,1,1,1]           #[1]*ninstances #for each instance 
models        = [OMT_mtz, 
                OMT_flow_1, 
                OMT_flow_2, 
                OMT_subelim, 
                OMT_mtz_covering, 
                OMT_flow_1_covering, 
                OMT_flow_2_covering, 
                OMT_subelim_covering,
                OMT_Benders_classic, 
                OMT_Benders_modern,
                OMT_Benders_mixed]


for N in list_N:
    lamb      = [1]*N  #[0]*(N-1) + [1]
    init_sols = []
    list_p    = [math.floor(N/4), math.floor(N/3), math.floor(N/2)]
    for p in list_p:
        instances_generator_txt(N,p,density,ninstances)
        instancias = instances_recuperator_txt(N,p,density)
        lanzadera_initsol_txt(N,p,lamb,density,heuristics,instancias)
        init_sols  = initsol_recuperator_txt(N,p,density)
        lanzadera_txt(N,p,lamb,density,tlimit,models,instancias,
                      heuristicos=heuristics, 
                      init_sols=init_sols, 
                      init_sols_pos=init_sols_pos, 
                      initial_bool = True)
        bounds_actualization(N,p,density)
        print("\n")
        print_results(N,p,density,tlimit,models)


# def worker(N,p,lamb):
#         instancias = instances_recuperator_txt(N,p,density)
#         lanzadera_initsol_txt(N,p,lamb,density,heuristics,instancias)
#         init_sols  = initsol_recuperator_txt(N,p,density)
#         lanzadera_txt(N,p,lamb,density,tlimit,models,instancias,
#                       heuristicos=heuristics, 
#                       init_sols=init_sols, 
#                       init_sols_pos=init_sols_pos, 
#                       initial_bool = True)
#         bounds_actualization(N,p,density)

# if __name__ == '__main__':
#     jobs = []
#     for N in list_N:
#         lamb = [1]*N  #[0]*(N-1) + [1]
#         init_sols  = []
#         for p in list_p:
#             instances_generator_txt(N,p,density,ninstances)
#             p = multiprocessing.Process(target=worker, args=(N,p,lamb,))
#             jobs.append(p)
#             p.start()

# print("\n")
# print_results(N,p,density,tlimit,models)
