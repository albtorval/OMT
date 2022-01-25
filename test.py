#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 01:38:09 2021

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

###############
### Testing ###
###############

######################### MODELS LOOPS ########################################
## Function parameters: 
##   + criteria (list): how many criterias are going to be studied
##   + criterion (int): lambda criteria used for computations
##      - 1: median criterion
##      - 2: k-centrum criterion
##      - 3: k-mean trimmed criterion
##   + server (int): which server is going to be used for computation. 
##       - 1: MIGUEL/SEMINARIO 
##       - 2: CARLOS/TAMIR
##       - 3: IMUS/WINDOWS
##   + list_N (list): node sizes N to use
##   + list_p (list): of server sizes p to use
##   + density (int): density of the graphs to use
##   + ctype (int): cost type used for computations
##      - 1: integer costs between [1,100]
##      - 2: integer costs between [1,10000]
##   + ninstances (int): if instances have to be build, number of instances to build
##   + tlimit (int): limit of time for model solving instances
##   + heuristics (list): list of the heuristic functions for computing initial solution
##   + init_sol_pos (list): list of integers with len number of instances indicating which init_sol for each instance
##   + fix_cov_bool (list): list indicating for each model whether to use covering+preprocessing or not
##   + models (list): list of model functions for computing final results

## Workflow for testing:
##  - Generate the instances using instances_generator_txt(.) and save them in txt
##    If instances are already generated, comment the line
##  - Retrieve previously generated instances using instances_recuperator_txt(.)
##  - Compute initial solution using heuristics using lanzadera_initsol_txt(.)
##    and save them in a txt file.
##    If initial solutions are already calculated, comment the line
##  - Retrieve previously generated instances using instances_recuperator_txt(.)
##  - Compute solutions for each model/instance pair indicating the initial solution 
##    you wish to use (init_sol_pos) using lanzadera_txt(.)
##  - Once solutions are computed, they will be saved in a txt
##  - Do the bounds actualization in the instances.tx using bounds_actualization(.)
##  - Finally, print results using print_results(.), which saves them in a final txt file

###### LOOP FUNCTION ########
criterion     = 1
criteria      = [1,2,3]
server        = 1
list_N        = [10]
density       = 1
ctype         = 2 #cost type
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
        instances_generator_txt(criteria,N,p,density,ctype,ninstances)
        instancias = instances_recuperator_txt(criterion,N,p,density,ctype)
        lanzadera_initsol_txt(criterion,N,p,density,ctype,lamb,heuristics,instancias)
        init_sol   = initsol_recuperator_txt(criterion,N,p,density,ctype)
        lanzadera_fix_covering_txt(criterion,N,p,density,ctype,lamb,instancias)
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
    
## Observations:
## - We are giving as initial solution the computed by the PMEDT+OM heuristic 
##   because even if we change the lambda vector, PMEDT + OM would give a solution,
##   OM+MST, although might always find optimal solution for any lambda, can return
##   can return no feasible solution
        
########################## INDIVIDUAL TESTS ###################################

## General parameters
#N       = 10
#p       = 5
#density = 1
#lamb    = [1]*N
#tL      = 60  
#
## Benders warm parameters
#Bw_MP_time  = 0
#Bw_MP_gap   = 0
#Bw_time     = 0
#Bw_gap      = 0
#
## Instance
#instancias = random_instances_generator(Nnodes = N, Ndensity = density, Ninstances = 1)
#instancia  = instancias[0]
##instancia  = {
## (1, 2): 2,
## (1, 3): 1,
## (1, 4): 3,
## (2, 3): 4,
## (2, 4): 5,
## (3, 4): 2, 
## (1, 1): 0,
## (2, 2): 0,
## (3, 3): 0,
## (4, 4): 0,
## (2, 1): 2,
## (3, 1): 1,
## (4, 1): 3,
## (3, 2): 4,
## (4, 2): 5,
## (4, 3): 2
##}

#t01 = OMT_heuristic_OMMST(N,p,lamb,instancia,timeLimit=tL)
#t02 = OMT_heuristic_PMEDTOM(N,p,lamb,instancia,timeLimit=tL)
#t03 = OMT_relaxed(N,p,lamb,instancia,timeLimit=tL)
#t04 = OMT_relaxed_covering(N,p,lamb,instancia,timeLimit=tL)
#t05 = OMT_mtz(N,p,lamb,instancia,timeLimit=tL)
#t06 = OMT_flow_1(N,p,lamb,instancia,timeLimit=tL)
#t07 = OMT_flow_2(N,p,lamb,instancia,timeLimit=tL)
#t08 = OMT_subelim_1(N,p,lamb,instancia,timeLimit=tL)
#t09 = OMT_subelim_2(N,p,lamb,instancia,timeLimit=tL)
#t10 = OMT_mtz_old(N,p,lamb,instancia,timeLimit=tL)
#t11 = OMT_flow_1_old(N,p,lamb,instancia,timeLimit=tL)
#t12 = OMT_flow_2_old(N,p,lamb,instancia,timeLimit=tL)
#t13 = OMT_subelim_1_old(N,p,lamb,instancia,timeLimit=tL)
#t14 = OMT_mtz_covering(N,p,lamb,instancia,timeLimit=tL)
#t15 = OMT_flow_1_covering(N,p,lamb,instancia,timeLimit=tL)
#t16 = OMT_flow_2_covering(N,p,lamb,instancia,timeLimit=tL)
#t17 = OMT_mtz_covering_old(N,p,lamb,instancia,timeLimit=tL)
#t18 = OMT_flow_1_covering_old(N,p,lamb,instancia,timeLimit=tL)
#t19 = OMT_flow_2_covering_old(N,p,lamb,instancia,timeLimit=tL)
#t20 = OMT_Benders_classic(N,p,lamb,instancia,timeLimit=tL)
#t21 = OMT_Benders_classic_covering(N,p,lamb,instancia,timeLimit=tL)
#t22 = OMT_Benders_classic_covering_warm(N,p,lamb,instancia,Bw_MP_time,Bw_MP_gap,Bw_time,Bw_gap,timeLimit=tL)
#t23 = OMT_Benders_modern(N,p,lamb,instancia,timeLimit=tL)
#t24 = OMT_Benders_modern_covering(N,p,lamb,instancia,timeLimit=tL)
#t25 = OMT_Benders_modern_covering_warm(N,p,lamb,instancia,Bw_MP_time,Bw_MP_gap,Bw_time,Bw_gap,timeLimit=tL)

#print("\n OM+MST Heuristic")
#print(t01)
##print_solution_graph(N,t01,instancia,print_edge_labels = True)
#print("\n PMED+OM Heuristic")
#print(t02)
##print_solution_graph(N,t02,instancia,print_edge_labels = True)
#print("\n Linear relaxation")
#print(t03)
#print("\n Linear relaxation (covering)")
#print(t04)
#print("\n OMT MTZ")
#print(t05)
#print_solution_graph(N,t05,instancia)
#print("\n OMT FLOW 1")
#print(t06)
#print_solution_graph(N,t06,instancia)
#print("\n OMT FLOW 2")
#print(t07)
#print_solution_graph(N,t07,instancia)
#print("\n OMT SUBELIM 1")
#print(t08)
#print_solution_graph(N,t08,instancia)
#print("\n OMT SUBELIM 2")
#print(t09)
##print_solution_graph(N,t09,instancia)
#print("\n OMT MTZ (old version)")
#print(t10)
##print_solution_graph(N,t10,instancia)
#print("\n OMT FLOW 1 (old version)")
#print(t11)
##print_solution_graph(N,t11,instancia)
#print("\n OMT FLOW 2 (old version)")
#print(t12)
##print_solution_graph(N,t12,instancia)
#print("\n OMT SUBELIM 1 (old version)")
#print(t13)
##print_solution_graph(N,t13,instancia)
#print("\n OMT MTZ COVERING")
#print(t14)
##print_solution_graph(N,t14,instancia)
#print("\n OMT FLOW 1 COVERING")
#print(t15)
##print_solution_graph(N,t15,instancia)
#print("\n OMT FLOW 2 COVERING")
#print(t16)
##print_solution_graph(N,t16,instancia)
#print("\n OMT MTZ COVERING (old version)")
#print(t17)
##print_solution_graph(N,t17,instancia)
#print("\n OMT FLOW 1 COVERING (old version)")
#print(t18)
##print_solution_graph(N,t18,instancia)
#print("\n OMT FLOW 2 COVERING (old version)")
#print(t19)
##print_solution_graph(N,t19,instancia)
#print("\n OMT BENDERS CLASSIC")
#print(t20)
##print_solution_graph(N,t20,instancia)
#print("\n OMT BENDERS CLASSIC COVERING")
#print(t21)
##print_solution_graph(N,t21,instancia)
#print("\n OMT BENDERS CLASSIC COVERING WARMED")
#print(t22)
##print_solution_graph(N,t22,instancia)
#print("\n OMT BENDERS MODERN")
#print(t23)
##print_solution_graph(N,t23,instancia)
#print("\n OMT BENDERS MODERN COVERING")
#print(t24)
##print_solution_graph(N,t24,instancia)
#print("\n OMT BENDERS MODERN COVERING WARMED")
#print(t25)
##print_solution_graph(N,t25,instancia)


## Observations:
## - If the graph is not complete, or does not have a density of edges considered,
## OMMST can be infeasible
## - If lambda! = [1] * N then the PMEDTOM does not return the optimal solution
## (for this, the part of PMEDT would have to be changed to that of the problem associated with the lambda)