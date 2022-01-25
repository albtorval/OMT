#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:26:33 2021

@author: albtorval
"""
import gurobipy as gp
from gurobipy import GRB

import random
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from instancias import *
from results import *
from omt_relax import *
from omt_heuristics import *
from omt_exact import *

## Modern Benders algorithm with a warm-up phase

def OMT_MST(N,p,lamb,instan,
            l_bool = True,  
            xlist  = [],
            slist  = []):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    ## Retrieving the list of servers
    if l_bool:
        servers = [i for (i,j) in xlist if i==j]
    else: 
        servers = slist
    ## Specification of the cost matrix between previous servers
    C = np.zeros((N,N))
    for i in nodes: 
        for j in nodes:
            if i in servers:
                if j in servers:
                    if (i,j) in arcs:
                        C[i-1][j-1] = costs[i,j]
            else: 
                C[i-1][j-1] = 0
    C   = csr_matrix(C)
    ## MST computing
    MST = minimum_spanning_tree(C)
    MST = MST.todok()
    ## Returning the edges between servers (z vars)
    z = dict(MST.items()) #dic of edges and costs associated
    zlist = [tuple((i+1,j+1)) for (i,j) in z.keys()] #list of edges
    return zlist

def OMT_Benders_modern_covering_warm(N,p,lamb,instan,
                                     max_MP_time,
                                     max_MP_gap,
                                     max_time,
                                     max_gap,
                bool_init_solution=False,
                init_solution=[],
                timeLimit=5000):
    costs   = instan
    nodes   = range(1,N+1)
    arcs    = [*costs]
    sucosts = sorted(np.unique(list(costs.values())))
    H       = range(1,len(sucosts))
    ## Initial bounds:    
    initial_sol = OMT_heuristic_PMEDTOM(N, p, lamb, instan)
    relaxed_sol = OMT_relaxed_covering(N, p, lamb, instan)
    UB = initial_sol[1]
    LB = relaxed_sol
    t_init = time.time()
    # Set the callback to use in the warm-start phase:
    def terminate_sub(model, where):
        if where == GRB.Callback.MIPSOL:
            objbst  = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            objbnd  = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            if abs(objbst - objbnd) < max_MP_gap * (1.0 + abs(objbst)):
                model.terminate()
    # Set lazy-constraints to add the optimality cuts:
    def add_benders_opt_cut(model, where):
#        # Fractonary cuts
#        if where == GRB.Callback.MIPNODE:
#            x_vals = model.cbGetNodeRel(model._x)
#            servers_vals   = {i:x_vals[i,j] for (i,j) in x_vals if i==j}
#            servers_vals_s = dict(sorted(servers_vals.items(), key=lambda item: item[1]))
#            frac_cut_tresh = sum(list(servers_vals_s.values())[-p:])
#            if frac_cut_tresh > p-1:
#                servers_list = list(servers_vals_s.keys())[-p:]
#                z_list = OMT_MST(N,p,lamb,instan,
#                                 l_bool = False,
#                                 slist  = servers_list)
#                obj_SP = sum([costs[i,j] for (i,j) in z_list])
#                model.cbLazy((obj_SP/(p-1))*(sum([model._x[i,i] for i in servers_list])-(p-1)) <= model._mu)
#                model._n_b_cuts = model._n_b_cuts + 1
        ## Integer cuts    
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(model._x)        
            x_list = [(i,j) for i, j in model._x.keys() if x_vals[i, j] > 0.5]
            z_list = OMT_MST(N,p,lamb,instan,
                             l_bool = True,
                             xlist  = x_list)
            obj_SP = sum([costs[i,j] for (i,j) in z_list])
            model.cbLazy((obj_SP/(p-1))*(sum([model._x[i,j] for (i,j) in x_list if i==j])-(p-1)) <= model._mu)
            model._n_b_cuts = model._n_b_cuts + 1 
    # Model:
    m = gp.Model("OMT_Benders_modern_warm")
    m.Params.LogToConsole = 0
    # Variable declaration:
    x  = {} 
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    us = {}
    for k in nodes:
        for h in H:
            us[k,h] = m.addVar(vtype=GRB.BINARY, name="u_"+str(k)+"_"+str(h))
    mu = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")    
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) + mu,
                   GRB.MINIMIZE)
    # Constraints:
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)    
    for (i,j) in arcs:
        if i!=j:
            m.addConstr(x[i,j] <= x[j,j])
    for h in H:        
        larcscosth = [(i,j) for (i,j) in arcs if costs[i,j] >= sucosts[h]]
        m.addConstr(sum([us[k,h] for k in nodes]) == sum([x[i,j] for (i,j) in larcscosth]))
#    for h in H:
        for k in nodes:
            if k < len(nodes):
                m.addConstr(us[k,h] <= us[k+1,h])
    # Initial solution:
    if bool_init_solution:
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
    # Lazy contraints:
    m._x  = x
    m._mu = mu
    m._n_b_cuts = 0
    m.Params.lazyConstraints = 1
    ## Warm-start phase:
    n_b_cuts_pre = 0
    t_warm = time.time() - t_init
    while t_warm < max_time:
        if (abs(UB - LB) < max_gap * (1.0 + abs(UB))): break
        m.setParam("TimeLimit", max_MP_time)
        m.optimize(terminate_sub)
        xlist   = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
        zlist   = OMT_MST(N,p,lamb,instan,
                          l_bool = True,
                          xlist  = xlist)
        obj_OM  = (1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h].x for h in H if us[k,h].x>0.5]) for k in nodes])
        obj_SP  = sum([costs[i,j] for (i,j) in zlist])
        obj_FIN = obj_OM + (1/(p-1))*obj_SP
        if obj_FIN < UB: UB = obj_FIN
        m.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)
        n_b_cuts_pre = n_b_cuts_pre + 1
        t_warm = time.time() - t_init
    ## Modern algorithm phase:
    m.setParam("TimeLimit", timeLimit-t_warm)
    # Optimizer:
    m.optimize(add_benders_opt_cut)
    t_fin = time.time()
    # Solution selection:
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = OMT_MST(N,p,lamb,instan,
                     l_bool = True,
                     xlist  = xlist)
    ulist  = [(k,h) for k in nodes for h in H if us[k,h].x>0.5]
    # Solution featuring:    
    runtime   = t_fin - t_init
    objVal    = m.objVal
    objBound  = m.objBound
    relaxVal  = relaxed_sol
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr  = m.getAttr('NodeCount')
    selection = [xlist, zlist, ulist]
    n_b_cuts_pos = m._n_b_cuts
    n_b_cuts     = n_b_cuts_pos + n_b_cuts_pre    
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts, n_b_cuts_pre, n_b_cuts_pos

#N       = 10
#p       = 4
#density = 1
#lamb    = [1]*N
#MP_tim  = 1
#MP_gap  = 0.3
#tim     = 10
#gap     = 0.3
#tL      = 60
#instancias = random_instances_generator(Nnodes = N, Ndensity = density, Ninstances = 1)
#instancia  = instancias[0]
#t = OMT_Benders_modern_covering_warm(N, p, lamb, instancia, 
#                                     MP_tim, 
#                                     MP_gap, 
#                                     tim, 
#                                     gap, 
#                                     timeLimit = tL)
#print(t)
#t1  = OMT_flow_1(N,p,lamb,instancia,timeLimit=tL) # OPT
#print(t1)

# Saving tests:
def lanzadera_txt_0_Benders_2(N, p, lamb, density, tlimit, modelos, instancias, t1, g1, t2, g2,
                  heuristicos=[], init_sols=[], init_sols_pos=[], initial_bool = False):
    with open("data/OMT_Benders/tune_modern/OMT_%s_%s_%s_%s_%s_%s_%s_results.txt" % (N, p, density, t1, g1, t2, g2), 'w') as f:
        for i in range(len(instancias)):
            for j in range(len(modelos)):
                print('Solving instance ' + str(i+1) + ' using model '+ str(j+1))
                if initial_bool:
                    init_solut = init_sols[(init_sols_pos[i]+i*len(heuristicos))-1][2][6]
                else:
                    init_solut = []
                resultado = modelos[j](N,p,lamb,instancias[i],t1,g1,t2,g2,
                                       bool_init_solution = initial_bool,
                                       init_solution      = init_solut,
                                       timeLimit          = tlimit)
                f.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;" % (i+1,j+1,
                                                     resultado[0],
                                                     resultado[1],
                                                     resultado[2],
                                                     resultado[3],
                                                     resultado[4],
                                                     resultado[5],
                                                     resultado[7],
                                                     resultado[8],
                                                     resultado[9],
                                                     t1,g1,t2,g2)+"\n")
                print('-Solved instance ' + str(i+1) + ' using model '+ str(j+1))
        print('End')
# Returns: 0 instance, 
#          1 model, 
#          2 time, 
#          3 U, 
#          4 L, 
#          5 R, 
#          6 optimality, 
#          7 nodes explored, 
#          8 cuts introduced, 
#          9 cuts pre, 
#          10 cuts post
#           + 4 parametros

def print_results_0_Benders_2(N, p, density, t1, g1, t2, g2, tlimit, modelos):
    def gap_1(x,y): return 100*abs(x-y)/y
    def gap_2(x,y): return 100*abs(x-y)/x
    df_instances = pd.read_csv("data instance/instances/OMT_%s_%s_%s.txt" % (N,p,density), 
                               sep=';', header = None, na_values=["-"])
    df_instances = np.array(df_instances)
    df_results = pd.read_csv("data/OMT_Benders/tune_modern/OMT_%s_%s_%s_%s_%s_%s_%s_results.txt" % (N, p, density, t1, g1, t2, g2), 
                             sep=';', header = None, na_values=["-"])
    df_results = np.array(df_results)
    data = []
    for l in range(df_results.shape[0]):
        l_N           = N
        l_p           = p
        l_density     = density
        l_numinstance = int(df_results[l][0])
        l_model       = int(df_results[l][1])
        l_time        = df_results[l][2]
        l_objU        = round(df_results[l][3],2)
        l_objL        = round(df_results[l][4],2)
        l_objR        = round(df_results[l][5],2)
        l_BobjU       = round(df_instances[math.floor(l/len(modelos))][0],2)
        l_BobjL       = round(df_instances[math.floor(l/len(modelos))][1],2)
        l_gapUL       = round(gap_1(l_objU, l_objL),2)
        l_gapBUL      = round(gap_2(l_BobjU,l_objL),2)
        l_gapUBL      = round(gap_2(l_objU, l_BobjL),2)
        l_gapBUR      = round(gap_2(l_BobjU,l_objR),2)
        l_optimal     = int(df_results[l][6])
        l_nodesexplr  = df_results[l][7]
        l_cuts_total  = df_results[l][8]
        l_cuts_pre    = df_results[l][9]
        l_cuts_pos    = df_results[l][10]
        l_time_MP     = df_results[l][11]
        l_gap_MP      = df_results[l][12]
        l_time_GE     = df_results[l][13]
        l_gap_GE      = df_results[l][14]
        
        data.append([
                l_model, l_N, l_density, l_p, l_numinstance, l_gapUL, l_gapBUL, l_gapUBL, l_gapBUR,
                l_time, l_objU, l_objL, l_objR, l_BobjU, l_BobjL, l_optimal, l_nodesexplr,
                l_cuts_total, l_cuts_pre, l_cuts_pos, l_time_MP, l_gap_MP, l_time_GE, l_gap_GE
                ])
    header = ["Model", "N", "density", "p", "instance", "gapUL", "gapBUL", "gapUBL", "gapBUR", "CPU",
              "objU", "objL", "objR", "Best objU", "Best objL", "optimality", "Nodes explored", 
              "Cuts total", "Cuts pre", "Cuts pos", "Time subpre", "Gap subpre", "Time pre", "Gap pre"] 
    table  = pd.DataFrame(data, columns = header)
    table.to_csv("data/OMT_Benders/tune_modern/OMT_%s_%s_%s_%s_%s_%s_%s_final.txt" % (N, p, density, t1, g1, t2, g2),
                 sep=';')
    print(table)
    print("Tiempo límite de cálculo: "+str(tlimit))
    print("Número de nodos de las instancias: " + str(N))
    print("Número de servidores en las instancias: " + str(p))
    print("Densidad de aristas: " + str(density))
    for l in range(1,len(modelos)+1):
        tmodelo = [data[row][9] for row in range(len(data)) if data[row][0] == l]
        print("El tiempo medio modelo "+ str(modelos[l-1].__name__) + " es: " + str(statistics.mean(tmodelo)))
    optimalies = 0
    for i in range(df_results.shape[0]):
        optimalies = optimalies + data[i][15]
    print("El número de soluciones óptimas encontradas es: "+str(optimalies))

list_N     = [20,30,40,50]
density    = 1
ninstances = 5     #careful, should match instances len already spec
tlimit     = 3600  #only for model computations
heuristics    = [OMT_heuristic_PMEDTOM]
init_sols_pos = [1,1,1,1,1]              #[1]*ninstances
models        = [OMT_Benders_modern_covering_warm]

list_gap_MP  = [0.1, 0.3, 0.5]
list_gap_GE  = [0.3, 0.4, 0.5]
list_time_MP = [math.floor(tlimit/100), math.floor(tlimit/50), math.floor(tlimit/10)]
list_time_GE = [math.floor(tlimit/10),  math.floor(tlimit/5),  math.floor(tlimit/3)]

for N in list_N:
    lamb      = [1]*N  #[0]*(N-1) + [1]
    list_p    = [math.floor(N/4), math.floor(N/3), math.floor(N/2)] 
    for p in list_p:
        for t1 in list_time_MP:
            for t2 in list_time_GE:
                for g1 in list_gap_MP:
                    for g2 in list_gap_GE:
                        instancias = instances_recuperator_txt(N,p,density)
                        init_sols  = initsol_recuperator_txt(N,p,density)
                        lanzadera_txt_0_Benders_2(N,p,lamb,density,tlimit,
                                                  models,instancias,
                                                  t1,g1,t2,g2,
                                                  heuristicos   = heuristics, 
                                                  init_sols     = init_sols, 
                                                  init_sols_pos = init_sols_pos, 
                                                  initial_bool  = True)
                        print_results_0_Benders_2(N,p,density,t1,g1,t2,g2,tlimit,models)
                        
                        
                        
                        