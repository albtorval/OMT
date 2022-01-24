# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:17:56 2021

@author: atorrejon
"""
import gurobipy as gp
from gurobipy import GRB

import random
import numpy as np
import pandas as pd
import statistics
import math
import ast
import time

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from omt_heuristics import *
from omt_relax import *
from omt_exact import *
from instancias import *
from results import *

## Algoritmos
def OMT_MST(N,p,lamb,instan,xlist):
    costs = instan
    nodes = range(1,N+1)
    arcs = [*costs]
    ## Retrieving the list of servers
    servers = [i for (i,j) in xlist if i==j]
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


def OMT_Benders_classic_covering(N,p,lamb,instan,
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
    relaxed_sol = OMT_relaxed(N,p,lamb,instan)
    UB = initial_sol[1]
    LB = relaxed_sol
    ## Set the time
    t_init = time.time()
    ## Master Problem (OM + mu)
    MP = gp.Model("OMT_MP")
    MP.Params.LogToConsole = 0
    x = {}
    for (i,j) in arcs:
        x[i,j] = MP.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="x_"+str(i)+"_"+str(j))
    us = {}
    for k in nodes:
        for h in H: 
                us[k,h] =  MP.addVar(vtype=GRB.BINARY, name="u_"+str(k)+"_"+str(h))
    mu =  MP.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")    
    MP.setObjective( (1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) + mu,
                   GRB.MINIMIZE)        
    MP.addConstr(sum([x[k,k] for k in nodes]) == p)
    for i in nodes: 
        MP.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    for (i,j) in arcs:
        MP.addConstr(x[i,j] <= x[j,j])
    for h in H:        
        larcscosth = [(i,j) for (i,j) in arcs if costs[i,j] >= sucosts[h]]
        MP.addConstr(sum([us[k,h] for k in nodes]) == sum([x[i,j] for (i,j) in larcscosth]))
    for h in H:
        for k in nodes:
            if k < len(nodes):
                MP.addConstr(us[k,h] <= us[k+1,h])
    # Initial solution:
    if bool_init_solution:
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0                
    n_b_cuts = 0
    ## Classic Benders algorithm:
    while UB > LB:
        ## Check time limit and reset time limit for computation
        t_loop = time.time() - t_init
        if t_loop > timeLimit: break
        MP.setParam("TimeLimit", timeLimit-t_loop)
        ## MP solution (OM)
        MP.optimize()
        xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
        ulist  = [(k,h) for k in nodes for h in H if us[k,h].x>0.5]
        obj_OM = (1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h].x for h in H if us[k,h].x>0.5]) for k in nodes])
        obj_MP = obj_OM + mu.x  
        ## LB actualization
        if obj_MP > LB and MP.status == GRB.OPTIMAL: LB = obj_MP
        ## SP solution (MST)
        zlist   = OMT_MST(N,p,lamb,instan,xlist)
        obj_SP  = sum([costs[i,j] for (i,j) in zlist])    
        obj_FIN = obj_OM + (1/(p-1))*obj_SP
        ## UB actualization
        if obj_FIN < UB: UB = obj_FIN
        ## Add optimality cut
        MP.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)
        n_b_cuts = n_b_cuts + 1 
    t_final = time.time()
    # Solution featuring:
    runtime    = t_final-t_init
    objVal     = UB
    objBound   = LB
    relaxVal   = relaxed_sol
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr   = "NO"
    selection  = [xlist,zlist,ulist]
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts    
    
    

#def OMT_Benders_classic(N,p,lamb,instan,
#            bool_init_solution=False,
#            init_solution=[],
#            timeLimit=5000):
#    costs = instan
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    ## Initial bounds:    
#    initial_sol = OMT_heuristic_PMEDTOM(N, p, lamb, instan)
#    relaxed_sol = OMT_relaxed(N,p,lamb,instan)
#    UB = initial_sol[1]
#    LB = relaxed_sol
#    ## Set the time
#    t_init = time.time()
#    ## Master Problem (OM + mu)
#    MP = gp.Model("OMT_MP")
#    MP.Params.LogToConsole = 0
#    x = {}
#    for (i,j) in arcs:
#        x[i,j] = MP.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="x_"+str(i)+"_"+str(j))
#    sx = {}
#    for k in nodes:
#        for (i,j) in arcs: 
#            sx[i,j,k] =  MP.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
#    mu =  MP.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")
#    MP.setObjective( (1/sum(lamb))*sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) + mu, GRB.MINIMIZE)
#    MP.addConstr(sum([x[k,k] for k in nodes]) == p)
#    for i in nodes: 
#        MP.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
#    for (i,j) in arcs:
#        MP.addConstr(x[i,j] <= x[j,j])
#    for k in nodes:
#        MP.addConstr(sum([sx[i,j,k] for (i,j) in arcs]) <= 1)
#    for (i,j) in arcs:
#        MP.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
#    for k in nodes:
#        if k < len(nodes):
#            MP.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in arcs]) <= sum([costs[i,j]*sx[i,j,k+1] for (i,j) in arcs]))
#    # Initial solution:
#    if bool_init_solution:
#        for (i,j) in arcs:
#            if (i,j) in init_solution[0]:
#                x[i,j].start = 1
#            else: 
#                x[i,j].start = 0
#        for l in nodes:
#            for (i,j) in arcs:
#                if (i,j,l) in init_solution[2]:
#                    sx[i,j,l].start = 1
#                else: 
#                    sx[i,j,l].start = 0                   
#    n_b_cuts = 0
#    ## Classic Benders algorithm:
#    while UB > LB:
#        ## Check time limit and reset time limit for computation
#        t_loop = time.time() - t_init
#        if t_loop > timeLimit: break
#        MP.setParam("TimeLimit", timeLimit-t_loop)
#        ## MP solution (OM)
#        MP.optimize()
#        xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
#        sxlist = [(i,j,k) for k in nodes for (i,j) in arcs if sx[i,j,k].x>0.5]    
#        obj_OM = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist])        
#        obj_MP = obj_OM + mu.x  #MP.objVal
#        ## LB actualization
#        if obj_MP > LB and MP.status == GRB.OPTIMAL: LB = obj_MP
#        ## SP solution (MST)
#        zlist   = OMT_MST(N,p,lamb,instan,xlist)
#        obj_SP  = sum([costs[i,j] for (i,j) in zlist])    
#        obj_FIN = obj_OM + (1/(p-1))*obj_SP
#        ## UB actualization
#        if obj_FIN < UB: UB = obj_FIN
#        ## Add optimality cut
#        MP.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)
#        n_b_cuts = n_b_cuts + 1 
#    t_final = time.time()
#    # Solution featuring:
#    runtime   = t_final-t_init
#    objVal    = UB
#    objBound  = LB
#    relaxVal  = relaxed_sol
#    optimality_found = 0
#    if runtime < timeLimit:
#        optimality_found = 1
#    nodExplr   = "NO"
#    selection  = [xlist,zlist,sxlist]
#    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts

def OMT_Benders_modern_covering(N,p,lamb,instan,
                bool_init_solution=False,
                init_solution=[],
                timeLimit=5000):
    costs   = instan
    nodes   = range(1,N+1)
    arcs    = [*costs]
    sucosts = sorted(np.unique(list(costs.values())))
    H       = range(1,len(sucosts))
    # Lazy-constraints to add the optimality cuts
    def add_benders_opt_cut(model, where):
        if where == GRB.Callback.MIPSOL:
            x_vals = model.cbGetSolution(model._x)            
            x_list = [(i,j) for i, j in model._x.keys() if x_vals[i, j] > 0.5]
            z_list = OMT_MST(N,p,lamb,instan,x_list)
            obj_SP = sum([costs[i,j] for (i,j) in z_list])
            model.cbLazy((obj_SP/(p-1))*(sum([model._x[i,j] for (i,j) in x_list if i==j])-(p-1)) <= model._mu)
            model._n_b_cuts = model._n_b_cuts + 1 
    # Model:
    m  = gp.Model("OMT_Benders_modern")
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
    m.setObjective( (1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) + mu,
                   GRB.MINIMIZE)
    # Constraints:    
    for i in nodes: m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)    
    m.addConstr(sum([x[k,k] for k in nodes]) == p)    
    for (i,j) in arcs: m.addConstr(x[i,j] <= x[j,j])
    for h in H:        
        larcscosth = [(i,j) for (i,j) in arcs if costs[i,j] >= sucosts[h]]
        m.addConstr(sum([us[k,h] for k in nodes]) == sum([x[i,j] for (i,j) in larcscosth]))
    for h in H:
        for k in nodes:
            if k < len(nodes):
                m.addConstr(us[k,h] <= us[k+1,h]) 
    # Lazy contraints:
    m._x  = x
    m._mu = mu
    m._n_b_cuts = 0
    m.Params.lazyConstraints = 1
    # Initial solution:
    if bool_init_solution:
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
    m.setParam("TimeLimit", timeLimit)
    # Optimizer:
    m.optimize(add_benders_opt_cut)
    # Solution selection:
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    ulist  = [(k,h) for k in nodes for h in H if us[k,h].x>0.5]
    zlist  = OMT_MST(N,p,lamb,instan,xlist)
    # Solution featuring:    
    runtime   = m.runtime
    objVal    = m.objVal
    objBound  = m.objBound
    relaxVal  = OMT_relaxed_covering(N,p,lamb,instan)
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr  = m.getAttr('NodeCount')
    selection = [xlist, zlist, ulist]
    n_b_cuts  = m._n_b_cuts
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts


#def OMT_Benders_modern(N,p,lamb,instan,
#                bool_init_solution=False,
#                init_solution=[],
#                timeLimit=5000):
#    costs = instan
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    # Lazy-constraints to add the optimality cuts
#    def add_benders_opt_cut(model, where):
#        if where == GRB.Callback.MIPSOL:
#            x_vals = model.cbGetSolution(model._x)            
#            x_list = [(i,j) for i, j in model._x.keys() if x_vals[i, j] > 0.5]
#            z_list = OMT_MST(N,p,lamb,instan,x_list)
#            obj_SP = sum([costs[i,j] for (i,j) in z_list])
#            model.cbLazy((obj_SP/(p-1))*(sum([model._x[i,j] for (i,j) in x_list if i==j])-(p-1)) <= model._mu)
#            model._n_b_cuts = model._n_b_cuts + 1 
#    # Model:
#    m = gp.Model("OMT_Benders_modern")
#    m.Params.LogToConsole = 0
#    # Variable declaration:
#    x = {} 
#    for (i,j) in arcs:
#        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
#    sx = {}
#    for l in nodes:
#        for (i,j) in arcs: 
#            sx[i,j,l] =  m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
#    mu =  m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")
#    # Objective:
#    m.setObjective((1/sum(lamb))*sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) + mu, GRB.MINIMIZE)
#    # Constraints:    
#    for i in nodes: m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)    
#    m.addConstr(sum([x[k,k] for k in nodes]) == p)    
#    for (i,j) in arcs: m.addConstr(x[i,j] <= x[j,j])
#    for l in nodes: m.addConstr(sum([sx[i,j,l] for (i,j) in arcs]) <= 1)
#    for (i,j) in arcs: m.addConstr(sum([sx[i,j,l] for l in nodes]) == x[i,j])
#    for l in nodes:
#        if l < len(nodes):
#            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs]))
#    # Lazy contraints:
#    m._x  = x
#    m._mu = mu
#    m._n_b_cuts = 0
#    m.Params.lazyConstraints = 1
#    # Initial solution:
#    if bool_init_solution:
#        for (i,j) in arcs:
#            if (i,j) in init_solution[0]:
#                x[i,j].start = 1
#            else: 
#                x[i,j].start = 0
#        for l in nodes:
#            for (i,j) in arcs:
#                if (i,j,l) in init_solution[2]:
#                    sx[i,j,l].start = 1
#                else: 
#                    sx[i,j,l].start = 0
#    m.setParam("TimeLimit", timeLimit)
#    # Optimizer:
#    m.optimize(add_benders_opt_cut)
#    # Solution selection:
#    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
#    sxlist = [(i,j,l) for l in nodes for (i,j) in arcs if sx[i,j,l].x>0.5]
#    zlist  = OMT_MST(N,p,lamb,instan,xlist)
#    # Solution featuring:    
#    runtime   = m.runtime
#    objVal    = m.objVal
#    objBound  = m.objBound
#    relaxVal  = OMT_relaxed(N,p,lamb,instan)
#    optimality_found = 0
#    if runtime < timeLimit:
#        optimality_found = 1
#    nodExplr  = m.getAttr('NodeCount')
#    selection = [xlist, zlist, sxlist]
#    n_b_cuts  = m._n_b_cuts
#    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts


## Changing results functions
def lanzadera_txt_0_Benders(N, p, lamb, density, tlimit, modelos, instancias,
                  heuristicos=[], init_sols=[], init_sols_pos=[], initial_bool = False):
    with open("data/OMT_Benders/no_warm_up/OMT_%s_%s_%s_results.txt" % (N,p,density), 'w') as f:
        for i in range(len(instancias)):
            for j in range(len(modelos)):
                print('Solving instance ' + str(i+1) + ' using model '+ str(j+1))
                if initial_bool:
                    init_solut = init_sols[(init_sols_pos[i]+i*len(heuristicos))-1][2][6]
                else:
                    init_solut = []
                resultado = modelos[j](N,p,lamb,instancias[i],
                                   
                                       bool_init_solution = initial_bool,
                                       init_solution      = init_solut,
                                       timeLimit          = tlimit)
                f.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;" % (i+1,j+1,
                                                     resultado[0],
                                                     resultado[1],
                                                     resultado[2],
                                                     resultado[3],
                                                     resultado[4],
                                                     resultado[5],
                                                     resultado[7])+"\n")
                print('-Solved instance ' + str(i+1) + ' using model '+ str(j+1))
        print('End')
# Returns: 0 instance, 1 model, 2 time, 3 U, 4 L, 5 R, 6 optimality, 7 nodes explored, 8 cuts introduced
                
        
def print_results_0_Benders(N, p, density, tlimit, modelos):
    def gap_1(x,y): return 100*abs(x-y)/y
    def gap_2(x,y): return 100*abs(x-y)/x
    df_instances = pd.read_csv("instances/OMT_%s_%s_%s.txt" % (N,p,density), 
                               sep=';', header = None, na_values=["-"])
    df_instances = np.array(df_instances)
    df_results   = pd.read_csv("data/OMT_Benders/no_warm_up/OMT_%s_%s_%s_results.txt" % (N,p,density), 
                             sep=';', header = None, na_values=["-"])
    df_results   = np.array(df_results)
    data = []
    for l in range(df_results.shape[0]):
        l_N = N
        l_p = p
        l_density = density
        l_numinstance = int(df_results[l][0])
        l_model   = int(df_results[l][1])
        l_time    = df_results[l][2]
        l_objU    = round(df_results[l][3],2)
        l_objL    = round(df_results[l][4],2)
        l_objR    = round(df_results[l][5],2)
        l_BobjU   = round(df_instances[math.floor(l/len(modelos))][0],2)
        l_BobjL   = round(df_instances[math.floor(l/len(modelos))][1],2)
        l_gapUL   = round(gap_1(l_objU, l_objL),2)
        l_gapBUL  = round(gap_2(l_BobjU,l_objL),2)
        l_gapUBL  = round(gap_2(l_objU, l_BobjL),2)
        l_gapBUR  = round(gap_2(l_BobjU,l_objR),2)
        l_optimal = int(df_results[l][6])
        l_nodesexplr = df_results[l][7]
        l_cuts_added = df_results[l][8]
        data.append([
                l_model, l_N, l_density, l_p, l_numinstance, l_gapUL, l_gapBUL, l_gapUBL, l_gapBUR,
                l_time, l_objU, l_objL, l_objR, l_BobjU, l_BobjL, l_optimal, l_nodesexplr, l_cuts_added
                ])
    header = ["Model", "N", "density", "p", "instance", "gapUL", "gapBUL", "gapUBL", "gapBUR", "CPU",
              "objU", "objL", "objR", "Best objU", "Best objL", "optimality", "Nodes explored", "Cuts added"]
    table  = pd.DataFrame(data, columns = header)
    table.to_csv("data/OMT_Benders/no_warm_up/OMT_%s_%s_%s_final.txt" % (N,p,density), sep=';')
    print(table)
    print("Tiempo límite de cálculo: "+str(tlimit))
    print("Número de nodos de las instancias: " + str(N))
    print("Número de servidores en las instancias: " + str(p))
    print("Densidad de aristas: " + str(density))
    for l in range(1,len(modelos)+1):
        tmodelo = [data[row][9] for row in range(len(data)) if data[row][0] == l]
        print("El tiempo medio modelo "+ str(modelos[l-1].__name__) +" es: " + str(statistics.mean(tmodelo)))
    optimalies = 0
    for i in range(df_results.shape[0]):
        optimalies = optimalies + data[i][15]
    print("El número de soluciones óptimas encontradas es: "+str(optimalies))
    

##################
## Test workflow
################## 
    
list_N     = [20,30,40,50]
density    = 1
ninstances = 5     #careful, should match instances len already spec
tlimit     = 3600  #only for model computations
heuristics    = [OMT_heuristic_PMEDTOM]
init_sols_pos = [1,1,1,1,1]              #[1]*ninstances
models        = [OMT_Benders_classic_covering,
                 OMT_Benders_modern_covering]

for N in list_N:
    lamb      = [1]*N  #[0]*(N-1) + [1]
    list_p    = [math.floor(N/4), math.floor(N/3), math.floor(N/2)]
    for p in list_p:
        instancias = instances_recuperator_txt(N,p,density)
        init_sols  = initsol_recuperator_txt(N,p,density)
        lanzadera_txt_0_Benders(N,p,lamb,density,tlimit,models,instancias,
                      heuristicos   = heuristics, 
                      init_sols     = init_sols, 
                      init_sols_pos = init_sols_pos, 
                      initial_bool  = True)
        print_results_0_Benders(N,p,density,tlimit,models)
