# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:11:59 2021

@author: atorrejon
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import random
import time

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from omt_heuristics import *
from omt_relax import *

##########################
### Auxiliar functions ###
##########################
    
## Kruskal MST algorithm:
## This function recieves as inputs N,p,lamb and the instance given to the OMT problem
## To indicate the servers, the function has two possibilities implemented:
## - l_bool==True  -> servers are given as the full list of x vars
## - l_bool==False -> servers are specified in a list of p integers
## The return is the list of edges connecting the tree servers.
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

############################
### Benders decompositions
############################

## Classic Benders decomposition algorithm
## MP: OM + mu
## SP: MST
## While the bounds are different:
## - Compute the objective and xlist of the MP
## - Update LB if neccesary
## - Compute SP for the list of xlist form MP
## - Update UB if neccesary
## - Add the Bender optimality cut
## Observations: 
## - Allow some gap when no solving for optimality
## Classic Benders algorithm
## MP: OM + mu
## SP: MST
## While the bounds are different:
## - Compute the objective and xlist of the MP
## - Update LB if neccesary
## - Compute SP for the list of xlist form MP
## - Update UB if neccesary
## - Add the Bender optimality cut
## Observations: 
## - Allow some gap when no solving for optimality
def OMT_Benders_classic(N,p,lamb,instan,
            bool_init_solution=False,
            init_solution=[],
            timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
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
    sx = {}
    for k in nodes:
        for (i,j) in arcs: 
            sx[i,j,k] =  MP.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
    mu = MP.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")
    MP.setObjective((1/sum(lamb))*sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) + mu, GRB.MINIMIZE)
    MP.addConstr(sum([x[i,i] for i in nodes]) == p)
    for i in nodes: 
        MP.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    for (i,j) in arcs:
        if i!=j:
            MP.addConstr(x[i,j] <= x[j,j])
    for k in nodes:
        MP.addConstr(sum([sx[i,j,k] for (i,j) in arcs]) <= 1)
    for (i,j) in arcs:
        MP.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
    for k in nodes:
        if k < len(nodes):
            MP.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in arcs]) <= sum([costs[i,j]*sx[i,j,k+1] for (i,j) in arcs]))
    # Initial solution:
    if bool_init_solution:
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
        for l in nodes:
            for (i,j) in arcs:
                if (i,j,l) in init_solution[2]:
                    sx[i,j,l].start = 1
                else: 
                    sx[i,j,l].start = 0                   
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
        sxlist = [(i,j,k) for k in nodes for (i,j) in arcs if sx[i,j,k].x>0.5]    
        obj_OM = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist])        
        obj_MP = obj_OM + mu.x  #MP.objVal
        ## LB actualization
        if (obj_MP>LB) and MP.status == GRB.OPTIMAL: LB = obj_MP
        ## SP solution (MST)
        zlist   = OMT_MST(N,p,lamb,instan,
                          l_bool = True,
                          xlist  = xlist)
        obj_SP  = sum([costs[i,j] for (i,j) in zlist])    
        obj_FIN = obj_OM + (1/(p-1))*obj_SP
        ## UB actualization
        if obj_FIN < UB: UB = obj_FIN
        ## Add optimality cut
        MP.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)
        n_b_cuts = n_b_cuts + 1 
    t_final = time.time()
    # Solution featuring:
    runtime   = t_final-t_init
    objVal    = UB
    objBound  = LB
    relaxVal  = relaxed_sol
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr   = "NO"
    selection  = [xlist,zlist,sxlist]
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts


## Classic Benders decomposition using covering formulation
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
    relaxed_sol = OMT_relaxed_covering(N,p,lamb,instan)
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
            us[k,h] = MP.addVar(vtype=GRB.BINARY, name="u_"+str(k)+"_"+str(h))
    mu = MP.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")    
    MP.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) + mu, GRB.MINIMIZE)        
    MP.addConstr(sum([x[i,i] for i in nodes]) == p)
    for i in nodes: 
        MP.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    for (i,j) in arcs:
        if i!=j:
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
        obj_MP = obj_OM + mu.x  #MP.objVal
        ## LB actualization
        if obj_MP > LB and MP.status == GRB.OPTIMAL: LB = obj_MP
        ## SP solution (MST)
        zlist   = OMT_MST(N,p,lamb,instan,
                          l_bool = True,
                          xlist  = xlist)
        obj_SP  = sum([costs[i,j] for (i,j) in zlist])    
        obj_FIN = obj_OM + (1/(p-1))*obj_SP
        ## UB actualization
        if obj_FIN < UB: UB = obj_FIN
        ## Add optimality cut
        MP.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)
        n_b_cuts = n_b_cuts + 1 
    t_final = time.time()
    # Solution featuring:
    runtime   = t_final-t_init
    objVal    = UB
    objBound  = LB
    relaxVal  = relaxed_sol
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr   = "NO"
    selection  = [xlist,zlist,ulist]
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts    

## Classic Benders decomposition with a warm-start phase algorithm 
## We tune 4 parameters to obtainn the better for every N,p,lambda 
#def OMT_Benders_classic_warm(N,p,lamb,instan,
#                        max_MP_time, #% time limit for every solve of the MP
#                        max_MP_gap, #% gap limit for every solve of the MP
#                        max_time, #% time limit to introduce heuristics cuts 
#                        max_gap, #% gap limit to introduce heuristics cuts 
#                        bool_init_solution=False,
#                        init_solution=[],
#                        timeLimit=5000): 
#    costs = instan
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    ## Initial bounds:    
#    initial_sol = OMT_heuristic_PMEDTOM(N, p, lamb, instan)
#    relaxed_sol = OMT_relaxed(N,p,lamb,instan)
#    UB = initial_sol[1]
#    LB = relaxed_sol
#    ## Set time:
#    t_init = time.time()
#    ## Set the callback to use in the warm-start phase:
#    def terminate_sub(model, where):
#        if where == GRB.Callback.MIPSOL:        
#            objbst  = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
#            objbnd  = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
#            time    = model.cbGet(GRB.Callback.RUNTIME)
#            if abs(objbst - objbnd) < max_MP_gap * (1.0 + abs(objbst)):
#                model.terminate()
#            if time > max_MP_time:
#                model.terminate()
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
#    # Initial solution (for x,z and sx):
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
#    ## Set some parameters to use in the while loop:
#    MP.update()
#    n_b_cuts_0    = MP.getAttr('NumConstrs')
#    n_b_cuts_pre = 0
#    n_b_cuts_pos = 0
#    use_callback = True
#    ## Classic Benders algorithm with warm-start phase:
#    while UB > LB:
#        ## Check time limit and reset time limit for computation
#        t_loop = time.time() - t_init      
#        if t_loop > timeLimit: break
#        MP.setParam("TimeLimit", timeLimit-t_loop)      
#        ## MP OM solution (callback vs no callback)      
#        if t_loop > max_time:
#            use_callback = False
#        if abs(UB - LB) < max_gap * (1.0 + abs(UB)):    
#            use_callback = False
#        if use_callback:
#            MP.optimize(terminate_sub)
#        else:
#            MP.optimize()
#        ## Retrieving solution features
#        xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
#        sxlist = [(i,j,k) for k in nodes for (i,j) in arcs if sx[i,j,k].x>0.5] 
#        obj_OM = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist]) 
#        obj_MP = obj_OM + mu.x  
#        ## LB actualization
#        if obj_MP > LB and MP.status == GRB.OPTIMAL and not use_callback: LB = obj_MP
#        ## SP solution (MST)
#        zlist   = OMT_MST(N,p,lamb,instan,
#                          l_bool = True,
#                          xlist  = xlist)
#        obj_SP  = sum([costs[i,j] for (i,j) in zlist])    
#        obj_FIN = obj_OM + (1/(p-1))*obj_SP
#        ## UB actualization
#        if obj_FIN < UB: UB = obj_FIN
#        ## Add optimality cut
#        MP.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)        
#        MP.update()
#        if use_callback:
#            n_b_cuts_pre = n_b_cuts_pre + 1
#        else:
#            n_b_cuts_pos = n_b_cuts_pos + 1
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
#    n_b_cuts_1 = MP.getAttr('NumConstrs')
#    n_b_cuts   = n_b_cuts_1-n_b_cuts_0
#    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts, n_b_cuts_pre, n_b_cuts_pos    


## Parameters: time_MP, gap_MP, time_GE, gap_ME
def OMT_Benders_classic_covering_warm(N,p,lamb,instan,
                        max_MP_time, #% time limit for every solve of the MP
                        max_MP_gap,  #% gap limit for every solve of the MP
                        max_time,    #% time limit to introduce heuristics cuts 
                        max_gap,     #% gap limit to introduce heuristics cuts 
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
    relaxed_sol = OMT_relaxed_covering(N,p,lamb,instan)
    UB = initial_sol[1]
    LB = relaxed_sol
    ## Set time:
    t_init = time.time()
    ## Set the callback to use in the warm-start phase:
    def terminate_sub(model, where):
        if where == GRB.Callback.MIPSOL:        
            objbst  = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            objbnd  = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            time    = model.cbGet(GRB.Callback.RUNTIME)
            if abs(objbst - objbnd) < max_MP_gap * (1.0 + abs(objbst)):
                model.terminate()
            if time > max_MP_time:
                model.terminate()
    ## Master Problem (OM + mu)
    MP = gp.Model("OMT_MP")
    MP.Params.LogToConsole = 0
    x  = {}
    for (i,j) in arcs:
        x[i,j] = MP.addVar(vtype=GRB.BINARY, lb=0, ub=1, name="x_"+str(i)+"_"+str(j))
    us = {}
    for k in nodes:
        for h in H: 
            us[k,h] = MP.addVar(vtype=GRB.BINARY, name="u_"+str(k)+"_"+str(h))
    mu = MP.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")    
    MP.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) + mu, GRB.MINIMIZE)
    MP.addConstr(sum([x[i,i] for i in nodes]) == p)
    for i in nodes: 
        MP.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    for (i,j) in arcs:
        if i!=j:
            MP.addConstr(x[i,j] <= x[j,j])        
    for h in H:        
        larcscosth = [(i,j) for (i,j) in arcs if costs[i,j] >= sucosts[h]]
        MP.addConstr(sum([us[k,h] for k in nodes]) == sum([x[i,j] for (i,j) in larcscosth]))
    for h in H:
        for k in nodes:
            if k < len(nodes):
                MP.addConstr(us[k,h] <= us[k+1,h])        
    # Initial solution (for x,z and sx):
    if bool_init_solution:
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
    ## Set some parameters to use in the while loop:
    MP.update()
    n_b_cuts_0   = MP.getAttr('NumConstrs')
    n_b_cuts_pre = 0
    n_b_cuts_pos = 0
    use_callback = True
    ## Classic Benders algorithm with warm-start phase:
    while UB > LB:
        ## Check time limit and reset time limit for computation
        t_loop = time.time() - t_init      
        if t_loop > timeLimit: break
        MP.setParam("TimeLimit", timeLimit-t_loop)      
        ## MP OM solution (callback vs no callback)      
        if t_loop > max_time:
            use_callback = False
        if abs(UB - LB) < max_gap * (1.0 + abs(UB)):    
            use_callback = False
        if use_callback:
            MP.optimize(terminate_sub)
        else:
            MP.optimize()
        ## Retrieving solution features
        xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
        ulist  = [(k,h) for k in nodes for h in H if us[k,h].x>0.5]
        obj_OM = (1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h].x for h in H if us[k,h].x>0.5]) for k in nodes])
        obj_MP = obj_OM + mu.x  
        ## LB actualization
        if obj_MP > LB and MP.status == GRB.OPTIMAL and not use_callback: LB = obj_MP
        ## SP solution (MST)
        zlist   = OMT_MST(N,p,lamb,instan,
                          l_bool = True,
                          xlist  = xlist)
        obj_SP  = sum([costs[i,j] for (i,j) in zlist])    
        obj_FIN = obj_OM + (1/(p-1))*obj_SP
        ## UB actualization
        if obj_FIN < UB: UB = obj_FIN
        ## Add optimality cut
        MP.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)        
        MP.update()
        if use_callback:
            n_b_cuts_pre = n_b_cuts_pre + 1
        else:
            n_b_cuts_pos = n_b_cuts_pos + 1
    t_final = time.time()        
    # Solution featuring:
    runtime   = t_final-t_init
    objVal    = UB
    objBound  = LB
    relaxVal  = relaxed_sol
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr   = "NO"
    selection  = [xlist,zlist,ulist]    
    n_b_cuts_1 = MP.getAttr('NumConstrs')
    n_b_cuts   = n_b_cuts_1-n_b_cuts_0
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts, n_b_cuts_pre, n_b_cuts_pos


##############################################################################################################################


## Modern Benders algorithm
## In this algorithm, while performing the Branch and Cut tree:
## - If the solution found in a node is fractionary then keep branching
## - Once the solution is integer, compute the SP solution and add the optimality cut
def OMT_Benders_modern(N,p,lamb,instan,
                bool_init_solution=False,
                init_solution=[],
                timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # Lazy-constraints to add the optimality cuts
    def add_benders_opt_cut(model, where):
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
    m = gp.Model("OMT_Benders_modern")
    m.Params.LogToConsole = 0
    # Variable declaration:
    x = {} 
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    sx = {}
    for l in nodes:
        for (i,j) in arcs: 
            sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
    mu = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")
    # Objective:
    m.setObjective((1/sum(lamb))*sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) + mu, GRB.MINIMIZE)
    # Constraints:    
    for i in nodes: m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)    
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    for (i,j) in arcs: 
        if i!=j:
            m.addConstr(x[i,j] <= x[j,j])
    for l in nodes: 
        m.addConstr(sum([sx[i,j,l] for (i,j) in arcs]) <= 1)
    for (i,j) in arcs: 
        m.addConstr(sum([sx[i,j,l] for l in nodes]) == x[i,j])
    for l in nodes:
        if l < len(nodes):
            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs]))
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
        for l in nodes:
            for (i,j) in arcs:
                if (i,j,l) in init_solution[2]:
                    sx[i,j,l].start = 1
                else: 
                    sx[i,j,l].start = 0
    m.setParam("TimeLimit", timeLimit)
    # Optimizer:
    m.optimize(add_benders_opt_cut)
    # Solution selection:
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    sxlist = [(i,j,l) for l in nodes for (i,j) in arcs if sx[i,j,l].x>0.5]
    zlist  = OMT_MST(N,p,lamb,instan,
                     l_bool = True,
                     xlist  = xlist)
    # Solution featuring:    
    runtime   = m.runtime
    objVal    = m.objVal
    objBound  = m.objBound
    relaxVal  = OMT_relaxed(N,p,lamb,instan)
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr  = m.getAttr('NodeCount')
    selection = [xlist, zlist, sxlist]
    n_b_cuts  = m._n_b_cuts
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts


## Modern Benders algorithm using covering
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
            z_list = OMT_MST(N,p,lamb,instan,
                             l_bool = True,
                             xlist  = x_list)
            obj_SP = sum([costs[i,j] for (i,j) in z_list])
            model.cbLazy((obj_SP/(p-1))*(sum([model._x[i,j] for (i,j) in x_list if i==j])-(p-1)) <= model._mu)
            model._n_b_cuts = model._n_b_cuts + 1 
    # Model:
    m = gp.Model("OMT_Benders_modern")
    m.Params.LogToConsole = 0
    # Variable declaration:
    x = {} 
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    us = {}
    for k in nodes:
        for h in H: 
            us[k,h] = m.addVar(vtype=GRB.BINARY, name="u_"+str(k)+"_"+str(h))
    mu = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")    
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) + mu, GRB.MINIMIZE)
    # Constraints:    
    for i in nodes: m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)    
    m.addConstr(sum([x[i,i] for i in nodes]) == p)    
    for (i,j) in arcs: 
        if i<j:
            m.addConstr(x[i,j] <= x[j,j])
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
    zlist  = OMT_MST(N,p,lamb,instan,
                     l_bool = True,
                     xlist  = xlist)
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

### Modern benders decomposition algorithm with a warm-start phase
#def OMT_Benders_modern_warm(N,p,lamb,instan,
#                        max_MP_time, #% time limit for every solve of the MP
#                        max_MP_gap, #% gap limit for every solve of the MP
#                        max_time,  #% time limit to introduce heuristics cuts 
#                        max_gap,  #% gap limit to introduce heuristics cuts
#                bool_init_solution=False,
#                init_solution=[],
#                timeLimit=5000):
#    costs = instan
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    ## Initial bounds:    
#    initial_sol = OMT_heuristic_PMEDTOM(N, p, lamb, instan)
#    relaxed_sol = OMT_relaxed(N,p,lamb,instan)
#    UB = initial_sol[1]
#    LB = relaxed_sol
#    t_init = time.time()
#    # Set the callback to use in the warm-start phase:
#    def terminate_sub(model, where):
#        if where == GRB.Callback.MIPSOL:
#            objbst  = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
#            objbnd  = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
#            if abs(objbst - objbnd) < max_MP_gap * (1.0 + abs(objbst)):
#                model.terminate()
#    # Set lazy-constraints to add the optimality cuts:
#    def add_benders_opt_cut(model, where):
#        if where == GRB.Callback.MIPSOL:
#            x_vals = model.cbGetSolution(model._x)        
#            x_list = [(i,j) for i, j in model._x.keys() if x_vals[i, j] > 0.5]
#            z_list = OMT_MST(N,p,lamb,instan,
#                          l_bool = True,
#                          xlist  = x_list)
#            obj_SP = sum([costs[i,j] for (i,j) in z_list])
#            model.cbLazy((obj_SP/(p-1))*(sum([model._x[i,j] for (i,j) in x_list if i==j])-(p-1)) <= model._mu)
#            model._n_b_cuts = model._n_b_cuts + 1 
#    # Model:
#    m = gp.Model("OMT_Benders_modern_warm")
#    m.Params.LogToConsole = 0
#    # Variable declaration:
#    x = {} 
#    for (i,j) in arcs:
#        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
#    sx = {}
#    for l in nodes:
#        for (i,j) in arcs: 
#            sx[i,j,l] =  m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
#    mu = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")
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
#    ## Warm-start phase:
#    n_b_cuts_pre = 0
#    t_warm = time.time() - t_init
#    while t_warm < max_time or abs(UB - LB) < max_gap * (1.0 + abs(UB)):
#        m.setParam("TimeLimit", max_MP_time)
#        m.optimize(terminate_sub)
#        xlist   = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
#        sxlist  = [(i,j,k) for k in nodes for (i,j) in arcs if sx[i,j,k].x>0.5] 
#        zlist   = OMT_MST(N,p,lamb,instan,
#                          l_bool = True,
#                          xlist  = xlist)
#        obj_OM  = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist]) 
#        obj_SP  = sum([costs[i,j] for (i,j) in zlist])  
#        obj_FIN = obj_OM + (1/(p-1))*obj_SP
#        if obj_FIN < UB: UB = obj_FIN
#        m.addConstr((obj_SP/(p-1))*(sum([x[i,j] for (i,j) in xlist if i==j])-(p-1)) <= mu)
#        m.update()
#        n_b_cuts_pre = n_b_cuts_pre + 1
#        t_warm = time.time() - t_init
#    ## Modern algorithm phase:
#    # Lazy contraints:
#    m._x  = x
#    m._mu = mu
#    m._n_b_cuts = 0
#    m.Params.lazyConstraints = 1
#    m.setParam("TimeLimit", timeLimit-max_time)
#    # Optimizer:
#    m.optimize(add_benders_opt_cut)
#    t_fin = time.time()
#    # Solution selection:
#    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
#    sxlist = [(i,j,l) for l in nodes for (i,j) in arcs if sx[i,j,l].x>0.5]
#    zlist  = OMT_MST(N,p,lamb,instan,
#                     l_bool = True,
#                     xlist  = xlist)
#    # Solution featuring:    
#    runtime   = t_fin - t_init
#    objVal    = m.objVal
#    objBound  = m.objBound
#    relaxVal  = OMT_relaxed(N,p,lamb,instan)
#    optimality_found = 0
#    if runtime < timeLimit:
#        optimality_found = 1
#    nodExplr  = m.getAttr('NodeCount')
#    selection = [xlist, zlist, sxlist]
#    n_b_cuts_pos = m._n_b_cuts
#    n_b_cuts     = n_b_cuts_pos + n_b_cuts_pre    
#    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection, n_b_cuts, n_b_cuts_pre, n_b_cuts_pos


### Modern benders decomposition algorithm usnig covering formulation with a warm-start phase
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
        ## Fractonary cuts
        if where == GRB.Callback.MIPNODE:
            x_vals = model.cbGetNodeRel(model._x)
            servers_vals   = {i:x_vals[i,j] for (i,j) in x_vals if i==j}
            servers_vals_s = dict(sorted(servers_vals.items(), key=lambda item: item[1]))
            frac_cut_tresh = sum(list(servers_vals_s.values())[-p:])
            if frac_cut_tresh > p-1:
                servers_list = list(servers_vals_s.keys())[-p:]
                z_list = OMT_MST(N,p,lamb,instan,
                                 l_bool = False,
                                 slist  = servers_list)
                obj_SP = sum([costs[i,j] for (i,j) in z_list])
                model.cbLazy((obj_SP/(p-1))*(sum([model._x[i,i] for i in servers_list])-(p-1)) <= model._mu)
                model._n_b_cuts = model._n_b_cuts + 1
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
    x = {} 
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    us = {}
    for k in nodes:
        for h in H:
            us[k,h] = m.addVar(vtype=GRB.BINARY, name="u_"+str(k)+"_"+str(h))
    mu = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="mu")    
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) + mu, GRB.MINIMIZE)
    # Constraints:
    for i in nodes: m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)    
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    for (i,j) in arcs: 
        if i<j:
            m.addConstr(x[i,j] <= x[j,j])
    for h in H:        
        larcscosth = [(i,j) for (i,j) in arcs if costs[i,j] >= sucosts[h]]
        m.addConstr(sum([us[k,h] for k in nodes]) == sum([x[i,j] for (i,j) in larcscosth]))
    for h in H:
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
    ## Warm-start phase:
    n_b_cuts_pre = 0
    t_warm = time.time() - t_init
    while ((t_warm < max_time) and (abs(UB - LB) < max_gap * (1.0 + abs(UB)))):
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
        m.update()
        n_b_cuts_pre = n_b_cuts_pre + 1
        t_warm = time.time() - t_init
    ## Modern algorithm phase:
    # Lazy contraints:
    m._x  = x
    m._mu = mu
    m._n_b_cuts = 0
    m.Params.lazyConstraints = 1
    m.setParam("TimeLimit", timeLimit-max_time)
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
