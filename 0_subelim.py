#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:15:33 2021

@author: albtorval
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import random
import time

from instancias import *
from omt_heuristics import *
from omt_exact_covering import *

##########################
### Auxiliar functions ###
##########################

## Connected components:
def connected_components_networkz(nodes,arcs):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(arcs)
    CC = list(nx.connected_components(G))
    return(CC)
    
def test_OMT_subelim(N,p,lamb,instan,
                bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs = [*costs]
    # Lazy-constraints to eliminate subtours
    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # Make a list of edges selected in the solution           
            x_vals = model.cbGetSolution(model._x)
            z_vals = model.cbGetSolution(model._z)
            x_selected = [(i, j) for i, j in model._x.keys() if x_vals[i, j] > 0.5]
            z_selected = [(i, j) for i, j in model._z.keys() if i<j if z_vals[i, j] > 0.5]
            # Search the connected components       
            CC = connected_components_networkz(nodes, x_selected + z_selected)            
            # If there is more than one connected component enter the loop
            if len(CC) > 1:
                S   = CC[0]
                VmS = set(nodes)-S
                model.cbLazy(
                        gp.quicksum(model._z[i,j] for i in S for j in VmS if i<j) +
                        gp.quicksum(model._z[j,i] for i in S for j in VmS if i>j) +
                        gp.quicksum(model._x[i,j] + model._x[j,i] for i in S for j in VmS)
                            >= 1)
    # Model:
    m = gp.Model("OMT_subelim_2")
    # Uncomment to show model information
    m.Params.LogToConsole = 0
    # Variable declaration:
    ## x[i,j]: 1 if client i served from server j
    x = {} 
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    ## z[i,j]: 1 if (i,j) is edge connecting servers
    z = {}
    for (i,j) in arcs:
        if i<j:
            z[i,j] = m.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,k) is ranked l
    sx = {}
    for l in nodes:
        for (i,j) in arcs: 
            sx[i,j,l] =  m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs if i<j]), GRB.MINIMIZE)
    # Constraints:    
    ## Exactly p servers
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    ## Each client assigned to exactly one server
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)      
    ## Relation client-server
    for (i,j) in arcs:
    ## No client served from a non-server
        if i!=j:
            m.addConstr(x[i,j] <= x[j,j])
    ## No edges between non-servers
        if i<j:
            m.addConstr(2*z[i,j] <= x[i,i] + x[j,j])
    # Servers connected as a tree:
    ## Exactly p-1 edges connecting servers:
    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)     
    # Ordering assignation to servers network:
    ## One sorted position is not assumed by more than one cost:
    for l in nodes:
        m.addConstr(sum([sx[i,j,l] for (i,j) in arcs]) <= 1)
    ## If i->j then only one order is assigned:
    for (i,j) in arcs:
        m.addConstr(sum([sx[i,j,l] for l in nodes]) == x[i,j])
    ## Correct sorting of the orders:
    for l in nodes:
        if l < len(nodes):
            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs]))
    # Lazy contraints
    # Pass variables to work with lazy constraints:
    m._nodes = nodes
    m._arcs  = arcs
    m._z     = z
    m._x     = x
    # Indicates we are working with lazy constraints
    m.Params.lazyConstraints = 1
    # Initial solution:
    if bool_init_solution:
        ## x[i,j]
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
        ## z[i,j]
        for (i,j) in arcs:
            if i<j:
                if (i,j) in init_solution[1]:
                    z[i,j].start = 1
                else: 
                    z[i,j].start = 0
        ## sx[i,j,l]
        for l in nodes:
            for (i,j) in arcs: 
                if (i,j,l) in init_solution[2]:
                    sx[i,j,l].start = 1
                else: 
                    sx[i,j,l].start = 0
    # Time limit
    m.setParam("TimeLimit", timeLimit)
    # Optimizer:
    m.optimize(subtourelim)
    # Solution featuring:
    # Vars
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = [(i,j) for (i,j) in arcs if i<j and z[i,j].x>0.5]
    sxlist = [(i,j,l) for l in nodes for (i,j) in arcs if sx[i,j,l].x>0.5]
    ## Linear relaxation of the model:
    rel = m.relax()
    rel.optimize()
    # Solution featuring:    
    runtime   = m.runtime
    objVal    = m.objVal
    objBound  = m.objBound
    relaxVal  = rel.objVal
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr  = m.getAttr('NodeCount')
    selection = [xlist,zlist,sxlist]
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection


## Test

N       = 10
p       = 6
density = 1
lamb    = [1]*N
tL      = 3600  #limit of time for all models
instancias = random_instances_generator(Nnodes = N, Ndensity = density, Ninstances = 1)
instancia  = instancias[0]
t_heuristic = OMT_heuristic_OMMST(N, p, lamb, instancia)
t_initial   = t_heuristic[6]
t           = test_OMT_subelim(N,p,lamb,instancia,
                    bool_init_solution   = True,  
                    init_solution        = t_initial,
                    timeLimit=5000)
print(t)

print("Solución óptima")
tOPT  = OMT_mtz_covering(N,p,lamb,instancia,timeLimit=tL)
print(tOPT)

#print_solution_graph(N,t1,instancia,print_edge_labels = True)



