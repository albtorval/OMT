# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:11:57 2021

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

## Connected components:
def connected_components_networkz(nodes,arcs):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(arcs)
    CC = list(nx.connected_components(G))
    return(CC)
    
## Kruskal MST algorithm:
## This function recieves as inputs N,p,lamb and the instance given to the OMT problem
## The list of the assignation variables (x vars) is also needed in order to compute MST
## only for the servers identified by previous models. 
## The return is the list of edges connecting the tree servers.
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

###################################
### Exact formulations (+ subelim)
###################################
    
## For this script we will aggregate x variables using sx 

## OMT, Miller-Tucker-Zemlin tree formulation
def OMT_mtz_2(N,p,lamb,instan,
            bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs = [*costs]
    # root node arbitrarily choosen:
    r = random.choice(nodes)
    # Model:
    m = gp.Model("OMT_mtz")
    m.Params.LogToConsole = 0
    # Variable declaration:
    ## z[i,j]: 1 if (i,j) is edge connecting servers
    ## y[i,j]: if (i,j) is arc connecting servers
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,l) is ranked l
    ## l[i]: position of i with respect to root node r
    sx,z,y,l = {},{},{},{}
    for (i,j) in arcs:
        z[i,j] = m.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
        y[i,j] = m.addVar(vtype=GRB.BINARY, name="y_"+str(i)+"_"+str(j))
        for k in nodes: 
            sx[i,j,k] =  m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
    for i in nodes:
        l[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
    # Objective:
    m.setObjective((1/sum(lamb))*sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs for k in nodes]) +
                   (1/(p-1))*sum([costs[i,j]*z[i,j] for (i,j) in arcs]), GRB.MINIMIZE)
    # Constraints:
    m.addConstr(sum([sx[i,i,k] for i in nodes for k in nodes]) == p)
    m.addConstr(l[r]==1)
    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)    
    for i in nodes: 
        m.addConstr(sum([sx[i,j,k] for j in nodes for k in nodes if (i,j) in arcs]) == 1)
    for u in nodes: 
        if u != r:
            m.addConstr(2 <= l[u])
            m.addConstr(l[u] <= p)
            inset =  [(i,j) for (i,j) in arcs if j == u]
            m.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
    for k in nodes:
        m.addConstr(sum([sx[i,j,k] for (i,j) in arcs]) <= 1)
        if k < len(nodes):
            m.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,k+1] for (i,j) in arcs]))
    for (i,j) in arcs:
        m.addConstr(sum([sx[i,j,k] for k in nodes]) <= sum([sx[j,j,k] for k in nodes]))
        m.addConstr(2*z[i,j] <= sum([sx[i,i,k]+sx[j,j,k]  for k in nodes]))
    for (u,v) in arcs:
        if u<v:
            m.addConstr(y[u,v] + y[v,u] == z[u,v])
        if u!=v:
            m.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
    # Initial solution 
    if bool_init_solution:
        ## z[i,j]
        for (i,j) in arcs:
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
    m.optimize()
    # Vars
    xlist  = []
    zlist  = [(i,j) for (i,j) in arcs if z[i,j].x>0.5]
    sxlist = [(i,j,l) for l in nodes for (i,j) in arcs if sx[i,j,l].x>0.5]
    ## Linear relaxation of the model:
    r = m.relax()
    r.optimize()
    # Solution featuring:    
    runtime   = m.runtime
    objVal    = m.objVal
    objBound  = m.objBound
    relaxVal  = r.objVal
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    nodExplr  = m.getAttr('NodeCount')
    selection = [xlist,zlist,sxlist]
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection
              
