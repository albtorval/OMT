#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:50:15 2021

@author: albtorval
"""
import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np

##############################
## OMT relaxed formulations ##
##############################

## This script keep several linear relaxations of the OMT problem
## Call the script when the relaxation of the OMT problem is needed
## The functions return is the objective value of the relaxation

## - OMT linear relaxation using MTZ formulation
def OMT_relaxed(N,p,lamb,instan,
            bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs = [*costs]
    r = random.choice(nodes)
    # Model:
    m = gp.Model("OMT_relaxed")
    m.Params.LogToConsole = 0
    # Variable declaration:
    x = {}
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x_"+str(i)+"_"+str(j))
    z = {}
    for (i,j) in arcs:
        if i<j:
            z[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z_"+str(i)+"_"+str(j))
    y = {}
    for (i,j) in arcs:
        y[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y_"+str(i)+"_"+str(j))
    l = {}
    for i in nodes:
        l[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
    sx = {}
    for k in nodes:
        for (i,j) in arcs: 
                sx[i,j,k] =  m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
    # Objective:
    m.setObjective( (1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs if i<j]) , GRB.MINIMIZE)
    # Constraints:
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    for (i,j) in arcs:
        if i!=j:
            m.addConstr(x[i,j] <= x[j,j])
        if i<j:
            m.addConstr(2*z[i,j] <= x[i,i] + x[j,j])
    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
    for u in nodes:
        if u != r:
            inset =  [(i,j) for (i,j) in arcs if j == u]
            m.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
    for (u,v) in arcs:
        if u<v:
            m.addConstr(y[u,v] + y[v,u] == z[u,v])
    for (u,v) in arcs: 
        if u!=v:
            m.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
    m.addConstr(l[r]==1)
    if p>1:
        for u in nodes: 
            if u != r:
                m.addConstr(2 <= l[u])
                m.addConstr(l[u] <= p)
    for k in nodes:
        m.addConstr(sum([sx[i,j,k] for (i,j) in arcs]) <= 1)
    for (i,j) in arcs:
        m.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
    for k in nodes:
        if k < len(nodes):
            m.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,k+1] for (i,j) in arcs]))
    # Optimizer:
    m.optimize()
    objVal    = m.objVal
    return objVal


## Covering relaxation
def OMT_relaxed_covering(N,p,lamb,instan,
            bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs   = instan
    nodes   = range(1,N+1)
    arcs    = [*costs]
    # List of unique costs sorted
    sucosts = sorted(np.unique(list(costs.values())))
    H = range(1,len(sucosts))
    # Root node arbitrarily choosen:
    r = random.choice(nodes)
    # Model:
    m = gp.Model("OMT_relax_covering")
    m.Params.LogToConsole = 0
    # Variable declaration:
    x = {}
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name="x_"+str(i)+"_"+str(j))
    z = {}
    for (i,j) in arcs:
        if i<j:
            z[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name="z_"+str(i)+"_"+str(j))
    y = {}
    for (i,j) in arcs:
        y[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name="y_"+str(i)+"_"+str(j))
    l = {}
    for i in nodes:
        l[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
    us = {}
    for k in nodes:
        for h in H: 
                us[k,h] =  m.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(k)+"_"+str(h))
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs if i<j]) , GRB.MINIMIZE)
    # Constraints:
    m.addConstr(sum([x[i,i] for k in nodes]) == p)
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    for (i,j) in arcs:
        if i!=j:
            m.addConstr(x[i,j] <= x[j,j])
        if i<j:
            m.addConstr(2*z[i,j] <= x[i,i] + x[j,j])
    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
    for u in nodes:
        if u != r:
            inset =  [(i,j) for (i,j) in arcs if j == u]
            m.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
    for (u,v) in arcs:
        if u<v:
            m.addConstr(y[u,v] + y[v,u] == z[u,v])
        if u!=v:
            m.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
    m.addConstr(l[r]==1)
    if p>1:
        for u in nodes:
            if u != r:
                m.addConstr(2 <= l[u])
                m.addConstr(l[u] <= p)
    for h in H:        
        larcscosth = [(i,j) for (i,j) in arcs if costs[i,j] >= sucosts[h]]
        m.addConstr(sum([us[k,h] for k in nodes]) == sum([x[i,j] for (i,j) in larcscosth]))
    for h in H:
        for k in nodes:
            if k < len(nodes):
                m.addConstr(us[k,h] <= us[k+1,h])
    # Initial a solution (for x,z and u):
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
        ## u initial solution?
    # Time limit
    m.setParam("TimeLimit", timeLimit)
    # Optimizer:
    m.optimize()
    objVal = m.objVal
    return objVal



#############################
## Old functions (21/12/21)
#############################


### Linear relaxation using MTZ formulation
#def OMT_relaxed(N,p,lamb,instan,
#            bool_init_solution=False,init_solution=[],timeLimit=5000):
#    costs = instan
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    r = random.choice(nodes)
#    # Model:
#    m = gp.Model("OMT_relaxed")
#    m.Params.LogToConsole = 0
#    # Variable declaration:
#    x = {}
#    for (i,j) in arcs:
#        x[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x_"+str(i)+"_"+str(j))
#    z = {}
#    for (i,j) in arcs:
#        z[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="z_"+str(i)+"_"+str(j))
#    y = {}
#    for (i,j) in arcs:
#        y[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y_"+str(i)+"_"+str(j))
#    l = {}
#    for i in nodes:
#        l[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
#    sx = {}
#    for k in nodes:
#        for (i,j) in arcs: 
#                sx[i,j,k] =  m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
#    # Objective:
#    m.setObjective( (1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
#                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs]) , GRB.MINIMIZE)
#    # Constraints:
#    for i in nodes: 
#        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
#    m.addConstr(sum([x[k,k] for k in nodes]) == p)
#    for (i,j) in arcs:
#        m.addConstr(x[i,j] <= x[j,j])
#        m.addConstr(z[i,j] <= x[i,i])
#        m.addConstr(z[i,j] <= x[j,j])
#    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
#    for u in nodes:
#        if u != r:
#            inset =  [(i,j) for (i,j) in arcs if j == u]
#            m.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
#    for (u,v) in arcs:
#        if u<v:
#            m.addConstr(y[u,v] + y[v,u] == z[u,v])
#    for (u,v) in arcs: 
#        if u!=v:
#            m.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
#    m.addConstr(l[r]==1)
#    if p>1:
#        for u in nodes: 
#            if u != r:
#                m.addConstr(2 <= l[u])
#                m.addConstr(l[u] <= p)
#    for k in nodes:
#        m.addConstr(sum([sx[i,j,k] for (i,j) in arcs]) <= 1)
#    for (i,j) in arcs:
#        m.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
#    for k in nodes:
#        if k < len(nodes):
#            m.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in arcs]) <= 
#                        sum([costs[i,j]*sx[i,j,k+1] for (i,j) in arcs]))
#    # Optimizer:
#    m.optimize()
#    objVal    = m.objVal
#    return objVal

### Covering linear relaxation
#def OMT_relaxed_covering(N,p,lamb,instan,
#            bool_init_solution=False,init_solution=[],timeLimit=5000):
#    costs   = instan
#    nodes   = range(1,N+1)
#    arcs    = [*costs]
#    # List of unique costs sorted
#    sucosts = sorted(np.unique(list(costs.values())))
#    H = range(1,len(sucosts))
#    # Root node arbitrarily choosen:
#    r = random.choice(nodes)
#    # Model:
#    m = gp.Model("OMT_relax_covering")
#    m.Params.LogToConsole = 0
#    # Variable declaration:
#    ## x[i,j]: 1 if client i served from server j
#    x = {}
#    for (i,j) in arcs:
#        x[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name="x_"+str(i)+"_"+str(j))
#    ## z[i,j]: 1 if (i,j) is edge connecting servers
#    z = {}
#    for (i,j) in arcs:
#        z[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name="z_"+str(i)+"_"+str(j))
#    ## y[i,j]: if (i,j) is arc connecting servers
#    y = {}
#    for (i,j) in arcs:
#        y[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name="y_"+str(i)+"_"+str(j))
#    # l[i]: position of i with respect to root node r
#    l = {}
#    for i in nodes:
#        l[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
#    ## u[l,h]: if the l-th assingment cost is at least c(h)
#    us = {}
#    for k in nodes:
#        for h in H: 
#                us[k,h] =  m.addVar(vtype=GRB.CONTINUOUS, name="u_"+str(k)+"_"+str(h))
#    # Objective:
#    m.setObjective( (1/sum(lamb)) * sum([sum([lamb[k-1]*(sucosts[h]-sucosts[h-1])*us[k,h] for h in H]) for k in nodes]) +
#                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs]) , GRB.MINIMIZE)
#    # Constraints:
#    ## Each client assigned to exactly one server
#    for i in nodes: 
#        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
#    ## Exactly p servers
#    m.addConstr(sum([x[k,k] for k in nodes]) == p)
#    ## Relations between servers and clients
#    for (i,j) in arcs:
#        ## No client served from a non-server
#        m.addConstr(x[i,j] <= x[j,j])
#        ## No edges between non-servers
#        m.addConstr(z[i,j] <= x[i,i])
#        m.addConstr(z[i,j] <= x[j,j])
#    # Servers connected as a tree:
#    ## Exactly p-1 edges connecting servers:
#    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
#    ## Only one arc goes into a non-root:
#    for u in nodes: ##  nodes.remove(r)
#        if u != r:
#            inset =  [(i,j) for (i,j) in arcs if j == u]
#            m.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
#    ## If considering an edge, only take one arc in wether sense
#    for (u,v) in arcs:
#        if u<v:
#            m.addConstr(y[u,v] + y[v,u] == z[u,v])
#    ## Position of the nodes:
#    ### If (u,v) arc is considered the position of v must be higher than u:
#    for (u,v) in arcs: 
#        if u!=v:
#            m.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
#    ### Root node in first position:
#    m.addConstr(l[r]==1)
#    ### Non root nodes position bounded between 2 and N:
#    if p>1:
#        for u in nodes: ##nodes.remove(r):
#            if u != r:
#                m.addConstr(2 <= l[u])
#                m.addConstr(l[u] <= p)
#    # Ordering costs:
#    for h in H:        
#        larcscosth = [(i,j) for (i,j) in arcs if costs[i,j] >= sucosts[h]]
#        m.addConstr(sum([us[k,h] for k in nodes]) == sum([x[i,j] for (i,j) in larcscosth]))
#    for h in H:
#        for k in nodes:
#            if k < len(nodes):
#                m.addConstr(us[k,h] <= us[k+1,h])
#    # Initial a solution (for x,z and u):
#    if bool_init_solution:
#        ## x[i,j]
#        for (i,j) in arcs:
#            if (i,j) in init_solution[0]:
#                x[i,j].start = 1
#            else: 
#                x[i,j].start = 0
#        ## z[i,j]
#        for (i,j) in arcs:
#            if (i,j) in init_solution[1]:
#                z[i,j].start = 1
#            else: 
#                z[i,j].start = 0
#        ## u initial solution?
#    # Time limit
#    m.setParam("TimeLimit", timeLimit)
#    # Optimizer:
#    m.optimize()
#    objVal    = m.objVal
#    return objVal





















































