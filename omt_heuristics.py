#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:57:58 2021

@author: albtorval
"""
import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import time

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

#################################
### OMT heuristics algorithms ###
#################################

# 1) OMT+MST algorithm:
# - Perform Ordered Median (OM) computation
# - Obtain the server nodes list
# - Perform MST over the servers

def OMT_heuristic_OMMST(N, p, lamb, instan,timeLimit=5000):
    t_init = time.time()
    costs = instan
    nodes = range(1,N+1)
    arcs = [*costs]
    # Model:
    m = gp.Model("OMT_heuristic_1")
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
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,j) is ranked l
    sx = {}
    for l in nodes:
        for (i,j) in arcs:
            sx[i,j,l] =  m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
    # Objective:
    m.setObjective(sum([sum([lamb[l-1]*sx[i,j,l]*costs[i,j] for (i,j) in arcs]) for l in nodes]), 
                   GRB.MINIMIZE)
    # Constraints:    
    ## Exactly p servers
    m.addConstr(sum([x[i,i] for i in nodes]) == p)    
    ## Each client assigned to exactly one server
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)    
    ## No client served from a non-server
    for (i,j) in arcs:
        if i!=j:
            m.addConstr(x[i,j] <= x[j,j])
    # Compesating assignation to servers network:
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
    # Time limit
    m.setParam("TimeLimit", timeLimit)
    # Optimizer:
    m.optimize()
    # X vars
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    # SX vars
    sxlist = [(i,j,l) for l in nodes for (i,j) in arcs if sx[i,j,l].x>0.5]
    # MST to connect servers using Kruskal
    servers = [i for (i,j) in xlist if i==j]
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
    MST = minimum_spanning_tree(C)
    MST = MST.todok() # convert to dictionary of keys format
    # Z vars
    z = dict(MST.items())
    #zlist = [tuple(sorted((i+1,j+1))) for (i,j) in z.keys()]
    zlist = [tuple((i+1,j+1)) for (i,j) in z.keys()]
    t_fin = time.time()
    # Solution featuring:    
    runtime   = t_fin-t_init
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    objVal = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist]) + (1/(p-1))*sum([costs[i,j] for (i,j) in zlist])
    if optimality_found == 1:
        objBound = objVal
    else:
        objBound = "-" #np.nan
    relaxVal  = "-" #np.nan
    nodExplr  = "-" #np.nan
    selection = [xlist,zlist,sxlist]
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection


### 2) PMEDT
### - Perform PMEDT 
### - Fix tree structure between servers and assignation to servers
### - Perform OM over the fixed servers and assignations
    
def OMT_heuristic_PMEDTOM(N, p, lamb, instan,timeLimit=5000):
    t_init = time.time()
    costs = instan
    nodes = range(1,N+1)
    arcs = [*costs]
    ### 1) PMEDT witz mtz
    # root node arbitrarily choosen:
    r = random.choice(nodes)
    # PMEDT Model:
    mPMEDT = gp.Model("PMEDT")
    mPMEDT.Params.LogToConsole = 0
    # Variable declaration:
    ## x[i,j]: 1 if client i served from server j
    x = {}
    for (i,j) in arcs:
        x[i,j] = mPMEDT.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    ## z[i,j]: 1 if (i,j) is edge connecting servers
    z = {}
    for (i,j) in arcs:
        if i<j:
            z[i,j] = mPMEDT.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
    ## y[i,j]: if (i,j) is arc connecting servers
    y = {}
    for (i,j) in arcs:
        y[i,j] = mPMEDT.addVar(vtype=GRB.BINARY, name="y_"+str(i)+"_"+str(j))
    # l[i]: position of i with respect to root node r
    l = {}
    for i in nodes:
        l[i] = mPMEDT.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
    # Objective:
    mPMEDT.setObjective(sum([costs[i,j]*x[i,j] for (i,j) in arcs]), 
                        GRB.MINIMIZE)
    # Constraints:
    ## Exactly p servers
    mPMEDT.addConstr(sum([x[i,i] for i in nodes]) == p)
    ## Each client assigned to exactly one server
    for i in nodes:
        mPMEDT.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    ## Relation server/client
    for (i,j) in arcs: 
        ## No client served from a non-server
        if i!=j:
            mPMEDT.addConstr(x[i,j] <= x[j,j])
        ## No edges between non-servers
        if i<j:
            mPMEDT.addConstr(2*z[i,j] <= x[i,i] + x[j,j])
    # Servers connected as a tree:
    ## Exactly p-1 edges connecting servers:
    mPMEDT.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
    ## Only one arc goes into a non-root:
    for u in nodes: ##  nodes.remove(r)
        if u != r:
            inset =  [(i,j) for (i,j) in arcs if j == u]
            mPMEDT.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
    ## If considering an edge, only take one arc in wether sense
    for (u,v) in arcs:
        if u<v:
            mPMEDT.addConstr(y[u,v] + y[v,u] == z[u,v])
    ## Position of the nodes:
    ### If (u,v) arc is considered the position of v must be higher than u:
    for (u,v) in arcs: 
        if u!=v:
            mPMEDT.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
    ### Root node in first position:
    mPMEDT.addConstr(l[r]==1)
    ### Non root nodes position bounded between 2 and N:
    if p>1:
        for u in nodes: ##nodes.remove(r):
            if u != r:
                mPMEDT.addConstr(2 <= l[u])
                mPMEDT.addConstr(l[u] <= p)
    # Time limit
    mPMEDT.setParam("TimeLimit", timeLimit)
    # Optimizer:
    mPMEDT.optimize()
    # Solution featuring:
    # X vars
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    # Z vars
    zlist  = [(i,j) for (i,j) in arcs if i<j if z[i,j].x>0.5]
    # 2) OMT fixing servers
    # OM Model:
    mOM = gp.Model("OM")
    mOM.Params.LogToConsole = 0
    # Variable declaration:
    ## x[i,j]: 1 if client i served from server j (heredit)
    x = {} 
    for (i,j) in xlist:
        x[i,j] = mOM.addVar(vtype=GRB.BINARY, lb=1, ub=1, name="x_"+str(i)+"_"+str(j))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,j) is ranked k
    sx = {}
    for (i,j) in xlist: 
        for k in nodes:
            sx[i,j,k] = mOM.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
    # Objective:
    mOM.setObjective(sum([sum([lamb[k-1]*sx[i,j,k]*costs[i,j] for (i,j) in xlist]) for k in nodes]), GRB.MINIMIZE)
    # Constraints:    
    # Compesating assignation to servers network:
    ## One sorted position is not assumed by more than one cost:
    for k in nodes:
        mOM.addConstr(sum([sx[i,j,k] for (i,j) in xlist]) <= 1)
    ## If i->j then only one order is assigned:
    for (i,j) in xlist:
        mOM.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
    ## Correct sorting of the orders:
    for k in nodes:
        if k < len(nodes):
            mOM.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in xlist]) <= sum([costs[i,j]*sx[i,j,k+1] for (i,j) in xlist]))
    # Time limit
    mOM.setParam("TimeLimit", timeLimit)
    # Optimizer:
    mOM.optimize()
    # SX vars
    sxlist = [(i,j,l) for l in nodes for (i,j) in xlist if sx[i,j,l].x>0.5]
    t_fin = time.time()
        # Solution featuring:    
    runtime   = t_fin-t_init
    optimality_found = 0
    if runtime < timeLimit:
        optimality_found = 1
    objVal = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist]) + (1/(p-1))*sum([costs[i,j] for (i,j) in zlist])   
    if optimality_found == 1:
        objBound = objVal
    else:
        objBound = "-" #np.nan
    relaxVal  = "-" #np.nan
    nodExplr  = "-" #np.nan
    selection = [xlist,zlist,sxlist]
    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection



#############################
## Old functions (21/12/21)
#############################

## 1) OMT+MST algorithm:
#def OMT_heuristic_OMMST(N, p, lamb, instan,timeLimit=5000):
#    t_init = time.time()
#    costs = instan
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    # Model:
#    m = gp.Model("OMT_heuristic_1")
#    # Uncomment to show model information
#    m.Params.LogToConsole = 0
#    # Variable declaration:
#    ## x[i,j]: 1 if client i served from server j
#    x = {} 
#    for (i,j) in arcs:
#        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
#    ## z[i,j]: 1 if (i,j) is edge connecting servers
#    z = {}
#    for (i,j) in arcs:
#        z[i,j] = m.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
#    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,j) is ranked l
#    sx = {}
#    for l in nodes:
#        for (i,j) in arcs:
#            sx[i,j,l] =  m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
#    # Objective:
#    m.setObjective(sum([sum([lamb[l-1]*sx[i,j,l]*costs[i,j] for (i,j) in arcs]) for l in nodes]), 
#                   GRB.MINIMIZE)
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
#    # Compesating assignation to servers network:
#    ## One sorted position is not assumed by more than one cost:
#    for l in nodes:
#        m.addConstr(sum([sx[i,j,l] for (i,j) in arcs]) <= 1)
#    ## If i->j then only one order is assigned:
#    for (i,j) in arcs:
#        m.addConstr(sum([sx[i,j,l] for l in nodes]) == x[i,j])
#    ## Correct sorting of the orders:
#    for l in nodes:
#        if l < len(nodes):
#            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs]))  
#    # Time limit
#    m.setParam("TimeLimit", timeLimit)
#    # Optimizer:
#    m.optimize()
#    # X vars
#    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
#    # SX vars
#    sxlist = [(i,j,l) for l in nodes for (i,j) in arcs if sx[i,j,l].x>0.5]
#    # MST to connect servers using Kruskal
#    servers = [i for (i,j) in xlist if i==j]
#    C = np.zeros((N,N))
#    for i in nodes: 
#        for j in nodes:
#            if i in servers:
#                if j in servers:
#                    if (i,j) in arcs:
#                        C[i-1][j-1] = costs[i,j]
#            else: 
#                C[i-1][j-1] = 0
#    C   = csr_matrix(C)
#    MST = minimum_spanning_tree(C)
#    MST = MST.todok() # convert to dictionary of keys format
#    # Z vars
#    z = dict(MST.items())
#    #zlist = [tuple(sorted((i+1,j+1))) for (i,j) in z.keys()]
#    zlist = [tuple((i+1,j+1)) for (i,j) in z.keys()]
#    t_fin = time.time()
#    # Solution featuring:    
#    runtime   = t_fin-t_init
#    optimality_found = 0
#    if runtime < timeLimit:
#        optimality_found = 1
#    objVal = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist]) + (1/(p-1))*sum([costs[i,j] for (i,j) in zlist])
#    if optimality_found == 1:
#        objBound = objVal
#    else:
#        objBound = "-" #np.nan
#    relaxVal  = "-" #np.nan
#    nodExplr  = "-" #np.nan
#    selection = [xlist,zlist,sxlist]
#    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection
#
#
#### 2) PMEDT    
#def OMT_heuristic_PMEDTOM(N, p, lamb, instan,timeLimit=5000):
#    t_init = time.time()
#    costs = instan
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    ### 1) PMEDT witz mtz
#    # root node arbitrarily choosen:
#    r = random.choice(nodes)
#    # PMEDT Model:
#    mPMEDT = gp.Model("PMEDT")
#    mPMEDT.Params.LogToConsole = 0
#    # Variable declaration:
#    ## x[i,j]: 1 if client i served from server j
#    x = {}
#    for (i,j) in arcs:
#        x[i,j] = mPMEDT.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
#    ## z[i,j]: 1 if (i,j) is edge connecting servers
#    z = {}
#    for (i,j) in arcs:
#        z[i,j] = mPMEDT.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
#    ## y[i,j]: if (i,j) is arc connecting servers
#    y = {}
#    for (i,j) in arcs:
#        y[i,j] = mPMEDT.addVar(vtype=GRB.BINARY, name="y_"+str(i)+"_"+str(j))
#    # l[i]: position of i with respect to root node r
#    l = {}
#    for i in nodes:
#        l[i] = mPMEDT.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
#    # Objective:
#    mPMEDT.setObjective(sum([costs[i,j]*x[i,j] for (i,j) in arcs]), 
#                        GRB.MINIMIZE)
#    # Constraints:
#    ## Each client assigned to exactly one server
#    for i in nodes: 
#        mPMEDT.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
#    ## Exactly p servers
#    mPMEDT.addConstr(sum([x[k,k] for k in nodes]) == p)
#    ## Relations between servers and clients
#    for (i,j) in arcs:
#        ## No client served from a non-server
#        mPMEDT.addConstr(x[i,j] <= x[j,j])
#        ## No edges between non-servers
#        mPMEDT.addConstr(z[i,j] <= x[i,i])
#        mPMEDT.addConstr(z[i,j] <= x[j,j])
#    # Servers connected as a tree:
#    ## Exactly p-1 edges connecting servers:
#    mPMEDT.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
#    ## Only one arc goes into a non-root:
#    for u in nodes: ##  nodes.remove(r)
#        if u != r:
#            inset =  [(i,j) for (i,j) in arcs if j == u]
#            mPMEDT.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
#    ## If considering an edge, only take one arc in wether sense
#    for (u,v) in arcs:
#        if u<v:
#            mPMEDT.addConstr(y[u,v] + y[v,u] == z[u,v])
#    ## Position of the nodes:
#    ### If (u,v) arc is considered the position of v must be higher than u:
#    for (u,v) in arcs: 
#        if u!=v:
#            mPMEDT.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
#    ### Root node in first position:
#    mPMEDT.addConstr(l[r]==1)
#    ### Non root nodes position bounded between 2 and N:
#    if p>1:
#        for u in nodes: ##nodes.remove(r):
#            if u != r:
#                mPMEDT.addConstr(2 <= l[u])
#                mPMEDT.addConstr(l[u] <= p)
#    # Time limit
#    mPMEDT.setParam("TimeLimit", timeLimit)
#    # Optimizer:
#    mPMEDT.optimize()
#    # Solution featuring:
#    # X vars
#    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
#    # Z vars
#    zlist  = [(i,j) for (i,j) in arcs if z[i,j].x>0.5]
#    # 2) OMT fixing servers
#    # OM Model:
#    mOM = gp.Model("OM")
#    mOM.Params.LogToConsole = 0
#    # Variable declaration:
#    ## x[i,j]: 1 if client i served from server j (heredit)
#    x = {} 
#    for (i,j) in xlist:
#        x[i,j] = mOM.addVar(vtype=GRB.BINARY, lb=1, ub=1, name="x_"+str(i)+"_"+str(j))
#    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,j) is ranked k
#    sx = {}
#    for (i,j) in xlist: 
#        for k in nodes:
#            sx[i,j,k] =  mOM.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
#    # Objective:
#    mOM.setObjective(sum([sum([lamb[k-1]*sx[i,j,k]*costs[i,j] for (i,j) in xlist]) for k in nodes]), GRB.MINIMIZE)
#    # Constraints:    
#    # Compesating assignation to servers network:
#    ## One sorted position is not assumed by more than one cost:
#    for k in nodes:
#        mOM.addConstr(sum([sx[i,j,k] for (i,j) in xlist]) <= 1)
#    ## If i->j then only one order is assigned:
#    for (i,j) in xlist:
#        mOM.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
#    ## Correct sorting of the orders:
#    for k in nodes:
#        if k < len(nodes):
#            mOM.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in xlist]) <= sum([costs[i,j]*sx[i,j,k+1] for (i,j) in xlist]))
#    # Time limit
#    mOM.setParam("TimeLimit", timeLimit)
#    # Optimizer:
#    mOM.optimize()
#    # SX vars
#    sxlist = [(i,j,l) for l in nodes for (i,j) in xlist if sx[i,j,l].x>0.5]
#    t_fin = time.time()
#        # Solution featuring:    
#    runtime   = t_fin-t_init
#    optimality_found = 0
#    if runtime < timeLimit:
#        optimality_found = 1
#    objVal = (1/sum(lamb))*sum([lamb[k-1]*costs[i,j] for (i,j,k) in sxlist]) + (1/(p-1))*sum([costs[i,j] for (i,j) in zlist])   
#    if optimality_found == 1:
#        objBound = objVal
#    else:
#        objBound = "-" #np.nan
#    relaxVal  = "-" #np.nan
#    nodExplr  = "-" #np.nan
#    selection = [xlist,zlist,sxlist]
#    return runtime, objVal, objBound, relaxVal, optimality_found, nodExplr, selection

### *) Random algorithm:
### - Select p servers randomdly
### - Compute MST on the servers nodes
### - For each non server node compute costs to each client and 

#def OMT_heuristic_random(N,p,instancia):
#    
#    costs = instancia
#    nodes = range(1,N+1)
#    arcs = [*costs]
#    
#    # Randomly select p servers
#    servers = set(random.sample(nodes, p))
#    nonservers = set(nodes)-set(servers)
#
#    # Computing MST between servers
#    C = np.zeros((N,N))
#    for i in nodes: 
#        for j in nodes:
#            if i in servers:
#                if j in servers:
#                    if (i,j) in arcs:
#                        C[i-1][j-1] = costs[i,j]
#            else: 
#                C[i-1][j-1] = 0
#
#    C   = csr_matrix(C)
#    MST = minimum_spanning_tree(C)
#    MST = MST.todok() # convert to dictionary of keys format
#    
#    # For each non server, connect to the server with lesser cost and compensating
#    dic_ = {}        
#    for o in nonservers:
#        d = {(i,j):v for ((i,j),v) in costs.items() if i == o if j in servers}
#        minval = min(d.values())
#        res = set(filter(lambda x: d[x]==minval, d))
#        dic_[res.pop()] = minval
#    dic = sorted(dic_, key=dic_.get)
#
#    # Objective
#    objective = sum(dic_.values())
#    # Z vars
#    z = dict(MST.items())
#    z = [tuple(sorted((i+1,j+1))) for (i,j) in z.keys()]
#    # X vars
#    x = [(i,i) for i in servers] + dic 
#    # SX vars
#    sx = list(map(lambda f,l:(f[0],f[1],l),x,list(nodes)))
#    
#    return objective,x,z,sx





