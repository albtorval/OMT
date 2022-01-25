#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:20:58 2021

@author: albtorval
"""

#### EXAMPLE: covering fixing and its matrices

import numpy as np
import pandas as pd

from instancias import *
from results import *
from omt_exact_covering import *

def preprocessings_covering_fix_0(N, p, density, lamb, instan):
    costs   = instan
    nodes   = range(1,N+1)
    arcs    = [*costs]
    sucosts = sorted(np.unique(list(costs.values())))
    H   = range(1,len(sucosts))   
    H1h = []
    print("Fixing to 1")
    for h in H:        
        # Model:
        m1 = gp.Model("OMT_covering_fixing_1")
        m1.Params.LogToConsole = 0
        # Variable declaration:    
        x = {}
        for (i,j) in arcs: x[i,j] = m1.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
        # Objective:
        m1.setObjective(sum([x[i,j] for (i,j) in arcs]), GRB.MAXIMIZE)    
        # Constraints:
        m1.addConstr(sum([x[i,i] for i in nodes]) == p)
        for i in nodes: 
            m1.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) <= 1)
        for (i,j) in arcs: 
            if i!=j:
                m1.addConstr(x[i,j] <= x[j,j])
        for (i,j) in arcs:
        #   if i!=j:
                m1.addConstr(costs[i,j]*x[i,j] <= sucosts[h-1])
        # Time limit
        #m.setParam("TimeLimit", timeLimit)
        # Optimizer:
        m1.optimize()
        # Solution        
        H1h.append(m1.objVal+1)
        xlist = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
        print("For h="+str(h)+" model selected "+str(xlist))
    H0h = []
    print("Fixing to 02")
    for h in H:        
        # Model:
        m0 = gp.Model("OMT_covering_fixing_0")
        m0.Params.LogToConsole = 0
        # Variable declaration:    
        x = {}
        for (i,j) in arcs: 
            x[i,j] = m0.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
        # Objective:
        m0.setObjective(sum([x[i,j] for (i,j) in arcs]), GRB.MAXIMIZE)    
        # Constraints:
        m0.addConstr(sum([x[i,i] for i in nodes]) == p)
        for i in nodes: 
            m0.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) <= 1)
        for (i,j) in arcs: 
            if i!=j:
                m0.addConstr(x[i,j] <= x[j,j])
        for (i,j) in arcs:
            if i!=j:
                m0.addConstr(costs[i,j] >= sucosts[h]*x[i,j])
        # Time limit
        #m.setParam("TimeLimit", timeLimit)
        # Optimizer:
        m0.optimize()
        H0h.append(N-m0.objVal+p)
        xlist = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
        print("For h="+str(h)+" model selected "+str(xlist))
    return H1h, H0h

N       = 4
p       = 2
density = 1
lamb    = [1]*N  #[0]*(N-1)+[1] 
#instancias = random_instances_generator(Nnodes = N, Ndensity = density, Ninstances = 1)
#instancia  = instancias[0]
instancia  = {
 (1, 2): 2,
 (1, 3): 1,
 (1, 4): 3,
 (2, 3): 4,
 (2, 4): 5,
 (3, 4): 2, 
 (1, 1): 0,
 (2, 2): 0,
 (3, 3): 0,
 (4, 4): 0,
 (2, 1): 2,
 (3, 1): 1,
 (4, 1): 3,
 (3, 2): 4,
 (4, 2): 5,
 (4, 3): 2
}

print_graph(instancia,print_edge_labels = True)

cov = OMT_mtz_covering(N,p,lamb,instancia)
print(cov)
print_solution_graph(N,cov,instancia,print_edge_labels = True)

# Check
costs   = instancia
sucosts = sorted(np.unique(list(costs.values())))
nodes   = range(1,N+1)
arcs    = [*costs]
lH  = len(sucosts)-1 #remove 0
us  = cov[6][2]
costs_undirected = sorted([costs[i,j] for (i,j) in arcs if i<j])
print(costs_undirected[-5:]) #five last costs
print(len(costs_undirected)-len(sucosts)+1) #number of repeated costs (without 0)

# U solution matrix
U = np.zeros((N, lH))
for i in range(1,N+1):
    for j in range(1,lH+1):
        if (i,j) in us: 
            U[i-1,j-1] = 1
        else:
            U[i-1,j-1] = 0
dfU = pd.DataFrame(U)
dfU.columns = list(range(1,lH+1))
dfU.index   = list(range(1,N+1))
print(dfU)

# Preprocessing fix matrix
l_fix = preprocessings_covering_fix_0(N, p, density, lamb, instancia)
for l in l_fix:
    print(l)

P = np.zeros((N,lH))
for j in range(1,lH+1):
    for i in range(1,N+1):
        if i >= l_fix[0][j-1]:
            P[i-1,j-1] = 1
        elif i <= l_fix[1][j-1]:
            P[i-1,j-1] = 0
        else: #-1 represents it was not fixed 
            P[i-1,j-1] = -1

dfP = pd.DataFrame(P)
dfP.columns = list(range(1,lH+1))
dfP.index   = list(range(1,N+1))
print(dfP)    





























