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

###################################
### Exact formulations (+ subelim)
###################################

## OMT, Miller-Tucker-Zemlin tree formulation
def OMT_mtz(N,p,lamb,instan,
            bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # root node arbitrarily choosen:
    r = random.choice(nodes)
    # Model:
    m = gp.Model("OMT_mtz")
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
    ## y[i,j]: if (i,j) is arc connecting servers
    y = {}
    for (i,j) in arcs:
        y[i,j] = m.addVar(vtype=GRB.BINARY, name="y_"+str(i)+"_"+str(j))
    # l[i]: position of i with respect to root node r
    l = {}
    for i in nodes:
        l[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,l) is ranked l
    sx = {}
    for k in nodes:
        for (i,j) in arcs: 
                sx[i,j,k] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
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
    ## Only one arc goes into a non-root:
    for u in nodes: ##  nodes.remove(r)
        if u != r:
            inset = [(i,j) for (i,j) in arcs if j == u]
            m.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
    ## If considering an edge, only take one arc in wether sense
    for (u,v) in arcs:
        if u<v:
            m.addConstr(y[u,v] + y[v,u] == z[u,v])
    ## Position of the nodes:
    ### If (u,v) arc is considered the position of v must be higher than u:
    for (u,v) in arcs: 
        if u!=v:
            m.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
    ### Root node in first position:
    m.addConstr(l[r]==1)
    ### Non root nodes position bounded between 2 and N:
    for u in nodes: ##nodes.remove(r):
        if u != r:
            m.addConstr(2 <= l[u])
            m.addConstr(l[u] <= p)
    # Compesating assignation to servers network:
    ## One sorted position is not assumed by more than one cost:
    for k in nodes:
        m.addConstr(sum([sx[i,j,k] for (i,j) in arcs]) <= 1)
    ## If i->j then only one order is assigned:
    for (i,j) in arcs:
        m.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
    ## Correct sorting of the orders:
    for k in nodes:
        if k < len(nodes):
            m.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,k+1] for (i,j) in arcs]))
    # Initial solution (for x,z and sx):
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
    m.optimize()
    # Vars
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = [(i,j) for (i,j) in arcs if i<j if z[i,j].x>0.5]
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

## OMT, flow based tree structure with additional variable r
def OMT_flow_1(N,p,lamb,instan,
               bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # Model:
    m = gp.Model("OMT_flow_1")
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
    ## y[i,j]: if there exist flow through (i,j) 
    f = {}
    for (i,j) in arcs:
        if i!=j:
            f[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_"+str(i)+"_"+str(j))
    ## r[i]: 1 if i node is selected as root for the connection structure
    r = {}
    for i in nodes: 
        r[i] = m.addVar(vtype=GRB.BINARY, name="r_"+str(i))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,k) is ranked l
    sx = {}
    for l in nodes:
        for (i,j) in arcs: 
                sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
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
    ## Only one root node selected:
    m.addConstr(sum([r[i] for i in nodes]) == 1)
    ## Only servers nodes can be activated as a root: 
    for i in nodes:
        m.addConstr(r[i] <= x[i,i])
    ## Flow distribution:
    for u in nodes:
        inset  = [(j,i) for (j,i) in arcs if i == u if i!=j]
        outset = [(i,j) for (i,j) in arcs if i == u if i!=j]
        m.addConstr((sum([f[i,j] for (i,j) in outset]) - sum([f[i,j] for (i,j) in inset])) == r[u]*(p-1)-(x[u,u]-r[u]))
    ## All variables sending flow must be activated:
    for (u,v) in arcs:
        if u<v:
            m.addConstr(f[u,v] <= (p-1)*z[u,v])
            m.addConstr(f[v,u] <= (p-1)*z[u,v])
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
            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs])) 
    # Initial a solution (for x,z and sx):
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
    m.optimize()
    # Vars
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = [(i,j) for (i,j) in arcs if i<j if z[i,j].x>0.5]
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

## OMT, flow based tree structure without additional variables
def OMT_flow_2(N,p,lamb,instan,
               bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # Source node arbitrarily choosen:
    r = random.choice(nodes)
    # Model:
    m = gp.Model("OMT_flow_2")
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
    ## y[i,j]: if there exist flow through (i,j) 
    f = {}
    for (i,j) in arcs:
        if i!=j:
            f[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_"+str(i)+"_"+str(j))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,k) is ranked l
    sx = {}
    for l in nodes:
        for (i,j) in arcs: 
                sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
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
    ## Flow distribution:
    ### - If root is non server: one flow vars must give p flow to the next node, that would we a server
    ###                          and that server would distributes p-1 flow
    ### - If root is server: directly distributes p-1 flow
    for u in nodes: 
        if (r,u) in arcs:
            inset  = [(j,i) for (j,i) in arcs if i == u if i!=j]
            outset = [(i,j) for (i,j) in arcs if i == u if i!=j]
            m.addConstr((sum([f[i,j] for (i,j) in outset]) - sum([f[i,j] for (i,j) in inset])) == p*x[r,u]-x[u,u])
    ## All variables sending flow must be activated:
    for (u,v) in arcs:
        if u<v:
            m.addConstr(f[v,u] <= (p-1)*z[u,v])
            m.addConstr(f[u,v] <= (p-1)*z[u,v])   
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
            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs]))
    # Initial a solution (for x,z and sx):
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
    m.optimize()
    # Solution featuring:
    # Vars
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = [(i,j) for (i,j) in arcs if i<j if z[i,j].x>0.5]
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

### Subtour elimination using lazy constraints and cycle elimination constraints
def OMT_subelim_1(N,p,lamb,instan,
                bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # Lazy-constraints to eliminate subtours
    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # Make a list of edges selected in the solution
            x_vals = model.cbGetSolution(model._x)
            z_vals = model.cbGetSolution(model._z)
            z_selected = [(i, j) for i, j in model._z.keys() if z_vals[i, j] > 0.5]
            servers_selected = [i for i, j in model._x.keys() if x_vals[i, j] > 0.5 if i==j]
            # Determine connected components
            CC = connected_components_networkz(servers_selected, z_selected)
            # If there is more than one connected component enter the loop
            # Otherwise we have found a solution
            if len(CC) > 1:
                # For each C connected component find the edges
                # If there is at leats one cycle (Cnodes<=Cedges)
                # then add the respective constraint
                for c_nodes in CC:
                    c_edges = [(i,j) for i,j in z_selected if i in c_nodes if j in c_nodes if i<j]
                    if len(c_nodes) <= len(c_edges):
                        model.cbLazy(gp.quicksum(model._z[i,j] for (i,j) in c_edges) <= (len(c_nodes) - 1))
                        # !!!! Just add the constraint for the first cycle identified
                        break
    # Model:
    m = gp.Model("OMT_subelim_1")
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
            sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
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
    ## Adding subtour elimination constraint only for unitary sets
    ## (Already tested, no improvements)
    ## for i in nodes: m.addConstr(z[i,i] <= 0)
    # Lazy contraints
    # Pass variables to work with lazy constraints:
    m._nodes = nodes
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
    zlist  = [(i,j) for (i,j) in arcs if i<j if z[i,j].x>0.5]
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

# Subtour elimination using second connectiion constraint
def OMT_subelim_2(N,p,lamb,instan,
                bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
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
            sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs if i<j]), GRB.MINIMIZE)
    # Constraints:    
    ## Exactly p servers
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    ## Each client assigned to exactly one server
    for i in nodes: m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)      
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



#############################
## Old functions (21/12/21)
#############################
    

## OMT, Miller-Tucker-Zemlin tree formulation
def OMT_mtz_old(N,p,lamb,instan,
            bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # root node arbitrarily choosen:
    r = random.choice(nodes)
    # Model:
    m = gp.Model("OMT_mtz")
    m.Params.LogToConsole = 0
    # Variable declaration:
    ## x[i,j]: 1 if client i served from server j
    x = {}
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    ## z[i,j]: 1 if (i,j) is edge connecting servers
    z = {}
    for (i,j) in arcs:
        z[i,j] = m.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
    ## y[i,j]: if (i,j) is arc connecting servers
    y = {}
    for (i,j) in arcs:
        y[i,j] = m.addVar(vtype=GRB.BINARY, name="y_"+str(i)+"_"+str(j))
    # l[i]: position of i with respect to root node r
    l = {}
    for i in nodes:
        l[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0 ,name="l_"+str(i))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,l) is ranked l
    sx = {}
    for k in nodes:
        for (i,j) in arcs: 
                sx[i,j,k] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(k))
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs]), GRB.MINIMIZE)
    # Constraints:
    ## Exactly p servers
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    ## Each client assigned to exactly one server
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    ## Relations between servers and clients
    for (i,j) in arcs:
        ## No client served from a non-server
        m.addConstr(x[i,j] <= x[j,j])
        ## No edges between non-servers (previosuly two sepparate)
        m.addConstr(2*z[i,j] <= x[i,i] + x[j,j])
    # Servers connected as a tree:
    ## Exactly p-1 edges connecting servers:
    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
    ## Only one arc goes into a non-root:
    for u in nodes: ##  nodes.remove(r)
        if u != r:
            inset =  [(i,j) for (i,j) in arcs if j == u]
            m.addConstr(sum([y[v,u] for (v,u) in inset]) == 1)
    ## If considering an edge, only take one arc in wether sense
    for (u,v) in arcs:
        if u<v:
            m.addConstr(y[u,v] + y[v,u] == z[u,v])
    ## Position of the nodes:
    ### If (u,v) arc is considered the position of v must be higher than u:
    for (u,v) in arcs: 
        if u!=v:
            m.addConstr(l[v] >= l[u]+1-N*(1-y[u,v]))
    ### Root node in first position:
    m.addConstr(l[r]==1)
    ### Non root nodes position bounded between 2 and N:
    for u in nodes: ##nodes.remove(r):
        if u != r:
            m.addConstr(2 <= l[u])
            m.addConstr(l[u] <= p)
    # Compesating assignation to servers network:
    ## One sorted position is not assumed by more than one cost:
    for k in nodes:
        m.addConstr(sum([sx[i,j,k] for (i,j) in arcs]) <= 1)
    ## If i->j then only one order is assigned:
    for (i,j) in arcs:
        m.addConstr(sum([sx[i,j,k] for k in nodes]) == x[i,j])
    ## Correct sorting of the orders:
    for k in nodes:
        if k < len(nodes):
            m.addConstr(sum([costs[i,j]*sx[i,j,k] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,k+1] for (i,j) in arcs]))
    # Initial solution (for x,z and sx):
    if bool_init_solution:
        ## x[i,j]
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
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
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = [(i,j) for (i,j) in arcs if z[i,j].x>0.5]
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

## OMT, flow based tree structure with additional variable r
def OMT_flow_1_old(N,p,lamb,instan,
               bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # Model:
    m = gp.Model("OMT_flow_1")
    m.Params.LogToConsole = 0
    # Variable declaration:
    ## x[i,j]: 1 if client i served from server j
    x = {}
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    ## z[i,j]: 1 if (i,j) is edge connecting servers
    z = {}
    for (i,j) in arcs:
        z[i,j] = m.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
    ## y[i,j]: 1 if there exist flow through (i,j) 
    f = {}
    for (i,j) in arcs:
        f[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_"+str(i)+"_"+str(j))
    ## r[i]: 1 if i node is selected as root for the connection structure
    r = {}
    for i in nodes: 
        r[i] = m.addVar(vtype=GRB.BINARY, name="r_"+str(i))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,k) is ranked l
    sx = {}
    for l in nodes:
        for (i,j) in arcs: 
                sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs]), GRB.MINIMIZE)
    # Constraints:
    ## Exactly p servers
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    ## Each client assigned to exactly one server
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    ## Relations between servers and clients
    for (i,j) in arcs:
        ## No client served from a non-server
        m.addConstr(x[i,j] <= x[j,j])
        ## No edges between non-servers
        m.addConstr(2*z[i,j] <= x[i,i] + x[j,j])
    # Servers connected as a tree:
    ## Exactly p-1 edges connecting servers:
    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
    ## Only one root node selected:
    m.addConstr(sum([r[i] for i in nodes]) == 1)
    ## Only servers nodes can be activated as a root: 
    for i in nodes:
        m.addConstr(r[i] <= x[i,i])
    ## Flow distribution:
    for u in nodes:
        inset  = [(j,i) for (j,i) in arcs if i == u if i!=j]
        outset = [(i,j) for (i,j) in arcs if i == u if i!=j]
        m.addConstr((sum([f[i,j] for (i,j) in outset]) - sum([f[i,j] for (i,j) in inset])) == r[u]*(p-1)-(x[u,u]-r[u]))
    ## All variables sending flow must be activated:
    for (u,v) in arcs:
        if u<v:
            m.addConstr(f[u,v] <= (p-1)*z[u,v])
            m.addConstr(f[v,u] <= (p-1)*z[u,v])
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
            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs])) 
    # Initial a solution (for x,z and sx):
    if bool_init_solution:
        ## x[i,j]
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
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
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = [(i,j) for (i,j) in arcs if z[i,j].x>0.5]
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

## OMT, flow based tree structure without additional variables
def OMT_flow_2_old(N,p,lamb,instan,
               bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # Source node arbitrarily choosen:
    r = random.choice(nodes)
    # Model:
    m = gp.Model("OMT_flow_2")
    m.Params.LogToConsole = 0
    # Variable declaration:
    ## x[i,j]: 1 if client i served from server j
    x = {}
    for (i,j) in arcs:
        x[i,j] = m.addVar(vtype=GRB.BINARY, name="x_"+str(i)+"_"+str(j))
    ## z[i,j]: 1 if (i,j) is edge connecting servers
    z = {}
    for (i,j) in arcs:
        z[i,j] = m.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
    ## y[i,j]: if there exist flow through (i,j) 
    f = {}
    for (i,j) in arcs:
        if i!=j:
            f[i,j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="f_"+str(i)+"_"+str(j))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,k) is ranked l
    sx = {}
    for l in nodes:
        for (i,j) in arcs: 
            #if i!=j:
                sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs]), GRB.MINIMIZE)
    # Constraints:
    ## Exactly p servers
    m.addConstr(sum([x[i,i] for i in nodes]) == p)
    ## Each client assigned to exactly one server
    for i in nodes: 
        m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    ## Relations between servers and clients
    for (i,j) in arcs:
        ## No client served from a non-server
        m.addConstr(x[i,j] <= x[j,j])
        ## No edges between non-servers
        m.addConstr(2*z[i,j] <= x[i,i] + x[j,j])
    # Servers connected as a tree:
    ## Exactly p-1 edges connecting servers:
    m.addConstr(sum([z[i,j] for (i,j) in arcs if i<j]) == p-1)
    ## Flow distribution:
    ### - If root is non server: one flow vars must give p flow to the next node, that would we a server
    ###                          and that server would distributes p-1 flow
    ### - If root is server: directly distributes p-1 flow
    for u in nodes: 
        if (r,u) in arcs:
            inset  = [(j,i) for (j,i) in arcs if i == u if i!=j]
            outset = [(i,j) for (i,j) in arcs if i == u if i!=j]
            m.addConstr((sum([f[i,j] for (i,j) in outset]) - sum([f[i,j] for (i,j) in inset])) == p*x[r,u]-x[u,u])
    ## All variables sending flow must be activated:
    for (u,v) in arcs:
        if u<v:
            m.addConstr(f[v,u] <= (p-1)*z[u,v])
            m.addConstr(f[u,v] <= (p-1)*z[u,v])   
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
            m.addConstr(sum([costs[i,j]*sx[i,j,l] for (i,j) in arcs]) <= 
                        sum([costs[i,j]*sx[i,j,l+1] for (i,j) in arcs]))
    # Initial a solution (for x,z and sx):
    if bool_init_solution:
        ## x[i,j]
        for (i,j) in arcs:
            if (i,j) in init_solution[0]:
                x[i,j].start = 1
            else: 
                x[i,j].start = 0
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
    # Solution featuring:
    # Vars
    xlist  = [(i,j) for (i,j) in arcs if x[i,j].x>0.5]
    zlist  = [(i,j) for (i,j) in arcs if z[i,j].x>0.5]
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

### Final subtour elimination using lazy constraints
def OMT_subelim_1_old(N,p,lamb,instan,
                bool_init_solution=False,init_solution=[],timeLimit=5000):
    costs = instan
    nodes = range(1,N+1)
    arcs  = [*costs]
    # Lazy-constraints to eliminate subtours
    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # Make a list of edges selected in the solution
            x_vals = model.cbGetSolution(model._x)
            z_vals = model.cbGetSolution(model._z)
            z_selected = [(i, j) for i, j in model._z.keys() if z_vals[i, j] > 0.5]
            servers_selected = [i for i, j in model._x.keys() if x_vals[i, j] > 0.5 if i==j]
            # Determine connected components
            CC = connected_components_networkz(servers_selected, z_selected)
            # If there is more than one connected component enter the loop
            # Otherwise we have found a solution
            if len(CC) > 1:
                # For each C connected component find the edges
                # If there is at leats one cycle (Cnodes<=Cedges)
                # then add the respective constraint
                for c_nodes in CC:
                    c_edges = [(i,j) for i,j in z_selected if i in c_nodes if j in c_nodes]
                    if len(c_nodes) <= len(c_edges):
                        model.cbLazy(gp.quicksum(model._z[s, t] for (s, t) in c_edges) <= (len(c_nodes) - 1))
                        # !!!! Just add the constraint for the first cycle identified
                        break
    # Model:
    m = gp.Model("OMT_subelim")
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
        z[i,j] = m.addVar(vtype=GRB.BINARY, name="z_"+str(i)+"_"+str(j))
    ## sx[i,j,l]: 1 if client i is assigned to server j and assignation cost (i,k) is ranked l
    sx = {}
    for l in nodes:
        for (i,j) in arcs: 
            sx[i,j,l] = m.addVar(vtype=GRB.BINARY, name="sx_"+str(i)+"_"+str(j)+"_"+str(l))
    # Objective:
    m.setObjective((1/sum(lamb)) * sum([sum([lamb[k-1]*costs[i,j]*sx[i,j,k] for (i,j) in arcs]) for k in nodes]) +
                   (1/(p-1)) * sum([costs[i,j]*z[i,j] for (i,j) in arcs]), GRB.MINIMIZE)
    # Constraints:    
    ## Exactly p servers
    m.addConstr(sum([x[i,i] for i in nodes]) == p)    
    ## Each client assigned to exactly one server
    for i in nodes: m.addConstr(sum([x[i,j] for j in nodes if (i,j) in arcs]) == 1)
    ## Relations between servers and clients
    for (i,j) in arcs:
        ## No client served from a non-server
        m.addConstr(x[i,j] <= x[j,j])
        ## No edges between non-servers
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
    ## Adding subtour elimination constraint only for unitary sets
    ## (Already tested, no improvements)
    ## for i in nodes: m.addConstr(z[i,i] <= 0)
    # Lazy contraints
    # Pass variables to work with lazy constraints:
    m._nodes = nodes
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
































