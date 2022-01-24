#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:51:36 2021

@author: albtorval
"""
import math
import random
import sys

import networkx as nx
import matplotlib.pyplot as plt

##########################
### Instances generation
##########################

# Creates a list of edges given the number of nodes and the density of the graph
def random_density_connected_graph(Nnodes, Ndensity, ins=1):
    Kedges = Nnodes*(Nnodes-1)/2
    random.seed(ins)
    if Ndensity*Kedges >= Nnodes-1: 
        # Create the random Eulerian path:
        def random_path(Nnodes):
            E,L = [],[i for i in range(1,Nnodes+1)]
            i = random.choice(L)
            L.remove(i)
            while(L != []):
                j = random.choice(L)
                E.append(tuple(sorted((i,j))))
                L.remove(j)
                i = j
            return E        
        E = random_path(Nnodes)
        # Adding random edges until the density is fulfilled:
        Nwant = math.ceil(Kedges*Ndensity)
        Nrest = Nwant - len(E)       
        while Nrest != 0:
            i = random.choice(range(1, Nnodes))
            j = random.choice(range(i+1, Nnodes+1))
            if (i,j) not in E:
                E.append((i,j))
                Nrest = Nrest-1
        return E
    else: sys.exit("Error: density not enough to connect all nodes")
      
# Given a list of edges, creates random costs on that list
def random_costs_assignation(E, Nnodes, lower_bound=1, upper_bound=100,
                            all_costs_same_value=False, all_value=0, 
                            symmetry = True, ins=1):
    Nedges = len(E)
    random.seed(ins)
    Costs = {}
    # Want all costs to same value?
    if all_costs_same_value:
        for i in range(Nedges):
            Costs[E[i]] = all_value
    # Want all costs randomly assigned?
    if not all_costs_same_value:
        for i in range(Nedges):
            Costs[E[i]] = random.choice(range(lower_bound, upper_bound))    
    # Adding (k,k) edges with 0 cost:
    for k in range(1, Nnodes+1):
        Costs[k,k] = 0
    # Simmetry:
    if symmetry:
        for (i,j) in E: 
            Costs[j,i] = Costs[i,j]
    return Costs

# Creates random instances (graph with assigned costs) given the number of nodes
def random_instances_generator(Nnodes, Ndensity=1, Ninstances = 100, 
                        lower_costs_bound = 1, upper_costs_bound = 100, 
                        symmetry = True, all_costs_same_value = False, all_value = 0):
    Instances = []
    for k in range(Ninstances):
        ins = random.randint(0,Ninstances*1000)
        E = random_density_connected_graph(Nnodes,Ndensity,ins)
        ECosts = random_costs_assignation(E, Nnodes,
                            lower_costs_bound, upper_costs_bound,
                            all_costs_same_value, all_value, 
                            symmetry, ins)
        Instances.append(ECosts)
    return Instances

#################
### Visualization
#################

# Visualize the graph given as instance:
def print_graph(Costs, print_edge_labels = False):
    E = [*Costs]
    G = nx.Graph()
    G.add_edges_from(E)
    if print_edge_labels:
        nx.draw(G, pos=E, with_labels = True)
        nx.draw_networkx_edge_labels(G, pos=E, edge_labels=Costs, 
            label_pos=0.75, font_color='red', font_size=10)
    if not print_edge_labels:
        nx.draw(G, with_labels = True)
    plt.show()
    
# Visualize the solution graph:
def print_solution_graph(N,selection,instan,print_edge_labels = False):
    sel = selection[6][0] + selection[6][1]
    costs = instan
    ARISTAS   = [(i,j) for (i,j) in sel if i!=j] 
    NODOS_HUB = [i for (i,j) in sel if i==j]
    COSTES = {} 
    for (i,j) in ARISTAS: COSTES[i,j] = costs[i,j]
    G = nx.Graph()
    G.add_nodes_from(list(range(1,N+1)))
    G.add_edges_from(ARISTAS) 
    color_map = ['red' if node in NODOS_HUB else 'orange' for node in G]
    if print_edge_labels:
        nx.draw(G, pos=[*costs], with_labels = True, node_color = color_map)
        nx.draw_networkx_edge_labels(G, pos=[*costs], edge_labels=COSTES, 
                                     label_pos=0.75, font_color='red', font_size=10)
    if not print_edge_labels:
        nx.draw(G, with_labels = True, node_color = color_map)
    plt.show()



              


