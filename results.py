#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:18:32 2021

@author: albtorval
"""
import numpy as np
import pandas as pd
import math
import ast
import statistics

from omt_exact_covering import preprocessings_covering_fix
from instancias import random_instances_generator

pd.set_option('max_rows', 100000)  # or 1000
pd.set_option('max_columns', None)  # or 1000
#pd.set_option('max_colwidth', None)  # or 199


#####################
### Test loop results
#####################    

## Creates random instances (graph with assigned costs) given the number of nodes and add them to a txt file
def instances_generator_txt(criteria,N,p,density,ctype,ninstances):
    if ctype == 1:
        lcb, ucb = 1,100
    if ctype == 2:
        lcb, ucb = 1,100000
    #add other type of costs here
    instancias = random_instances_generator(Nnodes = N, Ndensity= density, Ninstances = ninstances,
                                            lower_costs_bound = lcb, upper_costs_bound = ucb)
    for crit in criteria:
        with open("experiments/data/criterion_%s/instances/OMT_%s_%s_%s_c%s.txt" % (crit,N,p,density,ctype), 'w') as f: 
            for l in range(len(instancias)):
                f.write("%s;%s;%s" % (float("+inf"), float ("-inf"),instancias[l])+"\n")

## Retrieves the instances previously generated
def instances_recuperator_txt(criterion,N,p,density,ctype):
    df = pd.read_csv("experiments/data/criterion_%s/instances/OMT_%s_%s_%s_c%s.txt" % (criterion,N,p,density,ctype), sep=';', header = None)
    df = np.array(df)
    instancias = []
    for s in range(df.shape[0]):
        instancias.append(ast.literal_eval(df[s][2]))
    return instancias

## Computes initial solutions for each heuristic/instance pair adding them to a txt file:
def lanzadera_initsol_txt(criterion,N,p,density,ctype,lamb,heuristicos,instancias,timeLimit=5000):
    with open("experiments/data/criterion_%s/initsol/OMT_%s_%s_%s_c%s_initsol.txt" % (criterion,N,p,density,ctype), 'w') as f:
        for i in range(len(instancias)):
            for j in range(len(heuristicos)):
                print('INITSOL INSTANCE ' + str(i+1) + ' HEURISTIC '+ str(j+1))
                resultado = heuristicos[j](N,p,lamb,instancias[i],timeLimit=5000)
                f.write("%s;%s;%s" % (i+1,j+1,list(resultado))+"\n")
                print('- END')
        print('LOOP END')

## Retrieves the initial solution previously generated
def initsol_recuperator_txt(criterion,N,p,density,ctype):
    df = pd.read_csv("experiments/data/criterion_%s/initsol/OMT_%s_%s_%s_c%s_initsol.txt" % (criterion,N,p,density,ctype),
                     sep=';', header = None)
    df = np.array(df)
    iSol = []
    for s in df:
        enlista = ast.literal_eval(s[2])
        iSol.append([int(s[0]),int(s[1]),enlista])
    return iSol

## Computes the fixing preprocessings for each heuristic/instance pair adding them to a txt file:
def lanzadera_fix_covering_txt(criterion,N,p,density,ctype,lamb,instancias):
    with open("experiments/data/criterion_%s/fixing_covering/OMT_%s_%s_%s_c%s_fixing_covering.txt" % (criterion,N,p,density,ctype), 'w') as f:
        for i in range(len(instancias)):
                print('COVERING FIX INSTANCE ' + str(i+1))
                resultado = preprocessings_covering_fix(N,p,density,lamb,instancias[i])                
                f.write("%s;%s;%s" % (i+1,resultado[0],resultado[1])+"\n")
                print('- END')
        print('LOOP END')
        
## Retrieves the fixing preprocessings solutions previously generated        
def fix_covering_recuperator_txt(criterion,N,p,density,ctype):
    df = pd.read_csv("experiments/data/criterion_%s/fixing_covering/OMT_%s_%s_%s_c%s_fixing_covering.txt" % (criterion,N,p,density,ctype),
                     sep=';', header = None)
    df = np.array(df)
    iFix = []
    for s in df:
        to1 = ast.literal_eval(s[1])
        to0 = ast.literal_eval(s[2])
        iFix.append([to1, to0])
    return iFix
    
# Computes solutions for each model/instance pair adding them to a txt file:
def lanzadera_txt(server, N, p, density, ctype, lamb, tlimit, modelos, instancias, heuristicos = [],
                  init_sols   = [], init_sols_pos = [], init_bool = False,
                  fixing_sols = [], fixing_bool   = []):
    with open("experiments/results/server_%s/OMT_%s_%s_%s_c%s_results.txt" % (server,N,p,density,ctype), 'w') as f:
        for i in range(len(instancias)):
            for j in range(len(modelos)):
                print('SOLUTION INSTANCE ' + str(i+1) + ' MODEL '+ str(j+1))                
                if init_bool:
                    init_solut = init_sols[(init_sols_pos[i]+i*len(heuristicos))-1][2][6]
                else:
                    init_solut = []                
                if fixing_bool[j]:
                    fixing_solut = fixing_sols[i]
                    resultado = modelos[j](N,p,lamb,instancias[i],
                                           bool_init_solution   = init_bool,
                                           init_solution        = init_solut,
                                           bool_fixing_covering = True,
                                           fixing_covering      = fixing_solut,
                                           timeLimit            = tlimit)
                else:
                    resultado = modelos[j](N,p,lamb,instancias[i],
                                           bool_init_solution   = init_bool,
                                           init_solution        = init_solut,
                                           timeLimit            = tlimit)                
                f.write("%s;%s;%s;%s;%s;%s;%s;%s" % (i+1,j+1,
                                                     resultado[0],
                                                     resultado[1],
                                                     resultado[2],
                                                     resultado[3],
                                                     resultado[4],
                                                     resultado[5])+"\n")
                print('- END')
        print('LOOP END')
# Returns: 
#   0 instance, 1 model, 2 time, 3 U, 4 L, 5 optimality, 6 nodes explored, (NO) 7 vars selected
    
# Bounds actualization:
# Reads instances txt, if better bounds are found in the result computations then actualize
def bounds_actualization(criterion,server,N,p,density,ctype):    
    df_instances = pd.read_csv("experiments/data/criterion_%s/instances/OMT_%s_%s_%s_c%s.txt" % (criterion,N,p,density,ctype), 
                               sep=';', header = None, na_values=["-"])
    df_instances = np.array(df_instances)
    df_results   = pd.read_csv("experiments/results/server_%s/OMT_%s_%s_%s_c%s_results.txt" % (server,N,p,density,ctype), 
                             sep=';', header = None, na_values=["-"])
    df_results   = np.array(df_results)
    with open("experiments/data/criterion_%s/instances/OMT_%s_%s_%s_c%s.txt" % (criterion,N,p,density,ctype), 'w') as f: 
        for i in range(df_instances.shape[0]):
            for l in range(df_results.shape[0]):
                U = df_instances[i][0]
                L = df_instances[i][1]
                if df_results[l][0]==i+1:
                    if df_results[l][3]<U:
                        df_instances[i][0] = df_results[l][3]
                    if df_results[l][4]>L:
                        df_instances[i][1] = df_results[l][4]
            f.write("%s;%s;%s" % (df_instances[i][0], df_instances[i][1], df_instances[i][2])+"\n")

# Visualize as table the results computed in lanzadera and saves them
## TXT STRUCTURES 
## - OMT....txt
##  0 BestU, 1 BestL, 2 instance
## - OMT...results.txt
##  0 #instance, 1 #model, 2 time, 3 U, 4 L, 5 objRL, 6 optimality, 7 nodes explored
def print_results(criterion,server,N,p,density,ctype,tlimit,modelos):
    def gap_1(x,y): return 100*abs(x-y)/y
    def gap_2(x,y): return 100*abs(x-y)/x
    df_instances = pd.read_csv("experiments/data/criterion_%s/instances/OMT_%s_%s_%s_c%s.txt" % (criterion,N,p,density,ctype), 
                               sep=';', header = None, na_values=["-"])
    df_instances = np.array(df_instances)
    df_results   = pd.read_csv("experiments/results/server_%s/OMT_%s_%s_%s_c%s_results.txt" % (server,N,p,density,ctype), 
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
        data.append([
                l_model, l_N, l_density, l_p, l_numinstance, l_gapUL, l_gapBUL, l_gapUBL, l_gapBUR,
                l_time, l_objU, l_objL, l_objR, l_BobjU, l_BobjL, l_optimal, l_nodesexplr
                ])
    header = ["Model", "N", "density", "p", "instance", "gapUL", "gapBUL", "gapUBL", "gapBUR", "CPU",
              "objU", "objL", "objR", "Best objU", "Best objL", "optimality", "Nodes explored"] # Seleccion
    table  = pd.DataFrame(data, columns = header)
    table.to_csv("experiments/results/server_%s/OMT_%s_%s_%s_c%s_final.txt" % (server,N,p,density,ctype), sep=';')
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



