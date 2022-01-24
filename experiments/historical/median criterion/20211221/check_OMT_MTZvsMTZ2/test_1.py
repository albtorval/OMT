#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:01:40 2021

@author: albtorval
"""

# Tests for MTZ using x,z,sx,y,l vs aggregating x and sx

from instancias import *
from results import *
from omt_heuristics import *
from omt_exact_1 import *
from omt_exact_2 import *

import math

list_N     = [60, 70, 80] 
density    = 1
ninstances = 5
tlimit     = 3600
heuristics    = [OMT_heuristic_PMEDTOM]
init_sols_pos = [1,1,1,1,1]
models        = [OMT_mtz,
                 OMT_mtz_2]

for N in list_N:
    lamb      = [1]*N
    list_p    = [math.floor(N/4), math.floor(N/3), math.floor(N/2)]
    for p in list_p:
        instancias = instances_recuperator_txt(N,p,density)
        init_sols  = initsol_recuperator_txt(N,p,density)
        lanzadera_txt(N,p,lamb,density,tlimit,models,instancias,
                      heuristicos=heuristics, 
                      init_sols=init_sols, 
                      init_sols_pos=init_sols_pos, 
                      initial_bool = True)
        #bounds_actualization(N,p,density)
        print("\n")
        print_results(N,p,density,tlimit,models)
