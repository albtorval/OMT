# Ordered Median Tree Problem

This is the repository for the code implementing different formulations and algorithms from simplest p-median version problem to the Ordered Median Tree problem using Python. Different formulations presented would be compared using Gurobi solver.

## Description of the scipts

- `instancias.py`. Includes functions to create random instances generations for the problem.
    
    - `random_density_connected_graph()`. Given the number of nodes and a density about the edges computes a random connected graph. If density is set to 1, then we are considering a complete graph as instance. Only edges are considered, i.e. `(i,j)` with `i<j`. The instance is generated as follows:
        + First we find an eulerian path over the nodes.
        + We consider the remaining edges once the eulerian path is found.
        + We randomly add these edges until the density is fulfilled.
    - `random_costs_assignation()`. Adds random costs between [`lower_bound`,`upper_bound`] to the edges generated before. There exists the possibility to consider the same cost for all edges (`all_costs_same_value=True`). Also, if `(i,j)` is in the list, the random generation does not include arc `(j,i)`, but we create the `(j,i)` arc within this function with same costs as `(i,j)` (setting `symmetry = False` avoid considering `(j,i)`). As output we will have a dictionary with edges as keys and costs as attribute.
    - `random_instances_generator()`. Creates `Ninstances` using previous steps. Output would be a list of instances to use as inputs for our algorithms.
    - `instances_generator_txt()`. Creates the random instances and saves them in a txt file.
    - `instances_recuperator_txt()`. Retrieves the instances previously generated.
    - `initsol_recuperator_txt()`. Retrieves the initial solution previously generated.

- `omt.py`. Includes formulations of the ordered median tree problem we want to study in detail. In this problem we want to compensate the costs of assigning clients to the servers so that those clients who find it more difficult to join the network decide not to leave it. To do this we define a non-decreasing compensation vector `lambda`. As an inheritance from the previous section, several OMT formulations are presented according to the connection structure used:
    - `OMT_subelim()`. OMT using subtour elimination tree formulation.
    - `OMT_mtz()`. OMT using Miller-Tucker_Zemlin tree formulation (no arbitrary root node selection fix needed).
    - `OMT_flow_1()`. OMT using flow based tree formulation fixing arbitrary root node selection using additional variable r.
    - `OMT_flow_2()`. OMT using flow based tree formulation fixing arbitrary root node selection without using additional variables.
    - `OMT_subelim_covering()`. OMT using subtour elimination tree formulation with covering variables.
    - `OMT_mtz_covering()`. OMT using Miller-Tucker_Zemlin tree formulation (no arbitrary root node selection fix needed) with covering variables.
    - `OMT_flow_1_covering()`. OMT using flow based tree formulation fixing arbitrary root node selection using additional variable r with covering variables.
    - `OMT_flow_2_covering()`. OMT using flow based tree formulation fixing arbitrary root node selection without using additional variables with covering variables.

- `omt_heuristics.py`. Includes two heuristic algorithms for computing the initial solution to pass to the exact formulations in `omt.py`:
    + **OMT+MST algorithm**:
        - Perform Ordered Median (OM) computation.
        - Obtain the server nodes list.
        - Perform MST over the servers.
    + **PMEDT + OM**
        - Perform PMEDT 
        - Fix tree structure between servers and assignation to servers
        - Perform OM over the fixed servers and assignations

- `results.py`. Includes functions for visualizing the results in different ways: 
    - `print_graph()`. Visualize the graph given as instance.
    - `print_solution_graph()`. Visualize the solution graph.
    - `lanzadera_initsol_txt()`. Computes initial solutions for each heuristic/instance pair adding them to a txt file.
    - `lanzadera_txt()`. Computes solutions for each model/instance pair adding them to a txt file.
    - `bounds_actualization()`. Reads instances txt, if better bounds are found in the result computations then actualize.
    - `print_results()`. Computes solutions for each model/instance pair using `lanzadera` and returns visualization results as a table lanzadera. Includes: 
        + For each model/instance pair computes time and objective output.
        + Number of nodes of the instances
        + Number of servers of the instances
        + Density of edges used in the problem.
        + For each model, the average time of computation.
        + Lower bound and relaxation objective.
        + If the optimality is found.
        + Nodes explored in the branch and cut tree.

- `test.py`. Added for testing.