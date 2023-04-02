# Surrogate Level-Based Lagrangian Relaxation Implementations

This repository contains multiple implementations of the Surrogate Level-Based Lagrangian Relaxation (SLBLR) method for mixed-integer linear programming (MILP) problems, as presented in the following research papers:

    M. A. Bragin, and E. L. Tucker, "Surrogate 'Level-Based' Lagrangian Relaxation for Mixed-Integer Linear Programming," Scientific Reports, Volume 12, Issue 1, 22417, 2022. DOI: 10.1038/s41598-022-26264-1
    M. A. Bragin, P. B. Luh, B. Yan, and X. Sun, "A Scalable Solution Methodology for Mixed-Integer Linear Programming Problems Arising in Automation," IEEE Transaction on Automation Science and Engineering, vol. 16, issue 2, April 2019, pp. 531-541. DOI: 10.1109/TASE.2018.2835298.
    M. A. Bragin, "Lagrangian Relaxation for Mixed-Integer Linear Programming: Importance, Challenges, Recent Advancements, and Opportunities," https://arxiv.org/abs/2301.00573 

# Implementations

The implementations provided in this repository cover several examples from the cited papers:

* `SLBLR_Ex1.zip` - CPLEX OPL implementation of Example 1 from [1,2]. This example demonstrates convergence of multiplier only - one of the key aspects of the method 
* `SLBLR_GAP_d201600.zip` - CPLEX OPL implementation of the generalized assignment problem type D with 20 machines and 1600 jobs, which is Example 2 from [1,2]. This example demonstrates how a separable discrete (ILP) programming problem is solved. Key steps of the implementatin include: subproblem solving, stepsize (*key*) and multiplier update as well as generic heuristics for primal recovery. 
* `SLBLR_Python_Cplex_040223.py` - CPLEX Python implementation of Example 1 from [1,2]. This example demonstrates convergence of multiplier only - one of the key aspects of the method 


# Usage

To use the provided implementations, you need to have the appropriate software installed, such as CPLEX Optimization Studio for OPL files and the CPLEX Python API for Python files.

    Download the desired implementation from this repository.
    Open the file using the corresponding software (e.g., CPLEX Optimization Studio for OPL files).
    Run the code to solve the example problem using the SLBLR method.

# Contributing

If you have any additional examples, improvements, or bug fixes related to the Surrogate Level-Based Lagrangian Relaxation method, feel free to submit a pull request or open an issue.
