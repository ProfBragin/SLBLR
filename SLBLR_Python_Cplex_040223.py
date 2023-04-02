# ------------------------------------------------------------------------------------------------ #
# If you are going to use this code for publication purposes, please cite the following papers:    #
# ------------------------------------------------------------------------------------------------ #
# 1. M. A. Bragin, and E. L. Tucker, "Surrogate 'Level-Based' Lagrangian Relaxation for            #
#    Mixed-Integer Linear Programming," Scientific Reports, Volume 12, Issue 1, 22417, 2022.       #
#    DOI: 10.1038/s41598-022-26264-1                                                               #
#                                                                                                  #
# 2. M. A. Bragin, P. B. Luh, B. Yan, and X. Sun, "A Scalable Solution Methodology for             #
#    Mixed-Integer Linear Programming Problems Arising in Automation," IEEE Transaction on         #
#    Automation Science and Engineering, vol. 16, issue 2, April 2019, pp. 531-541.                #
#    DOI: 10.1109/TASE.2018.2835298.                                                               #
#                                                                                                  #
# For better understanding of the Lagrangian methods, the following reference is recommended:      #
# 3. M. A. Bragin, "Lagrangian Relaxation for Mixed-Integer Linear Programming: Importance,        #
#    Challenges, Recent Advancements, and Opportunities," https://arxiv.org/abs/2301.00573         #
#                                                                                                  #
# This code implements Example 1 from both papers, which serves as a justification for citing      #
# these works in any publication that uses this code.                                              #
# ------------------------------------------------------------------------------------------------ #

# Import necessary libraries and modules
import numpy as np                  # For numerical operations and manipulations
import time                         # For measuring execution time
import docplex.mp.model as cpx      # For using CPLEX optimization solver

class Coordinator(object):
    '''
    Coordinator for updating and broadcasting multipliers.
    '''
    def __init__(self, M = 1, nums = 100, threshold = 0.1):
        super().__init__()
        self._M = M
        self.nums = nums
        self.threshold = threshold
        self._k = None
        self._g = None
        self._s = None
        self._lambda = None
        self._solutions = None
        self._surrogate_dual = None
        self._lower_bound = None
        self._dual_gap = None
        self._subproblems = None
        self._q_star = 15.6
        self._q_lev = 20
        self._q_candidate_lev = -1000
        n_rows = 1000
        n_cols = 2
        self._s_lambda = [[None for j in range(n_cols)] for i in range(n_rows)]
        self._nbsubproblems = 6

    def init(self, para, subproblems):
        '''
        Initialize parameters, multipliers and stepsize, all subproblems are required to be initialized 
        and started in advance.

        para: Dict to describe the MILP problem
                    {'objective': [a1,a2,...,aI],
                    'constraints': [[c11,c12,...,c1I],[c21,c22,...,c2I],...,[cm1,cm2,...cmI]]
                    'rhs': [b1,b2,....,bm]}

        subproblems: Iteriable predifined subproblems [sub1, sub2, .... , subI]
        '''
        # Initialize objective coefficients, constraint coefficients, and rhs of constraints from the given parameters
        self.objective = para['objective']
        self.constraints = para['constraints']
        self.bs = para['rhs']

        # Initialize the subproblems and solutions
        self._subproblems = subproblems
        self._solutions = [None for x in range(len(self._subproblems))]

        # Calculate the initial lambda values
        self._k, self._lambda = self._init_lambda()

        # Run each subproblem and store the result in the solutions list
        for index, sub in enumerate(self._subproblems):
            self._solutions[index] = sub.run((self._k, self._lambda))

        # Reset the k parameter
        self._k = 0

        # Update the subgradient with the initial solution x^0
        self._g = self._get_g()

        # Update the surrogate dual with the initial solution x^0 and the initial subgradient g^0
        self._surrogate_dual = self._get_surrogate_dual()

        # Set the initial stepsize
        self._s = 0.01

    def run(self):
        '''
        Start and keep updating multipliers.
        '''
        self.logs = []
        start_time = time.time()
        # Ignore the stop criteria for debugging purposes
        self._dual_gap = 99999
        # Start from k = 1
        while self._k < self.nums and self._dual_gap >= self.threshold:
            # Update lambda
            index, x = sequence[self._k], self._subproblems[sequence[self._k]].run((self._k, self._lambda))
            self._solutions[index] = x
            self._g = self._get_g()  # Compute the gradient
            self._surrogate_dual = self._get_surrogate_dual()  # Compute the surrogate dual value
            self._k = self._k + 1  # Increment the iteration counter
            self._s = self._get_s()  # Compute the step size
            self._lambda = self._lambda + self._s * self._g  # Update the Lagrangian multipliers
            self._project_lambda()  # Project the multipliers onto the feasible set (positive orthant)

            # Calculate the time difference for the current iteration
            current_time_diff = time.time() - start_time

            # Print the progress log
            print(f"Step {self._k} Stepsize {self._s:.6f} Multipliers {self._lambda[0]:.6f}\t{self._lambda[1]:.6f} "
                  f"Surrogate dual {self._surrogate_dual:.6f} Level Value {self._q_lev:.6f} Candidate Lev. Value {self._q_candidate_lev:.6f} "
                  f"Time {current_time_diff}")

        # Calculate the total time taken for the algorithm
        end_time = time.time() - start_time

        
    def _init_lambda(self):
        # Initialize the Lagrangian multipliers vector
        # with a tuple containing -1 and an array of 0.1s
        # with the same length as the number of constraints
        return (-1, np.zeros(shape=len(self.constraints)) + 0.1)
    # The first element of the tuple is set to -1. This value might be problem-specific or chosen based on prior
    # experience with similar optimization problems. It could also be an arbitrary starting point for the optimization algorithm.

    # The second element is a NumPy array filled with 0.1s, with the same length as the number of constraints in the optimization problem.
    # Initializing the Lagrangian multipliers with small positive values helps to ensure that the algorithm starts from a feasible region
    # and promotes exploration of the solution space. Small positive values are used to avoid potential numerical issues or overemphasizing
    # certain constraints in the initial iterations of the algorithm.
    
    def _get_g(self):
        '''
        This method can only be used after updating x
        '''

        # Compute the constraint violations vector g
        return np.array(
            [
                # For each Lagrangian multiplier, calculate the constraint violation
                np.sum(
                    [
                        # Multiply the constraint value by the corresponding solution value
                        self.constraints[i][x] * self._solutions[x]
                        # Iterate through all the subproblems
                        for x in range(len(self._subproblems))
                    ]
                )
                # Add the corresponding b value for the constraint
                + self.bs[i]
                # Iterate through all the Lagrangian multipliers
                for i in range(len(self._lambda))
            ]
        )

    def _get_surrogate_dual(self):
        '''
        This method can only be used after updating x and g!
        '''
        # Calculate the first term (subsystem cost) of the surrogate dual function: sum(a_i * x_i)
        term1 = np.sum([self._solutions[x] * self.objective[x] for x in range(len(self._subproblems))])

        # Calculate the second term - relaxed term of the surrogate dual function: sum(lambda_i * g_i)
        term2 = np.sum([self._lambda[i] * self._g[i] for i in range(len(self._lambda))])

        # Return the surrogate dual value (Lagrangian function evaluated at subproblem solutions)
        return term1 + term2

    def _get_s(self):
        # Check if k equals to 0
        if self._k == 0:
            return self._s

        # Store the current values of self._lambda in self._s_lambda
        self._s_lambda[self._M][0] = self._lambda[0]
        self._s_lambda[self._M][1] = self._lambda[1]

        # Create a CPLEX model instance for the "multiplier divergence detection" problem
        model4 = cpx.Model(name="MIP Model")

        # Define decision variables for the "multiplier divergence detection" problem
        lambda1_var = model4.continuous_var(lb=0, name="lambda1")
        lambda2_var = model4.continuous_var(lb=0, name="lambda2")

        # Add constraints for the "multiplier divergence detection" problem
        for m in range(1, self._M - 1):
            model4.add_constraint(
                1 * (self._s_lambda[m][0] - lambda1_var) * (self._s_lambda[m][0] - lambda1_var) +
                1 * (self._s_lambda[m][1] - lambda2_var) * (self._s_lambda[m][1] - lambda2_var) >=
                (self._s_lambda[m + 1][0] - lambda1_var) * (self._s_lambda[m + 1][0] - lambda1_var) +
                (self._s_lambda[m + 1][1] - lambda2_var) * (self._s_lambda[m + 1][1] - lambda2_var))
            model4.objective_expr = 1
        self._M += 1

        # Update candidate for the level value if necessary
        if (self._q_candidate_lev < 6 * self._s * np.linalg.norm(self._g, ord=2) + self._surrogate_dual):
            self._q_candidate_lev = 6 * self._s * np.linalg.norm(self._g, ord=2) + self._surrogate_dual

        # Solve the CPLEX model
        solution = model4.solve()
        # Check if the problem is infeasible, if so, the level value needs to be updated at the latest
        # candidate level value
        if solution is None:
            if self._q_lev > self._q_candidate_lev:
                self._q_lev = self._q_candidate_lev
                self._q_candidate_lev = -1000
                self._M = 0

        # Update stepsize based on the latest level value and surrogate dual value
        s_new = 1 / 6 / 1.5 * (self._q_lev - self._surrogate_dual) / np.linalg.norm(self._g, ord=2) / np.linalg.norm(
            self._g, ord=2)
        return s_new

    def _project_lambda(self):
        # Project lambda values onto the non-negative orthant
        for i, x in enumerate(self._lambda):
            if x < 0:
                self._lambda[i] = 0


class Solver():
    '''
    Solver for solving subproblems and sending them back to coordinator.
    
    index: int value indicates the number of subproblem.
    '''
    def __init__(self, index):
        self._lambda = None
        self._x = None
        self._k = 0
        self._s_lambda = []
        
        self._lambda_index = -9999
        
        self._index = index
        self._name = 'x%d'%self._index
        
    def init(self, para, coordinator):
        '''
        para: Dict to describe the subproblem
              objective_g is generated from the objective function of original problem.
              objective_c is generated from the constraints of original problem.
                    {'objective_g': [g1,g2,...,gI],
                    'objective_c': [oc1,oc2,...,ocI],
                    'constraints': [[c11,c12,...,c1I],[c21,c22,...,c2I],...,[cm1,cm2,...cmI]]
                    'bs': [b1,b2,....,bm]}

        coordinator: Predifined coordinator
        '''
        self._k = -1
        self.objective_g = para['objective_g']
        self.objective_c = para['objective_c']
        self.constraints = para['constraints']
        self.bs = para['bs']
        self.ub = para['ub']
        self.lb = para['lb']
        
    def run(self, message):
        if message[0] > self._lambda_index:
            self._lambda = message[1]
            self._lambda_index = message[0]

            ## Solve solution with latest lambda
            self.solve()
            self._k = self._k + 1

            ## simulating the time delay of 5g transmission
            time.sleep(0.001)
            return self._x
        
                
    def solve(self,):
        model = cpx.Model(name="MIP Model")
        model.context.cplex_parameters.threads = 1
        x_var = model.continuous_var(lb = self.lb, ub = self.ub, name = self._name)
        ## pass | set constraints latter
        objective = model.sum([self._lambda[i] * self.objective_c[i] * x_var for i in range(len(self._lambda))]) + self.objective_g * x_var
        model.minimize(objective)
        result = model.solve()
        self._x = result._get_all_values()[0]
    
    def __str__(self):
        return self._name
    

def fix_timestamp(logs):
    fix = list(logs)

    fix_x = [x[0] // 0.1 for x in fix]
    fix_y = [x[1] for x in fix]

    fix = dict()

    for i in range(len(fix_x)):
        fix[fix_x[i]] = fix_y[i]

    # logs = []
    # for key in fix:
    #     logs.append([key / 10, fix[key]])
    # return logs



para = {
    'objective': [1,2,3,1,2,3],
    'constraints': [[-1,-3,-5,-1,-3,-5], [-2,-1.5,-5,-2,-0.5,-1]],
    'rhs': [26, 16]
}
sub_paras = []
for i in range(6):
    sub_para = {
        'objective_g': para['objective'][i],
        'objective_c': [para['constraints'][0][i], para['constraints'][1][i]],
        'constraints': [],
        'bs': [],
        'ub': 3,
        'lb': 0
    }
    sub_paras.append(sub_para)

# Create a sequence of randomly chosen integers from 0 to 5 with a length of 1000
sequence = np.random.randint(0, 6, 1000)

# Calculate the number of times the sequence should be repeated
n_repeats = 1000 // 6

# Create a sequence of deterministically and sequentially repeating integers the range from 0 to 5 for n_repeats times
sequence = np.tile(np.arange(0, 6), n_repeats)

# Initialize a Coordinator object with the length of the sequence as the number of subproblems
coordinator = Coordinator(nums=len(sequence))

subproblems = [Solver(i) for i in range(6)]
for index, subproblem in enumerate(subproblems):
    subproblem.init(sub_paras[index], coordinator)

coordinator.init(para, subproblems)
coordinator.run()
logs = fix_timestamp(coordinator.logs)
logs_all.append(logs)


f = open('SLR_100.txt','w')
f.write(str(logs_all))
f.close()