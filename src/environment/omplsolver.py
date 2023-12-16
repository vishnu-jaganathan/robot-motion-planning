from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

import numpy as np
import math

class PathLengthWorkspaceObjective(ob.PathLengthOptimizationObjective):
    def __init__(self, si, robotsp) -> None:
        super().__init__(si)
        self.robotsp = robotsp

    def motionCost(self, s1, s2):
        s1_w = np.asarray(self.robotsp.get_robot_points(s1))
        s2_w = np.asarray(self.robotsp.get_robot_points(s2))
        cost_val = np.linalg.norm(s2_w - s1_w)
        return ob.Cost(cost_val)

    def motionCostHeuristic(self, s1, s2):
        return self.motionCost(s1, s2)

class OMPLSolver():
    def __init__(self, non_ompl_validitychecker = None, dim=2, bounds_vec=[[0,0],[1,1]], max_action_perdim=0.25, robotsp=None) -> None:
        # create an state space
        self.use_simple_setup = True
        self.dim = dim
        self.space = ob.RealVectorStateSpace(self.dim)
        self.timelimit = 0.1
        self.planner_name = 'BITstar'

        # set lower and upper bounds
        assert dim == len(bounds_vec[0]) == len(bounds_vec[1])
        bounds = ob.RealVectorBounds(self.dim)
        for i in range(self.dim):
            bounds.setLow(i, bounds_vec[0][i])
            bounds.setHigh(i, bounds_vec[1][i])
        self.space.setBounds(bounds)
        self.max_action_perdim = max_action_perdim
        # self.space.setLongestValidSegmentFraction(0.34)  # 0.25*sqrt(2)  # useless

        # set the planner
        self.si = ob.SpaceInformation(self.space)
        self.non_ompl_validitychecker = non_ompl_validitychecker
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
        self.si.setStateValidityCheckingResolution(0.005)  # NOTE: fraction of state-space extent NOT path segment
        self.si.setup()

        # create a simple setup object
        if self.use_simple_setup:
            self.ss = og.SimpleSetup(self.si)
            self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
            optimization_obj = ob.PathLengthOptimizationObjective(self.si)
            # optimization_obj = PathLengthWorkspaceObjective(self.si, robotsp)
            self.ss.setOptimizationObjective(optimization_obj)
        else:
            # set problem definition
            self.optimization_obj = ob.PathLengthOptimizationObjective(self.si)
            self.pdef = ob.ProblemDefinition(self.si)
            self.pdef.setOptimizationObjective(self.optimization_obj)

        # set planner
        if self.planner_name == "RRTstar": self.planner = og.RRTstar(self.si)
        elif self.planner_name == "ABITstar": self.planner = og.ABITstar(self.si)
        elif self.planner_name == 'BITstar': self.planner = og.BITstar(self.si, 'kBITstar')
        elif self.planner_name == "RRTsharp": self.planner = og.RRTsharp(self.si)
        elif self.planner_name == "PRM": self.planner = og.PRM(self.si)
        else: raise ValueError('planner name not defined')

        # self.planner.setRange(self.max_action_perdim * math.sqrt(self.dim))  #NOTE: only works with RRTstar
        if self.use_simple_setup:
            self.ss.setPlanner(self.planner)
        
        # if want less information
        ou.setLogLevel(ou.LOG_WARN)
    
    def setStartGoal(self, start_st=None, goal_st=None):
        """Sets the start and goal for the ompl planner
        """
        start = ob.State(self.space)
        goal = ob.State(self.space)
        # we can pick a random start and state...
        start.random()
        goal.random()

        if start_st is not None and goal_st is not None:
            for i in range(self.space.getDimension()):
                start[i] = start_st[i]
                goal[i] = goal_st[i]
        if self.use_simple_setup: 
            self.ss.setStartAndGoalStates(start, goal)
        else:
            self.pdef.setStartAndGoalStates(start, goal)

    def isStateValid(self, state):
        """Funciton to act as state validity checker 

        Args:
            state (list): state to check for validity

        Returns:
            bool: True if valid
        """
        if self.non_ompl_validitychecker is not None:
            state_list = np.empty(self.dim)
            for i in range(self.dim):
                state_list[i] = state[i]
            return self.non_ompl_validitychecker(state_list)
        
        else: # somthing random
            return state[0] < 1.0
    
    def plan(self):
        """Plan a valid solution
        """
        solve_status = None
        time_cond = ob.timedPlannerTerminationCondition(self.timelimit)
        pdef = self.ss.getProblemDefinition() if self.use_simple_setup else self.pdef
        convergence_cond = ob.CostConvergenceTerminationCondition(pdef, solutionsWindow=5, epsilon=0.05)
        terminating_cond = ob.plannerOrTerminationCondition(time_cond, convergence_cond)
        if self.use_simple_setup:
            self.ss.clear()
            solve_status = self.ss.solve(terminating_cond)
        else:
            self.planner.clear()
            self.planner.setProblemDefinition(self.pdef)
            self.planner.setup()
            solve_status = self.planner.solve(terminating_cond)
        
        solution = []
        if solve_status.asString() == 'Exact solution':
            sol_path = None
            if self.use_simple_setup:
                self.ss.simplifySolution()
                sol_path = self.ss.getSolutionPath()
            else:
                sol_path = self.pdef.getSolutionPath()
            # extra info:
            # print('planner: ', self.planner.getName())
            # print('cost is: {:.3}'.format(sol_path.cost(self.ss.getOptimizationObjective()).value()))
            # print('solution lenth is:', sol_path.getStateCount())
            
            # print('before')
            for i in range(sol_path.getStateCount()):
                state = sol_path.getState(i)
                state_list = []
                for j in range(self.space.getDimension()):
                    state_list.append(state[j])
                # print(state_list)
                solution.append(state_list)
            
            ### limit max path length
            i = 0 
            max_act_val = self.max_action_perdim * math.sqrt(self.dim)
            while i < len(solution) - 1:
                action = np.array(solution[i + 1]) - np.array(solution[i])
                clipped_action = np.clip(action, -self.max_action_perdim, self.max_action_perdim)
                if not np.array_equal(action, clipped_action):
                    action_mag = np.linalg.norm(action)
                    alpha = max_act_val if action_mag > max_act_val else action_mag
                    new_state = np.array(solution[i]) + alpha * action / action_mag
                    solution.insert(i+1, new_state.tolist())
                i += 1
            return solution
        return solution
 
 
if __name__ == "__main__":
    solver = OMPLSolver()
    solver.setStartGoal([.1, .1], [.8, .8])
    # solver.setStartGoal()     # choose randomly
    print(solver.plan(0.01))

    solver.setStartGoal([.2, .1], [.5, .8])
    print(solver.plan(0.01))