import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import plot_line_segments
from matplotlib.colors import ListedColormap, BoundaryNorm

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy,logger,
                 resolution=1,distance='normL2'):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init       
        self.logger = logger              
        self.logger.warn(f"astar resolution {resolution}")
        self.logger.warn(f"astar x_init {x_init}")
        self.logger.warn(f"astar x_goal {x_goal}")
        self.x_init = x_init    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state
        # FOR DEBUG
        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.logger = logger
        self.logger.warn(f"astar adding root {self.x_init}")
        self.logger.warn(f"astar adding goal {self.x_goal}")
        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.distance = eval(f"self.{distance}")
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x, obs_margin):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
            margin: margin for obstacle avoidance
        Output:
            Boolean True/False
        """
        return self.statespace_lo[0] < x[0] < self.statespace_hi[0]\
            and self.statespace_lo[1] < x[1] < self.statespace_hi[1]\
            and self.occupancy.is_free(np.array(x)) # The stochastic grid has no margin

    # The name of the distance function was change to allow generalization
    def normL2(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        """
        return np.linalg.norm(np.array(x1)-np.array(x2))

    def normL1(self, x1, x2):
        """
        Computes the L1 norm between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float L1 norm

         """
        return np.sum(np.abs(np.array(x1)-np.array(x2)))

    def normLinfinity(self, x1, x2):
        """
        Computes the L-infinity norm between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float L-infinity norm

         """
        return np.max(np.abs(np.array(x1)-np.array(x2)))

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            An array that represents the closest point to x on the discrete state grid
        """
        return np.array((
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        ))

    def get_neighbors(self, x, obstacle_margin=0.1):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES
        """
        # Filling neighbors in clockwise order, starting from North
        all_neighs = [
            self.snap_to_grid((x[0],x[1]+self.resolution)),
            self.snap_to_grid((x[0]+self.resolution,x[1]+self.resolution)),
            self.snap_to_grid((x[0]+self.resolution,x[1])),
            self.snap_to_grid((x[0]+self.resolution,x[1]-self.resolution)),
            self.snap_to_grid((x[0],x[1]-self.resolution)),
            self.snap_to_grid((x[0]-self.resolution,x[1]-self.resolution)),
            self.snap_to_grid((x[0]-self.resolution,x[1])),
            self.snap_to_grid((x[0]-self.resolution,x[1]+self.resolution))
        ]
        return [n for n in all_neighs if self.occupancy.is_free(n)]



    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [tuple(self.x_goal)]
        current = path[0]
        while current != self.x_init:
            path.insert(0,self.came_from[current])
            current = path[0]
        self.logger.warn(f"reconstruct_path returning with length {len(path)}")   
        return path

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        #self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != tuple(self.x_goal)]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != tuple(self.x_goal)]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

    def solve(self,call_no,obstacle_margin=0.05):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found
        """
        if not self.occupancy.is_free(self.x_goal):
            self.logger.warn(f"DEBUG: astar.solve: Goal is occupied {self.x_goal}, search will return False")
            return False
        visit_count = 0
        self.logger.warn(f"DEBUG: open_set={self.open_set}, call_no={call_no}")
        # Plot Stochastic Occupancy grid with frontier to explore
        fig,ax = plt.subplots(1)
        cmap = ListedColormap(['purple', 'green', 'yellow'])
        bounds = [-1.0, 0.0, 0.5, 1.0]   # boundaries between bins
        norm = BoundaryNorm(bounds, cmap.N, clip=True)
        ax.imshow(self.occupancy.probs, origin='lower',cmap=cmap,norm=norm)
        x_curr = self.snap_to_grid(self.find_best_est_cost_through())
        x_grid = self.occupancy.state2grid(x_curr)
        ax.plot(x_grid[0], x_grid[1], 'r*')
        x_goal_grid = self.snap_to_grid(self.x_goal)
        ax.plot(x_goal_grid[0], x_goal_grid[1], 'r^')
        #ax.plot(frontier_states[:,0], frontier_states[:,1], 'b+')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        plt.savefig(f"occ_{call_no:03d}.png")
        while len(self.open_set) > 0:
            visit_count += 1
            #self.logger.warn(f"astar.solve() visit {visit_count}")
            x_curr = self.snap_to_grid(self.find_best_est_cost_through())
            if np.all(x_curr == self.x_goal):
                self.logger.warn(f"astar.solve() Returning True after {visit_count} visits")
                self.path = self.reconstruct_path()
                return True
            x_t = tuple(x_curr) # x_curr[0],x_curr[1])
            self.open_set.remove(x_t)
            self.closed_set.add(x_t)
            if visit_count == 1:
                self.logger.warn(f"DEBUG: x_t={x_t}, len(neighbors)={len(self.get_neighbors(x_t,obstacle_margin))}")
                self.logger.warn(f"DEBUG: is_free(x_t)={self.occupancy.is_free(np.array(x_t))}")
            for x_neigh in self.get_neighbors(x_t,obstacle_margin):
                x_n = tuple(x_neigh)
                if x_n in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_t] + self.distance(x_t,x_n)
                #self.logger.warn(f"astar.solve() tentative_cost_to_arrive = {tentative_cost_to_arrive}")
                if x_n not in self.open_set:
                    self.open_set.add(x_n)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_n]:
                    continue
                self.came_from[x_n] = x_t
                self.cost_to_arrive[x_n] = tentative_cost_to_arrive
                self.est_cost_through[x_n] = tentative_cost_to_arrive + self.distance(x_n,self.x_goal)
        self.logger.warn(f"astar.solve() Returning False after {visit_count} visits")
        return False
            

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x, margin=0.1):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - margin and \
               x[0] <= obs[1][0] + margin and \
               x[1] >= obs[0][1] - margin and \
               x[1] <= obs[1][1] + margin:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))
