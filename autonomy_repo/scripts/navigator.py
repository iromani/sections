#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg
import scipy
import rclpy
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl,TurtleBotState
from asl_tb3_lib.navigation import BaseNavigator,TrajectoryPlan, NavMode
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from scipy.interpolate import splev
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from astar import AStar
import typing as T
import time
from scipy.signal import convolve2d
from enum import Enum
import heapq
from rclpy import Parameter
from matplotlib.colors import ListedColormap, BoundaryNorm

   
class Navigator(BaseNavigator):
    def __init__(self, kpx: float = 1.5, kpy: float=1.5, kdx: float=1., kdy: float=1.) -> None:
        super().__init__()
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        self.declare_parameter("kp", 2.)
        self.kp = 2. # The above was failing
        self.V_PREV_THRES = 0.0001
        self.set_parameters([Parameter('plan_resolution', Parameter.Type.DOUBLE, 0.02)])
        self.reset()
        self.solve_call_no = -1
        self.sensor_range = 10.

    def reset(self) -> None:
        self.V_prev = 0.
        self.t_prev = 0.

    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs:
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d = splev(t, plan.path_x_spline, der=0)
        y_d = splev(t, plan.path_y_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        yd_d = splev(t, plan.path_y_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        ydd_d = splev(t, plan.path_y_spline, der=2)

        # avoid singularity
        V_prev = self.V_prev
        if abs(V_prev) < self.V_PREV_THRES:
            V_prev = self.V_PREV_THRES

        th = state.theta
        xd = V_prev*np.cos(th)
        yd = V_prev*np.sin(th)

        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-state.x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-state.y) + self.kdy*(yd_d-yd)])

        # compute real controls
        J = np.array([[np.cos(th), -V_prev*np.sin(th)],
                          [np.sin(th), V_prev*np.cos(th)]])
        a, om = linalg.solve(J, u)
        V = V_prev + a*dt
        ########## Code ends here ##########

        # apply control limits
        control = TurtleBotControl(v=V,omega=om)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V

        return control

    def compute_heading_control(self,current: TurtleBotState,desired: TurtleBotState) -> TurtleBotControl:
        eps = wrap_angle(wrap_angle(desired.theta) - wrap_angle(current.theta))
        omega = self.kp * eps
        control = TurtleBotControl()
        #self.get_logger().warn(f"compute_heading_control eps: {eps}, omega: {omega}")
        control.omega = omega
        return control

    def compute_smooth_plan(self, astar, v_desired=0.15, spline_alpha=0.05) -> TrajectoryPlan:
        # Ensure path is a numpy array
        path = np.asarray(astar.path)

        # Compute and set the following variables:
        #   1. ts: 
        #      Compute an array of time stamps for each planned waypoint assuming some constant 
        #      velocity between waypoints. 
        #
        #   2. path_x_spline, path_y_spline:
        #      Fit cubic splines to the x and y coordinates of the path separately
        #      with respect to the computed time stamp array.
        #      Hint: Use scipy.interpolate.splrep
        
        ts = [0]
        for i in range(1,len(path)):
            ts.append(ts[i-1] + astar.distance(path[i],path[i-1]) / v_desired)
        
        return TrajectoryPlan(
            path=path,
            path_x_spline=scipy.interpolate.splrep(ts,path[:,0],s=spline_alpha),
            path_y_spline=scipy.interpolate.splrep(ts,path[:,1],s=spline_alpha),
            duration=ts[-1],
    )
     
    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting

        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (DetOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon

        Returns:
            T.Optional[TrajectoryPlan]:
        """
        self.get_logger().warn("compute_trajectory_plan Occupancy grid:")
        self.get_logger().warn(f"compute_trajectory_plan occ.probs:\n{occupancy.probs}")
        self.get_logger().warn(f"compute_trajectory_plan size_xy = {occupancy.size_xy}")
        self.get_logger().warn(f"compute_trajectory_plan origin_xy = {occupancy.origin_xy}")
        occupancy.window_size = 12
        self.get_logger().warn(f"compute_trajectory_plan window_size = {occupancy.window_size}")
        self.get_logger().warn(f"compute_trajectory_plan thresh = {occupancy.thresh}")
        xy_lower = occupancy.origin_xy
        xy_no_cells = occupancy.size_xy
        grid_size = np.array([resolution * xy_no_cells[0], resolution * xy_no_cells[1]])
        xy_upper = xy_lower + grid_size
 
        # ### Starting and final positions

        x_init = (state.x, state.y)
        x_goal = (goal.x, goal.y)

        # ### Run A* planning
        # What to do with the parameter "horizon"?
        astar = AStar(xy_lower, xy_upper, x_init, x_goal, occupancy, self.get_logger(), 
                      resolution=resolution)
        self.get_logger().warn(f"compute_trajectory_plan x_init: {x_init}, x_goal: {x_goal}")
        self.get_logger().warn(f"compute_trajectory_plan will start astar search")
        #time.sleep(10000)
        self.solve_call_no += 1
        if not astar.solve(self.solve_call_no) or len(astar.path) < 4:
            return None
        self.reset()

        plt.rcParams['figure.figsize'] = [5, 5]
        #astar.plot_path()
        astar.plot_tree()
        return self.compute_smooth_plan(astar) 
    
    def explore(self, occupancy, current_state, resolution):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """

        window_size = 13 # e.g., if resolution is 0.1 m/cell, window_size is 10 cells (1 m x 1 m window)    
        unknown_masked = occupancy.probs.clip(max=0.0)
        unknown_window = -np.ones((window_size,window_size))
        unknown_pct_window = convolve2d(unknown_masked,unknown_window,mode='same') / window_size**2
        occupied_masked = np.where(occupancy.probs < 0.5, 0., 1.)
        full_window = np.ones((window_size,window_size))
        occupied_count = convolve2d(occupied_masked,full_window,mode='same')
        known_unoccupied_masked = (1+unknown_masked) * (1-occupied_masked) # element-wise product
        known_unoccupied_pct = convolve2d(known_unoccupied_masked,full_window,mode='same') / window_size**2
        rows1,cols1 = np.where(unknown_pct_window >= .2)
        selection1 = set(zip(cols1,rows1))
        rows2,cols2 = np.where(occupied_count < 1)
        selection2 = set(zip(cols2,rows2)) # less than one cell occupied
        rows3,cols3 = np.where(known_unoccupied_pct >= .3)
        selection3 = set(zip(cols3,rows3))
        frontier_grid_points = list(sorted(selection3.intersection(selection2.intersection(selection1))))
        return [self.occupancy.grid2state(np.array(p)) for p in frontier_grid_points]
    
    def center_of_mass(self, points):
        if not points:
            return None
        # Compute the center of mass as the mean position of the frontier states
        com_x = np.mean([s[0] for s in points])
        com_y = np.mean([s[1] for s in points])
        return np.array((com_x, com_y))
    
    def compute_goals_to_explore(self) -> list:
        # Compute a goal to explore unknown areas of the map 
        resolution = self.occupancy.resolution
        current_state = (self.state.x, self.state.y)
        frontier_states = self.explore(self.occupancy, current_state, resolution)
        if not frontier_states:
            return []
        com_state = self.center_of_mass(frontier_states)
        dists = [(np.linalg.norm(com_state-x),i) for i,x in enumerate(frontier_states)]
        #self.get_logger().warn(f"DEBUG: compute_goals_to_explore, dists={dists}")
        # Plot Stochastic Occupancy grid with frontier to explore
        fig,ax = plt.subplots(1)
        cmap = ListedColormap(['purple', 'green', 'yellow'])
        bounds = [-1.0, 0.0, 0.5, 1.0]   # boundaries between bins
        norm = BoundaryNorm(bounds, cmap.N, clip=True)
        ax.imshow(self.occupancy.probs, origin='lower',cmap=cmap,norm=norm)
        x_grid = self.occupancy.state2grid(np.array(current_state))
        ax.plot(x_grid[0], x_grid[1], 'r*')
        grid_frontier_xy = self.occupancy.state2grid(np.array(frontier_states))
        ax.plot(grid_frontier_xy[:,0], grid_frontier_xy[:,1], 'b+')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        plt.savefig(f"occ_{self.solve_call_no+1:03d}_f.png")
        return [frontier_states[i] for (d,i) in sorted(dists)]
    
    def update_known_grid_points(self):
        if self.state is None:
            return
        state = np.array([self.state.x,self.state.y])
        for i in range(self.occupancy.size_xy[1]):
            for j in range(self.occupancy.size_xy[0]):
                xy = self.occupancy.grid2state(np.array([j,i]))
                x = xy[0]
                y = xy[1]
                if np.linalg.norm(np.array((x,y)-state)) <= 1.:
                    if self.occupancy.probs[i,j] < 0.5:
                        self.occupancy.probs[i,j] = 0.
                        #self.get_logger().warn(f"DEBUG: update_known_grid_points ({x},{y})/({i},{j}), prob={self.occupancy.probs[i,j]}")
        fig,ax = plt.subplots(1)
        cmap = ListedColormap(['purple', 'green', 'yellow'])
        bounds = [-1.0, 0.0, 0.5, 1.0]   # boundaries between bins
        norm = BoundaryNorm(bounds, cmap.N, clip=True)
        ax.imshow(self.occupancy.probs, origin='lower',cmap=cmap,norm=norm)
        grid_xy = self.occupancy.state2grid(state)
        ax.plot(grid_xy[0], grid_xy[1], 'r*')
        #ax.plot(frontier_states[:,0], frontier_states[:,1], 'b+')
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        plt.savefig(f"occ_update.png")


    def map_callback(self, msg: OccupancyGridMsg) -> None:
        super().map_callback(msg)
        self.update_known_grid_points()
        self.get_logger().warn(f"DEBUG: map_callback: mode: {self.mode}")
        if self.state is None:
            self.get_logger().warn("DEBUG: map_callback: self.state is None")
        elif self.mode == NavMode.IDLE or self.mode == NavMode.PARK\
            or (self.goal is not None and not self.occupancy.is_free(np.array((self.goal.x,self.goal.y))))\
            or (self.mode == NavMode.TRACK and not self.is_planned):
                # ^ Sometimes a new map reveals that the goal is actually occuppied
            goals = self.compute_goals_to_explore()
            if not goals:
                self.get_logger().warn("DEBUG: map_callback: No more goals to explore, remaining in IDLE")
                return
            for goal in goals:
                self.get_logger().warn(f"Checking goal {goal}")
                if self.occupancy.is_free(goal):
                    self.get_logger().warn(f"DEBUG compute_goal_to_explore, x is free: {goal}")
                    self.replan(TurtleBotState(x=goal[0],y=goal[1],theta=0.0))
                    if self.is_planned:
                        return
            self.get_logger().warn("Explorer found no unoccupied frontier states")
        else:
            self.get_logger().warn("map_callback: No action")
        return None

    # def explore_callback(self,
    #     occupancy: StochOccupancyGrid2D,
    #     resolution: float, current_state: T.Tuple[float, float]):
    #     # Call to explore function
    #     state_xy = self.explore(occupancy)
    #     print(resolution)
    #     grid_xy = occupancy.state2grid(state_xy)

    #     # Plot Stochastic Occupancy grid with frontier to explore
    #     fig,ax = plt.subplots(1)
    #     ax.imshow(occupancy.probs, origin='lower')
    #     ax.plot(current_state[0]/resolution, current_state[1]/resolution, 'r*')
    #     ax.plot(grid_xy[:,0], grid_xy[:,1], 'b+')
    #     ax.set_ylabel('y')
    #     ax.set_xlabel('x')
    #     #ax.set_yticklabels([])
    #     #ax.set_xticklabels([])
    #     plt.show()        

def main():
    rclpy.init()
    node = Navigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()