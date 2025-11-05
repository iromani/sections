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
from astar import AStar
import typing as T
import time

class Navigator(BaseNavigator):
    def __init__(self, kpx: float = 1., kpy: float=1., kdx: float=0.1, kdy: float=0.1) -> None:
        super().__init__()
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        self.declare_parameter("kp", 2.)
        self.kp = 2. # The above was failing
        self.V_PREV_THRES = 0.0001
        self.reset()

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
        self.get_logger().warn(f"compute_heading_control eps: {eps}, omega: {omega}")
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
        self.get_logger().warn(f"compute_trajectory_plan state_to_grid: {occupancy.state2grid(np.array(x_init))}")
        self.get_logger().warn(f"compute_trajectory_plan will start astar search")
        #time.sleep(10000)
        if not astar.solve() or len(astar.path) < 4:
            return None

        self.reset()

        plt.rcParams['figure.figsize'] = [5, 5]
        #astar.plot_path()
        astar.plot_tree()
        return self.compute_smooth_plan(astar) 

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