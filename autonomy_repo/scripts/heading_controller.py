#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController 
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl,TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__()
        self.declare_parameter("kp", 2.)

    @property
    def kp(self) -> float:
        """ Get real-time parameter value of the proportional gain kp

        Returns:
            float: latest parameter value of the gain kp
        """
        return self.get_parameter("kp").value

    def compute_control_with_goal(self,current: TurtleBotState,desired: TurtleBotState) -> TurtleBotControl:
        eps = wrap_angle(wrap_angle(desired.theta) - wrap_angle(current.theta))
        omega = wrap_angle(self.kp * eps)
        control = TurtleBotControl()
        print(f"eps: {eps}, omega: {omega}")
        control.omega = omega
        return control

def main():
    rclpy.init()
    node = HeadingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()