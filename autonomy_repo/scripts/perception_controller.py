#!/usr/bin/env python3
from ast import In
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController 
from asl_tb3_msgs.msg import TurtleBotControl,TurtleBotState
from std_msgs.msg import Bool

# In the PerceptionController node, create a boolean variable image_detected and set its value to False (this is not a ROS parameter, just an attribute of the class). In that same node, create a subscriber with a callback function that listens to the /detector_bool topic and sets image_detected to True when a stop sign object has been detected.


class PerceptionController(BaseHeadingController):

    def __init__(self):
        super().__init__("perception_controller")
        self.declare_parameter("active", True)
        self.image_detected = False
        self.create_subscription(Bool, "/detector_bool", self.detector_callback, 10)

    @property
    def active(self) -> bool:
        """ Get real-time parameter value of the active state
        """
        return self.get_parameter("active").value

    def compute_control_with_goal(self,current: TurtleBotState,desired: TurtleBotState) -> TurtleBotControl:
        control = TurtleBotControl()
        if not self.image_detected:
            control.omega = 0.2
        else:
            control.omega = 0.0
        return control

    def detector_callback(self, msg: Bool):
        self.image_detected = msg.data

def main():
    rclpy.init()
    node = PerceptionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()