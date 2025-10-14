#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Int64, Bool


class Heartbeat(Node):
    def __init__(self) -> None:
    # initialize base class (must happen before everything else)
        super().__init__("heartbeat")
				
	    # a heartbeat counter
        self.hb_counter = 0

	    # create publisher with: self.create_publisher(<msg type>, <topic>, <qos>)
        self.hb_pub = self.create_publisher(Int64, "/heartbeat", 10)
        
        # create a timer with: self.create_timer(<second>, <callback>)
        self.hb_timer = self.create_timer(1.0, self.hb_callback)

    def hb_callback(self) -> None:
        """
        Heartbeat callback triggered by the timer
        """
        # construct heartbeat message
        msg = Int64()
        msg.data = self.hb_counter

        # publish heartbeat counter
        self.hb_pub.publish(msg)

	    # increment counter
        self.hb_counter += 1

    def health_callback(self, msg: Bool) -> None:
        """
        Sensor health callback triggered by subscription
        """
        if not msg.data:
            self.get_logger().fatal("Heartbeat stopped")
            self.hb_timer.cancel()

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Heartbeat()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context
