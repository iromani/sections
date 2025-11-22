#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Int64, Bool

from geometry_msgs.msg import Twist 


class Heartbeat(Node):
    def __init__(self) -> None:
    # initialize base class (must happen before everything else)
        super().__init__("heartbeat")
				
	    # a heartbeat counter
        self.hb_counter = 0

	    # create publisher with: self.create_publisher(<msg type>, <topic>, <qos>)
        self.hb_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # create a timer with: self.create_timer(<second>, <callback>)
        self.hb_timer = self.create_timer(1.0, self.hb_callback)

        self.motor_sub = self.create_subscription(Bool, "/kill", self.kill_callback, 10)


    def hb_callback(self) -> None:
        """
        Heartbeat callback triggered by the timer
        """
        # construct heartbeat message
        # msg = Int64()
        # msg.data = self.hb_counter

        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 0.5 # rad/sec

        # publish heartbeat counter
        self.hb_pub.publish(msg)

	    # increment counter
        self.hb_counter += 1

    def kill_callback(self, msg: Bool) -> None:
        """
        /kill subscription callback. If a True is received,
        stop the timer and publish a zero Twist to /cmd_vel.
        """
        if msg.data:
            self.get_logger().info("Received kill signal: stopping heartbeat and publishing zero Twist.")
            # stop periodic publications
            if self.hb_timer is not None:
                self.hb_timer.cancel()
                self.hb_timer = None

            # publish zero command to stop the robot
            stop_msg = Twist()
            # linear and angular are zero by default, but set explicitly
            stop_msg.linear.x = 0.0
            stop_msg.linear.y = 0.0
            stop_msg.linear.z = 0.0
            stop_msg.angular.x = 0.0
            stop_msg.angular.y = 0.0
            stop_msg.angular.z = 0.0

            self.hb_pub.publish(stop_msg)
        else:
            # Optional: if you want to restart on False, implement here.
            self.get_logger().info("Received kill=False; no action taken.")

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = Heartbeat()  # instantiate the heartbeat node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    node.destroy_node()
    rclpy.shutdown()    # cleanly shutdown ROS2 context
