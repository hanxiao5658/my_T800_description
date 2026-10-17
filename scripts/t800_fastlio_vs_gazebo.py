#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare FAST-LIO IMU/body odometry to Gazebo MID360 link ground truth.

Use in ROS1 Noetic A2sim after both LiDAR/IMU and FAST-LIO are stable.
Only reports relative trajectories (initial pose and heading removed).
No Gazebo state or robot commands are written.
"""
import math
import threading

import rospy
from gazebo_msgs.msg import LinkStates
from nav_msgs.msg import Odometry


def yaw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def local_delta(p, p0, initial_yaw):
    dx, dy = p[0] - p0[0], p[1] - p0[1]
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    return c*dx + s*dy, -s*dx + c*dy


def position_and_orientation(pose):
    p, q = pose.position, pose.orientation
    return ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))


class Monitor:
    def __init__(self):
        self.gt_link = rospy.get_param('~gt_link', 't800::mid360_link')
        self.root_link = rospy.get_param('~root_link', 't800::dummy_root')
        self.odom_topic = rospy.get_param('~odom_topic', '/Odometry')
        self.report_hz = float(rospy.get_param('~report_hz', 2.0))
        self.lock = threading.Lock()
        self.latest_gt = None
        self.latest_root = None
        self.latest_est = None
        self.baseline = None
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.gt_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)

    def gt_cb(self, msg):
        try:
            si, ri = msg.name.index(self.gt_link), msg.name.index(self.root_link)
        except ValueError:
            rospy.logwarn_throttle(5, 'GT links not found; expecting %s and %s',
                                   self.gt_link, self.root_link)
            return
        gt = position_and_orientation(msg.pose[si])
        root = position_and_orientation(msg.pose[ri])
        with self.lock:
            self.latest_gt, self.latest_root = gt, root

    def odom_cb(self, msg):
        if not msg.pose.pose.orientation.w and not any(
            (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z)):
            return  # invalid all-zero quaternion
        est = position_and_orientation(msg.pose.pose)
        with self.lock:
            self.latest_est = est

    def run(self):
        rospy.loginfo('Comparing Gazebo %s, root %s, FAST-LIO %s',
                      self.gt_link, self.root_link, self.odom_topic)
        rospy.loginfo('Start with robot stationary and FAST-LIO initialized; then rotate slowly.')
        rate = rospy.Rate(max(self.report_hz, 0.2))
        while not rospy.is_shutdown():
            with self.lock:
                gt, root, est = self.latest_gt, self.latest_root, self.latest_est
            if not (gt and root and est):
                rospy.logwarn_throttle(5, 'Waiting for Gazebo LinkStates and FAST-LIO Odometry')
                rate.sleep()
                continue
            if self.baseline is None:
                self.baseline = (gt, root, est)
                rospy.loginfo('Initial poses captured. Relative comparison begins.')
            gt0, root0, est0 = self.baseline
            dg = local_delta(gt[0], gt0[0], yaw(gt0[1]))
            dr = local_delta(root[0], root0[0], yaw(root0[1]))
            de = local_delta(est[0], est0[0], yaw(est0[1]))
            gt_yaw = wrap(yaw(gt[1]) - yaw(gt0[1]))
            est_yaw = wrap(yaw(est[1]) - yaw(est0[1]))
            err = math.hypot(de[0]-dg[0], de[1]-dg[1])
            yaw_err = wrap(est_yaw - gt_yaw)
            rospy.loginfo(
                '\n Gazebo sensor dXY=(%+.3f,%+.3f)m; root dXY=(%+.3f,%+.3f)m'
                '\n FAST-LIO     dXY=(%+.3f,%+.3f)m; deltaXY error=%.3fm'
                '\n deltaYaw Gazebo=%+.1fdeg FAST-LIO=%+.1fdeg; yaw error=%+.1fdeg',
                dg[0], dg[1], dr[0], dr[1], de[0], de[1], err,
                math.degrees(gt_yaw), math.degrees(est_yaw), math.degrees(yaw_err))
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('t800_fastlio_vs_gazebo', anonymous=True)
    Monitor().run()
