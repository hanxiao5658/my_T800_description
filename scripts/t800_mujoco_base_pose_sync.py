#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hard-sync the Gazebo Classic T800 model pose to MuJoCo world pose.

ROS1 Noetic inside the A2sim container.

Subscribe: /my_ros2/simulation/base_link (geometry_msgs/PoseStamped)
Service:   /gazebo/set_model_state (gazebo_msgs/SetModelState)

This writes the whole Gazebo model's WORLD pose directly, without PID.
It is independent from the existing joint-angle node, which calls
/gazebo/set_model_configuration.

Assumes MuJoCo's pose refers to LINK_BASE in world coordinates, and that the
Gazebo model's root frame (dummy_root -> LINK_BASE) has zero offset.
"""

import math
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class T800MuJoCoBasePoseSync:
    def __init__(self):
        rospy.init_node('t800_mujoco_base_pose_sync', anonymous=False)
        self.input_topic = rospy.get_param(
            '~input_topic', '/my_ros2/simulation/base_link')
        self.service_name = rospy.get_param(
            '~service_name', '/gazebo/set_model_state')
        self.model_name = rospy.get_param('~model_name', 't800')
        self.rate = float(rospy.get_param('~sync_rate', 30.0))
        self.stale_timeout = float(rospy.get_param('~stale_timeout', 0.5))
        self.allow_non_world_frame = bool(rospy.get_param(
            '~allow_non_world_frame', False))
        if not (math.isfinite(self.rate) and self.rate > 0.0):
            raise ValueError('sync_rate must be positive')
        if not (math.isfinite(self.stale_timeout) and self.stale_timeout > 0.0):
            raise ValueError('stale_timeout must be positive')

        self._lock = threading.Lock()
        self._latest = None
        self._last_rx_wall = None
        self._proxy = None
        self._success_reported = False
        self._sub = rospy.Subscriber(
            self.input_topic, PoseStamped, self._on_pose,
            queue_size=1, tcp_nodelay=True)
        rospy.loginfo(
            'T800 world-pose sync: %s -> %s (model=%s, %.1f Hz)',
            self.input_topic, self.service_name, self.model_name, self.rate)

    def _on_pose(self, msg):
        # ROS1 Subscriber callback runs in another thread.
        with self._lock:
            self._latest = msg
            self._last_rx_wall = time.monotonic()

    @staticmethod
    def _pose_is_valid(pose):
        p, q = pose.position, pose.orientation
        vals = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(float(v)) for v in vals):
            return False
        return (q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w) > 1e-12

    def _connect(self):
        if self._proxy is not None:
            return True
        try:
            rospy.wait_for_service(self.service_name, timeout=1.0)
            self._proxy = rospy.ServiceProxy(
                self.service_name, SetModelState, persistent=True)
            rospy.loginfo('Connected to Gazebo: %s', self.service_name)
            return True
        except rospy.ROSException:
            rospy.logwarn_throttle(5.0, 'Waiting for Gazebo service %s', self.service_name)
            return False

    def _sync_once(self):
        with self._lock:
            msg = self._latest
            received_at = self._last_rx_wall
        if msg is None or received_at is None:
            return
        if time.monotonic() - received_at > self.stale_timeout:
            rospy.logwarn_throttle(5.0, 'Stale MuJoCo base pose; skipping Gazebo update')
            return
        if msg.header.frame_id and msg.header.frame_id != 'world':
            if not self.allow_non_world_frame:
                rospy.logwarn_throttle(
                    5.0, 'Pose frame is %s, expected world. No frame transform '
                    'is performed; skipping.', msg.header.frame_id)
                return
        if not self._pose_is_valid(msg.pose):
            rospy.logwarn_throttle(5.0, 'Invalid pose/quaternion; skipping')
            return
        if not self._connect():
            return

        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = 'world'  # Gazebo's inertial world frame
        state.pose.position = msg.pose.position
        # Normalize quaternion as a guard against floating-point drift.
        q = msg.pose.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        state.pose.orientation.x = q.x / norm
        state.pose.orientation.y = q.y / norm
        state.pose.orientation.z = q.z / norm
        state.pose.orientation.w = q.w / norm
        # No source Twist in PoseStamped. Zero velocities reduce physics drift;
        # we directly overwrite pose again on the next update.
        # state.twist remains all zeros.
        try:
            result = self._proxy(model_state=state)
            if not result.success:
                rospy.logwarn_throttle(
                    5.0, 'Gazebo rejected base pose: %s', result.status_message)
            elif not self._success_reported:
                rospy.loginfo('T800 Gazebo base pose hard-sync working')
                self._success_reported = True
        except (rospy.ServiceException, rospy.ROSException) as exc:
            rospy.logwarn_throttle(5.0, 'Gazebo service failed: %s', exc)
            try:
                self._proxy.close()
            except Exception:
                pass
            self._proxy = None  # Reconnect if Gazebo was restarted.

    def run(self):
        # Wall clock, not rospy.Rate: Gazebo's /use_sim_time may be paused.
        interval = 1.0 / self.rate
        next_tick = time.monotonic()
        while not rospy.is_shutdown():
            self._sync_once()
            next_tick += interval
            now = time.monotonic()
            if next_tick < now:
                next_tick = now  # Do not queue old service calls.
            time.sleep(max(0.0, next_tick - now))


def main():
    T800MuJoCoBasePoseSync().run()


if __name__ == '__main__':
    main()
