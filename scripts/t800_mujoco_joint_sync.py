#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T800 MuJoCo -> Gazebo Classic joint-position hard synchronization (ROS1 Noetic).

Input (already bridged from ROS2 to ROS1, topic name unchanged):
    /my_ros2/hardware/joint_state   sensor_msgs/JointState

Output (Gazebo Classic ROS1 service):
    /gazebo/set_model_configuration gazebo_msgs/SetModelConfiguration

IMPORTANT:
    This service sets internal joint angles directly, not PID setpoints.
    /gazebo/set_model_state is for the base/world pose and cannot set joint angles.

Assumptions:
    - The Gazebo spawned model is named "t800".
    - The input JointState.name entries match the names in the T800 URDF.
    - The input JointState.position entries are in radians.
"""

import math
import threading
import time

import rospy
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest

# Joint names verified against the official T800 URDF. Fixed joints are omitted.
T800_JOINT_NAMES = [
    'J00_HIP_PITCH_L', 'J01_HIP_ROLL_L', 'J02_HIP_YAW_L',
    'J03_KNEE_PITCH_L', 'J04_ANKLE_PITCH_L', 'J05_ANKLE_ROLL_L',
    'J06_HIP_PITCH_R', 'J07_HIP_ROLL_R', 'J08_HIP_YAW_R',
    'J09_KNEE_PITCH_R', 'J10_ANKLE_PITCH_R', 'J11_ANKLE_ROLL_R',
    'J12_TORSO_YAW',
    'J13_SHOULDER_PITCH_L', 'J14_SHOULDER_ROLL_L',
    'J15_SHOULDER_YAW_L', 'J16_ELBOW_PITCH_L', 'J17_ELBOW_YAW_L',
    'J18_SHOULDER_PITCH_R', 'J19_SHOULDER_ROLL_R',
    'J20_SHOULDER_YAW_R', 'J21_ELBOW_PITCH_R', 'J22_ELBOW_YAW_R',
    'J23_HEAD_PITCH', 'J24_HEAD_YAW',
]


class T800MuJoCoJointSync:
    def __init__(self):
        self.input_topic = rospy.get_param('~input_topic', '/my_ros2/hardware/joint_state')
        self.service_name = rospy.get_param('~service_name', '/gazebo/set_model_configuration')
        self.model_name = rospy.get_param('~model_name', 't800')
        self.urdf_param_name = rospy.get_param('~urdf_param_name', 'robot_description')
        self.sync_rate = float(rospy.get_param('~sync_rate', 30.0))
        self.stale_timeout = float(rospy.get_param('~stale_timeout', 0.5))

        if not 0.0 < self.sync_rate <= 100.0:
            raise ValueError('~sync_rate must be in (0, 100] Hz')
        if self.stale_timeout <= 0:
            raise ValueError('~stale_timeout must be positive')

        self.joint_order = tuple(T800_JOINT_NAMES)
        self.lock = threading.Lock()
        self.latest_positions = None
        self.latest_arrival = None
        self.srv = None
        self.has_succeeded = False

        # queue_size=1 keeps only the newest state instead of building a backlog.
        # tcp_nodelay=True helps reduce ROS1 TCP transport latency.
        self.sub = rospy.Subscriber(
            self.input_topic, JointState, self.on_joint_state,
            queue_size=1, tcp_nodelay=True
        )
        rospy.loginfo('T800 joint sync: %s -> %s (model=%s, rate=%.1f Hz)',
                      self.input_topic, self.service_name, self.model_name,
                      self.sync_rate)

    def on_joint_state(self, msg):
        """Read name-position pairs; never assume an incoming array's order."""
        if len(msg.name) != len(msg.position):
            rospy.logwarn_throttle(5, 'JointState name/position length mismatch: %d/%d',
                                   len(msg.name), len(msg.position))
            return

        positions_by_name = {}
        allowed = set(self.joint_order)
        for name, value in zip(msg.name, msg.position):
            if name in allowed and math.isfinite(value):
                positions_by_name[name] = float(value)

        # Accept partial states: only change joints contained in this message.
        if not positions_by_name:
            rospy.logwarn_throttle(5, 'No T800 joint names matched the incoming JointState')
            return

        with self.lock:
            self.latest_positions = positions_by_name
            self.latest_arrival = time.monotonic()

    def connect_service(self):
        """Wait until Gazebo exposes its service; reconnect after restarts."""
        try:
            rospy.wait_for_service(self.service_name, timeout=2.0)
            # Persistent TCP service connection avoids a new handshake every cycle.
            self.srv = rospy.ServiceProxy(
                self.service_name, SetModelConfiguration, persistent=True
            )
            return True
        except rospy.ROSException:
            rospy.logwarn_throttle(5, 'Waiting for Gazebo service %s', self.service_name)
            return False

    def run(self):
        """Call the service at a wall-clock frequency (works with paused /clock)."""
        period = 1.0 / self.sync_rate
        sleeper = threading.Event()

        while not rospy.is_shutdown():
            cycle_start = time.monotonic()
            if self.srv is None and not self.connect_service():
                sleeper.wait(0.2)
                continue

            with self.lock:
                positions = (dict(self.latest_positions)
                             if self.latest_positions is not None else None)
                arrival = self.latest_arrival

            if positions is not None and arrival is not None:
                if time.monotonic() - arrival <= self.stale_timeout:
                    # Maintain a deterministic order without relying on upstream ordering.
                    names = [n for n in self.joint_order if n in positions]
                    values = [positions[n] for n in names]

                    req = SetModelConfigurationRequest()
                    req.model_name = self.model_name
                    req.urdf_param_name = self.urdf_param_name
                    req.joint_names = names
                    req.joint_positions = values

                    try:
                        response = self.srv(req)
                        if response.success:
                            if not self.has_succeeded:
                                rospy.loginfo('Gazebo joint sync working: %d joints updated',
                                              len(names))
                                self.has_succeeded = True
                        else:
                            rospy.logwarn_throttle(3, 'Gazebo rejected joint update: %s',
                                                   response.status_message)
                    except rospy.ServiceException as exc:
                        rospy.logwarn('Gazebo service call failed: %s; reconnecting', exc)
                        self.srv = None
                else:
                    rospy.logwarn_throttle(5, 'Source JointState is stale; skipping Gazebo updates')

            # Use monotonic wall time instead of rospy.Rate; /use_sim_time may pause.
            sleeper.wait(max(0.0, period - (time.monotonic() - cycle_start)))


def main():
    rospy.init_node('t800_mujoco_joint_sync', anonymous=False)
    T800MuJoCoJointSync().run()


if __name__ == '__main__':
    main()
