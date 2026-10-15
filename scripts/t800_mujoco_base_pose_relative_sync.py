#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T800 MuJoCo -> Gazebo Classic: increment-based base-pose hard synchronization.

Runs in: A2sim Docker / ROS1 Noetic, package 't800_description'.

Subscribe:
  /my_ros2/simulation/base_link    geometry_msgs/PoseStamped (bridged ROS2 -> ROS1)

Gazebo services:
  /gazebo/get_model_state          gazebo_msgs/GetModelState  (read initial pose)
  /gazebo/set_model_state          gazebo_msgs/SetModelState  (write current pose)

Optional reset service:
  /my_ros1/t800/reset_base_pose_origin    std_srvs/Trigger

Pose alignment (3D rigid transformation):
  T_G(t) = T_G0 * inverse(T_M0) * T_M(t)

  p_G(t) = p_G0 + R(q_G0 * inverse(q_M0)) * (p_M(t) - p_M0)
  q_G(t) = q_G0 * inverse(q_M0) * q_M(t)

Note: This is NOT a frame-by-frame accumulated increment. Every output is
computed from the initial anchors to avoid cumulative numerical drift.
Does not affect joints. Keep the separate joint sync node running.
"""

import copy
import math
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from std_srvs.srv import Trigger, TriggerResponse


def normalize_quaternion(q):
    """Return normalized quaternion (x,y,z,w), or None if invalid."""
    a = (float(q.x), float(q.y), float(q.z), float(q.w))
    if not all(math.isfinite(v) for v in a):
        return None
    norm = math.sqrt(sum(v * v for v in a))
    if norm < 1e-9:
        return None
    return tuple(v / norm for v in a)


def pose_ok(pose):
    p = pose.position
    return (all(math.isfinite(float(v)) for v in (p.x, p.y, p.z))
            and normalize_quaternion(pose.orientation) is not None)


def quat_mul(a, b):
    """Hamilton product; quaternion order is ROS (x,y,z,w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_inv_unit(q):
    """Inverse of unit quaternion."""
    return (-q[0], -q[1], -q[2], q[3])


def rotate_vector(q, v):
    """Rotate a 3D vector using a normalized (x,y,z,w) quaternion."""
    # Numerically efficient form: v' = v + w * t + cross(q.xyz, t),
    # where t = 2 * cross(q.xyz, v).
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


class T800RelativeBasePoseSync:
    def __init__(self):
        self.model_name = rospy.get_param('~model_name', 't800')
        self.input_topic = rospy.get_param(
            '~input_topic', '/my_ros2/simulation/base_link')
        self.get_service = rospy.get_param(
            '~get_service', '/gazebo/get_model_state')
        self.set_service = rospy.get_param(
            '~set_service', '/gazebo/set_model_state')
        self.reset_service = rospy.get_param(
            '~reset_service', '/my_ros1/t800/reset_base_pose_origin')
        self.sync_rate = float(rospy.get_param('~sync_rate', 30.0))
        self.stale_timeout = float(rospy.get_param('~stale_timeout', 0.5))
        self.expected_frame = str(rospy.get_param('~expected_frame', 'world'))
        if not 0.0 < self.sync_rate <= 100.0:
            raise ValueError('~sync_rate must be between 0 and 100 Hz')
        if not self.stale_timeout > 0.0:
            raise ValueError('~stale_timeout must be positive')

        self.lock = threading.Lock()
        self.latest = None
        self.last_rx = None  # monotonic wall time: independent of /use_sim_time
        self.needs_anchor = True
        self.anchor_mu_pos = None
        self.anchor_gz_pos = None
        self.q_align = None
        self.get_proxy = None
        self.set_proxy = None
        self.first_success = False

        self.sub = rospy.Subscriber(
            self.input_topic, PoseStamped, self.on_pose,
            queue_size=1, tcp_nodelay=True)
        self.reset_srv = rospy.Service(self.reset_service, Trigger, self.on_reset)
        rospy.loginfo('T800 relative base sync: %s -> %s, model=%s, %.1f Hz',
                      self.input_topic, self.set_service,
                      self.model_name, self.sync_rate)
        rospy.loginfo('Place T800 at the desired Gazebo spawn pose BEFORE starting.')

    def on_pose(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.expected_frame:
            rospy.logwarn_throttle(
                5.0, 'Unexpected pose frame_id=%s (expected %s); skipping',
                msg.header.frame_id, self.expected_frame)
            return
        if not pose_ok(msg.pose):
            rospy.logwarn_throttle(5.0, 'Invalid MuJoCo base pose/quaternion')
            return
        with self.lock:
            self.latest = copy.deepcopy(msg.pose)
            self.last_rx = time.monotonic()

    def on_reset(self, _req):
        """Re-anchor after repositioning Gazebo or resetting MuJoCo.

        The next update reads Gazebo's CURRENT model pose and the most recent
        MuJoCo pose, then resumes without moving the model from that pose.
        """
        with self.lock:
            self.needs_anchor = True
            self.anchor_mu_pos = None
            self.anchor_gz_pos = None
            self.q_align = None
        rospy.loginfo('Reset requested: will capture new MuJoCo/Gazebo anchors')
        return TriggerResponse(success=True, message='Re-anchor requested')

    def latest_pose(self):
        with self.lock:
            pose = copy.deepcopy(self.latest)
            arrived = self.last_rx
        if pose is None or arrived is None:
            return None
        if time.monotonic() - arrived > self.stale_timeout:
            rospy.logwarn_throttle(5.0, 'MuJoCo base pose stale; waiting for fresh data')
            return None
        return pose

    def connect_services(self):
        try:
            if self.get_proxy is None:
                rospy.wait_for_service(self.get_service, timeout=1.0)
                self.get_proxy = rospy.ServiceProxy(
                    self.get_service, GetModelState, persistent=True)
            if self.set_proxy is None:
                rospy.wait_for_service(self.set_service, timeout=1.0)
                self.set_proxy = rospy.ServiceProxy(
                    self.set_service, SetModelState, persistent=True)
            return True
        except rospy.ROSException:
            rospy.logwarn_throttle(5.0, 'Waiting for Gazebo get/set model state services')
            return False

    def capture_anchors(self):
        """Read initial Gazebo model world pose, then fresh MuJoCo source pose."""
        if self.latest_pose() is None or not self.connect_services():
            return False
        try:
            response = self.get_proxy(self.model_name, 'world')
            if not response.success:
                rospy.logwarn_throttle(5.0, 'GetModelState rejected: %s',
                                       response.status_message)
                return False
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(5.0, 'Gazebo get_model_state failed: %s', exc)
            self.get_proxy = None
            return False
        if not pose_ok(response.pose):
            rospy.logwarn_throttle(5.0, 'Gazebo returned invalid initial pose')
            return False

        # Capture the latest source after the Gazebo query, minimizing time skew.
        mu = self.latest_pose()
        if mu is None:
            return False
        gm = response.pose
        q_m0 = normalize_quaternion(mu.orientation)
        q_g0 = normalize_quaternion(gm.orientation)
        q_align = quat_mul(q_g0, quat_inv_unit(q_m0))
        q_norm = math.sqrt(sum(v * v for v in q_align))
        q_align = tuple(v / q_norm for v in q_align)

        with self.lock:
            self.anchor_mu_pos = (mu.position.x, mu.position.y, mu.position.z)
            self.anchor_gz_pos = (gm.position.x, gm.position.y, gm.position.z)
            self.q_align = q_align
            self.needs_anchor = False
        rospy.loginfo('Captured MuJoCo origin xyz=(%.3f, %.3f, %.3f)',
                      *self.anchor_mu_pos)
        rospy.loginfo('Captured Gazebo origin xyz=(%.3f, %.3f, %.3f)',
                      *self.anchor_gz_pos)
        rospy.loginfo('Initial orientations aligned; following RELATIVE 3D motion')
        self.first_success = False
        return True

    def sync_once(self):
        with self.lock:
            needs_anchor = self.needs_anchor
        if needs_anchor:
            self.capture_anchors()
            return

        mu = self.latest_pose()
        if mu is None or not self.connect_services():
            return
        with self.lock:
            if self.needs_anchor:  # reset was requested while reading source
                return
            mu0 = self.anchor_mu_pos
            gz0 = self.anchor_gz_pos
            qa = self.q_align

        delta = (mu.position.x - mu0[0],
                 mu.position.y - mu0[1],
                 mu.position.z - mu0[2])
        dx, dy, dz = rotate_vector(qa, delta)
        q_new = quat_mul(qa, normalize_quaternion(mu.orientation))
        q_length = math.sqrt(sum(v*v for v in q_new))
        q_new = tuple(v/q_length for v in q_new)

        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = 'world'
        state.pose.position.x = gz0[0] + dx
        state.pose.position.y = gz0[1] + dy
        state.pose.position.z = gz0[2] + dz
        (state.pose.orientation.x, state.pose.orientation.y,
         state.pose.orientation.z, state.pose.orientation.w) = q_new
        # Twist remains zero (visual-only hard sync; no Gazebo PID).

        try:
            answer = self.set_proxy(model_state=state)
            if answer.success:
                if not self.first_success:
                    rospy.loginfo('T800 relative base pose sync WORKING')
                    self.first_success = True
            else:
                rospy.logwarn_throttle(5.0, 'SetModelState rejected: %s',
                                       answer.status_message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(5.0, 'Gazebo set_model_state failed: %s', exc)
            self.set_proxy = None
            # If Gazebo restarted / model was respawned, re-read its new pose.
            with self.lock:
                self.needs_anchor = True

    def run(self):
        # Use wall time to continue working when Gazebo /clock is paused.
        interval = 1.0 / self.sync_rate
        next_tick = time.monotonic()
        while not rospy.is_shutdown():
            self.sync_once()
            next_tick += interval
            now = time.monotonic()
            if now > next_tick:
                next_tick = now
            time.sleep(max(0.0, next_tick - now))


def main():
    rospy.init_node('t800_mujoco_base_pose_relative_sync', anonymous=False)
    T800RelativeBasePoseSync().run()


if __name__ == '__main__':
    main()
