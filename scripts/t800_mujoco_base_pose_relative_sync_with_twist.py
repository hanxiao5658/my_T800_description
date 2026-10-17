#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS1 Noetic: MuJoCo PoseStamped -> Gazebo pose AND twist, relative alignment.

Experimental FAST-LIO diagnostic upgrade from t800_mujoco_base_pose_relative_sync.py.

Input: /my_ros2/simulation/base_link  geometry_msgs/PoseStamped
Services: /gazebo/get_model_state, /gazebo/set_model_state
Optional reset: /my_ros1/t800/reset_base_pose_origin  std_srvs/Trigger

Gazebo relative spawn is preserved:
  p_G = p_G0 + R_align * (p_M - p_M0)
  q_G = q_align * q_M
  v_G = R_align * v_M (world coordinates)
  w_G = R_align * w_M (world coordinates)

The twist estimate uses DIFFERENCES OF SOURCE MESSAGES, not repeated Gazebo
service calls. It is useful for diagnosing simulated IMU angular velocity, but
cannot reproduce chest joint acceleration, sub-frame motion or synchronized IMU.
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


def norm_q(q):
    v = (float(q.x), float(q.y), float(q.z), float(q.w))
    if not all(math.isfinite(t) for t in v):
        return None
    n = math.sqrt(sum(t*t for t in v))
    return tuple(t / n for t in v) if n > 1e-9 else None


def valid_pose(p):
    return all(math.isfinite(z) for z in (p.position.x, p.position.y, p.position.z)) and norm_q(p.orientation) is not None


def qmul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w*X + x*W + y*Z - z*Y,
            w*Y - x*Z + y*W + z*X,
            w*Z + x*Y - y*X + z*W,
            w*W - x*X - y*Y - z*Z)


def qinv(q):
    return (-q[0], -q[1], -q[2], q[3])


def qrotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx + w*tx + (y*tz-z*ty),
            vy + w*ty + (z*tx-x*tz),
            vz + w*tz + (x*ty-y*tx))


def delta_omega_world(q_prev, q_curr, dt):
    """World-frame angular velocity from q_curr * q_prev^-1; shortest arc."""
    d = qmul(q_curr, qinv(q_prev))
    if d[3] < 0:
        d = tuple(-x for x in d)
    u = math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
    if u < 1e-10:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(u, min(1.0, max(-1.0, d[3])))
    scale = angle / (u * dt)
    return (d[0]*scale, d[1]*scale, d[2]*scale)


class RelativeSyncWithTwist:
    def __init__(self):
        gp = rospy.get_param
        self.model_name = gp('~model_name', 't800')
        self.input_topic = gp('~input_topic', '/my_ros2/simulation/base_link')
        self.get_service = gp('~get_service', '/gazebo/get_model_state')
        self.set_service = gp('~set_service', '/gazebo/set_model_state')
        self.reset_service = gp('~reset_service', '/my_ros1/t800/reset_base_pose_origin')
        self.frame = gp('~expected_frame', 'world')
        self.rate = float(gp('~sync_rate', 30.0))
        self.stale_timeout = float(gp('~stale_timeout', 0.5))
        self.twist_alpha = float(gp('~twist_alpha', 0.6))
        self.max_linear = float(gp('~max_linear_speed', 5.0))
        self.max_angular = float(gp('~max_angular_speed', 12.0))
        self.use_header_stamp = bool(gp('~use_header_stamp', True))
        if not 0.0 < self.rate <= 200.0 or not 0.0 < self.twist_alpha <= 1.0:
            raise ValueError('Invalid ~sync_rate or ~twist_alpha')
        self.lock = threading.RLock()
        self.latest_pose = None
        self.latest_time = None
        self.latest_v = (0., 0., 0.)
        self.latest_w = (0., 0., 0.)
        self.prev_pose = None
        self.prev_time = None
        self.prev_stamp = None
        self.need_anchor = True
        self.mu0 = None
        self.gz0 = None
        self.qalign = None
        self.get_proxy = None
        self.set_proxy = None
        self.logged_success = False
        self.sub = rospy.Subscriber(self.input_topic, PoseStamped,
                                    self.on_source, queue_size=1, tcp_nodelay=True)
        self.srv = rospy.Service(self.reset_service, Trigger, self.on_reset)
        rospy.loginfo('T800 relative POSE+TWIST sync: model=%s topic=%s %.1fHz',
                      self.model_name, self.input_topic, self.rate)
        rospy.logwarn('Experimental: source pose derivatives are NOT a full simulated IMU.')

    def on_source(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.frame:
            rospy.logwarn_throttle(5, 'Pose frame %s != expected %s', msg.header.frame_id, self.frame)
            return
        if not valid_pose(msg.pose):
            rospy.logwarn_throttle(5, 'Rejected non-finite pose/quaternion')
            return
        t = time.monotonic()
        st = msg.header.stamp.to_sec()
        with self.lock:
            if self.prev_pose is not None and self.prev_time is not None:
                dt_arrival = t - self.prev_time
                dt = dt_arrival
                if self.use_header_stamp and self.prev_stamp is not None and st > 0 and self.prev_stamp > 0:
                    dt_stamp = st - self.prev_stamp
                    if 0.001 <= dt_stamp <= 0.3:
                        dt = dt_stamp
                if 0.001 <= dt <= 0.3:
                    old = self.prev_pose
                    new = msg.pose
                    v = tuple((a - b) / dt for a, b in zip(
                        (new.position.x, new.position.y, new.position.z),
                        (old.position.x, old.position.y, old.position.z)))
                    w = delta_omega_world(norm_q(old.orientation), norm_q(new.orientation), dt)
                    if math.sqrt(sum(x*x for x in v)) <= self.max_linear and math.sqrt(sum(x*x for x in w)) <= self.max_angular:
                        a = self.twist_alpha
                        self.latest_v = tuple(a*x + (1-a)*y for x, y in zip(v, self.latest_v))
                        self.latest_w = tuple(a*x + (1-a)*y for x, y in zip(w, self.latest_w))
                    else:
                        # Teleport / resetting MuJoCo should not produce an IMU spike.
                        self.latest_v, self.latest_w = (0., 0., 0.), (0., 0., 0.)
                        rospy.logwarn_throttle(5, 'Pose discontinuity; discarded velocity estimate')
                else:
                    self.latest_v, self.latest_w = (0., 0., 0.), (0., 0., 0.)
            self.latest_pose = copy.deepcopy(msg.pose)
            self.latest_time = t
            self.prev_pose = copy.deepcopy(msg.pose)
            self.prev_time = t
            self.prev_stamp = st

    def on_reset(self, _):
        with self.lock:
            self.need_anchor = True
            self.latest_v, self.latest_w = (0., 0., 0.), (0., 0., 0.)
            self.prev_pose = None
            self.prev_time = None
            self.prev_stamp = None
        return TriggerResponse(success=True, message='Re-anchor requested')

    def fresh_source(self):
        with self.lock:
            if self.latest_pose is None or self.latest_time is None:
                return None
            if time.monotonic() - self.latest_time > self.stale_timeout:
                rospy.logwarn_throttle(5, 'MuJoCo pose stale')
                return None
            return (copy.deepcopy(self.latest_pose), self.latest_v, self.latest_w)

    def connect(self):
        try:
            if self.get_proxy is None:
                rospy.wait_for_service(self.get_service, timeout=1.0)
                self.get_proxy = rospy.ServiceProxy(self.get_service, GetModelState, persistent=True)
            if self.set_proxy is None:
                rospy.wait_for_service(self.set_service, timeout=1.0)
                self.set_proxy = rospy.ServiceProxy(self.set_service, SetModelState, persistent=True)
            return True
        except rospy.ROSException:
            rospy.logwarn_throttle(5, 'Waiting for Gazebo model services')
            return False

    def anchor(self):
        if self.fresh_source() is None or not self.connect():
            return
        try:
            r = self.get_proxy(self.model_name, 'world')
            if not r.success or not valid_pose(r.pose):
                rospy.logwarn_throttle(5, 'Cannot read valid Gazebo initial model state')
                return
        except (rospy.ServiceException, rospy.ROSException) as ex:
            self.get_proxy = None
            rospy.logwarn_throttle(5, 'GetModelState: %s', ex)
            return
        data = self.fresh_source()
        if data is None:
            return
        mp, _, _ = data
        q_a = qmul(norm_q(r.pose.orientation), qinv(norm_q(mp.orientation)))
        s = math.sqrt(sum(x*x for x in q_a))
        q_a = tuple(x/s for x in q_a)
        with self.lock:
            self.mu0 = (mp.position.x, mp.position.y, mp.position.z)
            self.gz0 = (r.pose.position.x, r.pose.position.y, r.pose.position.z)
            self.qalign = q_a
            self.need_anchor = False
            self.logged_success = False
        rospy.loginfo('Anchored: MuJoCo(%.3f %.3f %.3f) => Gazebo(%.3f %.3f %.3f)',
                      *self.mu0, *self.gz0)

    def tick(self):
        with self.lock:
            init = self.need_anchor
        if init:
            self.anchor()
            return
        data = self.fresh_source()
        if data is None or not self.connect():
            return
        pose, v_m, w_m = data
        with self.lock:
            if self.need_anchor:
                return
            mu0, gz0, qa = self.mu0, self.gz0, self.qalign
        p = pose.position
        disp = qrotate(qa, (p.x-mu0[0], p.y-mu0[1], p.z-mu0[2]))
        q = qmul(qa, norm_q(pose.orientation))
        n = math.sqrt(sum(x*x for x in q))
        q = tuple(x/n for x in q)
        vg = qrotate(qa, v_m)
        wg = qrotate(qa, w_m)

        m = ModelState()
        m.model_name = self.model_name
        m.reference_frame = 'world'
        m.pose.position.x, m.pose.position.y, m.pose.position.z = tuple(gz0[i]+disp[i] for i in range(3))
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
        m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = vg
        m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = wg
        try:
            r = self.set_proxy(model_state=m)
            if not r.success:
                rospy.logwarn_throttle(5, 'SetModelState rejected: %s', r.status_message)
            elif not self.logged_success:
                rospy.loginfo('T800 relative pose+twist sync working')
                self.logged_success = True
        except (rospy.ServiceException, rospy.ROSException) as ex:
            self.set_proxy = None
            with self.lock:
                self.need_anchor = True
            rospy.logwarn_throttle(5, 'SetModelState failed, will reanchor: %s', ex)

    def run(self):
        step = 1.0 / self.rate
        deadline = time.monotonic()
        while not rospy.is_shutdown():
            self.tick()
            deadline += step
            now = time.monotonic()
            if deadline < now:
                deadline = now
            time.sleep(max(0., deadline-now))


def main():
    rospy.init_node('t800_mujoco_base_pose_relative_sync_with_twist')
    RelativeSyncWithTwist().run()


if __name__ == '__main__':
    main()
