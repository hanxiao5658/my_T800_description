#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ROS1 导航 YAML 向已运行的 Gazebo Classic 生成仅可视化的 Waypoint 标记。

每个标记：黄色薄圆柱，只有 <visual>，没有 <collision>。
不订阅/发布机器人控制话题；不会影响 MID360 射线或导航控制。
"""

import math
import os
import re
import sys

import rospy
import yaml
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from tf.transformations import quaternion_from_euler


def load_waypoints(path):
    """兼容常用的 {frame_id: world, waypoints: [{id,x,y,yaw}, ...]} 格式。"""
    if not os.path.isfile(path):
        raise IOError('找不到 Waypoint YAML 文件: {}'.format(path))

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError('YAML 根节点必须是字典，包含 waypoints 列表')

    frame_id = str(data.get('frame_id', 'world')).lstrip('/')
    if frame_id not in ('world', 'map'):
        rospy.logwarn('YAML frame_id=%s; 标记仍按 Gazebo world 坐标生成，请确认坐标一致', frame_id)

    points = data.get('waypoints')
    if not isinstance(points, list) or not points:
        raise ValueError('YAML 需要非空 waypoints 列表，例如 waypoints: [{id: wp1, x: 1, y: 2, yaw: 0}]')

    parsed = []
    for i, wp in enumerate(points, 1):
        if not isinstance(wp, dict):
            raise ValueError('第 {} 个 Waypoint 应为字典'.format(i))
        try:
            x = float(wp['x'])
            y = float(wp['y'])
            yaw = float(wp['yaw'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('第 {} 个 Waypoint 的 x/y/yaw 缺失或不是数字: {}'.format(i, exc))
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError('第 {} 个 Waypoint 存在无穷大或 NaN'.format(i))
        parsed.append({'id': str(wp.get('id', 'wp{}'.format(i))), 'x': x, 'y': y, 'yaw': yaw})
    return parsed


def build_marker_sdf(radius, height):
    """黄色圆柱薄片；注意刻意不包含 <collision>。"""
    return '''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="waypoint_marker">
    <static>true</static>
    <link name="marker_link">
      <gravity>false</gravity>
      <visual name="yellow_disc">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <cylinder>
            <radius>{radius:.8f}</radius>
            <length>{height:.8f}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.95 0.0 1.0</ambient>
          <diffuse>1.0 0.92 0.0 1.0</diffuse>
          <emissive>0.30 0.25 0.0 1.0</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''.format(radius=radius, height=height)


def main():
    rospy.init_node('spawn_waypoint_markers')

    yaml_path = os.path.expanduser(rospy.get_param('~waypoint_file'))
    radius = float(rospy.get_param('~radius', 0.30))
    height = float(rospy.get_param('~height', 0.01))
    # 模型中心高度：默认 0.015m，圆柱底面约 0.01m，避免地板遮挡。
    marker_z = float(rospy.get_param('~marker_z', 0.015))
    model_prefix = str(rospy.get_param('~model_prefix', 'waypoint_marker'))
    replace_existing = bool(rospy.get_param('~replace_existing', True))

    if radius <= 0 or height <= 0 or not all(math.isfinite(v) for v in (radius, height, marker_z)):
        rospy.logerr('半径、高度必须为正有限值，marker_z 必须为有限值')
        sys.exit(1)
    if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', model_prefix):
        rospy.logerr('model_prefix 只能包含字母、数字、下划线，且以字母开头')
        sys.exit(1)

    try:
        waypoints = load_waypoints(yaml_path)
    except (IOError, ValueError, yaml.YAMLError) as exc:
        rospy.logerr('加载 Waypoints 失败: %s', exc)
        sys.exit(1)

    rospy.loginfo('读取 %d 个 Waypoint: %s', len(waypoints), yaml_path)
    rospy.loginfo('等待 Gazebo 服务 /gazebo/spawn_sdf_model ...')
    try:
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=30.0)
    except rospy.ROSException:
        rospy.logerr('30 秒内未找到 /gazebo/spawn_sdf_model，请先启动 Gazebo World')
        sys.exit(1)

    spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    get_world = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    existing = set()
    try:
        world = get_world()
        if world.success:
            existing = set(world.model_names)
    except rospy.ServiceException as exc:
        rospy.logwarn('获取模型列表失败，将直接尝试生成: %s', exc)

    xml = build_marker_sdf(radius, height)
    succeeded = 0

    for idx, wp in enumerate(waypoints, 1):
        if rospy.is_shutdown():
            break
        name = '{}_{:03d}'.format(model_prefix, idx)
        if name in existing:
            if not replace_existing:
                rospy.logwarn('跳过已存在的模型 %s（replace_existing=false）', name)
                continue
            try:
                response = delete_model(name)
                if not response.success:
                    rospy.logerr('删除旧标记 %s 失败: %s', name, response.status_message)
                    continue
            except rospy.ServiceException as exc:
                rospy.logerr('删除旧标记 %s 异常: %s', name, exc)
                continue

        pose = Pose()
        pose.position.x = wp['x']
        pose.position.y = wp['y']
        pose.position.z = marker_z
        q = quaternion_from_euler(0.0, 0.0, wp['yaw'])
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q

        try:
            result = spawn_model(name, xml, '', pose, 'world')
            if not result.success:
                rospy.logerr('生成 %s 失败: %s', name, result.status_message)
                continue
        except rospy.ServiceException as exc:
            rospy.logerr('生成 %s 失败: %s', name, exc)
            continue

        succeeded += 1
        rospy.loginfo('已生成 %s [%s]  x=%.3f  y=%.3f  yaw=%.3f rad',
                      name, wp['id'], wp['x'], wp['y'], wp['yaw'])

    rospy.loginfo('Waypoint 标记生成完成：%d/%d。标记将保留在 Gazebo，直到删除模型或重启 World。',
                  succeeded, len(waypoints))


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
