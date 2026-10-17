#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在当前已验证的 T800 可视化 URDF 上安装 MID360 + IMU（Gazebo Classic / ROS1 Noetic）。

默认输入:  urdf/serial_t800_visual_sync.urdf
默认输出:  urdf/serial_t800_visual_mid360_imu.urdf

原则：
  * 保留原 T800 所有关节、link、mesh、Gazebo 现有设置；不改动关节硬同步。
  * 将 MID360 固定在 LINK_WAIST_YAW（胸部随腰关节旋转）。
  * 仅用一个 fixed joint 设置雷达/IMU 的共同安装位姿。
  * MID360 ray 和 IMU sensor 的 <pose> 都是 0 0 0 0 0 0，
    两者 <frameName> 都是 mid360_link，严格共点同轴，静态外参为单位变换。
  * 只产生新的输出 URDF，不覆盖你的旧 URDF。

依赖：Python3 标准库；如果安装了 go2w_description 包，会尝试复制其 MID360 DAE。
"""
import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


def sub(parent, tag, text=None, **attributes):
    el = ET.SubElement(parent, tag, attributes)
    if text is not None:
        el.text = str(text)
    return el


def locate_and_copy_mesh(package_dir):
    dst = package_dir / 'meshes' / 'mid360' / 'livox_mid360.dae'
    if dst.is_file():
        return True, dst

    candidates = [
        package_dir.parent / 'go2w_description' / 'meshes' / 'mid360' / 'livox_mid360.dae',
        Path.home() / 'catkin_ws' / 'src' / 'go2w_description' / 'meshes' / 'mid360' / 'livox_mid360.dae',
    ]
    try:
        pkg_path = subprocess.check_output(
            ['rospack', 'find', 'go2w_description'], stderr=subprocess.DEVNULL,
            text=True, timeout=3).strip()
        candidates.insert(0, Path(pkg_path) / 'meshes' / 'mid360' / 'livox_mid360.dae')
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    for src in candidates:
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print('[OK] 自动复制 MID360 可视化模型:', dst)
            return True, dst
    return False, dst


def add_sensor_visual_and_joint(robot, *, parent_name, xyz, rpy, has_mesh):
    # mid360_link 原点代表激光和 IMU 共用的测量中心。
    link = sub(robot, 'link', name='mid360_link')
    inertial = sub(link, 'inertial')
    sub(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
    sub(inertial, 'mass', value='0.1')
    sub(inertial, 'inertia', ixx='0.0001', ixy='0', ixz='0',
        iyy='0.0001', iyz='0', izz='0.0001')

    visual = sub(link, 'visual')
    if has_mesh:
        # Go2W URDF 的 ray/imu 原点比 mesh 原点高 0.05 m。
        # 这里将 visual 下移 0.05 m，令传感器 frame 就是真实测量中心。
        sub(visual, 'origin', xyz='0 0 -0.05', rpy='0 0 0')
        geometry = sub(visual, 'geometry')
        sub(geometry, 'mesh', filename='package://t800_description/meshes/mid360/livox_mid360.dae')
    else:
        # 备用可见外形，避免没有 DAE 时生成 Gazebo 无法显示的模型。
        sub(visual, 'origin', xyz='0 0 -0.02', rpy='0 0 0')
        geometry = sub(visual, 'geometry')
        sub(geometry, 'cylinder', radius='0.04', length='0.07')
        mat = sub(visual, 'material', name='mid360_placeholder_gray')
        sub(mat, 'color', rgba='0.3 0.3 0.3 1')

    joint = sub(robot, 'joint', name='t800_chest_to_mid360', type='fixed')
    sub(joint, 'origin', xyz=' '.join(map(str, xyz)), rpy=' '.join(map(str, rpy)))
    sub(joint, 'parent', link=parent_name)
    sub(joint, 'child', link='mid360_link')

    # 防止 Gazebo Classic 自动将 fixed joint 两端 link 合并，便于查询 link/TF。
    gz_joint = sub(robot, 'gazebo', reference='t800_chest_to_mid360')
    sub(gz_joint, 'preserveFixedJoint', 'true')

    gz_link = sub(robot, 'gazebo', reference='mid360_link')
    sub(gz_link, 'gravity', 'false')
    sub(gz_link, 'selfCollide', 'false')


def add_lidar(robot, topic, visualize_rays):
    gz = sub(robot, 'gazebo', reference='mid360_link')
    sensor = sub(gz, 'sensor', type='ray', name='t800_mid360_sensor')
    sub(sensor, 'pose', '0 0 0 0 0 0')  # 与 IMU 完全相同
    sub(sensor, 'visualize', str(visualize_rays).lower())
    sub(sensor, 'always_on', 'true')
    sub(sensor, 'update_rate', '10')

    plugin = sub(sensor, 'plugin', name='t800_mid360_livox_plugin',
                 filename='liblivox_laser_simulation.so')
    # 与 Go2W 的 Livox 插件格式一致：ray 必须嵌在 plugin 内。
    ray = sub(plugin, 'ray')
    scan = sub(ray, 'scan')
    hori = sub(scan, 'horizontal')
    for k, v in [('samples', '100'), ('resolution', '1'),
                 ('min_angle', '-3.1415926535897931'), ('max_angle', '3.1415926535897931')]:
        sub(hori, k, v)
    vert = sub(scan, 'vertical')
    for k, v in [('samples', '50'), ('resolution', '1'),
                 ('min_angle', '-3.1415926535897931'), ('max_angle', '3.1415926535897931')]:
        sub(vert, k, v)
    rng = sub(ray, 'range')
    for k, v in [('min', '0.1'), ('max', '40.0'), ('resolution', '1.0')]:
        sub(rng, k, v)
    noise = sub(ray, 'noise')
    for k, v in [('type', 'gaussian'), ('mean', '0.0'), ('stddev', '0.0')]:
        sub(noise, k, v)

    sub(plugin, 'visualize', 'false')
    sub(plugin, 'samples', '20000')
    sub(plugin, 'downsample', '2')
    sub(plugin, 'csv_file_name', 'mid360-real-centr.csv')
    sub(plugin, 'publish_pointcloud_type', '3')  # sensor_msgs/PointCloud2
    sub(plugin, 'ros_topic', topic)
    sub(plugin, 'frameName', 'mid360_link')


def add_imu(robot, topic):
    gz = sub(robot, 'gazebo', reference='mid360_link')
    sensor = sub(gz, 'sensor', type='imu', name='t800_mid360_imu_sensor')
    sub(sensor, 'pose', '0 0 0 0 0 0')  # 与 MID360 完全相同
    sub(sensor, 'always_on', 'true')
    sub(sensor, 'update_rate', '200')
    sub(sensor, 'visualize', 'false')
    sub(sensor, 'topic', '__default_topic__')

    plugin = sub(sensor, 'plugin', name='t800_mid360_imu_plugin',
                 filename='libgazebo_ros_imu_sensor.so')
    sub(plugin, 'topicName', topic)
    sub(plugin, 'frameName', 'mid360_link')
    sub(plugin, 'updateRateHZ', '200.0')
    sub(plugin, 'gaussianNoise', '0.0')
    sub(plugin, 'xyzOffset', '0 0 0')
    sub(plugin, 'rpyOffset', '0 0 0')
    sub(plugin, 'initialOrientationAsReference', 'false')


def main():
    parser = argparse.ArgumentParser(description='为 T800 当前 visual_sync URDF 添加 MID360+IMU')
    package_dir = Path(__file__).resolve().parent
    parser.add_argument('--input', type=Path,
        default=package_dir / 'urdf' / 'serial_t800_visual_sync.urdf')
    parser.add_argument('--output', type=Path,
        default=package_dir / 'urdf' / 'serial_t800_visual_mid360_imu.urdf')
    parser.add_argument('--parent', default='LINK_WAIST_YAW',
        help='安装的胸部 link，默认 LINK_WAIST_YAW')
    parser.add_argument('--xyz', nargs=3, type=float, default=[0.50, 0.0, 0.25],
        metavar=('X', 'Y', 'Z'), help='传感器中心相对胸部 link 的偏移，单位 m')
    parser.add_argument('--rpy', nargs=3, type=float, default=[0., 0., 0.],
        metavar=('R', 'P', 'Y'), help='传感器相对胸部 link 的欧拉角，单位 rad')
    parser.add_argument('--lidar-topic', default='/livox/lidar')
    parser.add_argument('--imu-topic', default='/livox/imu')
    parser.add_argument('--visualize-rays', action='store_true')
    args = parser.parse_args()
    src, dst = args.input.resolve(), args.output.resolve()
    if not src.is_file():
        parser.error(f'找不到原 URDF: {src}')
    if src == dst:
        parser.error('为保护已调好的模型，输入文件和输出文件不能相同')

    tree = ET.parse(src)
    robot = tree.getroot()
    if robot.tag != 'robot':
        parser.error('输入文件不是 URDF 的 <robot> 根标签')
    links = {x.get('name') for x in robot.findall('link')}
    joints = {x.get('name') for x in robot.findall('joint')}
    if args.parent not in links:
        parser.error(f'没有找到父 link: {args.parent}')
    if 'mid360_link' in links or 't800_chest_to_mid360' in joints:
        parser.error('输入 URDF 已含 mid360_link；请基于无雷达的旧 URDF 重新生成，避免重复插件')

    has_mesh, mesh_path = locate_and_copy_mesh(package_dir)
    add_sensor_visual_and_joint(robot, parent_name=args.parent, xyz=args.xyz,
                                rpy=args.rpy, has_mesh=has_mesh)
    add_lidar(robot, args.lidar_topic, args.visualize_rays)
    add_imu(robot, args.imu_topic)

    xml = ET.tostring(robot, encoding='utf-8')
    pretty = minidom.parseString(xml).toprettyxml(indent='  ')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('\n'.join(x for x in pretty.splitlines() if x.strip()) + '\n',
                   encoding='utf-8')

    # 对输出进行 XML 结构验证。
    out_robot = ET.parse(dst).getroot()
    assert len([x for x in out_robot.findall('link') if x.get('name') == 'mid360_link']) == 1
    sensors = out_robot.findall(".//gazebo[@reference='mid360_link']/sensor")
    assert sorted(s.get('type') for s in sensors) == ['imu', 'ray']
    assert all(s.findtext('pose') == '0 0 0 0 0 0' for s in sensors)
    print(f'[OK] 完整 T800 URDF 已生成: {dst}')
    print(f'[OK] 传感器父 link: {args.parent}, xyz={args.xyz}, rpy={args.rpy}')
    print('[OK] MID360 与 IMU 位置姿态相同，frameName 均为 mid360_link')
    print(f'[OK] lidar: {args.lidar_topic}, imu: {args.imu_topic}')
    if not has_mesh:
        print(f'[WARN] MID360 DAE 未找到: {mesh_path}')
        print('[WARN] 暂用圆柱体占位显示；复制 mesh 后重新执行本脚本即可')


if __name__ == '__main__':
    main()
