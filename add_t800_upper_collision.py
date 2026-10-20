#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add LiDAR-occluding collision meshes only to T800 upper-body links.

Uses each link's original <visual> <origin> + <geometry> so the collision
geometry matches the existing mirrored/rotated meshes exactly.

Input:  urdf/serial_t800_visual_mid360_imu.urdf
Output: urdf/serial_t800_visual_mid360_imu_upper_collision.urdf

No external Python packages required. The original URDF is not overwritten.
"""

import argparse
import copy
from pathlib import Path
from typing import Tuple, List
import re
import sys
import xml.etree.ElementTree as ET

# Confirmed in my_T800_description: this link's visual uses LINK_TORSO_YAW.dae.
# Excludes LINK_BASE (pelvis) and all HIP/KNEE/ANKLE/FOOT links.
UPPER_LINKS = (
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_YAW_L",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_SHOULDER_ROLL_R",
    "LINK_SHOULDER_YAW_R",
    "LINK_ELBOW_PITCH_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_PITCH",
    "LINK_HEAD_YAW",
)

# Both LINK_WRIST_END_L/R have no <visual> mesh in the current public URDF.
# Adding an invisible guessed sphere would create spurious LiDAR returns, so skip.


def make_collision_from_visual(visual_xml: str, link_name: str) -> str:
    visual = ET.fromstring(visual_xml)
    geo = visual.find("geometry")
    if geo is None:
        raise ValueError("{}: <visual> contains no <geometry>".format(link_name))
    element = ET.Element("collision", {"name": "lidar_occlusion_" + link_name})
    origin = visual.find("origin")
    if origin is not None:
        element.append(copy.deepcopy(origin))
    element.append(copy.deepcopy(geo))

    def xml_line(e):
        return ET.tostring(e, encoding="unicode", short_empty_elements=True).strip()

    rows = ['    <collision name="lidar_occlusion_{}">'.format(link_name)]
    for child in element:
        child_text = xml_line(child)
        if child.tag == 'geometry':
            rows.append("      <geometry>")
            for geochild in child:
                rows.append("        " + xml_line(geochild))
            rows.append("      </geometry>")
        else:
            rows.append("      " + child_text)
    rows.append("    </collision>")
    return "\n".join(rows)


def transform(source: str) -> Tuple[str, List[str]]:
    # Preflight parse: original must be valid XML, with expected sensor assembly.
    root = ET.fromstring(source)
    if root.tag != 'robot':
        raise ValueError('Input is not a URDF <robot> document')
    links = {node.get('name'): node for node in root.findall('link')}
    missing = [n for n in UPPER_LINKS if n not in links]
    if missing:
        raise ValueError('Missing expected T800 links: ' + ', '.join(missing))

    j = next((n for n in root.findall('joint')
              if n.get('name') == 't800_chest_to_mid360'), None)
    if j is None or j.find('parent') is None or j.find('child') is None:
        raise ValueError('Missing expected MID360 fixed joint')
    if (j.find('parent').get('link') != 'LINK_WAIST_YAW'
            or j.find('child').get('link') != 'mid360_link'):
        raise ValueError('Unexpected MID360 joint parent/child: please inspect input')

    # Use textual insertion rather than XML rewriting: preserves *all* original
    # Gazebo blocks, plugins, comments, mesh materials, joint settings, etc.
    changed = []
    output = source
    for name in UPPER_LINKS:
        link_pattern = re.compile(
            r'(<link\s+name=["\']' + re.escape(name) +
            r'["\']\s*>)(.*?)(</link\s*>)', re.DOTALL)
        found = list(link_pattern.finditer(output))
        if len(found) != 1:
            raise ValueError('Expected one <link name="{}">, got {}'.format(name, len(found)))
        m = found[0]
        inside = m.group(2)
        if re.search(r'<collision(?:\s|>)', inside):
            print('Already has collision, leaving unchanged: ' + name)
            continue
        visual_match = re.search(r'<visual(?:\s[^>]*)?\s*>.*?</visual\s*>',
                                 inside, flags=re.DOTALL)
        if visual_match is None:
            raise ValueError('{} lacks <visual> geometry'.format(name))
        new_collision = make_collision_from_visual(visual_match.group(0), name)
        # Insert before closing </link>; retain original text byte-for-byte otherwise.
        close_start = m.start(3)
        output = output[:close_start] + "\n" + new_collision + "\n" + output[close_start:]
        changed.append(name)

    verify = ET.fromstring(output)
    verify_links = {link.get('name'): link for link in verify.findall('link')}
    for name in UPPER_LINKS:
        link = verify_links[name]
        collision = link.find('collision')
        visual = link.find('visual')
        if collision is None or visual is None:
            raise AssertionError('Missing collision or visual: ' + name)
        # Geometry and origin must match EXACTLY, including mirror RPY and mesh scale.
        for tag in ('origin', 'geometry'):
            v = visual.find(tag)
            c = collision.find(tag)
            if (v is None) != (c is None):
                raise AssertionError('Mismatch in ' + name + ' ' + tag)
            if v is not None and ET.tostring(v) != ET.tostring(c):
                # Whitespace differences from serialization are okay: normalize tree.
                def canon(x):
                    return (x.tag, tuple(sorted(x.attrib.items())),
                            tuple(canon(y) for y in x))
                if canon(v) != canon(c):
                    raise AssertionError('Mismatch in ' + name + ' ' + tag)
    for name, link in verify_links.items():
        if (name.startswith(('LINK_HIP_', 'LINK_KNEE_', 'LINK_ANKLE_', 'LINK_FOOT_'))
                or name == 'LINK_BASE') and link.find('collision') is not None:
            raise AssertionError('Lower body contains unexpected collision: ' + name)

    # Enforce no gravity for all existing upper links via Gazebo's existing blocks.
    for name in UPPER_LINKS:
        gazebos = [g for g in verify.findall('gazebo') if g.get('reference') == name]
        if not gazebos or not any(g.findtext('gravity') == 'false' for g in gazebos):
            raise AssertionError('{} lacks <gazebo reference="{}"><gravity>false</gravity>'.format(name, name))

    return output, changed


def main():
    here = Path(__file__).resolve().parent
    default_source = Path('urdf/serial_t800_visual_mid360_imu.urdf')
    if not default_source.exists():
        default_source = here / 'urdf/serial_t800_visual_mid360_imu.urdf'
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--input', type=Path, default=default_source,
                    help='Existing, working T800 URDF (never overwritten)')
    ap.add_argument('--output', type=Path, default=None,
                    help='Modified URDF path (default: beside input, *_upper_collision.urdf)')
    args = ap.parse_args()
    src = args.input.expanduser().resolve()
    if not src.exists():
        ap.error('Source URDF not found: {}'.format(src))
    out = (args.output.expanduser().resolve() if args.output else
           src.with_name(src.stem + '_upper_collision.urdf'))
    if src == out:
        ap.error('Output must differ from input to protect the original URDF')
    new_text, names = transform(src.read_text(encoding='utf-8'))
    out.write_text(new_text, encoding='utf-8')
    print('Created: {}'.format(out))
    print('Added collision to {} upper-body links:'.format(len(names)))
    for n in names:
        print('  ' + n)
    print('Original URDF unchanged.')
    print('Kept MID360/IMU fixed-joint, sensors and all existing Gazebo settings.')
    print('All upper-body gravity settings verified false. Lower body has no collision.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, ET.ParseError, AssertionError) as e:
        print('ERROR: ' + str(e), file=sys.stderr)
        sys.exit(1)
