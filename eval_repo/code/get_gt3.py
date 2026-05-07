import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re

SKIP_MODELS = {'ceiling', 'floor', 'walls'}
SKIP_NAME_PATTERNS = [r'^ground', r'^sun', r'^directional']

#unique labels needing complete remappings
EXACT_LABEL_MAP = {
    'table_and_chairs': 'Table',
    'office_chair':     'Chair',
    'hangout_chair':    'Chair',
    'office_box':       'Storage',
    'mini_fridge':      'Appliance',
    'reception_desk':   'Table',
    'corner':           'Structure',
}

#labels with common substrings to all be mapped to same label
SUBSTRING_RULES = [
    ('chair',       'Chair'),   
    ('computer',    'Appliance'),
    ('whiteboard',  'Wall_Decoration'),
    ('toilet',      'Thing'),    
    ('wastebasket', 'Storage'),  
    ('fridge',      'Appliance'),
    ('desk',        'Table'),    
    ('monitor',     'Appliance'),
    ('couch',       'Couch'),   
    ('sofa',        'Couch'),   
    ('shelf',       'Shelf'),  
]

#return normalised label or unknown if not a string
def normalise_label(raw_type):
    if not isinstance(raw_type, str):
        return 'unknown'
    if raw_type in EXACT_LABEL_MAP:
        return EXACT_LABEL_MAP[raw_type]
    lower = raw_type.lower()
    for substring, label in SUBSTRING_RULES:
        if substring in lower:
            return label
    return raw_type


def parse_pose(pose_text):
    if not pose_text:
        return None
    vals = [float(v) for v in pose_text.strip().split()]
    if len(vals) == 6:
        return vals  # x, y, z, roll, pitch, yaw
    return None

def apply_parent_transform(parent_pose, local_x, local_y, local_z):
    px, py, pz = parent_pose[0], parent_pose[1], parent_pose[2]
    pyaw = parent_pose[5]
    wx = px + local_x * np.cos(pyaw) - local_y * np.sin(pyaw)
    wy = py + local_x * np.sin(pyaw) + local_y * np.cos(pyaw)
    wz = pz + local_z
    return round(wx, 4), round(wy, 4), round(wz, 4)

def get_type_from_name(name):
    return re.sub(r'_\d+$', '', name).lower()

def get_type_from_uri(uri):
    if uri:
        return uri.replace('model://', '').split('/')[0].lower()
    return None

def extract_geometry(element):
    """
    Walk all <collision> and <visual> children of a link/model element and
    return the largest bounding box size found, plus geometry type string.
    Returns (geom_type, size_x, size_y, size_z) or all None if not found.
    """
    best = None
    best_vol = -1

    for search_tag in ('collision', 'visual'):
        for node in element.iter(search_tag):
            geom = node.find('geometry')
            if geom is None:
                continue

            # --- Box ---
            box = geom.find('box')
            if box is not None:
                size_text = box.findtext('size')
                if size_text:
                    sx, sy, sz = [float(v) for v in size_text.strip().split()]
                    vol = sx * sy * sz
                    if vol > best_vol:
                        best_vol = vol
                        best = ('box', sx, sy, sz)

            # --- Cylinder --- treat as bbox of enclosing box
            cyl = geom.find('cylinder')
            if cyl is not None:
                r = float(cyl.findtext('radius') or 0)
                l = float(cyl.findtext('length') or 0)
                vol = (2*r) * (2*r) * l
                if vol > best_vol:
                    best_vol = vol
                    best = ('cylinder', round(2*r, 4), round(2*r, 4), round(l, 4))

            # --- Sphere ---
            sph = geom.find('sphere')
            if sph is not None:
                r = float(sph.findtext('radius') or 0)
                vol = (2*r)**3
                if vol > best_vol:
                    best_vol = vol
                    best = ('sphere', round(2*r, 4), round(2*r, 4), round(2*r, 4))

            # --- Mesh: no size info in SDF, mark as mesh ---
            mesh = geom.find('mesh')
            if mesh is not None and best is None:
                scale_text = mesh.findtext('scale')
                if scale_text:
                    vals = [float(v) for v in scale_text.strip().split()]
                    best = ('mesh', vals[0], vals[1], vals[2])
                else:
                    best = ('mesh', None, None, None)

    if best:
        return best
    return (None, None, None, None)

def parse_world_file(world_path):

    #get .world content
    with open(world_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().lstrip()

    root = ET.fromstring(content)
    world = root.find('world') or root
    objects = []

    for room_model in world.findall('model'):
        room_name = room_model.get('name', '')

        if room_name in SKIP_MODELS:
            continue
        if any(re.match(p, room_name) for p in SKIP_NAME_PATTERNS):
            continue

        room_pose_text = room_model.findtext('pose')
        room_pose = parse_pose(room_pose_text) or [0, 0, 0, 0, 0, 0]

        #gets object info for both nested and frame models

        # --- Nested <model> elements ---
        for obj_model in room_model.findall('model'):
            obj_name = obj_model.get('name', '')
            obj_type = get_type_from_name(obj_name)

            local_pose_text = obj_model.findtext('pose')
            local_pose = parse_pose(local_pose_text)

            if local_pose:
                wx, wy, wz = apply_parent_transform(room_pose, local_pose[0], local_pose[1], local_pose[2])
                world_yaw = round(room_pose[5] + local_pose[5], 4)
                world_roll  = round(room_pose[3] + local_pose[3], 4)
                world_pitch = round(room_pose[4] + local_pose[4], 4)
            else:
                wx, wy, wz = room_pose[0], room_pose[1], room_pose[2]
                world_yaw   = room_pose[5]
                world_roll  = room_pose[3]
                world_pitch = room_pose[4]

            geom_type, sx, sy, sz = extract_geometry(obj_model)

            # centroid z: if we have a height and the pose z is the base, shift up
            centroid_z = round(wz + (sz / 2.0 if sz else 0), 4)

            #stores all relevant gt object data for evaluation
            objects.append({
                'name': obj_name,
                'type': obj_type,
                #'type': normalise_label(obj_type),
                'parent_room': room_name,
                'source': 'nested_model',
                'x': wx, 'y': wy, 'z': wz,
                'centroid_z': centroid_z,
                'roll': world_roll, 'pitch': world_pitch, 'yaw': world_yaw,
                'geom_type': geom_type,
                'bbox_x': sx, 'bbox_y': sy, 'bbox_z': sz,
            })

        # --- <frame> elements ---
        for frame in room_model.findall('frame'):
            frame_name = frame.get('name', '')
            if '::__model__' not in frame_name:
                continue

            obj_name = frame_name.replace('::__model__', '')
            obj_type = get_type_from_name(obj_name)

            local_pose_text = frame.findtext('pose')
            local_pose = parse_pose(local_pose_text)

            if local_pose:
                wx, wy, wz = apply_parent_transform(room_pose, local_pose[0], local_pose[1], local_pose[2])
                world_yaw   = round(room_pose[5] + local_pose[5], 4)
                world_roll  = round(room_pose[3] + local_pose[3], 4)
                world_pitch = round(room_pose[4] + local_pose[4], 4)
            else:
                wx, wy, wz = room_pose[0], room_pose[1], room_pose[2]
                world_yaw   = room_pose[5]
                world_roll  = room_pose[3]
                world_pitch = room_pose[4]

            model_type = None
            geom_type, sx, sy, sz = None, None, None, None
            link_name = obj_name + '::link'
            for link in room_model.findall('link'):
                if link.get('name') == link_name:
                    uri = link.findtext('.//uri')
                    model_type = get_type_from_uri(uri)
                    geom_type, sx, sy, sz = extract_geometry(link)
                    break

            centroid_z = round(wz + (sz / 2.0 if sz else 0), 4)

            #stores all relevant gt object data for evaluation
            objects.append({
                'name': obj_name,
                'type': model_type or obj_type,
                #'type': normalise_label(model_type or obj_type),
                'parent_room': room_name,
                'source': 'frame',
                'x': wx, 'y': wy, 'z': wz,
                'centroid_z': centroid_z,
                'roll': world_roll, 'pitch': world_pitch, 'yaw': world_yaw,
                'geom_type': geom_type,
                'bbox_x': sx, 'bbox_y': sy, 'bbox_z': sz,
            })

    return pd.DataFrame(objects)

#returns objects grouped by normalised label
def get_gt_by_label(world_path="../final.world"):

    #get objects from world file
    df = parse_world_file(world_path)

    #group into labels, store relevant object info for evaluation
    gt_by_label = {}
    for _, row in df.iterrows():
        label = row['type']
        label = normalise_label(row['type'])
        entry = {
            'position': [row['x'], row['y'], row['z']],
            'centroid': [row['x'], row['y'], row['centroid_z']],
            'yaw': row['yaw'],
            'bbox': [row['bbox_x'], row['bbox_y'], row['bbox_z']],
            'geom_type': row['geom_type'],
            'name': row['name'],
        }
        gt_by_label.setdefault(label, []).append(entry)


    return gt_by_label

#returns centroid and yaw of rooms in world file, used for room recognition evaluation
def get_gt_rooms(world_path="../final.world"):

    #room names not included in the rosbag
    del_names = ['PublicMeetingRoomA', 'PublicMeetingRoomC', 'BackEntrance', 'Cubicle']

    with open(world_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().lstrip()
    
    root = ET.fromstring(content)
    world = root.find('world') or root
    
    gt_rooms = {}
    #gets centroid, yaw and name from world file

    for room_model in world.findall('model'):
        room_name = room_model.get('name', '')
        
        if room_name in SKIP_MODELS:
            continue
        if any(re.match(p, room_name) for p in SKIP_NAME_PATTERNS):
            continue
        
        room_pose = parse_pose(room_model.findtext('pose')) or [0,0,0,0,0,0]
        
        #add info
        gt_rooms[room_name] = {
            'centroid': [room_pose[0], room_pose[1], room_pose[2]],
            'yaw': room_pose[5],
        }

    #remove unexplored rooms which would skew evaluation
    keys = []
    for key in gt_rooms:
        for name in del_names:
            if name in key:
                keys.append(key)
    
    for key in keys:
        del gt_rooms[key]

    #add large open space as not specified in world file but is a room in the rosbag
    gt_rooms['OpenSpace'] = {'centroid': [9.2, 12.2, 0.0], 'yaw': 0.0}
    
    return gt_rooms


if __name__ == "__main__":
    #specifies world path to gazebo file
    world_path = "../final.world"

    #parse would file
    df = parse_world_file(world_path)

    #print overview of object extraction
    print(f"Total objects parsed: {len(df)}")
    print(f"\nObject type counts:")
    if len(df) > 0:
        print(df['type'].value_counts())
        print(f"\nAll objects:")
        print(df.to_string())
        df.to_csv("ground_truth_objects.csv", index=False)
        print("\nSaved to ground_truth_objects.csv")
    else:
        print("No objects found - check world file path and structure")