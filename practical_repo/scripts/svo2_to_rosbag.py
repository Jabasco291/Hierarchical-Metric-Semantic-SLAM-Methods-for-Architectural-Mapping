"""
Convert exported ZED SVO2 data (RGB frames, depth frames, traj.txt) to a ROS2 bag.

Expected input directory structure (as produced by svo2_export.py):
    <output_folder>/
        results/
            frame000000.jpg
            frame000001.jpg
            ...
            depth000000.png
            depth000001.png
            ...
        traj.txt          <- one line per frame, 16 floats (row-major 4x4 c2w matrix)

Requirements:
    pip install rosbags opencv-python numpy

Usage:
    python to_ros2bag.py --input ../graphs/hccr --output hccr.bag [options]

Camera intrinsics: ZED 2 @ 720p defaults are baked in below.
Override them with --fx --fy --cx --cy --width --height if your resolution differs.
"""

import argparse
import os
import sys
import struct
import numpy as np
import cv2
from pathlib import Path
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros2_humble import (
    builtin_interfaces__msg__Time as Time,
    std_msgs__msg__Header as Header,
    sensor_msgs__msg__Image as Image,
    sensor_msgs__msg__CameraInfo as CameraInfo,
    nav_msgs__msg__Odometry as Odometry,
    geometry_msgs__msg__PoseWithCovariance as PoseWithCovariance,
    geometry_msgs__msg__TwistWithCovariance as TwistWithCovariance,
    geometry_msgs__msg__Pose as Pose,
    geometry_msgs__msg__Point as Point,
    geometry_msgs__msg__Quaternion as Quaternion,
    geometry_msgs__msg__Twist as Twist,
    geometry_msgs__msg__Vector3 as Vector3,
)

# scipy is only needed for rotation matrix -> quaternion conversion
try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to (x, y, z, w) quaternion."""
    if HAS_SCIPY:
        r = Rotation.from_matrix(R)
        x, y, z, w = r.as_quat()  # scipy returns (x, y, z, w)
        return x, y, z, w
    # Manual fallback (Shepperd's method)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def make_timestamp(frame_idx, fps=30.0):
    """Convert frame index to nanosecond timestamp."""
    t_sec = frame_idx / fps
    sec = int(t_sec)
    nanosec = int((t_sec - sec) * 1e9)
    return sec, nanosec


def to_ns(sec, nanosec):
    return int(sec) * 10**9 + int(nanosec)


def make_camera_info(header, fx, fy, cx, cy, width, height):
    """Build a CameraInfo message (no distortion assumed - ZED rectified output)."""
    K = [fx, 0.0, cx,
         0.0, fy, cy,
         0.0, 0.0, 1.0]
    R = [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
    P = [fx, 0.0, cx, 0.0,
         0.0, fy, cy, 0.0,
         0.0, 0.0, 1.0, 0.0]
    from rosbags.typesys.stores.ros2_humble import (
        sensor_msgs__msg__RegionOfInterest as RegionOfInterest,
    )
    roi = RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
    return CameraInfo(
        header=header,
        width=width,
        height=height,
        distortion_model='plumb_bob',
        d=np.zeros(5, dtype=np.float64),
        k=np.array(K, dtype=np.float64),
        r=np.array(R, dtype=np.float64),
        p=np.array(P, dtype=np.float64),
        binning_x=0,
        binning_y=0,
        roi=roi,
    )


def bgr_to_ros_image(bgr_img, header, encoding='rgb8'):
    """Convert an OpenCV BGR image to a ROS2 Image message."""
    if encoding == 'rgb8':
        img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        img = bgr_img
    h, w = img.shape[:2]
    step = w * img.shape[2] if len(img.shape) == 3 else w
    return Image(
        header=header,
        height=h,
        width=w,
        encoding=encoding,
        is_bigendian=0,
        step=step,
        data=img.flatten().astype(np.uint8),
    )


def depth_to_ros_image(depth_uint16, header):
    """
    Convert a uint16 depth image (mm) to a ROS2 Image message with encoding 16UC1.
    The ZED export script stored depth_m * 1000 as uint16, so units are millimetres,
    which is exactly what ROS depth_image_proc expects with 16UC1.
    """
    h, w = depth_uint16.shape
    return Image(
        header=header,
        height=h,
        width=w,
        encoding='16UC1',
        is_bigendian=0,
        step=w * 2,
        data=depth_uint16.flatten().view(np.uint8),
    )


def pose_matrix_to_odometry(c2w, header, child_frame_id='camera'):
    """Convert a 4x4 camera-to-world matrix to a nav_msgs/Odometry message."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    x, y, z, w = rotation_matrix_to_quaternion(R)
    zero_cov = np.zeros(36, dtype=np.float64)
    return Odometry(
        header=header,
        child_frame_id=child_frame_id,
        pose=PoseWithCovariance(
            pose=Pose(
                position=Point(x=float(t[0]), y=float(t[1]), z=float(t[2])),
                orientation=Quaternion(x=float(x), y=float(y), z=float(z), w=float(w)),
            ),
            covariance=zero_cov,
        ),
        twist=TwistWithCovariance(
            twist=Twist(
                linear=Vector3(x=0.0, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            covariance=zero_cov,
        ),
    )


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(input_dir, output_bag, fps, fx, fy, cx, cy, width, height, frame_id, start_frame, end_frame):
    input_dir = Path(input_dir)
    results_dir = input_dir / 'results'
    traj_file = input_dir / 'traj.txt'

    # Load poses
    if not traj_file.exists():
        print(f"ERROR: traj.txt not found at {traj_file}")
        sys.exit(1)

    poses_raw = np.loadtxt(traj_file)  # shape (N, 16)
    if poses_raw.ndim == 1:
        poses_raw = poses_raw[np.newaxis, :]

    n_poses = len(poses_raw)
    print(f"Loaded {n_poses} poses from traj.txt")

    # Discover frames
    rgb_files = sorted(results_dir.glob('frame*.jpg'))
    dep_files = sorted(results_dir.glob('depth*.png'))

    if len(rgb_files) != len(dep_files):
        print(f"WARNING: {len(rgb_files)} RGB frames but {len(dep_files)} depth frames")

    n_frames = min(len(rgb_files), len(dep_files), n_poses)
    print(f"Total usable frames: {n_frames}")

    end_frame = min(end_frame if end_frame >= 0 else n_frames, n_frames)
    frame_range = range(start_frame, end_frame)
    print(f"Converting frames {start_frame} to {end_frame - 1}")

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # Topic names expected by Hydra-ROS
    TOPICS = {
        'rgb':        ('/camera/rgb/image_rect_color',    'sensor_msgs/msg/Image'),
        'depth':      ('/camera/depth_registered/image_rect', 'sensor_msgs/msg/Image'),
        'cam_info':   ('/camera/rgb/camera_info',         'sensor_msgs/msg/CameraInfo'),
        'odometry':   ('/odom',                           'nav_msgs/msg/Odometry'),
    }

    if Path(output_bag).exists():
        import shutil
        shutil.rmtree(output_bag)

    with Writer(output_bag, version=9) as writer:
        connections = {}
        for key, (topic, msgtype) in TOPICS.items():
            connections[key] = writer.add_connection(topic, msgtype, typestore=typestore)

        for i in frame_range:
            sec, nanosec = make_timestamp(i, fps)
            timestamp_ns = to_ns(sec, nanosec)

            header = Header(
                stamp=Time(sec=sec, nanosec=nanosec),
                frame_id=frame_id,
            )

            # --- RGB ---
            bgr = cv2.imread(str(rgb_files[i]))
            if bgr is None:
                print(f"WARNING: Could not read {rgb_files[i]}, skipping frame {i}")
                continue
            rgb_msg = bgr_to_ros_image(bgr, header, encoding='rgb8')
            writer.write(connections['rgb'], timestamp_ns,
                         typestore.serialize_cdr(rgb_msg, 'sensor_msgs/msg/Image'))

            # --- Depth ---
            dep = cv2.imread(str(dep_files[i]), cv2.IMREAD_UNCHANGED)
            if dep is None:
                print(f"WARNING: Could not read {dep_files[i]}, skipping frame {i}")
                continue
            dep_msg = depth_to_ros_image(dep, header)
            writer.write(connections['depth'], timestamp_ns,
                         typestore.serialize_cdr(dep_msg, 'sensor_msgs/msg/Image'))

            # --- CameraInfo ---
            cam_info_msg = make_camera_info(header, fx, fy, cx, cy, width, height)
            writer.write(connections['cam_info'], timestamp_ns,
                         typestore.serialize_cdr(cam_info_msg, 'sensor_msgs/msg/CameraInfo'))

            # --- Odometry ---
            c2w = poses_raw[i].reshape(4, 4)
            odom_msg = pose_matrix_to_odometry(c2w, header)
            writer.write(connections['odometry'], timestamp_ns,
                         typestore.serialize_cdr(odom_msg, 'nav_msgs/msg/Odometry'))

            if i % 50 == 0:
                print(f"  Frame {i}/{end_frame - 1}...", end='\r')

    print(f"\nDone! Bag written to: {output_bag}")
    print(f"\nTo verify:  ros2 bag info {output_bag}")
    print(f"To play:    ros2 bag play {output_bag} --clock")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ZED 2 defaults at 720p (1280x720).
# Override with --fx etc. if you used a different resolution.
# To get exact values from your SVO: run `python to_ros2bag.py --print-intrinsics`
# after adding a small snippet to read them via pyzed (see bottom of this file).

ZED2_720P = dict(
    fx=1063.41845703125,
    fy=1063.41845703125,
    cx=969.0910034179688,
    cy=514.6602783203125,
    width=1920,
    height=1080,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ZED SVO2 export to ROS2 bag')
    parser.add_argument('--input',  required=True,
                        help='Path to the export folder (contains results/ and traj.txt)')
    parser.add_argument('--output', default='hccr.bag',
                        help='Output bag path (default: hccr.bag)')
    parser.add_argument('--fps',    type=float, default=30.0,
                        help='Frame rate used to synthesise timestamps (default: 30)')
    parser.add_argument('--fx',     type=float, default=ZED2_720P['fx'])
    parser.add_argument('--fy',     type=float, default=ZED2_720P['fy'])
    parser.add_argument('--cx',     type=float, default=ZED2_720P['cx'])
    parser.add_argument('--cy',     type=float, default=ZED2_720P['cy'])
    parser.add_argument('--width',  type=int,   default=ZED2_720P['width'])
    parser.add_argument('--height', type=int,   default=ZED2_720P['height'])
    parser.add_argument('--frame-id', default='camera', dest='frame_id',
                        help='ROS frame_id for all messages (default: camera)')
    parser.add_argument('--start',  type=int, default=0,
                        help='Start frame index (default: 0)')
    parser.parameter = None
    parser.add_argument('--end',    type=int, default=-1,
                        help='End frame index exclusive (default: all frames)')
    args = parser.parse_args()

    convert(
        input_dir=args.input,
        output_bag=args.output,
        fps=args.fps,
        fx=args.fx, fy=args.fy,
        cx=args.cx, cy=args.cy,
        width=args.width, height=args.height,
        frame_id=args.frame_id,
        start_frame=args.start,
        end_frame=args.end,
    )