import sys
import os
import numpy as np
import cv2
import pyzed.sl as sl


def export_data(svo_path, output_folder):
    # 1. Create Output Folders
    rgb_path = os.path.join(output_folder, "results")
    depth_path = os.path.join(output_folder, "results")
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    # 2. Open the ZED Camera in "SVO Playback" mode
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Best quality depth
    init_params.coordinate_units = sl.UNIT.METER  # Use Meters
    init_params.svo_real_time_mode = False
    #init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO: {err}")
        return
    
    cal = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    print(f"fx={cal.fx}, fy={cal.fy}, cx={cal.cx}, cy={cal.cy}, w={cal.image_size.width}, h={cal.image_size.height}")

    # 3. Positional Tracking
    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.enable_area_memory = True
    zed.enable_positional_tracking(tracking_params)
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    pose = sl.Pose()

    poses_list = []

    print(f"Processing {svo_path}")

    i = 0
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)

            if state == sl.POSITIONAL_TRACKING_STATE.OK:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                img_cv = image_zed.get_data()
                # RGBA -> RGB
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2RGB)

                # Depth Map
                zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                depth_data = depth_zed.get_data()

                depth_data[np.isnan(depth_data)] = 0
                depth_data[np.isinf(depth_data)] = 0
                depth_uint16 = (depth_data * 1000).astype(np.uint16)
                filename = f"{i:06d}"
                cv2.imwrite(f"{rgb_path}/frame{filename}.jpg", img_cv)
                cv2.imwrite(f"{depth_path}/depth{filename}.png", depth_uint16)
                c2w = pose.pose_data(sl.Transform()).m  # Get 4x4 numpy matrix

                flat_pose = c2w.flatten()
                poses_list.append(flat_pose)

                print(f"Frame {i} captured.", end='\r')
                i += 1
        else:
            break

    zed.close()
    print(f"\nWriting traj.txt with {len(poses_list)} frames...")
    with open(os.path.join(output_folder, "traj.txt"), "w") as f:
        for p in poses_list:
            line = " ".join(f"{val:.6f}" for val in p)
            f.write(line + "\n")

    print("Done!")


if __name__ == "__main__":
    #SVO_FILE = "....path_to_file/HD720_SN37409870_00-27-27.svo2"
    #OUTPUT_DIR = "....path_to_output/dir-name"

    SVO_FILE = "../../hccr3.svo2"
    OUTPUT_DIR = "../graphs/hccr3"

    export_data(SVO_FILE, OUTPUT_DIR)
