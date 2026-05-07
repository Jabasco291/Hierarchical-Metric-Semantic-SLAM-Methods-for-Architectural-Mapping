# practical_repo

This repository contains the instructions to deploy the three candidate methods onto the custom office and hccr dataset
---

## Prequisites:
These are the expected requirements to execute the rest of the guide

- System running Ubuntu 22.04 LTS
- Docker installed
- ROS2 Humble Installed
- Nvidia GPU with CUDA and TensorRT capabilities
- Install ZED SDK: https://www.stereolabs.com/en-gb/developers/release (for svo2 conversion only)

## Project Structure
```text
dsg_execution/
│
├── README.md
│
├── datasets/
│   ├── clio/
│   │   ├── custom_office_ros1.bag         #Clio
│   │   ├── tasks.yaml
│   │   └── region_tasks.yaml
│   ├── custom_office/
│   │   ├── custom_office_0.db3            #Hydra
│   │   ├── metadata.yaml
│   └── hccr/
│       ├── hccr.svo2                      #Raw ZED Recording
│       └── hccr3.bag                      #Pre-Converted Bag                      
│           ├── hccr3.bag.db3                  
│           └── metadata.yaml                      
│
├── hydra_ws/                  
│   └── src/
│
├── models/
│   └── ade20k-efficientvit_seg_l2.onnx    #Hydra semantic inference model
│
├── config/
│   └── clio/
│       ├── custom_office_main.launch
│       ├── house_clip.yaml
│       ├── pipeline.yaml
│       └── realsense_fine.yaml
│
└── scripts/
    ├── svo2_export.py
    ├── svo2_to_rosbag.py
    ├── s-graph_to_json.py
    ├── extract_times.py
    └── odom_to_tf.py
```

# DATASET DOWNLOAD:
### Due to large file sizes, datasets are not included. To run the methods, please follow these links to download the datasets. Once downloaded, insert to correct place in project structure.
Warning, they are large capacity.
- Custom Office : https://cf-my.sharepoint.com/:f:/g/personal/wilkesj5_cardiff_ac_uk/IgArdzA-BUlgSazG0jV-SvVMAVDUsocts74sYcEKOUEdvng?e=eIhh1e
- Clio: https://cf-my.sharepoint.com/:f:/g/personal/wilkesj5_cardiff_ac_uk/IgAz72HnCcX9SrFh_Zs_1QUnAVZaDku9tQ0asS8eKKaDnsI?e=oGVxBI
- HCCR: https://cf-my.sharepoint.com/:f:/g/personal/wilkesj5_cardiff_ac_uk/IgDu95GHhd-sS7wEDgQDn01eAcUCN0awiIPuFe3rp1PRnDE?e=2wP9cr

# DOCKER SETUP:
### Due to large file sizes, Docker images are not included. Please follow these instructions for installing the relevant Dockers for running graph building methods.

## Clio

1. Clone Repo from this [Link](https://github.com/chadrs2/ClioDocker)
2. Follow install instructions up to Build the Docker image (step included)

## Hydra

```bash
cd hydra_ws/src
git clone https://github.com/MIT-SPARK/Hydra-ROS.git hydra_ros

vcs import . < hydra_ros/install/ros2_docker.yaml
```
#### Note
If GitHub blocks concurrent requests with 
```
kex_exchange_identification: read: Connection reset by peer
```
run:
```
vcs import . < hydra_ros/install/ros2_docker.yaml --workers 1
```


# METHOD DEPLOYMENT:
## Vital Execution Note:
### All terminal commands must be executed from the root practical_repo directory unless stated otherwise. This ensures file paths are interpreted correctly


# S-Graphs+

### Install S-Graphs+ 
- Follow the [Official S-Graphs+ Installation Guide.](https://snt-arg.github.io/lidar_situational_graphs/1.installation/)
---

### Source Workspace (do this for every terminal)
```bash
source ~/workspaces/s_graphs/install/setup.bash
```

### Terminal 1: 
```bash
# Launch S-Graphs+
ros2 launch lidar_situational_graphs s_graphs_launch.py compute_odom:=false lidar_topic:=/velodyne2/velodyne_points2 odom_topic:=/odom base_frame:=base_link
```
### Terminal 2: 
```bash
# Publish transform
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link top
```
### Terminal 3: 
```bash
# Start Rviz2 for visualisation
rviz2 -d ~/workspaces/s_graphs/install/lidar_situational_graphs/share/lidar_situational_graphs/rviz/s_graphs_ros2.rviz
```
### Terminal 4: 
```bash
# Navigate to bag directory
cd datasets/custom_office

# Play ROS2 Bag
ros2 bag play custom_office_0.db3 --clock
```

# Hydra

### Start Docker
```bash
sudo systemctl start docker
```

### Build Image
```bash
cd hydra_ws/src/hydra_ros/docker
sudo make build PROFILE=dev
```

### Setup Environment (Terminal 1)
 
```bash
#Navigate to root of dsg_execution directory

#Set Dataset path 
echo "DATASETS_PATH=${PWD}" > hydra_ws/src/hydra_ros/docker/.env

#Allow Docker GUI access
xhost +local:root
```

### Start and Enter container (Terminal 2)
 
```bash
cd hydra_ws/src/hydra_ros/docker
sudo make up PROFILE=dev
sudo make shell PROFILE=dev
```

### Setup Hydra Config and Semantics (Terminal 1)

```bash
#Copy Semantic Inference Model into Container
sudo docker cp models/ade20k-efficientvit_seg_l2.onnx hydra-dev:/root/.semantic_inference/
```

### Build and Launch Hydra (Terminal 2)
 
```bash
#build
colcon build --symlink-install --continue-on-error
#source
source install/setup.bash

#Execute Launch depending on Dataset

#For custom office dataset
ros2 launch hydra_ros custom_office_main.launch.yaml

#For HCCR dataset
ros2 launch hydra_ros hccr.launch.yaml
```

## Dataset: custom_office

### Enter container and play bag (Terminal 3)
 
```bash
#Enter Container
cd hydra_ws/src/hydra_ros/docker
sudo make shell PROFILE=dev

#Set TF overrides
echo "/tf_static: {depth: 1, durability: transient_local}" > ~/.tf_overrides.yaml

#Play Bag
ros2 bag play /root/data/datasets/custom_office/custom_office_0.db3 --clock --qos-profile-overrides-path ~/.tf_overrides.yaml
```

## Dataset: HCCR

### Static Transform(Terminal 3)
```bash
#Enter Container
cd hydra_ws/src/hydra_ros/docker
sudo make shell PROFILE=dev

#Publish Transform
ros2 run tf2_ros static_transform_publisher \
  --x 0 --y 0 --z 0 \
  --qx 0 --qy 0 --qz 0 --qw 1 \
  --frame-id camera \
  --child-frame-id base_link
```

### Odometry to TF (Terminal 4)
```bash
#Enter Container
cd hydra_ws/src/hydra_ros/docker
sudo make shell PROFILE=dev

#Publish Transform
python3 /root/data/scripts/odom_to_tf.py
```

### Enter container and play bag (Terminal 5)
 
```bash
#Enter Container
cd hydra_ws/src/hydra_ros/docker
sudo make shell PROFILE=dev

#Set TF overrides
echo "/tf_static: {depth: 1, durability: transient_local}" > ~/.tf_overrides.yaml

#Play Bag
ros2 bag play /root/data/datasets/hccr/hccr3.bag/hccr3.bag.db3 --clock --qos-profile-overrides-path ~/.tf_overrides.yaml
```

# Clio

### Start Docker
```bash
sudo systemctl start docker
```

### Allow GUI Access (Terminal 1)
 
```bash
#Allow Docker GUI access
xhost +local:root
```

### Run Docker (Terminal 2)
```bash
#RUN FROM dsg_execution DIRECTORY
sudo docker run -it --user ros --network=host --ipc=host \
--name clio-container \
-v ${PWD}:/clio_dataset \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
--gpus all --runtime nvidia \
--env="QT_X11_NO_MITSHM=1" \
--env="NVIDIA_DRIVER_CAPABILITIES=all" \
--env="NVIDIA_VISIBLE_DEVICES=all" \
--device=/dev/dri:/dev/dri \
--env=DISPLAY \
clio_ros1
```

### Copy Config (Terminal 2)
```bash
#Copy launch file
cp /clio_dataset/config/clio/custom_office_main.launch /home/ros/catkin_ws/src/clio/clio_ros/launch/custom_office_main.launch 

#Copy Parameters
cp /clio_dataset/config/clio/house_clip.yaml /home/ros/catkin_ws/src/clio/clio_ros/config/segmentation/house_clip.yaml
cp /clio_dataset/config/clio/pipeline.yaml /home/ros/catkin_ws/src/clio/clio_ros/config/realsense/pipeline.yaml
cp /clio_dataset/config/clio/realsense_fine.yaml /home/ros/catkin_ws/src/clio/clio_ros/config/realsense/realsense_fine.yaml
```

### Source Workspace (Terminal 2)
```bash
source ~/catkin_ws/devel/setup.bash
source ~/environments/clio_ros/bin/activate
```

### Launch Clio (Terminal 2)
```bash
roslaunch clio_ros custom_office_main.launch \
     object_tasks_file:=/clio_dataset/datasets/clio/tasks.yaml \
     place_tasks_file:=/clio_dataset/datasets/clio/region_tasks.yaml
```

### Play Rosbag (Terminal 3)
```bash
#Start Container
sudo docker exec -it -u root clio-container bash

#Source ROS
source /opt/ros/noetic/setup.bash

#Play bag, only required topics and 0.25 rate
rosbag play /clio_dataset/datasets/clio/custom_office_ros1.bag --clock \
  --topics \
  /camera/realsense_d435/image_raw \
  /camera/realsense_d435/depth/image_raw \
  /camera/realsense_d435/camera_info \
  /odom /tf /tf_static \
  --rate 0.25
```


# DATA COLLECTION 
## ZED Camera to ROS2 Bag
These instructions demonstrate how to take a svo2 recording (from the ZED camera), and process it into a ROS2 bag for Hydra to use.

### Export Images form Svo2 file
```bash
#
#Path to svo2 and output need to be set in the script
python3 scripts/svo2_export.py
```
### Convert Images to ROS2 Bag:
```bash
python3 scripts/svo2_to_rosbag.py --input path/to/exported_folder --output datasets/hccr/hccr3.bag
```


## S-Graph+ Output to JSON (for saving the output of a S-Graphs+ run):

These steps take an actively running S-Graphs+ graph and saves it in JSON format

## PREREQUISITES:
- Run S-Graphs+ on a ROS2 Bag
- ROS2 Bag has finished playing, and S-Graphs+ is still running

## Save Graph to JSON File:
```bash
# (Ensure ROS2 workspace is sourced)
python3 scripts/s-graph_to_json.py --output output_filename.json
```

## Process Timing Logs:
```bash
python3 scripts/extract_times.py --log_dir ../graphs/s-graphs/your_run_folder/
```
