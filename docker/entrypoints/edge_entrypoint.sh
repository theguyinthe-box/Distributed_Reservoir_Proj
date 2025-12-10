#!/bin/bash

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/ros2_ws/install/distributed_reservoir/lib/python3.10/site-packages:$PYTHONPATH

python3 -c "from distributed_reservoir.edge_server import main; main()"