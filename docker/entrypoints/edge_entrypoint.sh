#!/bin/bash

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/ros2_ws/install/distributed_reservoir/lib/python3.10/site-packages:$PYTHONPATH

# Launch the edge server node
# All parameters must be set via environment variables in compose.yaml

ros2 launch distributed_reservoir edge_server.launch.py \
  func:=$FUNC \
  model_type:=$MODEL_TYPE \
  res_dim:=$RES_DIM \
  spectral_radius:=$SPECTRAL_RADIUS \
  leak_rate:=$LEAK_RATE \
  n_iterations:=$N_ITERATIONS \
  log_level:=$LOG_LEVEL