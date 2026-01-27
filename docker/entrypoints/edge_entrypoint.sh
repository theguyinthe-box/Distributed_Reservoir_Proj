#!/bin/bash

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/ros2_ws/install/distributed_reservoir/lib/python3.10/site-packages:$PYTHONPATH

# Launch the edge server node with optional parameters
# Default: lorenz reservoir
# Override with environment variables: FUNC, MODEL_TYPE, RES_DIM, SPECTRAL_RADIUS, LEAK_RATE, N_ITERATIONS

FUNC=${FUNC:-lorenz}
MODEL_TYPE=${MODEL_TYPE:-reservoir}
RES_DIM=${RES_DIM:-256}
SPECTRAL_RADIUS=${SPECTRAL_RADIUS:-1.1}
LEAK_RATE=${LEAK_RATE:-0.15}
N_ITERATIONS=${N_ITERATIONS:-20}

ros2 launch distributed_reservoir edge_server.launch.py \
  func:=$FUNC \
  model_type:=$MODEL_TYPE \
  res_dim:=$RES_DIM \
  spectral_radius:=$SPECTRAL_RADIUS \
  leak_rate:=$LEAK_RATE \
  n_iterations:=$N_ITERATIONS