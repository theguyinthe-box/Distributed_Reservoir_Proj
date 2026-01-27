#!/bin/bash

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/ros2_ws/install/distributed_reservoir/lib/python3.10/site-packages:$PYTHONPATH

# Launch the agent node with optional parameters
# Default: lorenz with standard settings
# Override with environment variables: FUNC, DT, INTEGRATOR, TRAINING_LENGTH, EVAL_LENGTH, BATCH_SIZE

FUNC=${FUNC:-lorenz}
DT=${DT:-0.01}
INTEGRATOR=${INTEGRATOR:-RK45}
TRAINING_LENGTH=${TRAINING_LENGTH:-100}
EVAL_LENGTH=${EVAL_LENGTH:-100}
BATCH_SIZE=${BATCH_SIZE:-2}

ros2 launch distributed_reservoir agent.launch.py \
  func:=$FUNC \
  dt:=$DT \
  integrator:=$INTEGRATOR \
  training_length:=$TRAINING_LENGTH \
  eval_length:=$EVAL_LENGTH \
  batch_size:=$BATCH_SIZE
