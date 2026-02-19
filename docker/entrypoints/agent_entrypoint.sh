#!/bin/bash

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/ros2_ws/install/distributed_reservoir/lib/python3.10/site-packages:$PYTHONPATH

# Launch the agent node
# All parameters must be set via environment variables in compose.yaml

ros2 launch distributed_reservoir agent.launch.py \
  func:=$FUNC \
  dt:=$DT \
  integrator:=$INTEGRATOR \
  training_length:=$TRAINING_LENGTH \
  eval_length:=$EVAL_LENGTH \
  batch_size:=$BATCH_SIZE \
  log_level:=$LOG_LEVEL
