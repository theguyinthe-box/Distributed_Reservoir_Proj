# QuickStart
from root directory, `docker-compose up`

`Ctrl+c` to kill, make sure to `docker-compose down` to cleanup

## Updating ROS2 node
```
cd ros2_ws
colcon build --symlink-install
```

## Parameters
Agent and edge node have configurable parameters below. Parameters are declared in ROS2 nodes, which means they can be set by (1) setting the environment variables (currently set in `*_entrypoint.sh`), (2) using launchfiles in the `launch` directory, or (3) using yaml config files in the `config` directory.
### Agent
`func`: function implemented (`rossler` or `lorenz`)\
`dt`:\
`integrator`:\
`training_length`:\
`eval_length`:\
`batch_size`:
### Edge
`FUNC`: function implemented (`rossler` or `lorenze`)\
`MODEL_TYPE`: model implemented (`reservoir` or `lstm`)\
`RES_DIM`:\
`SPECTRAL_RADIUS`:\
`LEAK_RATE`:\
`N_ITERATIONS`:
