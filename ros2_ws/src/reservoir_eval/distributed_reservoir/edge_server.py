# dependency imports
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from typing import List, Dict, Any, Optional

# codebase imports
from reservoir import Reservoir
from functions import dynamical_functions as d

class Edge_ROSNode(Node):
    def __init__(self, func: str = 'lorenz', 
                 res_dim: int = 500,
                 spectral_radius: float = 1.6,
                 leak_rate: float = 0.15,
                 n_iterations: int = 20):
        
        super().__init__('edge_server_ros_node')    
        
        # get GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # hyper parameters
        input_dim = d.function_dims(func)
        self.reservoir_params = {
            "input_dim": input_dim, 
            "res_dim": res_dim,
            "spectral_radius": spectral_radius,
            "leak_rate": leak_rate, 
            "iter": n_iterations
        }
        
        # Publishers / Subscribers
        self.param_publisher = self.create_publisher(dict, 'reservoir_params', queue_size = 1)
        self.subscription = self.create_subscription(Float32MultiArray, f'{func}_agent_msg', self._handle_input, queue_size = 10)
        self.output_publisher = self.create_publisher(Float32MultiArray, f'{func}_res_msg', queue_size = 10)

        # instantiate reservoir 
        self.model = Reservoir(res_dim, spectral_radius=spectral_radius, leak_rate=leak_rate)
        
        # publish parameters to agent
        self._publish_params()
        
        self.get_logger().info(f"Edge reservoir node ready. Reservoir: {res_dim}D, Input: {input_dim}D")

    def _msg_to_layer(self, msg):
        '''
        recieve message from agent
        tranlate to nodes
        '''
        return torch.tensor(msg.data, dtype=torch.float32, device=self.device)
   
    def _data_to_msg(self, u):
        '''
        translate data to ROS msg
        '''
        msg = Float32MultiArray()
        msg.data = np.asarray(u, dtype=np.float32).flatten().tolist()
        return msg
    
    def _publish_params(self):
        '''
        publish reservoir parameters to agent
        '''
        self.param_publisher.publish(self.reservoir_params)
        self.get_logger().info(f"Published reservoir parameters to agent")
    
    def _handle_input(self, msg):
        '''
        callback to receive data from agent
        run it through the reservoir
        send reservoir output back to agent with sequence number
        '''
        try:
            # Convert incoming message to tensor
            input_data = self._msg_to_layer(msg)
            self.get_logger().debug(f"Received input data (seq {msg.seq}) with shape: {input_data.shape}")
            
            # Process through reservoir
            output_data = self.model.forward(input_data, n_steps=self.reservoir_params['iter'])
            self.get_logger().debug(f"Reservoir output shape: {output_data.shape}")
            
            # Convert output to message and publish
            output_msg = self._data_to_msg(output_data)
            output_msg.seq = msg.seq  # Preserve sequence number
            self.output_publisher.publish(output_msg)
            self.get_logger().debug(f"Published reservoir output for seq {msg.seq}")
            
        except Exception as e:
            self.get_logger().error(f"Error in _handle_input: {e}")
            import traceback
            traceback.print_exc()
