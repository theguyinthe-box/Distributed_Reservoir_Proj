
# dependency imports
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# codebase imports
from reservoir import Reservoir
from functions import dynamical_functions as d

class Edge_ROSNode(Node):
    def __init__(self, 
                 func:str, 
                 res_dim = 500,
                 spectral_radius = 1.0,
                 leak_rate = 0.15,
                 iterations = 20):
        
        super().__init__('edge_server_ros_node')    
        
        #hyper parameters
        self.reservoir_params = {
            "input_dim": d.function_dims(func), 
            "res_dim": res_dim,
            "spectral_radius": spectral_radius,
            "leak_rate": leak_rate,
            "iter": iterations
        }

        self.runtime_params = {
            "model_path": "/ros2_ws/model.pt",
        }
        
        self.param_publisher = self.create_publisher(dict,'reservoir_params', queue_size = 10)

        # Publishers / Subscribers
        self.agent_subscription = self.create_subscription(Float64MultiArray, f'{func}_agent_msg', self._handle_input, queue_size = 10)
        self.output_publisher = self.create_publisher(Float64MultiArray, f'{func}_res_msg', queue_size = 10)
        
        self.get_logger().info(f"Edge reservoir node ready and waiting for input.")

        # instantiate reservoir 
        self.model = Reservoir(res_dim, spectral_radius=spectral_radius, leak_rate = leak_rate)

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
    
    def _handle_input(self, msg):

        input_data = self._msg_to_layer(msg)
        output_data = self.model.forward(input_data, self.reservoir_params('iter'))
        output_msg = self._data_to_msg(output_data)
        self.output_publisher(output_msg)


    '''
    TODO in later iterations
    function to handle new connections
    function to queue to incoming messages
    function to take from queue to reservoir and output 
    '''    
    