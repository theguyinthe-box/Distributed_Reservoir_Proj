
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
from logger import Logger

class Edge_ROSNode(Node):
    def __init__(self, func:str, 
                 res_dim):
        super().__init__('edge_server_ros_node')    
        #hyper parameters

        self.reservoir_params = {
            "input_dim": d.function_dims(func), 
            "res_dim": 500,
            "spectral_radius": 1.3,
            "leak_rate": 0.15, 
        }

        self.runtime_params = {
            "model_path": "/ros2_ws/model.pt",
        }
        
        self.param_publisher = self.create_publisher(dict, )

        # Publishers / Subscribers
        self.subscription = self.create_subscription(Float64MultiArray, f'{func}_agent_msg', self.handle_input, queue_size = 10)
        self.publisher = self.create_publisher(Float64MultiArray, f'{func}_res_msg', queue_size = 10)

        self.get_logger().info(f"Edge reservoir node ready and waiting for input.")

        # instantiate reservoir 
        self.model = Reservoir(self.reservoir_params('res_dim'), self.reservoir_params('sepctral_radius'), self.reservoir_params('leak_rate'))

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
    
    def _send_data(self, msg):
        '''
        send data to edge network
        '''
        self.data_publisher.publish(msg)
    

    '''
    TODO
    function to handle new connections
    function to queue to incoming messages
    function to take from queue to reservoir and output 
    function to 
    '''    
    