# Dependency imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# Codebase imports
from reservoir import Readout
from functions import dynamical_functions as d
from logger import Logger

class Agent_ROSNode(Node):
    def __init__(self, 
                 func: str, 
                 ic = [0.1,0.1,0.1], 
                 dt = .001,
                 integrator: str = 'RK45',
                 training_length = 500,
                 batch_size = 20):
    
        super().__init__(f'{func}_agent_node')
        # get GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # logger init
        self.get_logger().info("Agent started, waiting for ack signal from edge...")
        
        # store function type
        self.function = func
        self.io_dims = d.function_dims(function)
        self.func = d.return_function(function)
        
        self.logger = None
        self.connected_to_edge = False

        self.get_logger().info("Agent started, waiting to connect to Edge...")
        # Subscribe to Edge before creating publishers
        # Init Subscribers  
        # get params
        self.reservoir_param_subscriber = self.create_subscription(String, 'reservoir_params', self._handle_params)
        # subscribe to reservoir output
        self.data_subscription = self.create_subscription(Float32MultiArray, f'{func}_res_msg', self._handle_reservoir_data, queue_size = 10)
        
        # Publishers
        self.data_publisher = self.create_publisher(Float32MultiArray, f'{function}_agent_msg', self.run, queue_size = 10)        

        # generates the linear layers of the 
        self.model = Readout(self.res_dim, self.input_dim, self.io_dims)

        # initial condition of the system
        self.ic = ic

        # length of each ros message
        self.msg_length = batch_size

        # training params
        self.training_length = training_length

        # data generator params
        self.integrator = integrator
        self.dt = dt

    def _handle_reservoir_data(self, msg):
        '''
        callback to handle incoming reservoir data
        '''
        try:
            reservoir_output = self._msg_to_layer(msg)
            results = self.model(reservoir_output)
            # RESULTS GETS PASSED TO LOGGER
        except Exception as e:
            self.get_logger().error(f"Error handling reservoir data: {e}")

    def _handle_params(self, msg):
        '''
        callback to handle parameter updates from edge
        '''
        if self.logger is None:
            self.logger = Logger(self.function, params, self.get_logger())
        self.get_logger().info(f"Received parameters: {msg.data}")
        self.connected_to_edge = True

    def _msg_to_layer(self, msg):
        '''
        recieve message from edge
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
    
    def _generate_data(self, t0, ic):
        '''
        wrapper for rk45 
        '''
        tspan = np.linspace(t0, self.msg_length*self.dt, self.msg_length)
        sol = solve_ivp(self.func, [t0, self.msg_length*self.dt], ic, t_eval=tspan, method=self.integrator)
        return sol.y, sol.t

    def send_batch_to_res(self,t_curr):
        '''
        generate data and send batch of data to 
        '''
        # Generate data using the dynamical function
        data, time = self._generate_data(t_curr, self.ic)
        self.data_hist.append(data)
        self.time_hist.append(time)
        # Convert output to ROS message and send
        msg = self._data_to_msg(data)
        self._send_data(msg)
        # Update initial conditions
        self.ic = data[:, -1]  # Last state becomes next initial condition
        self.get_logger().debug(f"Generated and sent data batch at t=[{t_curr},{t_curr+self.msg_length*self.dt})")

    def run(self):
        '''
        Main node spin loop: generates data, feeds through reservoir, sends output
        '''
        t_curr = 0.0
        
        while len(self.time_hist) <= self.training_length:
            try:
                self.send_batch_to_res(t_curr)
                #update time for next iteration
                t_curr += self.msg_length * self.dt
            except Exception as e:
                self.get_logger().error(f"Error in node spin: {e}")
                break
    
      