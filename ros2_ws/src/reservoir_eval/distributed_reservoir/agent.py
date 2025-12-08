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
        self.training = False

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
        self.reservoir_param_subscriber = self.create_subscription(dict, 'reservoir_params', self._handle_params)
        # subscribe to reservoir output
        self.data_subscription = self.create_subscription(Float32MultiArray, f'{func}_res_msg', self._handle_reservoir_data, queue_size = 10)
        
        # Publishers
        self.data_publisher = self.create_publisher(Float32MultiArray, f'{function}_agent_msg', self._run, queue_size = 10)        

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
        
        # training setup
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.learning_rate = 0.001

    def _handle_reservoir_data(self, msg):
        '''
        callback to handle incoming reservoir data
        trains the readout layer on every message received
        '''
        if self.training:
            try:
                # Convert message to tensor
                reservoir_output = self._msg_to_layer(msg)
                
                # Get ground truth target from last generated data batch
                if len(self.data_hist) > 0:
                    # Use the last generated data as target
                    target = torch.tensor(self.data_hist[-1], dtype=torch.float32, device=self.device).T
                    # Train on this message immediately
                    self._train_readout(reservoir_output, target)
                    
            except Exception as e:
                self.get_logger().error(f"Error handling reservoir data: {e}")

    def _handle_params(self, msg):
        '''
        callback to handle parameter updates from edge
        '''
        if self.logger is None: ### You use logger like 6 other times and you dont do this there
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
        msg = Float32MultiArray() # is this required?
        msg.data = np.asarray(u, dtype=np.float32).flatten().tolist()
        return msg
    
    def _send_data(self, msg):
        '''
        send data to edge network
        '''
        self.data_publisher.publish(msg)
    
    def _train_readout(self, reservoir_output, target):
        '''
        train the readout layer on a single message
        reservoir_output: tensor of shape (batch_size * input_dim,) or (batch_size, input_dim)
        target: tensor of shape (batch_size, output_dim)
        '''
        try:
            # Ensure proper dimensions for batch training
            if reservoir_output.dim() == 1:
                reservoir_output = reservoir_output.unsqueeze(0)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            
            # Forward pass
            predictions = self.model(reservoir_output)
            
            # Compute loss
            loss = self.criterion(predictions, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.get_logger().debug(f"Training loss: {loss.item():.6f}")
            
        except Exception as e:
            self.get_logger().error(f"Error training readout layer: {e}")
    
    def _generate_data(self, t0, ic):
        '''
        wrapper for rk45 
        '''
        tspan = np.linspace(t0, self.msg_length*self.dt, self.msg_length)
        sol = solve_ivp(self.func, [t0, self.msg_length*self.dt], ic, t_eval=tspan, method=self.integrator)
        return sol.y, sol.t
    
    ### This is fine but technically inconsistent with your naming scheme
    ### Are these functions to meant be exposed? 
    ### Python convention is _ before function name for "private" internal functions
    def _send_batch_to_res(self,t_curr):
        '''
        generate data and send batch of data to 
        '''
        # Generate data using the dynamical function
        data, time = self._generate_data(t_curr, self.ic)
        self.data_hist.append(data)
        self.time_hist.append(time)
        # Convert output to ROS message and send
        ### Why is this 2 functions at this point anyways?
        ### Send Data is a single line anyways
        msg = self._data_to_msg(data)
        self._send_data(msg)
        # Update initial conditions
        self.ic = data[:, -1]  # Last state becomes next initial condition
        self.get_logger().debug(f"Generated and sent data batch at t=[{t_curr},{t_curr+self.msg_length*self.dt})")

    ###same as send_batch_to_res
    def _run(self):
        '''
        Main node spin loop: generates data, feeds through reservoir, sends output
        '''
        t_curr = 0.0
        
        while t_curr <= self.training_length:
            try:
                self.training = True
                self._send_batch_to_res(t_curr)
                #update time for next iteration
                t_curr += self.msg_length * self.dt
            except Exception as e:
                self.get_logger().error(f"Error in node spin: {e}")
                break
    
      