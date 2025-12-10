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
from typing import List, Dict, Any, Optional

# Codebase imports
from reservoir import Readout
from functions import dynamical_functions as d
from logger import Logger

class Agent_ROSNode(Node):
    def __init__(self, 
                 func: str, 
                 ic = [0.1,0.1,0.1], 
                 dt = .01,
                 integrator: str = 'RK45',
                 training_length = 100,
                 eval_length = 300,
                 batch_size = 20):
    
        super().__init__(f'{func}_agent_node')

        # get GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training = False

        self.res_dim = None

        # store function type
        self.function = func
        self.function_dims = d.function_dims(function)
        self.func = d.return_function(function)
        
        self.logger = Logger('{func}_agent_node')

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
        self.model = Readout(self.res_dim, self.function_dim)

        # initial condition of the system
        self.t_curr = 0.0
        self.ic = ic

        # length of each ros message
        self.msg_length = batch_size

        # training params
        self.training_length = training_length

        # data generator params
        self.integrator = integrator
        self.dt = dt
        self.t_curr = 0
        self.max_time = training_length + eval_length  # stop after time/dt runs

        # training setup
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.learning_rate = 0.01

        self.last_pred = None

    def _handle_reservoir_data(self, msg):
        '''
        callback to handle incoming reservoir data
        trains the readout layer on every message received
        '''
        if self.training:
            try:
                # Convert message to tensor
                reservoir_output = self._msg_to_layer(msg)
                #record rtt
                rtt_seconds = self.logger.record_roundtrip_time(msg.seq)
                self.get_logger().debug(f"Roundtrip time for seq {msg.seq}: {rtt_seconds*1000:.2f} ms")
                # Get ground truth target from last generated data batch
                if len(self.logger.data_hist) > 0:
                    # Use the last generated data as target
                    if self.msg_length > 1:
                        target = torch.tensor(self.logger.gst_data_hist[:,-self.msg_length:-1], dtype=torch.float32, device=self.device).T
                    else:
                        target = torch.tensor(self.logger.gst_data_hist[:,-1], dtype=torch.float32, device=self.device).T
                    # Train on this message immediately
                    self._train_readout(reservoir_output, target)
            except Exception as e:
                self.get_logger().error(f"Error handling reservoir data: {e}")
        else:
            try:
                # Convert message to tensor
                reservoir_output = self._msg_to_layer(msg)
                #record rtt
                rtt_seconds = self.logger.record_roundtrip_time(msg.seq)
                self.get_logger().debug(f"Roundtrip time for seq {msg.seq}: {rtt_seconds*1000:.2f} ms")
                # Predict through trained readout layer
                prediction = self.model(reservoir_output)
                self.logger.pred_hist.append(prediction)
                self.last_pred = prediction
            except Exception as e:
                self.get_logger().error(f"Error handling reservoir data: {e}")

    def _handle_params(self, msg):
        '''
        callback to handle parameter updates from edge
        extracts reservoir parameters from dict message
        '''
        if self.connected_to_edge:
            return
        try:
            # Extract reservoir dimensions and parameters from the dict
            self.res_dim = msg.get('res_dim')
            self.get_logger().info(f"Received reservoir parameters: res_dim={self.res_dim}")
            self.connected_to_edge = True
        except Exception as e:
            self.get_logger().error(f"Error handling parameters: {e}")

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
            
            self.logger.training_loss.append(loss.item)

            self.get_logger().debug(f"Training loss: {loss.item():.6f}")
            
        except Exception as e:
            self.get_logger().error(f"Error training readout layer: {e}")
    
    def _generate_data(self, t0, ic):
        '''
        wrapper for rk45
        take in some time and IC and advance the integration by  
        '''
        tspan = np.linspace(t0, t0 + self.msg_length*self.dt, self.msg_length)
        sol = solve_ivp(self.func, [t0, t0 + self.msg_length*self.dt], ic, t_eval=tspan, method=self.integrator)
        self.logger.gst_data_hist.append(sol.y)
        self.logger.gst_time_step_hist.append(sol.t)
        return sol.y, sol.t
    
    def _send_batch_to_res(self):
        '''
        generate data and send batch of data to reservoir
        '''
        # Generate data using the dynamical function
        data, _ = self._generate_data(self.t_curr, self.ic)
        
        # Get sequence number and record send time
        seq = self.logger.get_next_msg_seq()
        
        msg = self._data_to_msg(data)
        msg.seq = seq  # Add sequence number to message

        self.data_publisher.publish(msg)
        
        # Update initial conditions
        self.ic = data[:, -1]
        self.get_logger().debug(f"Sent batch {seq} at t=[{self.t_curr},{self.t_curr+self.msg_length*self.dt})")

    def _pred_with_res(self):
        _._ = self._generate_data(self.t_curr)


    ###same as send_batch_to_res
    def _run(self):
        '''
        Main node spin loop: generates data, feeds through reservoir, sends output
        '''
        if self.t_curr >= self.max_time:
            # Evaluation complete
            self.logger.summary()
            rclpy.shutdown()
        elif self.training:
            try:
                self._send_batch_to_res()
                self.t_curr += self.msg_length * self.dt
            except Exception as e:
                self.get_logger().error(f"Error in node spin: {e}")           
        elif not self.training:
            try:
                self._pred_with_res()
                self.t_curr += self.msg_length * self.dt
                self.run_count += 1
            except Exception as e:
                self.get_logger().error(f"Error in node spin: {e}")
        

