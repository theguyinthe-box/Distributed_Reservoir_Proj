# Dependency imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String
import json
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

# Codebase imports
from .reservoir import Readout
from .functions import dynamical_functions as d
from .logger import Logger

class Agent_ROSNode(Node):
    def __init__(self, 
                 func: str = 'lorenz', 
                 ic = None,
                 dt = .01,
                 integrator: str = 'RK45',
                 training_length = 100,
                 eval_length = 100,
                 batch_size = 2):

    
        super().__init__(f'{func}_agent_node')

        # Declare and read ROS2 parameters
        self.declare_parameter('func', func)
        self.declare_parameter('dt', dt)
        self.declare_parameter('integrator', integrator)
        self.declare_parameter('training_length', training_length)
        self.declare_parameter('eval_length', eval_length)
        self.declare_parameter('batch_size', batch_size)
        
        func = self.get_parameter('func').value
        dt = self.get_parameter('dt').value
        integrator = self.get_parameter('integrator').value
        training_length = self.get_parameter('training_length').value
        eval_length = self.get_parameter('eval_length').value
        batch_size = self.get_parameter('batch_size').value

        # get GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.training = True
        self.res_dim = None

        # store function type
        self.function = func
        self.function_dims = d.function_dims(self.function)
        self.func = d.return_function(self.function)
        
        self.get_logger().info("Agent started, initiating handshake with Edge...")
        # Subscribe to Edge before creating publishers
        # Init Subscribers
        # QOS profile
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)  
        # get params
        self.connected_to_edge = False
        self.handshake_complete = False
        self.reservoir_param_subscriber = self.create_subscription(String, 'reservoir_params', self._handle_params, qos_profile) 
        self.handshake_subscriber = self.create_subscription(String, 'edge_ready', self._handle_handshake, qos_profile)
        # subscribe to reservoir output
        self.data_subscription = self.create_subscription(Float32MultiArray, f'{self.function}_res_msg', self._handle_reservoir_data, qos_profile)
        # Publishers
        self.data_publisher = self.create_publisher(Float32MultiArray, f'{self.function}_agent_msg', qos_profile)
        self.readiness_publisher = self.create_publisher(String, 'agent_ready', qos_profile)        

        self.logger = Logger(f'{func}_agent_node', self.get_logger())

        # generates the linear layers of the 

        # initial condition of the system
        self.t_curr = 0.0
        self.ic = np.asarray(ic if ic is not None else [0.1, 0.1, 0.1], dtype=np.float32)

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
        
        self.last_pred = None

        # Burn-in phase: let the system reach the chaotic attractor before training
        self._burnin_phase()
        
        # Perform handshake with edge server
        self._handshake_with_edge()
        
        # Create a timer to periodically run the main loop
        self.create_timer(0.01, self._run)
    
    def _burnin_phase(self, burnin_time=150):
        '''
        Run the ODE integrator for a period to reach the chaotic attractor
        without sending any data through the distributed system
        '''
        self.get_logger().info(f"Starting burn-in phase ({burnin_time} seconds)...")
        burnin_steps = int(burnin_time / self.dt)
        tspan = np.linspace(0, burnin_time, burnin_steps)
        
        try:
            sol = solve_ivp(self.func, [0, burnin_time], self.ic, t_eval=tspan, method=self.integrator)
            # Update IC to the final state (on the attractor)
            self.ic = sol.y[:, -1]
            self.get_logger().info(f"Burn-in complete. IC updated to: {self.ic}")
        except Exception as e:
            self.get_logger().error(f"Error during burn-in phase: {e}")
            import traceback
            traceback.print_exc()

    def _handle_reservoir_data(self, msg):
        '''
        callback to handle incoming reservoir data
        trains the readout layer on every message received
        '''
        if self.training:
            try:
                # Convert message to tensor
                reservoir_output = self._msg_to_layer(msg)
                
                #TODO record rtt
                #rtt_seconds = self.logger.record_roundtrip_time(msg.seq)
                #self.get_logger().debug(f"Roundtrip time for seq {msg.seq}: {rtt_seconds*1000:.2f} ms")
                
                # Get ground truth target from last generated data batch
                if len(self.logger.gst_data_hist) > 0:
                    target = torch.tensor(self.logger.gst_data_hist[-1], dtype=torch.float32, device=self.device).T
                    # Train on this message immediately
                    self._train_readout(reservoir_output, target)
                    
            except Exception as e:
                self.get_logger().error(f"Error handling reservoir data in training: {e}")
        else:
            try:
                # Convert message to tensor
                reservoir_output = self._msg_to_layer(msg)
                
                #TODO record rtt
                #rtt_seconds = self.logger.record_roundtrip_time(msg.seq)
                #self.get_logger().debug(f"Roundtrip time for seq {msg.seq}: {rtt_seconds*1000:.2f} ms")
                
                # Predict through trained readout layer
                prediction = self.model(reservoir_output)
                self.logger.pred_hist.append(prediction)
                self.last_pred = prediction
                
            except Exception as e:
                self.get_logger().error(f"Error handling reservoir data in eval: {e}")
                import traceback
                traceback.print_exc()

    def _handle_params(self, msg):
        '''
        callback to handle parameter updates from edge
        extracts reservoir parameters from JSON string message
        '''
        if self.connected_to_edge:
            return
        try:
            # Parse JSON string to get parameters
            params = json.loads(msg.data)
            self.res_dim = params['res_dim']
            self.get_logger().info(f"Received reservoir parameters: res_dim={self.res_dim}")
            
            #instantiate readout
            self.model = Readout(self.res_dim, self.function_dims)
            #put readout on gpu
            self.model.to(self.device)

            # set connection var
            self.connected_to_edge = True
        except Exception as e:
            self.get_logger().error(f"Error handling parameters: {e}")
    
    def _handle_handshake(self, msg):
        '''
        callback to handle edge server readiness signal
        signals that the handshake is complete
        '''
        if msg.data == "READY":
            self.handshake_complete = True
            self.get_logger().info("Handshake complete: Edge server is ready")
    
    def _handshake_with_edge(self):
        '''
        perform handshake with edge server
        pauses execution until edge confirms readiness
        '''
        self.get_logger().info("Waiting for handshake with Edge server...")
        
        # Publish agent readiness signal
        msg = String()
        msg.data = "READY"
        self.readiness_publisher.publish(msg)
        self.get_logger().info("Published READY signal to edge server")
        
        # Spin with timeout until handshake complete
        timeout = 30.0  # 30 second timeout
        start_time = self.get_clock().now()
        while not self.handshake_complete:
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout:
                self.get_logger().error(f"Handshake timeout after {timeout} seconds")
                raise RuntimeError("Failed to establish handshake with Edge server")
        
        self.get_logger().info("Handshake successful, starting main loop")

    def _msg_to_layer(self, msg):
        '''
        recieve message from edge
        translate to nodes
        reshape to (batch_size, res_dim)
        '''
        data = torch.tensor(msg.data, dtype=torch.float32, device=self.device)
        # Reshape to (batch_size, res_dim)
        # res_dim should be known from handshake
        if self.res_dim is not None:
            batch_size = len(msg.data) // self.res_dim
            data = data.reshape(batch_size, self.res_dim)
        return data
    
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
            self.last_pred = predictions
            # Compute loss
            loss = self.model.criterion(predictions, target)
            
            # Backward pass
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            
            self.logger.training_loss.append(loss.item())

            self.get_logger().debug(f"Training loss: {loss.item():.6f}")
            
        except Exception as e:
            self.get_logger().error(f"Error training readout layer: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_data(self, t0, ic):
        '''
        wrapper for rk45
        take in some time and IC and advance the integration by msg_length *  dt
        '''
        try:
            tf = t0 + self.msg_length*self.dt
            tspan = np.linspace(t0, tf, self.msg_length, endpoint = True)
            # Ensure ic is 1D numpy array (handle PyTorch tensors on GPU)
            if hasattr(ic, 'cpu'):  # PyTorch tensor
                ic_flat = ic.cpu().detach().numpy().flatten().astype(np.float32)
            else:
                ic_flat = np.asarray(ic, dtype=np.float32).flatten()
            sol = solve_ivp(self.func, [t0,tf], ic_flat, t_eval=tspan, method=self.integrator)
            self.logger.gst_data_hist.append(sol.y)
            self.logger.gst_time_step_hist.append(sol.t)
            return sol.y, sol.t
        except Exception as e:
            self.get_logger().error(f"Error _generate_data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _send_batch_to_res(self):
        '''
        generate data and send batch of data to reservoir
        '''
        # Generate data using the dynamical function
        try:    
            data, _ = self._generate_data(self.t_curr, self.ic)
            if data is None:
                self.get_logger().warn("Data generation failed, skipping batch")
                return
            # Get sequence number and record send time
            seq = self.logger.get_next_msg_seq()
            msg = self._data_to_msg(data.T)
            
            # TODO Add sequence number to message 
            #msg.seq = seq  
            
            # publish msg 
            self.data_publisher.publish(msg)
            
            # Update initial conditions
            self.ic = data[:, -1]
            self.get_logger().debug(f"Sent batch {seq} at t=[{self.t_curr},{self.t_curr+self.msg_length*self.dt})")
        except Exception as e:
            self.get_logger().error(f"Error _send_batch_to_res: {e}")

    def _pred_with_res(self):
        try:    
            # Extract last point (last column) from last trajectory
            last_ic = self.logger.gst_data_hist[-1][:, -1]
            data, _ = self._generate_data(self.t_curr, last_ic)
            
            if data is None:
                self.get_logger().warn("Failed to generate data in _pred_with_res")
                return
                
            self.get_logger().debug(f"data generated sending prediction-based trajectory")
            # Send the generated trajectory data
            msg = self._data_to_msg(data.T)
            self.data_publisher.publish(msg)
            
            # Update initial conditions for next batch
            self.ic = data[:, -1]
            self.t_curr += self.msg_length * self.dt
        except Exception as e:
            self.get_logger().error(f"Error _pred_with_res: {e}")
            import traceback
            traceback.print_exc()


    ###same as send_batch_to_res
    def _run(self):
        '''
        Main node spin loop: generates data, feeds through reservoir, sends output
        '''
        try:
            if self.t_curr >= self.max_time:
                # Evaluation complete
                self.logger.summary()
                rclpy.shutdown()
            elif self.t_curr < self.training_length:
                #self.get_logger().debug("training")
                # Training phase: send real data
                self._send_batch_to_res()
                self.t_curr += self.msg_length * self.dt
                self.training = True
            else:
                #self.get_logger().info("evaluating")
                # Evaluation phase: use predictions as input
                if self.last_pred is not None:
                    self._pred_with_res()
                    self.t_curr += self.msg_length * self.dt
                    self.training = False
                else:
                    # Wait for first prediction before starting eval
                    self.get_logger().debug("Waiting for first prediction...")
        except Exception as e:
                self.get_logger().error(f"Error in _run: {e}")
        

def main(args=None):
    rclpy.init(args=args)
    
    # Default parameters (can be overridden via ROS2 launch files or environment variables)
    func = 'lorenz'  # or 'rossler'
    
    node = Agent_ROSNode(
        func=func
    )
    rclpy.spin(node)

if __name__ == '__main__':
    main()
