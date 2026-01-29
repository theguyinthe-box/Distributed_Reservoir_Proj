# dependency imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, String 
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import json

# codebase imports
from .lstm import LSTMModel
from .reservoir import Reservoir
from .functions import dynamical_functions as d


class Edge_ROSNode(Node):
    def __init__(self, 
                 func: str = 'lorenz',
                 model_type: str = 'reservoir',
                 res_dim: int = 256,
                 spectral_radius: float = 1.1,
                 leak_rate: float = 0.15,
                 n_iterations: int = 20,
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 3,
                 training_length: int = 100,
                 dt: float = 0.01,
                 batch_size: int = 2):
        
        super().__init__('edge_server_ros_node')    
        
        # Declare and read ROS2 parameters
        self.declare_parameter('func', func)
        self.declare_parameter('model_type', model_type)
        self.declare_parameter('res_dim', res_dim)
        self.declare_parameter('spectral_radius', spectral_radius)
        self.declare_parameter('leak_rate', leak_rate)
        self.declare_parameter('n_iterations', n_iterations)
        self.declare_parameter('lstm_hidden_size', lstm_hidden_size)
        self.declare_parameter('lstm_num_layers', lstm_num_layers)
        self.declare_parameter('training_length', training_length)
        self.declare_parameter('dt', dt)
        self.declare_parameter('batch_size', batch_size)
        
        func = self.get_parameter('func').value
        model_type = self.get_parameter('model_type').value
        res_dim = self.get_parameter('res_dim').value
        spectral_radius = self.get_parameter('spectral_radius').value
        leak_rate = self.get_parameter('leak_rate').value
        n_iterations = self.get_parameter('n_iterations').value
        lstm_hidden_size = self.get_parameter('lstm_hidden_size').value
        lstm_num_layers = self.get_parameter('lstm_num_layers').value
        training_length = self.get_parameter('training_length').value
        dt = self.get_parameter('dt').value
        batch_size = self.get_parameter('batch_size').value
        
        # get GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store configuration
        self.func = func
        self.model_type = model_type.lower()
        self.training = True  # Track training vs evaluation mode
        self.training_length = training_length  # Training duration for LSTM
        self.batch_count = 0  # Track number of batches received for training length logic
        self.dt = dt  # Time step for integration
        self.batch_size = batch_size  # Number of samples per batch
        
        if self.model_type not in ['reservoir', 'lstm']:
            raise ValueError(f"model_type must be 'reservoir' or 'lstm', got '{self.model_type}'")
        
        # hyper parameters

        
        # Publishers / Subscribers
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        self.agent_ready_subscriber = self.create_subscription(String, 'agent_ready', self._handle_agent_ready, qos_profile)
        self.param_publisher = self.create_publisher(String, 'reservoir_params', qos_profile)
        self.ready_publisher = self.create_publisher(String, 'edge_ready', qos_profile)
        self.subscription = self.create_subscription(Float32MultiArray, f'{self.func}_agent_msg', self._handle_input, qos_profile)
        self.output_publisher = self.create_publisher(Float32MultiArray, f'{self.func}_edge_msg', qos_profile)

        input_dim = d.function_dims(func)
        output_dim = d.function_dims(func)  # Separate output_dim for asymmetric functions
        self.reservoir_params = {
            "res_dim": res_dim,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "spectral_radius": spectral_radius,
            "leak_rate": leak_rate, 
            "iter": n_iterations,
            "model_type": self.model_type,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers
        }

        # instantiate model based on model_type
        self.model = self._build_model(self.model_type, self.reservoir_params).to(self.device)
        
        # publish parameters to agent
        self._publish_params()
        
        self.get_logger().info(f"Edge {self.model_type} node ready. Model: {res_dim}D, Input: {input_dim}D, Output: {output_dim}D, Function: {self.func}")

    def _build_model(self, model_type: str, params: Dict[str, Any]):
        '''
        Implementation-agnostic method to instantiate model based on type.
        Encapsulates all model-specific parameter assignment logic.
        For LSTM, also initializes training infrastructure (optimizer, loss function).
        
        Args:
            model_type: Type of model ('reservoir' or 'lstm')
            params: Dictionary containing model parameters
                    Must include: res_dim, input_dim
                    Reservoir-specific: spectral_radius, leak_rate
                    LSTM-specific: (can be extended)
        
        Returns:
            Instantiated model object
        '''
        if model_type == 'reservoir':
            return Reservoir(
                res_dim=params['res_dim'],
                input_dim=params['input_dim'],
                spectral_radius=params['spectral_radius'],
                leak_rate=params['leak_rate']
            )
        elif model_type == 'lstm':
            torch.manual_seed(42)
            np.random.seed(42)
            model = LSTMModel(
                input_size=params['input_dim'],
                hidden_size=params.get('lstm_hidden_size', 64),
                output_size=params['output_dim'],
                num_layers=params.get('lstm_num_layers', 3)
            )
            
            # Initialize LSTM training infrastructure
            self.loss_fn = nn.MSELoss()
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            self.training_loss_history = []  # Track loss across batches
            self.prev_input = None  # Buffer for previous input (used as training input, current input is target)
            
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")

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
        msg.data = torch.asarray(u, dtype=torch.float32).flatten().tolist()
        return msg
    
    def _publish_params(self):
        '''
        publish reservoir parameters to agent as JSON string
        '''
        msg = String()
        msg.data = json.dumps(self.reservoir_params)
        self.param_publisher.publish(msg)
        self.get_logger().info(f"Published reservoir parameters to agent")
    
    def _handle_input(self, msg):
        '''
        callback to receive data from agent
        Process through model (reservoir or LSTM) and send output back to agent
        Behavior differs based on training vs evaluation mode
        '''
        try:
            # Convert incoming message to tensor
            input_data = self._msg_to_layer(msg)

            # Reshape to (batch_size, input_dim)
            input_data = input_data.reshape(-1, self.reservoir_params['input_dim'])

            # Increment batch count and check if training is complete
            self.batch_count += 1
            # Estimate time elapsed based on batch count
            time_elapsed = self.batch_count * self.batch_size * self.dt
                    
            if time_elapsed > self.training_length:
                self.training = False
                self.get_logger().info(f"LSTM training complete after {self.batch_count} batches (t={time_elapsed:.2f}s)")
                    
            if self.training:
                # ============ TRAINING MODE ============
                # For LSTM: forward pass, compute loss, backprop, optimizer step
                # For Reservoir: reservoir doesn't train on edge (only agent readout trains)
                self.model.train()
                
                if self.model_type == 'lstm':
                    # LSTM training: time series prediction
                    # Model predicts next timestep from current timestep
                    # Input: previous timestep value, Target: current timestep value (ground truth)
                    
                    # Skip first batch (no previous input to use as training input)
                    if self.prev_input is None:
                        self.prev_input = input_data.clone()
                        self.get_logger().debug("LSTM: Buffered first input, skipping training for this batch")
                        # Still send output back to agent
                        with torch.no_grad():
                            output_data = self.model.process_input(
                                input_data,
                                n_steps=self.reservoir_params['iter']
                            )
                        output_msg = self._data_to_msg(output_data)
                        self.output_publisher.publish(output_msg)
                    else:
                        # Use previous input as model input, current input as target
                        train_input = self.prev_input
                        target_data = input_data  # Current input is ground truth for previous prediction
                        
                        # Forward pass on previous input
                        output_data = self.model.process_input(
                            train_input,
                            n_steps=self.reservoir_params['iter']
                        )  # [B, output_size]
                        
                        target_data = target_data.to(self.device, non_blocking=True).float()
                        
                        # Compute loss
                        loss = self.loss_fn(output_data, target_data)
                        
                        # Backward pass and optimizer step
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                        # Track loss
                        batch_loss = loss.item()
                        self.training_loss_history.append(batch_loss)
                        self.get_logger().debug(f"LSTM training batch loss: {batch_loss:.6f}")
                        
                        # Update buffer for next iteration
                        self.prev_input = input_data.clone()
                        
                        # Send prediction for current input back to agent
                        with torch.no_grad():
                            current_output = self.model.process_input(
                                input_data,
                                n_steps=self.reservoir_params['iter']
                            )
                        output_msg = self._data_to_msg(current_output)
                        self.output_publisher.publish(output_msg)
                        self.get_logger().debug(f"Published LSTM output (training)")
                elif self.model_type == 'reservoir':
                    # Reservoir: just forward pass, no training on edge
                    output_data = self.model.process_input(
                        input_data,
                        n_steps=self.reservoir_params['iter']
                    )
                    self.get_logger().debug(f"Model output shape: {output_data.shape}")

                    # Convert output to message and publish
                    output_msg = self._data_to_msg(output_data)
                    self.output_publisher.publish(output_msg)
                    self.get_logger().debug(f"Published model output (training)")
            else:
                # ============ EVALUATION MODE ============
                # Process through model using standard interface
                self.model.eval()
                output_data = self.model.process_input(
                    input_data,
                    n_steps=self.reservoir_params['iter']
                )
                self.get_logger().debug(f"Model output shape: {output_data.shape}")

                # Convert output to message and publish
                output_msg = self._data_to_msg(output_data)

                #output_msg.seq = msg.seq  # Preserve sequence number

                self.output_publisher.publish(output_msg)

                self.get_logger().debug(f"Published model output (evaluation)")

        except Exception as e:
            self.get_logger().error(f"Error in _handle_input: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_agent_ready(self, msg):
        '''
        callback to handle agent readiness signal
        respond with edge readiness to complete handshake
        '''
        if msg.data == "READY":
            self.get_logger().info("Received READY signal from agent, completing handshake...")
            response = String()
            response.data = "READY"
            self.ready_publisher.publish(response)
            self.get_logger().info("Sent READY signal to agent, handshake complete")

def main(args=None):
    rclpy.init(args=args)
    
    # Default parameters (can be overridden via ROS2 launch files or environment variables)
    func = 'lorenz'  # or 'rossler'
    model_type = 'reservoir'  # or 'lstm'
    
    node = Edge_ROSNode(
        func=func,
        model_type=model_type,
    )
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()