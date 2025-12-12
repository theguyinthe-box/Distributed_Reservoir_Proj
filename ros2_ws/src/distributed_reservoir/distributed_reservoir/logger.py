import statistics
import time
import os
import json
import math
import csv
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

class Logger:
    # Keys used to match for upsert/overwrite
    _MATCH_KEYS = ["units", "spectral_radius", "leaking_rate", "input_scaling", "ridge_alpha"]
    _FLOAT_KEYS = {"spectral_radius", "leaking_rate", "input_scaling", "ridge_alpha"}
    _INT_KEYS = {"units"}
    _FLOAT_TOL = 1e-9  # Tolerance for float equality checks

    def __init__(self, 
                 model_type: str, 
                 ros_logger,
                 results_dir: Optional[str] = None, 
                 results_filename: Optional[str] = None):
        
        self.model_type = model_type.upper()
        self.ros_logger = ros_logger
        self.model_size = None

        # Data tracking
        self.gst_time_step_hist = []
        self.gst_data_hist = []

        self.pred_time = []
        self.pred_hist = []
        self.training_loss = []

        # Roundtrip time tracking
        self.msg_seq = 0
        self.send_times = {}  # seq -> send_time
        self.warmup_count = 0
        self.mse_list = []

        self.pred_time_avgs = []
        self.pred_time_mins = []
        self.pred_time_maxs = []
        
        self.roundtrip_times = []

        self.experiment_start_time = time.time()
        

        # Path / filename
        if results_filename is None:
            results_filename = f"{self.model_type.lower()}_results.csv"
        base_dir = results_dir or os.getenv("RESULTS_DIR", "/results")
        os.makedirs(base_dir, exist_ok=True)
        self.results_path = os.path.join(base_dir, results_filename)

        self.ros_logger.info("=" * 60)
        self.ros_logger.info(f"Starting {self.model_type} Evaluation")
        self.ros_logger.info("=" * 60)
        self.ros_logger.info(f"Results will be stored at: {self.results_path}")

    def get_next_msg_seq(self) -> int:
        '''
        Get next message sequence number and record send time
        Call this when sending data to the reservoir
        Returns the sequence number to attach to your message
        '''
        self.msg_seq += 1
        self.send_times[self.msg_seq] = time.time()
        return self.msg_seq

    def record_roundtrip_time(self, seq: int) -> float:
        '''
        Record the roundtrip time for a message
        Call this when receiving processed data back from reservoir
        Returns the roundtrip time in seconds
        '''
        if seq not in self.send_times:
            self.ros_logger.warning(f"Sequence {seq} not found in send_times tracking")
            return 0.0
        
        roundtrip_time_s = time.time() - self.send_times[seq]
        self.roundtrip_times.append(roundtrip_time_s)
        del self.send_times[seq]  # Clean up
        
        return roundtrip_time_s

    def summary(self):
        '''
        Generate and save a summary of the experiment results
        '''
        try:
            experiment_time = time.time() - self.experiment_start_time
            
            # Calculate statistics
            avg_rtt = statistics.mean(self.roundtrip_times) if self.roundtrip_times else 0
            min_rtt = min(self.roundtrip_times) if self.roundtrip_times else 0
            max_rtt = max(self.roundtrip_times) if self.roundtrip_times else 0
            
            avg_loss = statistics.mean(self.training_loss) if self.training_loss else 0
            
            # Write summary to file
            summary_data = {
                "model_type": self.model_type,
                "experiment_duration_s": experiment_time,
                "num_messages": self.msg_seq,
                "avg_roundtrip_time_s": avg_rtt,
                "min_roundtrip_time_s": min_rtt,
                "max_roundtrip_time_s": max_rtt,
                "avg_training_loss": avg_loss,
                "num_predictions": len(self.pred_hist),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save as JSON
            with open(self.results_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            self.ros_logger.info("=" * 60)
            self.ros_logger.info(f"Experiment Complete: {self.model_type}")
            self.ros_logger.info(f"Duration: {experiment_time:.2f}s")
            self.ros_logger.info(f"Avg Roundtrip Time: {avg_rtt*1000:.2f}ms")
            self.ros_logger.info(f"Messages Sent: {self.msg_seq}")
            self.ros_logger.info("=" * 60)
            
            # Dump raw prediction data
            self._dump_predictions_csv()
            
            # Dump ground truth data
            self._dump_groundtruth_csv()
            
        except Exception as e:
            self.ros_logger.error(f"Error generating summary: {e}")

    def _dump_predictions_csv(self):
        '''
        Dump raw prediction data to CSV file
        Each prediction is a single 3D vector (one row per evaluation time step)
        '''
        try:
            if not self.pred_hist:
                self.ros_logger.warning("No prediction data to dump")
                return
            
            # Generate CSV filename
            base_dir = os.path.dirname(self.results_path)
            csv_filename = f"{self.model_type.lower()}_predictions.csv"
            csv_path = os.path.join(base_dir, csv_filename)
            
            # Convert predictions to list format
            # Each prediction tensor has shape (batch_size, 3)
            # We want to extract individual rows
            pred_data = []
            for batch_pred in self.pred_hist:
                # Handle both tensor and numpy array formats
                if hasattr(batch_pred, 'cpu'):  # PyTorch tensor
                    pred_array = batch_pred.detach().cpu().numpy()
                else:
                    pred_array = np.asarray(batch_pred)
                
                # pred_array should be (batch_size, 3)
                # Extract each row as a separate prediction
                if pred_array.ndim == 2:
                    for row in pred_array:
                        pred_data.append(row)
                else:
                    # If 1D, just add it
                    pred_data.append(pred_array)
            
            # Write to CSV
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                num_dims = len(pred_data[0]) if pred_data else 0
                writer.writerow([f"dim_{i}" for i in range(num_dims)])
                # Write data
                writer.writerows(pred_data)
            
            self.ros_logger.info(f"Prediction data saved to: {csv_path}")
            
        except Exception as e:
            self.ros_logger.error(f"Error dumping predictions to CSV: {e}")

    def _dump_groundtruth_csv(self):
        '''
        Dump ground truth data to CSV file
        gst_data_hist contains arrays of shape (n_dims, n_timepoints)
        '''
        try:
            if not self.gst_data_hist:
                self.ros_logger.warning("No ground truth data to dump")
                return
            
            # Generate CSV filename
            base_dir = os.path.dirname(self.results_path)
            csv_filename = f"{self.model_type.lower()}_groundtruth.csv"
            csv_path = os.path.join(base_dir, csv_filename)
            
            # Flatten and concatenate all ground truth data
            gst_data = []
            for batch in self.gst_data_hist:
                # batch has shape (n_dims, n_timepoints)
                if hasattr(batch, 'cpu'):  # PyTorch tensor
                    batch_array = batch.detach().cpu().numpy()
                else:
                    batch_array = np.asarray(batch)
                # Transpose to (n_timepoints, n_dims) and append
                gst_data.append(batch_array.T)
            
            # Concatenate all batches
            if gst_data:
                all_gst = np.vstack(gst_data)
                
                # Write to CSV
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    num_dims = all_gst.shape[1]
                    writer.writerow([f"dim_{i}" for i in range(num_dims)])
                    # Write data
                    writer.writerows(all_gst)
                
                self.ros_logger.info(f"Ground truth data saved to: {csv_path}")
                self.ros_logger.info(f"Total ground truth samples: {all_gst.shape[0]}")
            
        except Exception as e:
            self.ros_logger.error(f"Error dumping ground truth to CSV: {e}")