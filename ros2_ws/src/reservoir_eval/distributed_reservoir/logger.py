import statistics
import time
import os
import json
import math
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd

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