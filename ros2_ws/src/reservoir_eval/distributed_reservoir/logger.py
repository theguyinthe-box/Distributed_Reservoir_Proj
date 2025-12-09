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

    def __init__(self, model_type: str, ros_logger,
                 results_dir: Optional[str] = None, results_filename: Optional[str] = None):
        
        self.model_type = model_type.upper()
        self.ros_logger = ros_logger
        self.model_size = None

        self.time_step_hist = []
        self.gst_data_hist = []
        
        self.pred_time = []
        self.pred_hist = []

        self.training_loss = []

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

    def set_training_time(self, seconds: float):
        self.training_time_s = float(seconds)

    def set_model_size(self, size_mb: float):
        self.model_size = float(size_mb)

    def log_run(self, run_number, mse, pred_times_s: list, roundtrip_time_s: float):
        if not pred_times_s:
            self.ros_logger.warning(f"Run {run_number}: pred_times_s is empty; skipping pred stats.")
            avg_pred = float("nan")
            min_pred = float("nan")
            max_pred = float("nan")
        else:
            pred_times_ms = [t * 1000.0 for t in pred_times_s]
            avg_pred = sum(pred_times_ms) / len(pred_times_ms)
            min_pred = min(pred_times_ms)
            max_pred = max(pred_times_ms)

        roundtrip_ms = roundtrip_time_s * 1000.0

        self.ros_logger.info(f"\nRun {run_number}")
        self.ros_logger.info(f"MSE: {mse:.6f}")
        self.ros_logger.info(f"Prediction Time per Step (ms): avg: {avg_pred:.3f} | min: {min_pred:.3f} | max: {max_pred:.3f}")
        self.ros_logger.info(f"Roundtrip Time Agent→Edge→Agent (ms): {roundtrip_ms:.3f}")
        self.ros_logger.info("-" * 60)

        self.mse_list.append(float(mse))
        self.pred_time_avgs.append(float(avg_pred))
        self.pred_time_mins.append(float(min_pred))
        self.pred_time_maxs.append(float(max_pred))
        self.roundtrip_times.append(float(roundtrip_time_s))
        self.run_infos.append({
            'run': int(run_number),
            'mse': float(mse),
            'avg_pred_ms': float(avg_pred),
            'min_pred_ms': float(min_pred),
            'max_pred_ms': float(max_pred),
            'roundtrip_ms': float(roundtrip_ms),
        })

    # ---------- helpers ----------
    def _safe_mean(self, arr: List[float], default=float("nan")) -> float:
        return sum(arr) / len(arr) if arr else default

    def _safe_std(self, arr: List[float], default=float("nan")) -> float:
        return statistics.stdev(arr) if len(arr) > 1 else (0.0 if len(arr) == 1 else default)

    def _flatten_hyperparams(self) -> Dict[str, Any]:
        flat = {}
        def _recurse(prefix, val):
            if isinstance(val, dict):
                for k, v in val.items():
                    _recurse(f"{prefix}{k}.", v)
            elif isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    _recurse(f"{prefix}{i}.", v)
            else:
                flat[prefix[:-1]] = val
        _recurse("", self.hyperparams)
        return flat

    # ---- Matching on the 5 selected keys (robust normalization) ----
    def _norm_value(self, key: str, value: Any) -> Any:
        if key in self._INT_KEYS:
            try:
                return int(value)
            except Exception:
                return value
        if key in self._FLOAT_KEYS:
            try:
                return float(value)
            except Exception:
                return value
        return value

    def _equal_values(self, key: str, a: Any, b: Any) -> bool:
        if key in self._FLOAT_KEYS:
            try:
                fa, fb = float(a), float(b)
                return math.isclose(fa, fb, rel_tol=0.0, abs_tol=self._FLOAT_TOL)
            except Exception:
                return a == b
        if key in self._INT_KEYS:
            try:
                return int(a) == int(b)
            except Exception:
                return a == b
        return a == b

    def _selected_hp_dict(self, hp_flat: Dict[str, Any]) -> Dict[str, Any]:
        sel = {}
        for k in self._MATCH_KEYS:
            v = hp_flat.get(k, None)
            sel[k] = self._norm_value(k, v)
        return sel

    def _find_existing_index_by_selected_hparams(self, df_old: pd.DataFrame,
                                                 selected_hp: Dict[str, Any]) -> Optional[int]:
        """Find the index of a row whose hp.* columns exactly match the 5 selected hyperparameters."""
        if df_old is None or df_old.empty:
            return None

        # Ensure hp.* columns exist for the selection keys
        for k in self._MATCH_KEYS:
            col = f"hp.{k}"
            if col not in df_old.columns:
                df_old[col] = pd.NA

        # Row-wise comparison
        for idx, row in df_old.iterrows():
            ok = True
            for k in self._MATCH_KEYS:
                col = f"hp.{k}"
                a = selected_hp[k]
                b = row[col]
                # If b is missing, it's not a match
                if pd.isna(b):
                    ok = False
                    break
                if not self._equal_values(k, a, b):
                    ok = False
                    break
            if ok:
                return idx
        return None

    def _upsert_best(self, df_old: pd.DataFrame, df_new_row: pd.DataFrame,
                     hp_flat: Dict[str, Any], metric_col: str) -> pd.DataFrame:
        """
        Upsert based on the 5 key HPs:
        - If the same 5 HPs exist → overwrite only if the new metric is better (smaller)
        - Otherwise: append a new row
        """
        if df_old is None or not isinstance(df_old, pd.DataFrame):
            df_old = pd.DataFrame()

        df_old = df_old.copy()
        df_new_row = df_new_row.copy()

        # Union of columns
        for col in df_new_row.columns:
            if col not in df_old.columns:
                df_old[col] = pd.NA
        for col in df_old.columns:
            if col not in df_new_row.columns:
                df_new_row[col] = pd.NA

        selected_hp = self._selected_hp_dict(hp_flat)
        idx = self._find_existing_index_by_selected_hparams(df_old, selected_hp)

        if idx is not None:
            # Parameter set exists → check if better
            old_val = df_old.loc[idx, metric_col]
            new_val = df_new_row.iloc[0][metric_col]
            if pd.isna(old_val) or float(new_val) < float(old_val):
                df_old.loc[idx, df_new_row.columns] = df_new_row.iloc[0][df_new_row.columns]
            # else: keep old row unchanged
            return df_old
        else:
            # New parameter set → append row
            if df_old.empty:
                return df_new_row
            return pd.concat([df_old, df_new_row], ignore_index=True)

    # ---------- summary + persist ----------
    def summary(self):
        experiment_duration_time_s = time.time() - self.experiment_start_time

        self.ros_logger.info("\nFINAL SUMMARY (over all runs)")
        self.ros_logger.info("=" * 60)
        self.ros_logger.info(f"Model: {self.model_type}")

        # MSE
        if self.mse_list:
            avg_mse = sum(self.mse_list) / len(self.mse_list)
            std_mse = statistics.stdev(self.mse_list) if len(self.mse_list) > 1 else 0.0
            self.ros_logger.info(f"MSE: avg: {avg_mse:.6f} ± {std_mse:.6f} | min: {min(self.mse_list):.6f} | max: {max(self.mse_list):.6f}")
        else:
            self.ros_logger.info("MSE: no data")

        # Training time
        self.ros_logger.info(f"\nTraining Time: {self.training_time_s:.3f} seconds")

        # Model size
        if self.model_size is not None:
            self.ros_logger.info(f"Model size: {self.model_size:.6f} MB")

        # Prediction time stats
        if self.pred_time_avgs:
            avg_of_avgs = sum(self.pred_time_avgs) / len(self.pred_time_avgs)
            std_pred = statistics.stdev(self.pred_time_avgs) if len(self.pred_time_avgs) > 1 else 0.0
            self.ros_logger.info("\nPrediction Time per Step (ms):")
            self.ros_logger.info(f"   avg of avgs: {avg_of_avgs:.3f} ± {std_pred:.3f}")
            self.ros_logger.info(f"   min of mins: {min(self.pred_time_mins):.3f}")
            self.ros_logger.info(f"   max of maxs: {max(self.pred_time_maxs):.3f}")

        # Experiment duration
        self.ros_logger.info(f"\nExperiment duration (from logger init): {experiment_duration_time_s:.3f} seconds")

        # Roundtrip
        if self.roundtrip_times:
            roundtrip_ms = [rtt * 1000 for rtt in self.roundtrip_times]
            avg_rtt = sum(roundtrip_ms) / len(roundtrip_ms)
            std_rtt = statistics.stdev(roundtrip_ms) if len(roundtrip_ms) > 1 else 0.0
            self.ros_logger.info("\n🔁 Roundtrip Time Agent→Edge→Agent (ms):")
            self.ros_logger.info(f"   avg: {avg_rtt:.3f} ± {std_rtt:.3f}")
            self.ros_logger.info(f"   min: {min(roundtrip_ms):.3f}")
            self.ros_logger.info(f"   max: {max(roundtrip_ms):.3f}")

        # Fastest runs
        if not self.run_infos:
            self.ros_logger.warning("No runs recorded; nothing to persist.")
            return

        fastest_pred = min(self.run_infos, key=lambda x: x['avg_pred_ms'])
        fastest_rtt  = min(self.run_infos, key=lambda x: x['roundtrip_ms'])

        self.ros_logger.info(f"\n⚡ Fastest Prediction Run: #{fastest_pred['run']}")
        self.ros_logger.info(f"   Avg Prediction Time: {fastest_pred['avg_pred_ms']:.3f} ms")
        self.ros_logger.info(f"   Roundtrip Time: {fastest_pred['roundtrip_ms']:.3f} ms")
        self.ros_logger.info(f"   MSE: {fastest_pred['mse']:.6f}")

        self.ros_logger.info(f"\n🔁 Fastest Roundtrip Run: #{fastest_rtt['run']}")
        self.ros_logger.info(f"   Roundtrip Time: {fastest_rtt['roundtrip_ms']:.3f} ms")
        self.ros_logger.info(f"   Avg Prediction Time: {fastest_rtt['avg_pred_ms']:.3f} ms")
        self.ros_logger.info(f"   MSE: {fastest_rtt['mse']:.6f}")

        self.ros_logger.info("=" * 60)

        # ---- Prepare Excel rows ----
        base_row_common = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "training_time_s": float(self.training_time_s),
            "model_size_mb": float(self.model_size) if self.model_size is not None else float("nan"),
            "experiment_duration_s": float(experiment_duration_time_s),
            "runs_total_session": len(self.run_infos),
            "warmups_session": self.warmup_count,
        }
        hp_flat = self._flatten_hyperparams()
        hp_cols_prefixed = {f"hp.{k}": v for k, v in hp_flat.items()}

        # Sheet "Average Prediction Time" (best by average prediction latency)
        row_pred = {
            **base_row_common,
            "best_avg_pred_run": fastest_pred["run"],
            "mse": fastest_pred["mse"],
            "avg_pred_ms": fastest_pred["avg_pred_ms"],
            "min_pred_ms": fastest_pred["min_pred_ms"],
            "max_pred_ms": fastest_pred["max_pred_ms"],
            "roundtrip_ms": fastest_pred["roundtrip_ms"],
            **hp_cols_prefixed,
        }
        df_pred_new = pd.DataFrame([row_pred])

        # Sheet "Roundtrip" (best by roundtrip time)
        row_rtt = {
            **base_row_common,
            "best_roundtrip_run": fastest_rtt["run"],
            "mse": fastest_rtt["mse"],
            "avg_pred_ms": fastest_rtt["avg_pred_ms"],
            "min_pred_ms": fastest_rtt["min_pred_ms"],
            "max_pred_ms": fastest_rtt["max_pred_ms"],
            "roundtrip_ms": fastest_rtt["roundtrip_ms"],
            **hp_cols_prefixed,
        }
        df_rtt_new = pd.DataFrame([row_rtt])

        # Load existing file (if present)
        if os.path.exists(self.results_path):
            try:
                df_pred_old = pd.read_excel(self.results_path, sheet_name="Average Prediction Time")
            except Exception:
                df_pred_old = pd.DataFrame()
            try:
                df_rtt_old = pd.read_excel(self.results_path, sheet_name="Roundtrip")
            except Exception:
                df_rtt_old = pd.DataFrame()
        else:
            df_pred_old = pd.DataFrame()
            df_rtt_old = pd.DataFrame()

        # Upsert per hyperparameter set
        df_pred_upd = self._upsert_best(df_pred_old, df_pred_new, hp_flat=hp_flat, metric_col="avg_pred_ms")
        df_rtt_upd  = self._upsert_best(df_rtt_old,  df_rtt_new,  hp_flat=hp_flat, metric_col="roundtrip_ms")

        # Sorting
        with pd.option_context('mode.use_inf_as_na', True):
            if not df_pred_upd.empty and "avg_pred_ms" in df_pred_upd.columns:
                df_pred_upd = df_pred_upd.sort_values(by="avg_pred_ms", ascending=True, na_position="last")
            if not df_rtt_upd.empty and "roundtrip_ms" in df_rtt_upd.columns:
                df_rtt_upd = df_rtt_upd.sort_values(by="roundtrip_ms", ascending=True, na_position="last")

        # Write
        os.makedirs(os.path.dirname(self.results_path) or ".", exist_ok=True)
        with pd.ExcelWriter(self.results_path, engine="openpyxl") as writer:
            df_pred_upd.to_excel(writer, index=False, sheet_name="Average Prediction Time")
            df_rtt_upd.to_excel(writer,  index=False, sheet_name="Roundtrip")

        self.ros_logger.info(f"Results updated: {self.results_path}")
