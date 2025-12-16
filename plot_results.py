#!/usr/bin/env python3
"""
Script to plot ground truth and predictions from the distributed reservoir computing experiment
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_ground_truth_vs_predictions(results_dir='./results', output_file=None):
    """
    Plot ground truth and predictions from the reservoir computing experiment
    
    Parameters:
    -----------
    results_dir : str
        Directory containing the CSV files
    output_file : str, optional
        Path to save the figure. If None, displays the plot.
    """
    results_path = Path(results_dir)
    
    # Load data
    gt_file = results_path / 'rossler_agent_node_groundtruth.csv'
    pred_file = results_path / 'rossler_agent_node_predictions.csv'
    
    if not gt_file.exists():
        print(f"Error: Ground truth file not found: {gt_file}")
        return
    
    if not pred_file.exists():
        print(f"Error: Predictions file not found: {pred_file}")
        return
    
    # Read CSV files
    groundtruth = pd.read_csv(gt_file)
    predictions = pd.read_csv(pred_file)
    
    # Extract dimensions
    gt_dims = [col for col in groundtruth.columns if col.startswith('dim_')]
    pred_dims = [col for col in predictions.columns if col.startswith('dim_')]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Rossler System: Ground Truth vs Reservoir Predictions', fontsize=16, fontweight='bold')
    
    # Time indices
    gt_time = np.arange(len(groundtruth))
    pred_time = np.arange(len(predictions))
    
    # Plot each dimension
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dim_names = ['X', 'Y', 'Z']
    
    for i, (ax, color, dim_name) in enumerate(zip(axes, colors, dim_names)):
        if i < len(gt_dims):
            # Plot ground truth
            ax.plot(gt_time, groundtruth[gt_dims[i]], 'o-', 
                   label='Ground Truth', color='k', linewidth=1, markersize=1, alpha=0.7)
        
        if i < len(pred_dims):
            # Plot predictions (offset to show on same plot)
            ax.plot(pred_time, predictions[pred_dims[i]], 's--', 
                   label='Predictions', color=color, linewidth=2, markersize=2, alpha=0.7, linestyle='--')
        
        ax.set_ylabel(f'{dim_name} Value', fontsize=11)
        ax.set_title(f'Dimension {i} ({dim_name})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    axes[-1].set_xlabel('Time Step', fontsize=11)
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    # Print statistics
    print("\n=== Experiment Statistics ===")
    print(f"Ground Truth samples: {len(groundtruth)}")
    print(f"Prediction samples: {len(predictions)}")
    print(f"\nGround Truth dimensions: {groundtruth.shape}")
    print(f"Prediction dimensions: {predictions.shape}")
    
    # Calculate MSE between ground truth and predictions (for overlapping portion)
    min_len = min(len(groundtruth), len(predictions))
    if min_len > 0:
        print(f"\n=== MSE (for first {min_len} samples) ===")
        for i, (gt_col, pred_col) in enumerate(zip(gt_dims[:3], pred_dims[:3])):
            gt_vals = groundtruth[gt_col].values[:min_len]
            pred_vals = predictions[pred_col].values[:min_len]
            mse = np.mean((gt_vals - pred_vals) ** 2)
            rmse = np.sqrt(mse)
            print(f"Dimension {i}: MSE={mse:.6f}, RMSE={rmse:.6f}")

def plot_3d_trajectories(results_dir='./results', output_file=None):
    """
    Create a 3D plot of the trajectories in phase space
    
    Parameters:
    -----------
    results_dir : str
        Directory containing the CSV files
    output_file : str, optional
        Path to save the figure. If None, displays the plot.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    results_path = Path(results_dir)
    
    # Load data
    gt_file = results_path / 'rossler_agent_node_groundtruth.csv'
    pred_file = results_path / 'rossler_agent_node_predictions.csv'
    
    if not gt_file.exists() or not pred_file.exists():
        print("Error: Missing CSV files")
        return
    
    groundtruth = pd.read_csv(gt_file)
    predictions = pd.read_csv(pred_file)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract X, Y, Z
    gt_x, gt_y, gt_z = groundtruth['dim_0'], groundtruth['dim_1'], groundtruth['dim_2']
    pred_x, pred_y, pred_z = predictions['dim_0'], predictions['dim_1'], predictions['dim_2']
    
    # Plot trajectories
    ax.plot(gt_x, gt_y, gt_z, 'o-', label='Ground Truth', 
           linewidth=2, markersize=2, alpha=0.7, color='blue')
    ax.plot(pred_x, pred_y, pred_z, 's--', label='Predictions', 
           linewidth=2, markersize=1.5, alpha=0.7, color='red')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('Rossler Attractor: Ground Truth vs Predictions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to: {output_file}")
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot ground truth vs predictions')
    parser.add_argument('--results-dir', default='./results', help='Path to results directory')
    parser.add_argument('--output', help='Output file path (if not specified, displays plot)')
    parser.add_argument('--3d', action='store_true', help='Create 3D phase space plot')
    
    args = parser.parse_args()
    
    if args.__dict__['3d']:
        output_file = args.output or '/tmp/lorenz_3d_trajectories.png'
        plot_3d_trajectories(args.results_dir, output_file if args.output else None)
    else:
        output_file = args.output or '/tmp/lorenz_ground_truth_vs_predictions.png'
        plot_ground_truth_vs_predictions(args.results_dir, output_file if args.output else None)
