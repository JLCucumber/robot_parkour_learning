#!/usr/bin/env python3
"""
Loss Curve Plotting Script for Go2 Distillation Training
Plots loss curves from CSV files comparing AWBC vs no-AWBC training
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_and_process_csv(file_path, label):
    """Load CSV file and process the data"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {label}: {len(df)} data points")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Step range: {df['Step'].min()} - {df['Step'].max()}")
        print(f"Value range: {df['Value'].min():.4f} - {df['Value'].max():.4f}")
        print()
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_loss_curves(df_awbc, df_no_awbc, save_path=None):
    """Plot loss curves for both datasets"""
    plt.figure(figsize=(12, 8))
    
    # Plot both curves
    plt.plot(df_awbc['Step'], df_awbc['Value'], 
             label='With AWBC', color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(df_no_awbc['Step'], df_no_awbc['Value'], 
             label='Without AWBC', color='red', linewidth=1.5, alpha=0.8)
    
    # Add smoothed trend lines
    window_size = max(len(df_awbc) // 50, 10)  # Adaptive window size
    awbc_smooth = df_awbc['Value'].rolling(window=window_size, center=True).mean()
    no_awbc_smooth = df_no_awbc['Value'].rolling(window=window_size, center=True).mean()
    
    plt.plot(df_awbc['Step'], awbc_smooth, 
             label='With AWBC (smoothed)', color='darkblue', linewidth=2.5)
    plt.plot(df_no_awbc['Step'], no_awbc_smooth, 
             label='Without AWBC (smoothed)', color='darkred', linewidth=2.5)
    
    # Formatting
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Go2 Distillation Training: Loss Curves Comparison\nAWBC vs No-AWBC', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to log scale if values span multiple orders of magnitude
    y_min = min(df_awbc['Value'].min(), df_no_awbc['Value'].min())
    y_max = max(df_awbc['Value'].max(), df_no_awbc['Value'].max())
    if y_max / y_min > 10:
        plt.yscale('log')
        plt.ylabel('Loss Value (log scale)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def calculate_statistics(df_awbc, df_no_awbc):
    """Calculate and print comparison statistics"""
    print("=== Training Statistics Comparison ===")
    print(f"{'Metric':<20} {'With AWBC':<15} {'Without AWBC':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Final loss values (last 10% of training)
    awbc_final = df_awbc['Value'].tail(len(df_awbc)//10).mean()
    no_awbc_final = df_no_awbc['Value'].tail(len(df_no_awbc)//10).mean()
    improvement = ((no_awbc_final - awbc_final) / no_awbc_final * 100)
    
    print(f"{'Final Loss':<20} {awbc_final:<15.4f} {no_awbc_final:<15.4f} {improvement:<15.1f}%")
    
    # Minimum loss
    awbc_min = df_awbc['Value'].min()
    no_awbc_min = df_no_awbc['Value'].min()
    min_improvement = ((no_awbc_min - awbc_min) / no_awbc_min * 100)
    
    print(f"{'Minimum Loss':<20} {awbc_min:<15.4f} {no_awbc_min:<15.4f} {min_improvement:<15.1f}%")
    
    # Stability (standard deviation of last 20% of training)
    awbc_std = df_awbc['Value'].tail(len(df_awbc)//5).std()
    no_awbc_std = df_no_awbc['Value'].tail(len(df_no_awbc)//5).std()
    stability_improvement = ((no_awbc_std - awbc_std) / no_awbc_std * 100)
    
    print(f"{'Late Training Std':<20} {awbc_std:<15.4f} {no_awbc_std:<15.4f} {stability_improvement:<15.1f}%")
    print()

def main():
    # File paths in workspace
    workspace_dir = Path("/home/data/projects/robot_parkour_learning")
    awbc_file = workspace_dir / "go2_distill_awbc_Aug26_01-00-02_Go2_7skills_fromAug19_18-16-38.csv"
    no_awbc_file = workspace_dir / "go2_distill_no_awbc_Aug26_01-00-10_Go2_7skills_fromAug19_18-16-38.csv"
    
    print("Go2 Distillation Training Loss Curve Analysis")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(awbc_file):
        print(f"File not found: {awbc_file}")
        return
    if not os.path.exists(no_awbc_file):
        print(f"File not found: {no_awbc_file}")
        return
    
    # Load data
    df_awbc = load_and_process_csv(awbc_file, "With AWBC")
    df_no_awbc = load_and_process_csv(no_awbc_file, "Without AWBC")
    
    if df_awbc is None or df_no_awbc is None:
        print("Failed to load data files. Please check file paths and format.")
        return
    
    # Calculate statistics
    calculate_statistics(df_awbc, df_no_awbc)
    
    # Create plot
    save_path = workspace_dir / "loss_curves_comparison.png"
    plot_loss_curves(df_awbc, df_no_awbc, save_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
