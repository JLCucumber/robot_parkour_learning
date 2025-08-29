#!/usr/bin/env python3
"""
Enhanced Loss Curve Plotting Script for Go2 Distillation Training
Plots loss curves from CSV files comparing AWBC vs no-AWBC training with additional analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set matplotlib to use non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

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

def plot_loss_curves_enhanced(df_awbc, df_no_awbc, save_path=None):
    """Plot enhanced loss curves with multiple subplots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main plot - both curves
    ax1.plot(df_awbc['Step'], df_awbc['Value'], 
             label='With AWBC', color='blue', linewidth=1.2, alpha=0.7)
    ax1.plot(df_no_awbc['Step'], df_no_awbc['Value'], 
             label='Without AWBC', color='red', linewidth=1.2, alpha=0.7)
    
    # Add smoothed trend lines
    window_size = max(len(df_awbc) // 50, 10)
    awbc_smooth = df_awbc['Value'].rolling(window=window_size, center=True).mean()
    no_awbc_smooth = df_no_awbc['Value'].rolling(window=window_size, center=True).mean()
    
    ax1.plot(df_awbc['Step'], awbc_smooth, 
             label='With AWBC (smoothed)', color='darkblue', linewidth=2.5)
    ax1.plot(df_no_awbc['Step'], no_awbc_smooth, 
             label='Without AWBC (smoothed)', color='darkred', linewidth=2.5)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Training Loss Comparison: AWBC vs No-AWBC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale plot
    ax2.semilogy(df_awbc['Step'], df_awbc['Value'], 
                 label='With AWBC', color='blue', linewidth=1.2, alpha=0.7)
    ax2.semilogy(df_no_awbc['Step'], df_no_awbc['Value'], 
                 label='Without AWBC', color='red', linewidth=1.2, alpha=0.7)
    ax2.semilogy(df_awbc['Step'], awbc_smooth, 
                 label='With AWBC (smoothed)', color='darkblue', linewidth=2.5)
    ax2.semilogy(df_no_awbc['Step'], no_awbc_smooth, 
                 label='Without AWBC (smoothed)', color='darkred', linewidth=2.5)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss Value (log scale)')
    ax2.set_title('Training Loss Comparison (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training progress comparison (normalized)
    awbc_norm = (df_awbc['Value'] - df_awbc['Value'].iloc[0]) / df_awbc['Value'].iloc[0]
    no_awbc_norm = (df_no_awbc['Value'] - df_no_awbc['Value'].iloc[0]) / df_no_awbc['Value'].iloc[0]
    
    ax3.plot(df_awbc['Step'], awbc_norm, 
             label='With AWBC', color='blue', linewidth=1.5)
    ax3.plot(df_no_awbc['Step'], no_awbc_norm, 
             label='Without AWBC', color='red', linewidth=1.5)
    
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Normalized Loss Change')
    ax3.set_title('Normalized Loss Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Loss distribution histogram
    ax4.hist(df_awbc['Value'], bins=50, alpha=0.7, label='With AWBC', 
             color='blue', density=True)
    ax4.hist(df_no_awbc['Value'], bins=50, alpha=0.7, label='Without AWBC', 
             color='red', density=True)
    
    ax4.set_xlabel('Loss Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Loss Value Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced plot saved to: {save_path}")
    
    plt.close()  # Close the figure to free memory

def calculate_detailed_statistics(df_awbc, df_no_awbc):
    """Calculate detailed comparison statistics"""
    print("=== Detailed Training Statistics Comparison ===")
    print(f"{'Metric':<25} {'With AWBC':<15} {'Without AWBC':<15} {'Improvement':<15}")
    print("-" * 75)
    
    # Basic statistics
    awbc_mean = df_awbc['Value'].mean()
    no_awbc_mean = df_no_awbc['Value'].mean()
    mean_improvement = ((no_awbc_mean - awbc_mean) / no_awbc_mean * 100)
    print(f"{'Mean Loss':<25} {awbc_mean:<15.4f} {no_awbc_mean:<15.4f} {mean_improvement:<15.1f}%")
    
    # Final loss values (last 10% of training)
    awbc_final = df_awbc['Value'].tail(len(df_awbc)//10).mean()
    no_awbc_final = df_no_awbc['Value'].tail(len(df_no_awbc)//10).mean()
    improvement = ((no_awbc_final - awbc_final) / no_awbc_final * 100)
    print(f"{'Final Loss (10%)':<25} {awbc_final:<15.4f} {no_awbc_final:<15.4f} {improvement:<15.1f}%")
    
    # Minimum loss
    awbc_min = df_awbc['Value'].min()
    no_awbc_min = df_no_awbc['Value'].min()
    min_improvement = ((no_awbc_min - awbc_min) / no_awbc_min * 100)
    print(f"{'Minimum Loss':<25} {awbc_min:<15.4f} {no_awbc_min:<15.4f} {min_improvement:<15.1f}%")
    
    # Stability metrics
    awbc_std = df_awbc['Value'].std()
    no_awbc_std = df_no_awbc['Value'].std()
    stability_improvement = ((no_awbc_std - awbc_std) / no_awbc_std * 100)
    print(f"{'Overall Std Dev':<25} {awbc_std:<15.4f} {no_awbc_std:<15.4f} {stability_improvement:<15.1f}%")
    
    # Late training stability (last 20%)
    awbc_late_std = df_awbc['Value'].tail(len(df_awbc)//5).std()
    no_awbc_late_std = df_no_awbc['Value'].tail(len(df_no_awbc)//5).std()
    late_stability = ((no_awbc_late_std - awbc_late_std) / no_awbc_late_std * 100)
    print(f"{'Late Training Std':<25} {awbc_late_std:<15.4f} {no_awbc_late_std:<15.4f} {late_stability:<15.1f}%")
    
    # Convergence analysis
    awbc_initial = df_awbc['Value'].head(len(df_awbc)//10).mean()
    no_awbc_initial = df_no_awbc['Value'].head(len(df_no_awbc)//10).mean()
    awbc_convergence = (awbc_initial - awbc_final) / awbc_initial
    no_awbc_convergence = (no_awbc_initial - no_awbc_final) / no_awbc_initial
    
    print(f"\n=== Convergence Analysis ===")
    print(f"{'Method':<25} {'Initial Loss':<15} {'Final Loss':<15} {'Reduction %':<15}")
    print("-" * 75)
    print(f"{'With AWBC':<25} {awbc_initial:<15.4f} {awbc_final:<15.4f} {awbc_convergence*100:<15.1f}%")
    print(f"{'Without AWBC':<25} {no_awbc_initial:<15.4f} {no_awbc_final:<15.4f} {no_awbc_convergence*100:<15.1f}%")
    print()

def create_summary_report(df_awbc, df_no_awbc, workspace_dir):
    """Create a text summary report"""
    report_path = workspace_dir / "training_comparison_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Go2 Distillation Training Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"- With AWBC: {len(df_awbc)} data points, steps {df_awbc['Step'].min()} - {df_awbc['Step'].max()}\n")
        f.write(f"- Without AWBC: {len(df_no_awbc)} data points, steps {df_no_awbc['Step'].min()} - {df_no_awbc['Step'].max()}\n\n")
        
        # Key findings
        awbc_final = df_awbc['Value'].tail(len(df_awbc)//10).mean()
        no_awbc_final = df_no_awbc['Value'].tail(len(df_no_awbc)//10).mean()
        improvement = ((no_awbc_final - awbc_final) / no_awbc_final * 100)
        
        f.write("Key Findings:\n")
        if improvement > 0:
            f.write(f"- AWBC shows {improvement:.1f}% improvement in final loss\n")
        else:
            f.write(f"- No-AWBC shows {-improvement:.1f}% better final loss\n")
        
        awbc_min = df_awbc['Value'].min()
        no_awbc_min = df_no_awbc['Value'].min()
        if awbc_min < no_awbc_min:
            f.write(f"- AWBC achieves lower minimum loss: {awbc_min:.4f} vs {no_awbc_min:.4f}\n")
        else:
            f.write(f"- No-AWBC achieves lower minimum loss: {no_awbc_min:.4f} vs {awbc_min:.4f}\n")
        
        awbc_std = df_awbc['Value'].tail(len(df_awbc)//5).std()
        no_awbc_std = df_no_awbc['Value'].tail(len(df_no_awbc)//5).std()
        if awbc_std < no_awbc_std:
            f.write(f"- AWBC shows more stable training (lower std dev): {awbc_std:.4f} vs {no_awbc_std:.4f}\n")
        else:
            f.write(f"- No-AWBC shows more stable training (lower std dev): {no_awbc_std:.4f} vs {awbc_std:.4f}\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    # File paths in workspace
    workspace_dir = Path("/home/data/projects/robot_parkour_learning")
    awbc_file = workspace_dir / "go2_distill_awbc_Aug27_16-58-48_Go2_7skills_fromAug19_18-16-38.csv"
    no_awbc_file = workspace_dir / "go2_distill_no_awbc_Aug27_16-57-38_Go2_7skills_fromAug19_18-16-38.csv"
    
    print("Go2 Distillation Training Loss Curve Analysis (Enhanced)")
    print("=" * 60)
    
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
    
    # Calculate detailed statistics
    calculate_detailed_statistics(df_awbc, df_no_awbc)
    
    # Create enhanced plot
    enhanced_save_path = workspace_dir / "loss_curves_enhanced_comparison.png"
    plot_loss_curves_enhanced(df_awbc, df_no_awbc, enhanced_save_path)
    
    # Create summary report
    create_summary_report(df_awbc, df_no_awbc, workspace_dir)
    
    print("Enhanced analysis complete!")
    print(f"Check the following files:")
    print(f"- Enhanced plot: {enhanced_save_path}")
    print(f"- Summary report: {workspace_dir / 'training_comparison_report.txt'}")

if __name__ == "__main__":
    main()
