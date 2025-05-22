#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.model.model_evaluator import ModelEvaluationViz
from smart_pub.utils.helpers import get_logger
from smart_pub.config import RESULTS_DIR

logger = get_logger("results_viewer")

def show_evaluation_results():
    """
    Display evaluation results
    """
    evaluator = ModelEvaluationViz(RESULTS_DIR)
    results_df = evaluator.load_results()
    
    if results_df.empty:
        print("No evaluation results found.")
        return
    
    # Get best model
    best_model = evaluator.get_best_model()
    
    # Print results in a nice table
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    # Format dataframe for display
    display_df = results_df.copy()
    
    # Create a nice pandas styler
    def highlight_best_model(s):
        is_best = s['model_id'] == best_model
        return ['background-color: lightyellow' if is_best else '' for _ in s]
    
    # Display metrics
    print("\nMetrics by Model:\n")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    
    for _, row in display_df.iterrows():
        model_name = row['model_id']
        highlight = " (BEST MODEL)" if model_name == best_model else ""
        print(f"Model: {model_name}{highlight}")
        
        for col in display_df.columns:
            if col != 'model_id':
                if isinstance(row[col], float):
                    print(f"  {col}: {row[col]:.4f}")
                else:
                    print(f"  {col}: {row[col]}")
        print()
    
    # Show plots if they exist
    plots_file = os.path.join(RESULTS_DIR, "model_evaluation_plots.png")
    radar_file = os.path.join(RESULTS_DIR, "model_radar_chart.png")
    
    if os.path.exists(plots_file):
        print(f"\nPerformance plots saved to: {plots_file}")
    
    if os.path.exists(radar_file):
        print(f"Radar chart saved to: {radar_file}")
    
    # Show best model details
    best_model_file = os.path.join(RESULTS_DIR, "best_model.json")
    if os.path.exists(best_model_file):
        with open(best_model_file, 'r') as f:
            best_model_info = json.load(f)
        
        print("\n" + "="*40)
        print("BEST MODEL")
        print("="*40)
        print(f"Model ID: {best_model_info.get('model_id')}")
        print(f"Score: {best_model_info.get('score'):.4f}")
        
        metrics = best_model_info.get('metrics', {})
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    print("\n" + "="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="View Smart Pub Model Evaluation Results")
    parser.add_argument("--regenerate-plots", action="store_true", help="Regenerate plots from results")
    args = parser.parse_args()
    
    evaluator = ModelEvaluationViz(RESULTS_DIR)
    
    if args.regenerate_plots:
        if evaluator.create_plots():
            print("Successfully regenerated plots.")
        else:
            print("Failed to regenerate plots.")
    
    show_evaluation_results()

if __name__ == "__main__":
    main()