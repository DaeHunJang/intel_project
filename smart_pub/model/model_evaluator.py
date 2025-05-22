import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import json

from ..utils.helpers import get_logger, save_json
from ..config import RESULTS_DIR

logger = get_logger(__name__)

class ModelEvaluationViz:
    def __init__(self, results_dir=RESULTS_DIR):
        self.results_dir = results_dir
        self.results_file = os.path.join(results_dir, "model_evaluation_results.csv")
        self.best_model_file = os.path.join(results_dir, "best_model.json")
        
    def save_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Save evaluation results to CSV
        """
        try:
            df = pd.DataFrame(results)
            df.to_csv(self.results_file, index=False)
            logger.info(f"Saved evaluation results to {self.results_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return False
    
    def load_results(self) -> pd.DataFrame:
        """
        Load evaluation results from CSV
        """
        try:
            if os.path.exists(self.results_file):
                df = pd.DataFrame(pd.read_csv(self.results_file))
                logger.info(f"Loaded evaluation results from {self.results_file}")
                return df
            else:
                logger.warning(f"Results file not found: {self.results_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading evaluation results: {e}")
            return pd.DataFrame()
    
    def select_best_model(self, weights: Dict[str, float] = None) -> str:
        """
        Select best model based on weighted metrics
        """
        if weights is None:
            # Default weights for metrics (accuracy is most important)
            weights = {
                "accuracy": 0.5,
                "latency": 0.25,
                "throughput": 0.25
            }
        
        df = self.load_results()
        if df.empty:
            logger.warning("No evaluation results found")
            return ""
        
        # Normalize metrics (higher is better for all)
        df_norm = df.copy()
        
        # For latency, lower is better, so invert
        if "latency" in df_norm.columns:
            max_latency = df_norm["latency"].max()
            min_latency = df_norm["latency"].min()
            if max_latency > min_latency:
                df_norm["latency"] = 1 - (df_norm["latency"] - min_latency) / (max_latency - min_latency)
            else:
                df_norm["latency"] = 1.0
        
        # For throughput, higher is better
        if "throughput" in df_norm.columns:
            max_throughput = df_norm["throughput"].max()
            min_throughput = df_norm["throughput"].min()
            if max_throughput > min_throughput:
                df_norm["throughput"] = (df_norm["throughput"] - min_throughput) / (max_throughput - min_throughput)
            else:
                df_norm["throughput"] = 1.0
        
        # For accuracy, higher is better
        if "accuracy" in df_norm.columns:
            max_accuracy = df_norm["accuracy"].max()
            min_accuracy = df_norm["accuracy"].min()
            if max_accuracy > min_accuracy:
                df_norm["accuracy"] = (df_norm["accuracy"] - min_accuracy) / (max_accuracy - min_accuracy)
            else:
                df_norm["accuracy"] = 1.0
        
        # Calculate weighted score
        df_norm["score"] = 0
        for metric, weight in weights.items():
            if metric in df_norm.columns:
                df_norm["score"] += df_norm[metric] * weight
        
        # Select model with highest score
        best_idx = df_norm["score"].idxmax()
        best_model = df.loc[best_idx, "model_id"]
        best_score = df_norm.loc[best_idx, "score"]
        
        logger.info(f"Selected best model: {best_model} with score {best_score:.4f}")
        
        # Save best model info
        best_model_info = {
            "model_id": best_model,
            "score": float(best_score),
            "metrics": {
                metric: float(df.loc[best_idx, metric]) 
                for metric in ["accuracy", "latency", "throughput"] 
                if metric in df.columns
            }
        }
        
        save_json(best_model_info, self.best_model_file)
        logger.info(f"Saved best model info to {self.best_model_file}")
        
        return best_model
    
    def create_plots(self) -> bool:
        """
        Create visualization plots for model evaluation results
        """
        df = self.load_results()
        if df.empty:
            logger.warning("No evaluation results found")
            return False
        
        try:
            # Set style for plots
            sns.set(style="whitegrid")
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot accuracy
            if "accuracy" in df.columns:
                sns.barplot(x="model_id", y="accuracy", data=df, ax=axes[0])
                axes[0].set_title("Model Accuracy")
                axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
                axes[0].set_ylim(0, 1)
            
            # Plot latency
            if "latency" in df.columns:
                sns.barplot(x="model_id", y="latency", data=df, ax=axes[1])
                axes[1].set_title("Model Latency (lower is better)")
                axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
            
            # Plot throughput
            if "throughput" in df.columns:
                sns.barplot(x="model_id", y="throughput", data=df, ax=axes[2])
                axes[2].set_title("Model Throughput (tokens/second)")
                axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
            
            # Adjust layout and save
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, "model_evaluation_plots.png")
            plt.savefig(plot_file, dpi=300)
            logger.info(f"Saved evaluation plots to {plot_file}")
            
            # Create radar chart for comparison
            self._create_radar_chart()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            return False
    
    def _create_radar_chart(self) -> bool:
        """
        Create radar chart for model comparison
        """
        df = self.load_results()
        if df.empty:
            return False
        
        try:
            # Get metrics for radar chart
            metrics = ["accuracy", "latency", "throughput"]
            available_metrics = [m for m in metrics if m in df.columns]
            
            if len(available_metrics) < 2:
                logger.warning("Not enough metrics for radar chart")
                return False
            
            # Normalize metrics
            df_norm = df.copy()
            
            # For latency, lower is better, so invert
            if "latency" in df_norm.columns:
                max_val = df_norm["latency"].max()
                min_val = df_norm["latency"].min()
                if max_val > min_val:
                    df_norm["latency_norm"] = 1 - (df_norm["latency"] - min_val) / (max_val - min_val)
                else:
                    df_norm["latency_norm"] = 1.0
                df_norm = df_norm.drop("latency", axis=1)
                available_metrics[available_metrics.index("latency")] = "latency_norm"
            
            # For throughput and accuracy, higher is better
            for metric in ["throughput", "accuracy"]:
                if metric in df_norm.columns:
                    max_val = df_norm[metric].max()
                    min_val = df_norm[metric].min()
                    if max_val > min_val:
                        df_norm[f"{metric}_norm"] = (df_norm[metric] - min_val) / (max_val - min_val)
                    else:
                        df_norm[f"{metric}_norm"] = 1.0
                    df_norm = df_norm.drop(metric, axis=1)
                    available_metrics[available_metrics.index(metric)] = f"{metric}_norm"
            
            # Create radar chart
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of metrics
            N = len(available_metrics)
            
            # Angle for each metric
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Plot each model
            for i, model in enumerate(df["model_id"]):
                values = df_norm.loc[i, available_metrics].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels
            metric_labels = [m.replace("_norm", "") for m in available_metrics]
            plt.xticks(angles[:-1], metric_labels)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Save radar chart
            radar_file = os.path.join(self.results_dir, "model_radar_chart.png")
            plt.tight_layout()
            plt.savefig(radar_file, dpi=300)
            logger.info(f"Saved radar chart to {radar_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            return False
        
    def get_best_model(self) -> str:
        """
        Get the best model ID
        """
        try:
            if os.path.exists(self.best_model_file):
                with open(self.best_model_file, 'r') as f:
                    best_model_info = json.load(f)
                return best_model_info.get("model_id", "")
            else:
                logger.warning(f"Best model file not found: {self.best_model_file}")
                return ""
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            return ""