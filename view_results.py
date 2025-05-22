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

def load_latest_results():
    """Load the most recent evaluation results"""
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        print("No evaluation results directory found.")
        return None, None
    
    # Find the latest JSON file
    json_files = list(results_dir.glob("model_evaluation_results_*.json"))
    if not json_files:
        print("No evaluation results found.")
        return None, None
    
    latest_file = max(json_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Also try to load corresponding CSV
        csv_file = latest_file.with_suffix('.csv')
        df = None
        if csv_file.exists():
            df = pd.read_csv(csv_file)
        
        print(f"Loaded results from: {latest_file}")
        return data, df
    
    except Exception as e:
        print(f"Error loading results: {e}")
        return None, None

def print_performance_summary(data):
    """Print detailed performance summary"""
    models = data['models']
    
    print("\n" + "="*80)
    print("🏆 BEST PERFORMERS")
    print("="*80)
    
    # Find best performers
    best_accuracy = max(models, key=lambda x: x['accuracy'])
    best_latency = min(models, key=lambda x: x['latency'])
    best_throughput = max(models, key=lambda x: x['throughput'])
    best_memory = min(models, key=lambda x: x['ram_usage_mb'])
    
    print(f"🎯 Best Accuracy:   {best_accuracy['model_id']} ({best_accuracy['accuracy']:.3f})")
    print(f"⚡ Best Latency:    {best_latency['model_id']} ({best_latency['latency']:.2f}s)")
    print(f"🚀 Best Throughput: {best_throughput['model_id']} ({best_throughput['throughput']:.1f} tok/s)")
    print(f"💾 Best Memory:     {best_memory['model_id']} ({best_memory['ram_usage_mb']:.1f} MB)")

def print_optimization_analysis(data):
    """Print optimization effectiveness analysis"""
    print("\n" + "="*80)
    print("📈 OPTIMIZATION EFFECTIVENESS")
    print("="*80)
    
    grouped = data['grouped_by_base_model']
    
    for base_model, variants in grouped.items():
        print(f"\n📋 {base_model}:")
        
        if 'original' not in variants:
            continue
            
        original = variants['original']
        print(f"  🔵 Original: Acc={original['accuracy']:.3f}, "
              f"Latency={original['latency']:.2f}s, "
              f"RAM={original['ram_usage_mb']:.0f}MB")
        
        for opt_type in ['int8', 'int4']:
            if opt_type in variants:
                opt = variants[opt_type]
                
                # Calculate improvements
                speedup = original['latency'] / opt['latency']
                acc_change = opt['accuracy'] - original['accuracy']
                mem_change = ((opt['ram_usage_mb'] - original['ram_usage_mb']) / original['ram_usage_mb']) * 100
                gpu_change = ((opt['gpu_allocated_mb'] - original['gpu_allocated_mb']) / original['gpu_allocated_mb']) * 100 if original['gpu_allocated_mb'] > 0 else 0
                
                # Performance indicators
                speed_indicator = "🟢" if speedup > 1.5 else "🟡" if speedup > 1.1 else "🔴"
                acc_indicator = "🟢" if acc_change >= -0.05 else "🟡" if acc_change >= -0.15 else "🔴"
                mem_indicator = "🟢" if mem_change < -10 else "🟡" if abs(mem_change) < 10 else "🔴"
                
                icon = "🟡" if opt_type == "int8" else "🟠"
                
                print(f"  {icon} {opt_type.upper()}: "
                      f"Acc={opt['accuracy']:.3f} ({acc_change:+.3f}) {acc_indicator}, "
                      f"{speedup:.2f}x speed {speed_indicator}, "
                      f"{mem_change:+.1f}% RAM {mem_indicator}")
                
                if abs(gpu_change) > 5:
                    gpu_indicator = "🟢" if gpu_change < -10 else "🔴" if gpu_change > 10 else "🟡"
                    print(f"          GPU: {gpu_change:+.1f}% {gpu_indicator}")

def print_model_rankings(data):
    """Print comprehensive model rankings"""
    print("\n" + "="*80)
    print("📊 MODEL RANKINGS")
    print("="*80)
    
    models = data['models']
    
    # Calculate composite score
    for model in models:
        # Weighted scoring: accuracy (40%), speed (30%), memory efficiency (30%)
        accuracy_score = model['accuracy']
        speed_score = 1 / model['latency'] if model['latency'] > 0 else 0
        memory_score = 1 / (model['ram_usage_mb'] / 1000) if model['ram_usage_mb'] > 0 else 0
        
        # Normalize scores (0-1 range)
        max_speed = max(1 / m['latency'] for m in models if m['latency'] > 0)
        max_memory = max(1 / (m['ram_usage_mb'] / 1000) for m in models if m['ram_usage_mb'] > 0)
        
        normalized_speed = speed_score / max_speed if max_speed > 0 else 0
        normalized_memory = memory_score / max_memory if max_memory > 0 else 0
        
        composite_score = (
            accuracy_score * 0.4 + 
            normalized_speed * 0.3 + 
            normalized_memory * 0.3
        )
        
        model['composite_score'] = composite_score
    
    # Sort by composite score
    ranked_models = sorted(models, key=lambda x: x['composite_score'], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<45} {'Type':<8} {'Score':<6} {'Acc':<6} {'Speed':<8} {'Memory'}")
    print("-" * 85)
    
    for i, model in enumerate(ranked_models, 1):
        model_name = model['model_id']
        if len(model_name) > 42:
            model_name = model_name[:39] + "..."
        
        model_type = model['model_type']
        score = model['composite_score']
        accuracy = model['accuracy']
        latency = model['latency']
        memory = model['ram_usage_mb']
        
        # Performance indicators
        acc_icon = "🟢" if accuracy >= 0.5 else "🟡" if accuracy >= 0.3 else "🔴"
        speed_icon = "🟢" if latency <= 3.0 else "🟡" if latency <= 5.0 else "🔴"
        mem_icon = "🟢" if memory <= 4000 else "🟡" if memory <= 6000 else "🔴"
        
        print(f"{i:<4} {model_name:<45} {model_type:<8} {score:<6.3f} "
              f"{accuracy:<6.3f}{acc_icon} {latency:<6.2f}s{speed_icon} {memory:<6.0f}MB{mem_icon}")

def print_optimization_summary(data):
    """Print optimization type comparison"""
    print("\n" + "="*80)
    print("🔧 OPTIMIZATION TYPE COMPARISON")
    print("="*80)
    
    summary = data['summary']['optimization_comparison']
    
    for opt_type, stats in summary.items():
        type_name = {
            'original': 'Original Models',
            'int8': 'INT8 Quantized',
            'int4': 'INT4 Quantized'
        }.get(opt_type, opt_type)
        
        icon = {
            'original': '🔵',
            'int8': '🟡', 
            'int4': '🟠'
        }.get(opt_type, '⚪')
        
        print(f"\n{icon} {type_name} ({stats['count']} models):")
        print(f"  Average Accuracy: {stats['avg_accuracy']:.3f}")
        print(f"  Average Latency:  {stats['avg_latency']:.2f}s")
        print(f"  Average Memory:   {stats['avg_memory']:.0f}MB")
        
        # Relative performance vs original
        if opt_type != 'original' and 'original' in summary:
            orig_stats = summary['original']
            acc_diff = stats['avg_accuracy'] - orig_stats['avg_accuracy']
            speed_ratio = orig_stats['avg_latency'] / stats['avg_latency']
            mem_diff = ((stats['avg_memory'] - orig_stats['avg_memory']) / orig_stats['avg_memory']) * 100
            
            print(f"  vs Original:")
            print(f"    Accuracy: {acc_diff:+.3f}")
            print(f"    Speed:    {speed_ratio:.2f}x")
            print(f"    Memory:   {mem_diff:+.1f}%")

def print_recommendations(data):
    """Print usage recommendations"""
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    models = data['models']
    
    # Find best for different use cases
    best_overall = max(models, key=lambda x: x.get('composite_score', 0))
    best_accuracy = max(models, key=lambda x: x['accuracy'])
    best_speed = min(models, key=lambda x: x['latency'])
    best_balanced = max(models, key=lambda x: x['accuracy'] * (1/x['latency']) * 100)
    
    print(f"🏆 Best Overall:    {best_overall['model_id']}")
    print(f"   Composite Score: {best_overall.get('composite_score', 0):.3f}")
    print(f"   Use for: General purpose applications")
    
    print(f"\n🎯 Most Accurate:   {best_accuracy['model_id']}")
    print(f"   Accuracy: {best_accuracy['accuracy']:.3f}")
    print(f"   Use for: High-precision tasks")
    
    print(f"\n⚡ Fastest:         {best_speed['model_id']}")
    print(f"   Latency: {best_speed['latency']:.2f}s")
    print(f"   Use for: Real-time applications")
    
    print(f"\n⚖️ Best Balanced:   {best_balanced['model_id']}")
    print(f"   Accuracy: {best_balanced['accuracy']:.3f}, Latency: {best_balanced['latency']:.2f}s")
    print(f"   Use for: Production applications")
    
    # Optimization recommendations
    print(f"\n🔧 Optimization Recommendations:")
    grouped = data['grouped_by_base_model']
    
    for base_model, variants in grouped.items():
        if len(variants) > 1 and 'original' in variants:
            orig = variants['original']
            
            # Find best optimization
            best_opt = None
            best_opt_type = None
            best_improvement = -1
            
            for opt_type in ['int8', 'int4']:
                if opt_type in variants:
                    opt = variants[opt_type]
                    speedup = orig['latency'] / opt['latency']
                    acc_preserved = opt['accuracy'] >= orig['accuracy'] * 0.9  # 90% accuracy retention
                    
                    if speedup > 1.2 and acc_preserved:
                        improvement = speedup * opt['accuracy']
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_opt = opt
                            best_opt_type = opt_type
            
            if best_opt:
                speedup = orig['latency'] / best_opt['latency']
                print(f"  • {base_model}: Use {best_opt_type.upper()} ({speedup:.1f}x faster)")

def show_evaluation_results():
    """Display comprehensive evaluation results"""
    data, df = load_latest_results()
    
    if not data:
        print("No evaluation results found.")
        return
    
    # Print header
    print("=" * 80)
    print("🔬 SMART PUB MODEL EVALUATION RESULTS")
    print("=" * 80)
    print(f"📅 Evaluation Date: {data['evaluation_info']['timestamp']}")
    print(f"📊 Total Models: {data['evaluation_info']['total_models_evaluated']}")
    print(f"🧪 Test Cases: 50 (from test_cases.json)")
    
    # Show all analysis sections
    print_performance_summary(data)
    print_optimization_analysis(data)
    print_model_rankings(data)
    print_optimization_summary(data)
    print_recommendations(data)
    
    # Show file locations
    print("\n" + "="*80)
    print("📁 FILES")
    print("="*80)
    results_dir = Path(RESULTS_DIR)
    json_files = list(results_dir.glob("model_evaluation_results_*.json"))
    csv_files = list(results_dir.glob("model_evaluation_results_*.csv"))
    
    if json_files:
        latest_json = max(json_files, key=os.path.getctime)
        print(f"📄 Latest JSON: {latest_json}")
    
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)
        print(f"📊 Latest CSV:  {latest_csv}")
    
    print("="*80)

def create_visualization_plots(data):
    """Create visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        models = data['models']
        df = pd.DataFrame(models)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Smart Pub Model Evaluation Results', fontsize=16)
        
        # Plot 1: Accuracy vs Latency
        scatter_colors = {'original': 'blue', 'int8': 'orange', 'int4': 'red'}
        for model_type in ['original', 'int8', 'int4']:
            subset = df[df['model_type'] == model_type]
            if not subset.empty:
                axes[0,0].scatter(subset['latency'], subset['accuracy'], 
                                label=model_type.upper(), alpha=0.7, 
                                color=scatter_colors[model_type], s=100)
        
        axes[0,0].set_xlabel('Latency (seconds)')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Accuracy vs Latency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Throughput comparison
        df_sorted = df.sort_values('throughput', ascending=True)
        bars = axes[0,1].barh(range(len(df_sorted)), df_sorted['throughput'])
        axes[0,1].set_yticks(range(len(df_sorted)))
        axes[0,1].set_yticklabels([f"{row['model_id'].split('/')[-1][:20]}..." 
                                   if len(row['model_id']) > 20 else row['model_id'].split('/')[-1] 
                                   for _, row in df_sorted.iterrows()], fontsize=8)
        axes[0,1].set_xlabel('Throughput (tokens/second)')
        axes[0,1].set_title('Model Throughput Comparison')
        
        # Color bars by model type
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            bars[i].set_color(scatter_colors.get(row['model_type'], 'gray'))
        
        # Plot 3: Memory usage
        axes[1,0].bar(df['model_id'].str.split('/').str[-1].str[:15], df['ram_usage_mb'])
        axes[1,0].set_ylabel('RAM Usage (MB)')
        axes[1,0].set_title('Memory Usage by Model')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Optimization effectiveness
        base_models = list(data['grouped_by_base_model'].keys())
        opt_data = {'original': [], 'int8': [], 'int4': []}
        
        for base_model in base_models:
            variants = data['grouped_by_base_model'][base_model]
            for opt_type in ['original', 'int8', 'int4']:
                if opt_type in variants:
                    opt_data[opt_type].append(variants[opt_type]['accuracy'])
                else:
                    opt_data[opt_type].append(0)
        
        x = range(len(base_models))
        width = 0.25
        
        for i, (opt_type, accuracies) in enumerate(opt_data.items()):
            if any(acc > 0 for acc in accuracies):
                axes[1,1].bar([xi + i*width for xi in x], accuracies, 
                            width, label=opt_type.upper(), 
                            color=scatter_colors.get(opt_type, 'gray'), alpha=0.7)
        
        axes[1,1].set_xlabel('Base Model')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Optimization Impact on Accuracy')
        axes[1,1].set_xticks([xi + width for xi in x])
        axes[1,1].set_xticklabels([bm.split('-')[0][:10] for bm in base_models], rotation=45)
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(RESULTS_DIR) / "evaluation_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {plot_file}")
        
        # Show plot if in interactive mode
        try:
            plt.show()
        except:
            pass
            
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {e}")

def main():
    parser = argparse.ArgumentParser(description="View Smart Pub Model Evaluation Results")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument("--json-file", help="Specific JSON file to analyze")
    args = parser.parse_args()
    
    if args.json_file:
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = None
        except Exception as e:
            print(f"Error loading {args.json_file}: {e}")
            return
    else:
        data, df = load_latest_results()
        if not data:
            return
    
    show_evaluation_results()
    
    if args.plot:
        create_visualization_plots(data)

if __name__ == "__main__":
    main()