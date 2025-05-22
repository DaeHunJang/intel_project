#!/usr/bin/env python3
import os
import argparse
import sys
import time
import psutil
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import torch

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.model.llm_wrapper import LLMWrapper
from smart_pub.model.optimizer import ModelOptimizer
from smart_pub.model.model_evaluator import ModelEvaluationViz
from smart_pub.data_processing.drink_processor import DrinkProcessor
from smart_pub.utils.helpers import get_logger
from smart_pub.config import DATA_DIR, MODEL_DIR, MODEL_ID, MODEL_CANDIDATES, RESULTS_DIR

logger = get_logger("evaluator")

def get_local_model_path(model_id, model_dir):
    """Convert Hugging Face model ID to local path"""
    # Extract model name from HF model ID
    if "/" in model_id:
        model_name = model_id.split("/")[-1]
    else:
        model_name = model_id
    
    local_path = Path(model_dir) / model_name
    
    # Check if local model exists
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"Found local model: {local_path}")
        return str(local_path)
    else:
        logger.warning(f"Local model not found at {local_path}")
        logger.info(f"Available directories in {model_dir}:")
        model_dir_path = Path(model_dir)
        if model_dir_path.exists():
            for item in model_dir_path.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item.name}")
        return None

def check_model_files(model_path):
    """Check if model has required files"""
    model_path = Path(model_path)
    
    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    
    # Check required files
    for req_file in required_files:
        if not (model_path / req_file).exists():
            logger.error(f"Missing required file: {req_file}")
            return False
    
    # Check model weight files
    has_model_file = False
    for model_file in model_files:
        if (model_path / model_file).exists():
            has_model_file = True
            logger.info(f"Found model file: {model_file}")
            break
        
        # Check for split model files
        split_files = list(model_path.glob(f"{model_file.split('.')[0]}-*.{model_file.split('.')[1]}"))
        if split_files:
            has_model_file = True
            logger.info(f"Found split model files: {[f.name for f in split_files]}")
            break
    
    if not has_model_file:
        logger.error("No model weight files found")
        return False
    
    return True

class LLMEvaluator:
    def __init__(self, model_id=MODEL_ID, model_dir=MODEL_DIR):
        self.original_model_id = model_id
        self.model_dir = model_dir
        
        # Get local path for the model
        self.local_model_path = get_local_model_path(model_id, model_dir)
        
        if self.local_model_path:
            self.model_id = self.local_model_path
            logger.info(f"Using local model path: {self.model_id}")
        else:
            self.model_id = model_id
            logger.warning(f"Will attempt to use HF model ID: {self.model_id}")
        
        self.llm = None
        self.metrics = {}
    
    def load_model(self):
        logger.info(f"Loading model from {self.model_id}")
        
        # Verify model files if using local path
        if self.local_model_path and not check_model_files(self.local_model_path):
            logger.error(f"Model validation failed for {self.local_model_path}")
            return False
        
        try:
            self.llm = LLMWrapper(self.model_id, self.model_dir)
            result = self.llm.load_model()
            
            if result:
                logger.info(f"Successfully loaded model: {self.original_model_id}")
            else:
                logger.error(f"Failed to load model: {self.original_model_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading model {self.original_model_id}: {e}")
            return False
    
    def unload_model(self):
        if self.llm and self.llm.model:
            self.llm.model = None
            self.llm.tokenizer = None
            self.llm.pipeline = None
            
            torch.cuda.empty_cache()
            logger.info(f"Unloaded model {self.original_model_id} and cleared GPU cache")
    
    def measure_latency(self, prompt, num_runs=3):
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return 0
        
        total_time = 0
        
        for _ in tqdm(range(num_runs), desc="Measuring latency"):
            start_time = time.time()
            try:
                self.llm.generate(prompt)
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                return 0
            end_time = time.time()
            total_time += (end_time - start_time)
        
        avg_latency = total_time / num_runs
        logger.info(f"Average latency: {avg_latency:.4f} seconds")
        
        self.metrics["latency"] = avg_latency
        return avg_latency
    
    def measure_throughput(self, prompt, num_runs=3):
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return 0
        
        total_tokens = 0
        total_time = 0
        
        for _ in tqdm(range(num_runs), desc="Measuring throughput"):
            start_time = time.time()
            try:
                generated_text = self.llm.generate(prompt)
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                return 0
            end_time = time.time()
            
            token_count = len(generated_text.split())
            total_tokens += token_count
            total_time += (end_time - start_time)
        
        avg_throughput = total_tokens / total_time if total_time > 0 else 0
        logger.info(f"Average throughput: {avg_throughput:.2f} tokens/second")
        
        self.metrics["throughput"] = avg_throughput
        return avg_throughput
    
    def measure_memory(self):
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
        
        memory_stats = {}
        
        # Get process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_stats["ram_usage_mb"] = memory_info.rss / (1024 * 1024)
        
        # Get GPU memory info if available
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            memory_stats["gpu_allocated_mb"] = torch.cuda.memory_allocated(current_device) / (1024 * 1024)
            memory_stats["gpu_reserved_mb"] = torch.cuda.memory_reserved(current_device) / (1024 * 1024)
        
        logger.info(f"Memory usage: {memory_stats}")
        self.metrics.update(memory_stats)
        
        return memory_stats
    
    def measure_accuracy(self, test_cases):
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return 0
        
        correct = 0
        
        for i, case in enumerate(tqdm(test_cases, desc="Measuring accuracy")):
            prompt = case["prompt"]
            expected_keywords = case["expected_keywords"]
            
            try:
                generated_text = self.llm.generate(prompt)
            except Exception as e:
                logger.error(f"Error during generation for test case {i+1}: {e}")
                continue
            
            matches = sum(1 for keyword in expected_keywords if keyword.lower() in generated_text.lower())
            score = matches / len(expected_keywords) if expected_keywords else 0
            
            if score >= 0.7:
                correct += 1
            
            logger.debug(f"Test case {i+1}, Score: {score:.2f}")
        
        accuracy = correct / len(test_cases) if test_cases else 0
        logger.info(f"Accuracy: {accuracy:.2f}")
        
        self.metrics["accuracy"] = accuracy
        return accuracy
    
    def run_all_metrics(self, test_cases):
        # Short prompt for latency and throughput tests
        short_prompt = "오늘 기분이 좋아. 어떤 술이 좋을까?"
        
        self.measure_latency(short_prompt)
        self.measure_throughput(short_prompt)
        self.measure_memory()
        self.measure_accuracy(test_cases)
        
        self.metrics["model_id"] = self.original_model_id
        
        return self.metrics

def prepare_test_cases():
    # Simple test cases for emotion analysis and recommendation
    test_cases = [
        {
            "prompt": "오늘 기분이 너무 우울해. 뭐 마시면 좋을까?",
            "expected_keywords": ["기분전환", "위로"]
        },
        {
            "prompt": "친구들과 파티할 거라서 신나는 분위기가 필요해!",
            "expected_keywords": ["기분고조", "사교", "활력"]
        },
        {
            "prompt": "일이 너무 많아서 스트레스 받아. 도움되는 음료 추천해줘.",
            "expected_keywords": ["스트레스해소", "긴장완화"]
        },
        {
            "prompt": "오늘 승진했어! 축하할만한 특별한 음료 없을까?",
            "expected_keywords": ["특별한", "행복", "축하"]
        },
        {
            "prompt": "조용히 사색하고 싶은 밤이야. 어울리는 술 추천해줘.",
            "expected_keywords": ["사색", "차분함", "여유"]
        }
    ]
    
    return test_cases

def list_available_models(model_dir):
    """List available local models"""
    model_dir_path = Path(model_dir)
    available_models = []
    
    if not model_dir_path.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return available_models
    
    for item in model_dir_path.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            available_models.append(item.name)
    
    return available_models

def get_target_models():
    """Get the target models for evaluation from available local models"""
    target_model_names = [
        "HyperCLOVAX-SEED-Text-Instruct-1.5B",
        "ko-sbert-nli", 
        "HyperCLOVAX-SEED-Text-Instruct-0.5B",
        "ko-gpt-trinity-1.2B-v0.5",
        "polyglot-ko-1.3b",
        "Llama-3-Open-Ko-8B"
    ]
    
    available_models = list_available_models(MODEL_DIR)
    logger.info(f"Available local models: {available_models}")
    
    # Match target models with available models
    target_models = []
    for target in target_model_names:
        if target in available_models:
            target_models.append(target)
            logger.info(f"✓ Found target model: {target}")
        else:
            logger.warning(f"✗ Target model not found: {target}")
    
    return target_models

def save_all_results(results, output_file=None):
    """Save all evaluation results to a single comprehensive file"""
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"model_evaluation_results_{timestamp}.json"
    
    # Ensure results directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare comprehensive results
    comprehensive_results = {
        "evaluation_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models_evaluated": len(results),
            "evaluation_metrics": ["latency", "throughput", "ram_usage_mb", "gpu_allocated_mb", "gpu_reserved_mb", "accuracy"]
        },
        "models": results,
        "summary": {}
    }
    
    # Add summary statistics
    if results:
        metrics_summary = {}
        for metric in ["latency", "throughput", "ram_usage_mb", "gpu_allocated_mb", "accuracy"]:
            values = [r.get(metric, 0) for r in results if metric in r and isinstance(r[metric], (int, float))]
            if values:
                metrics_summary[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "best_model": None
                }
                
                # Find best model for each metric
                if metric in ["throughput", "accuracy"]:  # Higher is better
                    best_idx = values.index(max(values))
                else:  # Lower is better (latency, memory)
                    best_idx = values.index(min(values))
                
                metrics_summary[metric]["best_model"] = results[best_idx]["model_id"]
        
        comprehensive_results["summary"] = metrics_summary
    
    # Save to JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive results saved to: {output_file}")
        
        # Also save as CSV for easy viewing
        csv_file = output_file.with_suffix('.csv')
        if results:
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Results also saved as CSV: {csv_file}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None

def evaluate_all_models():
    """Evaluate all target models without optimization"""
    
    # Get target models from local directory
    target_models = get_target_models()
    
    if not target_models:
        logger.error("No target models found locally")
        return [], None
    
    logger.info(f"Will evaluate {len(target_models)} models: {target_models}")
    
    test_cases = prepare_test_cases()
    results = []
    
    for i, model_name in enumerate(target_models, 1):
        logger.info(f"[{i}/{len(target_models)}] Evaluating model: {model_name}")
        
        # Use model name directly (it's already the local directory name)
        evaluator = LLMEvaluator(model_name, MODEL_DIR)
        
        if evaluator.load_model():
            try:
                logger.info(f"Running metrics for {model_name}...")
                metrics = evaluator.run_all_metrics(test_cases)
                results.append(metrics)
                
                # Print intermediate results
                print(f"\n{'='*60}")
                print(f"Results for {model_name}:")
                print(f"{'='*60}")
                for k, v in metrics.items():
                    if k != "model_id" and isinstance(v, (int, float)):
                        print(f"  {k:<20}: {v:.4f}")
                    elif k != "model_id":
                        print(f"  {k:<20}: {v}")
                print(f"{'='*60}")
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            finally:
                evaluator.unload_model()  # Unload model to free up memory
                logger.info(f"Model {model_name} unloaded, GPU memory cleared")
                
        else:
            logger.error(f"Failed to load model {model_name}")
    
    if not results:
        logger.error("No models were successfully evaluated")
        return [], None
    
    # Save comprehensive results
    results_file = save_all_results(results)
    
    # Find best model based on combined score
    best_model = None
    if results:
        best_model = find_best_model(results)
    
    return results, best_model, results_file

def find_best_model(results):
    """Find the best model based on weighted scoring"""
    if not results:
        return None
    
    scored_models = []
    
    for result in results:
        model_id = result.get("model_id", "Unknown")
        score = 0
        
        # Scoring weights (can be adjusted)
        weights = {
            "accuracy": 0.4,      # 40% - most important for our use case
            "latency": 0.2,       # 20% - response time matters
            "throughput": 0.2,    # 20% - processing speed
            "ram_usage_mb": 0.1,  # 10% - memory efficiency
            "gpu_allocated_mb": 0.1  # 10% - GPU efficiency
        }
        
        # Normalize and score each metric
        for metric, weight in weights.items():
            if metric in result and isinstance(result[metric], (int, float)):
                value = result[metric]
                if metric in ["accuracy", "throughput"]:  # Higher is better
                    score += value * weight
                else:  # Lower is better (latency, memory)
                    # Use inverse scoring for "lower is better" metrics
                    if value > 0:
                        score += (1 / value) * weight * 1000  # Scale factor
        
        scored_models.append((model_id, score))
        logger.info(f"Model {model_id} total score: {score:.4f}")
    
    # Sort by score (highest first)
    scored_models.sort(key=lambda x: x[1], reverse=True)
    best_model = scored_models[0][0] if scored_models else None
    
    logger.info(f"Best model selected: {best_model}")
    return best_model

def update_config_with_best_model(best_model):
    if not best_model:
        logger.warning("No best model to update config with")
        return False
    
    try:
        config_path = os.path.join(Path(__file__).parent, "smart_pub", "config.py")
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        new_config = config_content.replace(
            f'MODEL_ID = "{MODEL_ID}"', 
            f'MODEL_ID = "{best_model}"'
        )
        
        with open(config_path, 'w') as f:
            f.write(new_config)
        
        logger.info(f"Updated config.py with best model: {best_model}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating config.py: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM models for Smart Pub")
    parser.add_argument("--model-id", help="Specific model ID to evaluate")
    parser.add_argument("--list-models", action="store_true", help="List available local models")
    parser.add_argument("--output-file", help="Output file path for results")
    args = parser.parse_args()
    
    if args.list_models:
        available_models = list_available_models(MODEL_DIR)
        target_models = get_target_models()
        
        print("Available local models:")
        for model in available_models:
            status = "✓ TARGET" if model in target_models else "  "
            print(f"  {status} {model}")
        
        print(f"\nTarget models for evaluation: {len(target_models)}")
        for model in target_models:
            print(f"  ✓ {model}")
        return
    
    if args.model_id:
        # Evaluate single model
        logger.info(f"Evaluating single model: {args.model_id}")
        
        evaluator = LLMEvaluator(args.model_id, MODEL_DIR)
        
        if evaluator.load_model():
            try:
                test_cases = prepare_test_cases()
                metrics = evaluator.run_all_metrics(test_cases)
                
                # Save single model result
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = args.output_file or (RESULTS_DIR / f"single_model_evaluation_{args.model_id}_{timestamp}.json")
                save_all_results([metrics], Path(output_file))
                
                # Print results
                print(f"\n{'='*60}")
                print(f"Evaluation Results for {args.model_id}")
                print(f"{'='*60}")
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k:<20}: {v:.4f}")
                    else:
                        print(f"  {k:<20}: {v}")
                print(f"{'='*60}")
                
            except Exception as e:
                logger.error(f"Error evaluating model {args.model_id}: {e}")
            finally:
                evaluator.unload_model()
        else:
            logger.error(f"Failed to load model {args.model_id}")
        return
    
    # Evaluate all target models
    logger.info("Starting comprehensive model evaluation...")
    results, best_model, results_file = evaluate_all_models()
    
    if not results:
        logger.error("No evaluation results obtained")
        return
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal Models Evaluated: {len(results)}")
    print(f"Results saved to: {results_file}")
    
    # Print detailed results for each model
    for i, model_metrics in enumerate(results, 1):
        model_id = model_metrics.get("model_id", "Unknown")
        print(f"\n[{i}] Model: {model_id}")
        print("-" * 60)
        for k, v in model_metrics.items():
            if k != "model_id" and isinstance(v, (int, float)):
                print(f"  {k:<20}: {v:.4f}")
        
        # Add performance indicators
        accuracy = model_metrics.get("accuracy", 0)
        latency = model_metrics.get("latency", float('inf'))
        
        if accuracy >= 0.8:
            print("  🟢 High Accuracy")
        elif accuracy >= 0.6:
            print("  🟡 Medium Accuracy") 
        else:
            print("  🔴 Low Accuracy")
            
        if latency <= 2.0:
            print("  🟢 Fast Response")
        elif latency <= 5.0:
            print("  🟡 Medium Response")
        else:
            print("  🔴 Slow Response")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    if results:
        accuracies = [r.get("accuracy", 0) for r in results]
        latencies = [r.get("latency", 0) for r in results if r.get("latency", 0) > 0]
        throughputs = [r.get("throughput", 0) for r in results if r.get("throughput", 0) > 0]
        
        if accuracies:
            print(f"Accuracy  - Best: {max(accuracies):.4f}, Worst: {min(accuracies):.4f}, Avg: {sum(accuracies)/len(accuracies):.4f}")
        if latencies:
            print(f"Latency   - Best: {min(latencies):.4f}s, Worst: {max(latencies):.4f}s, Avg: {sum(latencies)/len(latencies):.4f}s")
        if throughputs:
            print(f"Throughput- Best: {max(throughputs):.2f} tok/s, Worst: {min(throughputs):.2f} tok/s, Avg: {sum(throughputs)/len(throughputs):.2f} tok/s")
    
    print(f"\n🏆 BEST OVERALL MODEL: {best_model}")
    print("="*80)
    
    # Save output file path info
    if results_file:
        print(f"\n📁 Detailed results saved to:")
        print(f"   JSON: {results_file}")
        print(f"   CSV:  {Path(results_file).with_suffix('.csv')}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 {len(results)} models evaluated and saved to single file")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()