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
import gc

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.model.llm_wrapper import LLMWrapper
from smart_pub.model.optimizer import ModelOptimizer
from smart_pub.model.model_evaluator import ModelEvaluationViz
from smart_pub.data_processing.drink_processor import DrinkProcessor
from smart_pub.utils.helpers import get_logger
from smart_pub.config import DATA_DIR, MODEL_DIR, MODEL_ID, MODEL_CANDIDATES, RESULTS_DIR

logger = get_logger("evaluator")

def get_local_model_path(model_id: str, model_dir: str) -> str | None:
    """Convert Hugging Face model ID to local path"""
    if "/" in model_id:
        model_name = model_id.split("/")[-1]
    else:
        model_name = model_id
    
    local_path = Path(model_dir) / model_name
    
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"Found local model: {local_path}")
        return str(local_path)
    else:
        logger.warning(f"Local model not found at {local_path}")
        return None

def check_model_files(model_path: str) -> bool:
    """Check if model has required files"""
    model_path_obj = Path(model_path)
    
    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    
    for req_file in required_files:
        if not (model_path_obj / req_file).exists():
            logger.error(f"Missing required file: {req_file}")
            return False
    
    has_model_file = False
    for model_file in model_files:
        if (model_path_obj / model_file).exists():
            has_model_file = True
            logger.info(f"Found model file: {model_file}")
            break
        
        split_files = list(model_path_obj.glob(f"{model_file.split('.')[0]}-*.{model_file.split('.')[1]}"))
        if split_files:
            has_model_file = True
            logger.info(f"Found split model files: {[f.name for f in split_files]}")
            break
    
    if not has_model_file:
        logger.error("No model weight files found")
        return False
    
    return True

class LLMEvaluator:
    def __init__(self, model_id: str = MODEL_ID, model_dir: str = MODEL_DIR):
        self.original_model_id = model_id
        self.model_dir = model_dir
        
        self.local_model_path = get_local_model_path(model_id, model_dir)
        
        if self.local_model_path:
            self.model_id = self.local_model_path
            logger.info(f"Using local model path: {self.model_id}")
        else:
            self.model_id = model_id
            logger.warning(f"Will attempt to use HF model ID: {self.model_id}")
        
        self.llm = None
        self.metrics = {}

    def load_model(self) -> bool:
        logger.info(f"Loading model from {self.model_id}")
        
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
    
    def unload_model(self) -> None:
        if self.llm and self.llm.model:
            del self.llm.model
            del self.llm.tokenizer
            del self.llm.pipeline
            self.llm.model = None
            self.llm.tokenizer = None
            self.llm.pipeline = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"Unloaded model {self.original_model_id} and cleared memory")
    
    def measure_latency(self, prompts: list, num_runs: int = 3) -> float:
        """배치 처리를 사용한 지연 시간 측정"""
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return 0
        
        total_time = 0
        successful_runs = 0
        
        for run in tqdm(range(num_runs), desc="Measuring latency (batch)"):
            start_time = time.time()
            try:
                # 배치 처리 사용
                responses = self.llm.generate_batch(prompts, max_new_tokens=100, temperature=0.7, batch_size=2)
                
                # 성공한 응답 개수 카운트
                for response in responses:
                    if response and len(response.strip()) > 0:
                        successful_runs += 1
                        
            except Exception as e:
                logger.warning(f"Batch generation failed, trying individual: {e}")
                # 폴백: 개별 처리
                for prompt in prompts:
                    try:
                        response = self.llm.generate(prompt, max_new_tokens=100, temperature=0.7)
                        if response and len(response.strip()) > 0:
                            successful_runs += 1
                    except Exception as e2:
                        logger.warning(f"Individual generation failed: {e2}")
                        continue
            
            end_time = time.time()
            total_time += (end_time - start_time)
        
        if successful_runs == 0:
            logger.error("No successful generations")
            return 0
            
        avg_latency = total_time / successful_runs
        logger.info(f"Average latency (batch): {avg_latency:.4f} seconds ({successful_runs} successful runs)")
        
        self.metrics["latency"] = avg_latency
        self.metrics["successful_generations"] = successful_runs
        return avg_latency

    def measure_throughput(self, prompts: list, num_runs: int = 3) -> float:
        """배치 처리를 사용한 처리량 측정"""
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return 0
        
        total_tokens = 0
        total_time = 0
        successful_runs = 0
        
        for run in tqdm(range(num_runs), desc="Measuring throughput (batch)"):
            start_time = time.time()
            try:
                # 배치 처리 사용
                responses = self.llm.generate_batch(prompts, max_new_tokens=100, temperature=0.7, batch_size=2)
                
                for response in responses:
                    if response and len(response.strip()) > 0:
                        successful_runs += 1
                        token_count = len(response.split())
                        total_tokens += token_count
                        
            except Exception as e:
                logger.warning(f"Batch generation failed, trying individual: {e}")
                # 폴백: 개별 처리
                for prompt in prompts:
                    try:
                        response = self.llm.generate(prompt, max_new_tokens=100, temperature=0.7)
                        if response and len(response.strip()) > 0:
                            successful_runs += 1
                            token_count = len(response.split())
                            total_tokens += token_count
                    except Exception as e2:
                        logger.warning(f"Individual generation failed: {e2}")
                        continue
            
            end_time = time.time()
            total_time += (end_time - start_time)
        
        if total_time == 0 or successful_runs == 0:
            logger.error("No successful generations for throughput measurement")
            return 0
            
        avg_throughput = total_tokens / total_time
        logger.info(f"Average throughput (batch): {avg_throughput:.2f} tokens/second ({successful_runs} successful runs)")
        
        self.metrics["throughput"] = avg_throughput
        return avg_throughput

    def measure_accuracy(self, test_cases: list, batch_size: int = 3) -> float:
        """배치 처리를 사용한 정확도 측정 (선택적)"""
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return 0
        
        correct = 0
        total_attempted = 0
        failed_generations = 0
        
        # 배치 처리 시도
        try:
            # 배치 단위로 처리
            for i in tqdm(range(0, len(test_cases), batch_size), desc="Measuring accuracy (batch)"):
                batch = test_cases[i:i+batch_size]
                prompts = [case["prompt"] for case in batch]
                
                try:
                    # 배치 생성
                    responses = self.llm.generate_batch(prompts, max_new_tokens=200, temperature=0.7, batch_size=batch_size)
                    
                    # 각 응답 평가
                    for j, (case, response) in enumerate(zip(batch, responses)):
                        if not response or len(response.strip()) == 0:
                            failed_generations += 1
                            logger.warning(f"Empty response for test case {i+j+1}")
                            continue
                            
                        total_attempted += 1
                        expected_keywords = case["expected_keywords"]
                        
                        response_lower = response.lower()
                        matches = sum(1 for keyword in expected_keywords if keyword in response_lower)
                        score = matches / len(expected_keywords) if expected_keywords else 0
                        
                        if score >= 0.25:
                            correct += 1
                        
                        logger.debug(f"Test case {i+j+1}: Score: {score:.2f}, Keywords: {matches}/{len(expected_keywords)}")
                        
                except Exception as e:
                    logger.warning(f"Batch processing failed for batch {i//batch_size + 1}: {e}")
                    # 폴백: 개별 처리
                    for j, case in enumerate(batch):
                        try:
                            response = self.llm.generate(case["prompt"], max_new_tokens=200, temperature=0.7)
                            
                            if not response or len(response.strip()) == 0:
                                failed_generations += 1
                                logger.warning(f"Empty response for test case {i+j+1}")
                                continue
                                
                            total_attempted += 1
                            expected_keywords = case["expected_keywords"]
                            
                            response_lower = response.lower()
                            matches = sum(1 for keyword in expected_keywords if keyword in response_lower)
                            score = matches / len(expected_keywords) if expected_keywords else 0
                            
                            if score >= 0.25:
                                correct += 1
                            
                            logger.debug(f"Test case {i+j+1}: Score: {score:.2f}, Keywords: {matches}/{len(expected_keywords)}")
                            
                        except Exception as e2:
                            failed_generations += 1
                            logger.error(f"Error during individual generation for test case {i+j+1}: {e2}")
                            continue
        
        except Exception as e:
            logger.error(f"Batch accuracy measurement failed completely: {e}")
            # 완전 폴백: 기존 개별 처리 방식
            return self._measure_accuracy_individual(test_cases)
        
        if total_attempted == 0:
            logger.error("No successful test case evaluations")
            accuracy = 0
        else:
            accuracy = correct / total_attempted
        
        logger.info(f"Accuracy (batch): {accuracy:.4f} ({correct}/{total_attempted} correct, {failed_generations} failed)")
        
        self.metrics["accuracy"] = accuracy
        self.metrics["total_test_cases"] = len(test_cases)
        self.metrics["successful_evaluations"] = total_attempted
        self.metrics["failed_generations"] = failed_generations
        
        return accuracy

    def _measure_accuracy_individual(self, test_cases: list) -> float:
        """개별 처리 방식 정확도 측정 (폴백용)"""
        correct = 0
        total_attempted = 0
        failed_generations = 0
        
        for i, case in enumerate(tqdm(test_cases, desc="Measuring accuracy (individual)")):
            prompt = case["prompt"]
            expected_keywords = case["expected_keywords"]
            
            try:
                generated_text = self.llm.generate(prompt, max_new_tokens=200, temperature=0.7)
                
                if not generated_text or len(generated_text.strip()) == 0:
                    failed_generations += 1
                    logger.warning(f"Empty response for test case {i+1}")
                    continue
                    
                total_attempted += 1
                
                generated_lower = generated_text.lower()
                matches = sum(1 for keyword in expected_keywords if keyword in generated_lower)
                score = matches / len(expected_keywords) if expected_keywords else 0
                
                if score >= 0.25:
                    correct += 1
                
                logger.debug(f"Test case {i+1}: Score: {score:.2f}, Keywords: {matches}/{len(expected_keywords)}")
                
            except Exception as e:
                failed_generations += 1
                logger.error(f"Error during generation for test case {i+1}: {e}")
                continue
        
        if total_attempted == 0:
            logger.error("No successful test case evaluations")
            accuracy = 0
        else:
            accuracy = correct / total_attempted
        
        logger.info(f"Accuracy (individual): {accuracy:.4f} ({correct}/{total_attempted} correct, {failed_generations} failed)")
        
        self.metrics["accuracy"] = accuracy
        self.metrics["total_test_cases"] = len(test_cases)
        self.metrics["successful_evaluations"] = total_attempted
        self.metrics["failed_generations"] = failed_generations
        
        return accuracy
    
    def measure_memory(self) -> dict:
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return {}
        
        memory_stats = {}
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_stats["ram_usage_mb"] = memory_info.rss / (1024 * 1024)
        
        if torch.cuda.is_available():
            try:
                current_device = torch.cuda.current_device()
                memory_stats["gpu_allocated_mb"] = torch.cuda.memory_allocated(current_device) / (1024 * 1024)
                memory_stats["gpu_reserved_mb"] = torch.cuda.memory_reserved(current_device) / (1024 * 1024)
                memory_stats["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated(current_device) / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Could not get GPU memory stats: {e}")
        
        logger.info(f"Memory usage: {memory_stats}")
        self.metrics.update(memory_stats)
        
        return memory_stats
    
    def run_all_metrics(self, test_cases: list) -> dict:
        short_prompts = [
            "오늘 기분이 좋아. 어떤 술이 좋을까?",
            "스트레스 받아서 술 마시고 싶어.",
            "친구들과 파티할 거야."
        ]
        
        try:
            self.measure_latency(short_prompts)
            self.measure_throughput(short_prompts)
            self.measure_memory()
            self.measure_accuracy(test_cases)
        except Exception as e:
            logger.error(f"Error during metrics measurement: {e}")
        
        model_type, base_model = detect_model_type(self.original_model_id)
        
        self.metrics["model_id"] = self.original_model_id
        self.metrics["model_type"] = model_type
        self.metrics["base_model"] = base_model
        
        return self.metrics

def prepare_test_cases() -> list:
    """Load test cases from test_cases.json file"""
    try:
        from smart_pub.config import TEST_CASES_FILE
        
        if not TEST_CASES_FILE.exists():
            logger.error(f"Test cases file not found: {TEST_CASES_FILE}")
            return []
        
        with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = data.get("test_cases", [])
        logger.info(f"Loaded {len(test_cases)} test cases from {TEST_CASES_FILE}")
        
        return test_cases
        
    except Exception as e:
        logger.error(f"Error loading test cases: {e}")
        return []

def list_available_models(model_dir: str) -> list:
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

def get_target_models() -> list:
    """Get the target models for evaluation from available local models"""
    target_model_names = [
        "HyperCLOVAX-SEED-Text-Instruct-1.5B",
        "HyperCLOVAX-SEED-Text-Instruct-0.5B",
        "ko-gpt-trinity-1.2B-v0.5",
        "polyglot-ko-1.3b",
        "Llama-3-Open-Ko-8B"
    ]
    
    available_models = list_available_models(MODEL_DIR)
    logger.info(f"Available local models: {available_models}")
    
    target_models = []
    for target in target_model_names:
        if target in available_models:
            target_models.append(target)
            logger.info(f"✓ Found target model: {target}")
        
        for optimization in ["int8", "int4"]:
            optimized_name = f"{target}-{optimization}"
            if optimized_name in available_models:
                target_models.append(optimized_name)
                logger.info(f"✓ Found optimized model: {optimized_name}")
        
        if target not in available_models:
            logger.warning(f"✗ Target model not found: {target}")
    
    return target_models

def detect_model_type(model_name: str) -> tuple[str, str]:
    """Detect if model is original or optimized"""
    if model_name.endswith("-int8"):
        return "int8", model_name[:-5]
    elif model_name.endswith("-int4"):
        return "int4", model_name[:-5]
    else:
        return "original", model_name

def evaluate_all_models() -> tuple[list, str | None, str | None]:
    """Evaluate all target models"""
    
    target_models = get_target_models()
    
    if not target_models:
        logger.error("No target models found locally")
        return [], None, None
    
    logger.info(f"Will evaluate {len(target_models)} models: {target_models}")
    
    test_cases = prepare_test_cases()
    if not test_cases:
        logger.error("No test cases loaded")
        return [], None, None
    
    results = []
    
    for i, model_name in enumerate(target_models, 1):
        logger.info(f"[{i}/{len(target_models)}] Evaluating model: {model_name}")
        
        evaluator = LLMEvaluator(model_name, MODEL_DIR)
        
        if evaluator.load_model():
            try:
                logger.info(f"Running metrics for {model_name}...")
                metrics = evaluator.run_all_metrics(test_cases)
                results.append(metrics)
                
                print(f"\n{'='*60}")
                print(f"Results for {model_name}:")
                print(f"{'='*60}")
                for k, v in metrics.items():
                    if k not in ["model_id", "model_type", "base_model"] and isinstance(v, (int, float)):
                        print(f"  {k:<25}: {v:.4f}")
                    elif k not in ["model_id", "model_type", "base_model"]:
                        print(f"  {k:<25}: {v}")
                print(f"{'='*60}")
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            finally:
                evaluator.unload_model()
                logger.info(f"Model {model_name} unloaded, memory cleared")
                
        else:
            logger.error(f"Failed to load model {model_name}")
    
    if not results:
        logger.error("No models were successfully evaluated")
        return [], None, None
    
    results_file = save_all_results(results)
    best_model = find_best_model(results)
    
    return results, best_model, results_file

def save_all_results(results, output_file=None):
    """Save all evaluation results to a comprehensive file"""
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"model_evaluation_results_{timestamp}.json"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    grouped_results = {}
    for result in results:
        base_model = result.get("base_model", result.get("model_id", "Unknown"))
        model_type = result.get("model_type", "unknown")
        
        if base_model not in grouped_results:
            grouped_results[base_model] = {}
        
        grouped_results[base_model][model_type] = result
    
    comprehensive_results = {
        "evaluation_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models_evaluated": len(results),
            "optimization_types_tested": ["original", "int8", "int4"],
            "evaluation_metrics": ["latency", "throughput", "ram_usage_mb", "gpu_allocated_mb", "accuracy"]
        },
        "models": results,
        "grouped_by_base_model": grouped_results,
        "summary": {}
    }
    
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
                
                if metric in ["throughput", "accuracy"]:
                    best_idx = values.index(max(values))
                else:
                    best_idx = values.index(min(values))
                
                metrics_summary[metric]["best_model"] = results[best_idx]["model_id"]
        
        optimization_comparison = {}
        for opt_type in ["original", "int8", "int4"]:
            opt_results = [r for r in results if r.get("model_type") == opt_type]
            if opt_results:
                optimization_comparison[opt_type] = {
                    "count": len(opt_results),
                    "avg_latency": sum(r.get("latency", 0) for r in opt_results) / len(opt_results),
                    "avg_accuracy": sum(r.get("accuracy", 0) for r in opt_results) / len(opt_results),
                    "avg_memory": sum(r.get("ram_usage_mb", 0) for r in opt_results) / len(opt_results)
                }
        
        comprehensive_results["summary"] = {
            "overall_metrics": metrics_summary,
            "optimization_comparison": optimization_comparison
        }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comprehensive results saved to: {output_file}")
        
        csv_file = output_file.with_suffix('.csv')
        if results:
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Results also saved as CSV: {csv_file}")
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return None

def find_best_model(results: list) -> str | None:
    """Find the best model based on weighted scoring"""
    if not results:
        return None
    
    scored_models = []
    
    for result in results:
        model_id = result.get("model_id", "Unknown")
        score = 0
        
        weights = {
            "accuracy": 0.4,
            "latency": 0.2,
            "throughput": 0.2,
            "ram_usage_mb": 0.1,
            "gpu_allocated_mb": 0.1
        }
        
        for metric, weight in weights.items():
            if metric in result and isinstance(result[metric], (int, float)):
                value = result[metric]
                if metric in ["accuracy", "throughput"]:
                    score += value * weight
                else:
                    if value > 0:
                        score += (1 / value) * weight * 1000
        
        scored_models.append((model_id, score))
        logger.info(f"Model {model_id} total score: {score:.4f}")
    
    scored_models.sort(key=lambda x: x[1], reverse=True)
    best_model = scored_models[0][0] if scored_models else None
    
    logger.info(f"Best model selected: {best_model}")
    return best_model

def update_config_with_best_model(best_model: str | None) -> bool:
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM models for Smart Pub")
    parser.add_argument("--model-id", help="Specific model ID to evaluate")
    parser.add_argument("--list-models", action="store_true", help="List available local models")
    parser.add_argument("--output-file", help="Output file path for results")
    parser.add_argument("--update-config", action="store_true", help="Update config with best model")
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
        logger.info(f"Evaluating single model: {args.model_id}")
        
        evaluator = LLMEvaluator(args.model_id, MODEL_DIR)
        
        if evaluator.load_model():
            try:
                test_cases = prepare_test_cases()
                if not test_cases:
                    logger.error("No test cases loaded")
                    return
                    
                metrics = evaluator.run_all_metrics(test_cases)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = args.output_file or (RESULTS_DIR / f"single_model_evaluation_{args.model_id}_{timestamp}.json")
                save_all_results([metrics], Path(output_file))
                
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
        model_type = model_metrics.get("model_type", "unknown")
        base_model = model_metrics.get("base_model", "Unknown")
        
        type_indicator = ""
        if model_type == "original":
            type_indicator = "🔵 Original"
        elif model_type == "int8":
            type_indicator = "🟡 INT8"
        elif model_type == "int4":
            type_indicator = "🟠 INT4"
        
        print(f"\n[{i}] {type_indicator} Model: {model_id}")
        print("-" * 70)
        for k, v in model_metrics.items():
            if k not in ["model_id", "model_type", "base_model"] and isinstance(v, (int, float)):
                print(f"  {k:<25}: {v:.4f}")
        
        # Add performance indicators
        accuracy = model_metrics.get("accuracy", 0)
        latency = model_metrics.get("latency", float('inf'))
        successful_evals = model_metrics.get("successful_evaluations", 0)
        total_cases = model_metrics.get("total_test_cases", 0)
        
        if accuracy >= 0.6:
            print("  🟢 High Accuracy")
        elif accuracy >= 0.3:
            print("  🟡 Medium Accuracy") 
        else:
            print("  🔴 Low Accuracy")
            
        if latency <= 5.0:
            print("  🟢 Fast Response")
        elif latency <= 15.0:
            print("  🟡 Medium Response")
        else:
            print("  🔴 Slow Response")
            
        if successful_evals == total_cases:
            print("  🟢 All Tests Passed")
        elif successful_evals >= total_cases * 0.8:
            print("  🟡 Most Tests Passed")
        else:
            print("  🔴 Many Tests Failed")
    
    # Optimization comparison summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPARISON")
    print("="*80)
    
    optimization_stats = {}
    for opt_type in ["original", "int8", "int4"]:
        opt_results = [r for r in results if r.get("model_type") == opt_type]
        if opt_results:
            avg_latency = sum(r.get("latency", 0) for r in opt_results) / len(opt_results)
            avg_accuracy = sum(r.get("accuracy", 0) for r in opt_results) / len(opt_results)
            avg_memory = sum(r.get("ram_usage_mb", 0) for r in opt_results) / len(opt_results)
            avg_success_rate = sum(r.get("successful_evaluations", 0) / r.get("total_test_cases", 1) for r in opt_results) / len(opt_results)
            
            optimization_stats[opt_type] = {
                "count": len(opt_results),
                "avg_latency": avg_latency,
                "avg_accuracy": avg_accuracy,
                "avg_memory": avg_memory,
                "avg_success_rate": avg_success_rate
            }
            
            type_name = {"original": "Original Models", "int8": "INT8 Optimized", "int4": "INT4 Optimized"}[opt_type]
            print(f"\n{type_name} ({len(opt_results)} models):")
            print(f"  Average Latency    : {avg_latency:.4f}s")
            print(f"  Average Accuracy   : {avg_accuracy:.4f}")
            print(f"  Average Memory     : {avg_memory:.2f} MB")
            print(f"  Average Success Rate: {avg_success_rate:.4f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    if results:
        accuracies = [r.get("accuracy", 0) for r in results]
        latencies = [r.get("latency", 0) for r in results if r.get("latency", 0) > 0]
        throughputs = [r.get("throughput", 0) for r in results if r.get("throughput", 0) > 0]
        success_rates = [r.get("successful_evaluations", 0) / r.get("total_test_cases", 1) for r in results]
        
        if accuracies:
            print(f"Accuracy     - Best: {max(accuracies):.4f}, Worst: {min(accuracies):.4f}, Avg: {sum(accuracies)/len(accuracies):.4f}")
        if latencies:
            print(f"Latency      - Best: {min(latencies):.4f}s, Worst: {max(latencies):.4f}s, Avg: {sum(latencies)/len(latencies):.4f}s")
        if throughputs:
            print(f"Throughput   - Best: {max(throughputs):.2f} tok/s, Worst: {min(throughputs):.2f} tok/s, Avg: {sum(throughputs)/len(throughputs):.2f} tok/s")
        if success_rates:
            print(f"Success Rate - Best: {max(success_rates):.4f}, Worst: {min(success_rates):.4f}, Avg: {sum(success_rates)/len(success_rates):.4f}")
    
    print(f"\n🏆 BEST OVERALL MODEL: {best_model}")
    
    # Update config if requested
    if args.update_config and best_model:
        if update_config_with_best_model(best_model):
            print(f"✅ Config updated with best model: {best_model}")
        else:
            print(f"❌ Failed to update config")
    
    print("="*80)
    
    if results_file:
        print(f"\n📁 Detailed results saved to:")
        print(f"   JSON: {results_file}")
        print(f"   CSV:  {Path(results_file).with_suffix('.csv')}")
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📊 {len(results)} models evaluated")
    
    # Print optimization effectiveness
    print(f"\n📈 OPTIMIZATION EFFECTIVENESS:")
    base_models = list(set([r.get("base_model") for r in results]))
    for base_model in base_models:
        base_results = [r for r in results if r.get("base_model") == base_model]
        original = next((r for r in base_results if r.get("model_type") == "original"), None)
        
        if original and len(base_results) > 1:
            print(f"\n{base_model}:")
            optimized = [r for r in base_results if r.get("model_type") != "original"]
            
            for opt in optimized:
                opt_type = opt.get("model_type")
                
                # Speed improvement
                if opt.get("latency", 0) > 0 and original.get("latency", 0) > 0:
                    speedup = original.get("latency") / opt.get("latency")
                    if speedup > 1.1:
                        print(f"  ✅ {opt_type.upper()}: {speedup:.2f}x faster")
                    elif speedup > 0.9:
                        print(f"  ➖ {opt_type.upper()}: {speedup:.2f}x speed (similar)")
                    else:
                        print(f"  ❌ {opt_type.upper()}: {speedup:.2f}x speed (slower)")
                
                # Memory efficiency
                orig_mem = original.get("ram_usage_mb", 0)
                opt_mem = opt.get("ram_usage_mb", 0)
                if orig_mem > 0:
                    mem_change = (opt_mem - orig_mem) / orig_mem * 100
                    if mem_change < -10:
                        print(f"  ✅ {opt_type.upper()}: {abs(mem_change):.1f}% less memory")
                    elif mem_change < 10:
                        print(f"  ➖ {opt_type.upper()}: {abs(mem_change):.1f}% memory change (similar)")
                    else:
                        print(f"  ❌ {opt_type.upper()}: +{mem_change:.1f}% more memory")
                
                # Accuracy preservation
                acc_change = opt.get("accuracy", 0) - original.get("accuracy", 0)
                if acc_change >= -0.05:
                    print(f"  ✅ {opt_type.upper()}: Accuracy preserved ({acc_change:+.4f})")
                elif acc_change >= -0.15:
                    print(f"  ⚠️ {opt_type.upper()}: Minor accuracy loss ({acc_change:+.4f})")
                else:
                    print(f"  ❌ {opt_type.upper()}: Significant accuracy loss ({acc_change:+.4f})")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()