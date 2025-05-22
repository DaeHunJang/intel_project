#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.model.llm_wrapper import LLMWrapper
from smart_pub.utils.helpers import get_logger
from smart_pub.config import MODEL_DIR, RESULTS_DIR, MODEL_ID, TEST_CASES_FILE

logger = get_logger("accuracy_tester")

class AccuracyTester:
    def __init__(self, model_id=MODEL_ID, model_dir=MODEL_DIR):
        self.model_id = model_id
        self.model_dir = model_dir
        self.llm = None
        
    def load_model(self):
        """Load the LLM model (local only)"""
        logger.info(f"Loading model: {self.model_id}")
        
        # Handle local model paths
        if "/" in self.model_id:
            model_folder = self.model_id.split("/")[-1]
        else:
            model_folder = self.model_id

        model_path = Path(self.model_dir) / model_folder
        if not model_path.exists():
            logger.error(f"Local model not found: {model_path}")
            return False

        # Check if model has required files
        if not (model_path / "config.json").exists():
            logger.error(f"Model config.json not found in: {model_path}")
            return False

        try:
            # Use local path for LLMWrapper
            self.llm = LLMWrapper(str(model_path), self.model_dir)
            result = self.llm.load_model()

            if result:
                logger.info(f"Successfully loaded model: {self.model_id}")
            else:
                logger.error(f"Failed to load model: {self.model_id}")

            return result

        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            return False
    
    def load_test_cases(self, test_file=None):
        """Load test cases from JSON file"""
        if test_file is None:
            test_file_path = TEST_CASES_FILE
        else:
            test_file_path = Path(test_file)
        
        if not test_file_path.exists():
            logger.error(f"Test cases file not found: {test_file_path}")
            return []
        
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            test_cases = data.get("test_cases", [])
            logger.info(f"Loaded {len(test_cases)} test cases from {test_file_path}")
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            return []
    
    def test_single_case(self, test_case):
        """Test a single case and return result"""
        prompt = test_case["prompt"]
        expected_keywords = test_case["expected_keywords"]
        case_id = test_case.get("id", "unknown")
        category = test_case.get("category", "unknown")
        
        try:
            # Generate response
            generated_text = self.llm.generate(prompt)
            
            if not generated_text:
                return {
                    "id": case_id,
                    "category": category,
                    "prompt": prompt,
                    "expected": expected_keywords,
                    "generated": "",
                    "matches": 0,
                    "score": 0.0,
                    "passed": False,
                    "error": "Empty response"
                }
            
            # Check keyword matches
            matches = sum(1 for keyword in expected_keywords 
                         if keyword.lower() in generated_text.lower())
            
            score = matches / len(expected_keywords) if expected_keywords else 0
            passed = score >= 0.5  # 50% threshold for passing
            
            return {
                "id": case_id,
                "category": category,
                "prompt": prompt,
                "expected": expected_keywords,
                "generated": generated_text.strip(),
                "matches": matches,
                "score": score,
                "passed": passed,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error testing case {case_id}: {e}")
            return {
                "id": case_id,
                "category": category,
                "prompt": prompt,
                "expected": expected_keywords,
                "generated": "",
                "matches": 0,
                "score": 0.0,
                "passed": False,
                "error": str(e)
            }
    
    def run_accuracy_test(self, test_cases, verbose=False):
        """Run accuracy test on all test cases"""
        if not self.llm:
            logger.error("Model not loaded. Call load_model() first.")
            return None
        
        results = []
        correct_count = 0
        total_count = len(test_cases)
        
        logger.info(f"Starting accuracy test with {total_count} test cases...")
        
        # Test each case
        for i, test_case in enumerate(tqdm(test_cases, desc="Testing cases"), 1):
            result = self.test_single_case(test_case)
            results.append(result)
            
            if result["passed"]:
                correct_count += 1
            
            # Print progress for verbose mode
            if verbose:
                status = "PASS" if result["passed"] else "FAIL"
                print(f"[{i}/{total_count}] {status} - Case {result['id']}: {result['score']:.2f}")
                if result["error"]:
                    print(f"  ERROR: {result['error']}")
        
        # Calculate overall accuracy
        overall_accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Detect model type
        model_type, base_model = self.detect_model_type(self.model_id)
        
        # Create summary
        summary = {
            "model_id": self.model_id,
            "model_type": model_type,
            "base_model": base_model,
            "total_cases": total_count,
            "passed_cases": correct_count,
            "failed_cases": total_count - correct_count,
            "overall_accuracy": overall_accuracy,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Category-wise analysis
        categories = {}
        for result in results:
            category = result["category"]
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0}
            
            categories[category]["total"] += 1
            if result["passed"]:
                categories[category]["passed"] += 1
        
        # Calculate category accuracies
        category_accuracies = {}
        for category, stats in categories.items():
            accuracy = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            category_accuracies[category] = {
                "total": stats["total"],
                "passed": stats["passed"],
                "accuracy": accuracy
            }
        
        summary["category_breakdown"] = category_accuracies
        
        return {
            "summary": summary,
            "detailed_results": results
        }
    
    def detect_model_type(self, model_name):
        """Detect if model is original or optimized"""
        if model_name.endswith("-int8"):
            return "int8", model_name[:-5]
        elif model_name.endswith("-int4"):
            return "int4", model_name[:-5]
        else:
            return "original", model_name
    
    def save_results(self, test_results, output_file=None):
        """Save test results to file"""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            # Clean model name for filename
            model_name = self.model_id.replace("/", "_").replace("\\", "_")
            output_file = RESULTS_DIR / f"accuracy_test_{model_name}_{timestamp}.json"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Test results saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def print_summary(self, test_results):
        """Print test summary"""
        summary = test_results["summary"]
        
        print(f"\n{'='*60}")
        print(f"ACCURACY TEST RESULTS")
        print(f"{'='*60}")
        print(f"Model: {summary['model_id']}")
        print(f"Type: {summary['model_type']}")
        if summary['model_type'] != 'original':
            print(f"Base Model: {summary['base_model']}")
        print(f"Test Date: {summary['test_timestamp']}")
        print(f"Total Cases: {summary['total_cases']}")
        print(f"Passed: {summary['passed_cases']}")
        print(f"Failed: {summary['failed_cases']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.4f} ({summary['overall_accuracy']*100:.2f}%)")
        
        print(f"\n{'='*60}")
        print(f"CATEGORY BREAKDOWN")
        print(f"{'='*60}")
        
        for category, stats in summary["category_breakdown"].items():
            print(f"{category:15s}: {stats['passed']:2d}/{stats['total']:2d} ({stats['accuracy']*100:5.1f}%)")
        
        print(f"{'='*60}")

def list_available_models(model_dir):
    """List all available local models"""
    model_dir_path = Path(model_dir)
    available_models = []
    
    if not model_dir_path.exists():
        logger.warning(f"Model directory does not exist: {model_dir}")
        return available_models
    
    for item in model_dir_path.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            available_models.append(item.name)
    
    return available_models

def get_target_models(model_dir):
    """Get target models including optimized versions"""
    target_model_names = [
        "HyperCLOVAX-SEED-Text-Instruct-1.5B",
        "HyperCLOVAX-SEED-Text-Instruct-0.5B", 
        "ko-gpt-trinity-1.2B-v0.5",
        "polyglot-ko-1.3b",
        "Llama-3-Open-Ko-8B"
    ]
    
    available_models = list_available_models(model_dir)
    logger.info(f"Available local models: {available_models}")
    
    # Match target models with available models (including optimized versions)
    target_models = []
    for target in target_model_names:
        # Original model
        if target in available_models:
            target_models.append(target)
            logger.info(f"Found target model: {target}")
        
        # Optimized versions
        for optimization in ["int8", "int4"]:
            optimized_name = f"{target}-{optimization}"
            if optimized_name in available_models:
                target_models.append(optimized_name)
                logger.info(f"Found optimized model: {optimized_name}")
        
        if target not in available_models:
            logger.warning(f"Target model not found: {target}")
    
    return target_models

def test_all_models(test_cases, verbose=False):
    """Test all available local models"""
    target_models = get_target_models(MODEL_DIR)
    
    if not target_models:
        logger.error("No target models found locally")
        return []
    
    logger.info(f"Will test {len(target_models)} models: {target_models}")
    
    all_results = []
    
    for i, model_name in enumerate(target_models, 1):
        logger.info(f"[{i}/{len(target_models)}] Testing model: {model_name}")
        
        tester = AccuracyTester(model_name, MODEL_DIR)
        
        if tester.load_model():
            try:
                test_results = tester.run_accuracy_test(test_cases, verbose)
                
                if test_results:
                    all_results.append(test_results)
                    
                    # Save individual results
                    tester.save_results(test_results)
                    
                    # Print summary
                    tester.print_summary(test_results)
                    
                else:
                    logger.error(f"Failed to get test results for {model_name}")
                    
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error(f"Failed to load model {model_name}")
    
    return all_results

def save_comparison_results(all_results):
    """Save comparison results across all models"""
    if not all_results:
        return None
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"accuracy_comparison_all_models_{timestamp}.json"
    
    # Prepare comparison data
    comparison_data = {
        "comparison_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models_tested": len(all_results),
            "test_description": "Accuracy comparison across all local models"
        },
        "models": [result["summary"] for result in all_results],
        "detailed_results": all_results
    }
    
    # Calculate rankings
    models_with_accuracy = [(result["summary"]["model_id"], result["summary"]["overall_accuracy"]) 
                           for result in all_results]
    models_with_accuracy.sort(key=lambda x: x[1], reverse=True)
    
    comparison_data["rankings"] = {
        "by_accuracy": models_with_accuracy
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to: {output_file}")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error saving comparison results: {e}")
        return None

def print_comparison_summary(all_results):
    """Print comparison summary"""
    if not all_results:
        return
    
    print(f"\n{'='*80}")
    print(f"ALL MODELS ACCURACY COMPARISON")
    print(f"{'='*80}")
    
    # Sort by accuracy
    sorted_results = sorted(all_results, key=lambda x: x["summary"]["overall_accuracy"], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<40} {'Type':<8} {'Accuracy':<10} {'Pass Rate'}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results, 1):
        summary = result["summary"]
        model_name = summary["model_id"]
        model_type = summary["model_type"]
        accuracy = summary["overall_accuracy"]
        pass_rate = f"{summary['passed_cases']}/{summary['total_cases']}"
        
        print(f"{i:<4} {model_name:<40} {model_type:<8} {accuracy:<10.4f} {pass_rate}")
    
    print("=" * 80)
    
    # Best model by type
    print(f"\nBest Models by Type:")
    types = {}
    for result in all_results:
        model_type = result["summary"]["model_type"]
        accuracy = result["summary"]["overall_accuracy"]
        model_id = result["summary"]["model_id"]
        
        if model_type not in types or accuracy > types[model_type][1]:
            types[model_type] = (model_id, accuracy)
    
    for model_type, (model_id, accuracy) in types.items():
        print(f"  {model_type:<10}: {model_id} ({accuracy:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Test model accuracy on emotion analysis")
    parser.add_argument("--model-id", help="Specific model ID to test")
    parser.add_argument("--all", action="store_true", help="Test all available local models")
    parser.add_argument("--list-models", action="store_true", help="List available local models")
    parser.add_argument("--test-file", help="Test cases JSON file path")
    parser.add_argument("--output-file", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--show-details", action="store_true", help="Show detailed results")
    args = parser.parse_args()
    
    if args.list_models:
        available_models = list_available_models(MODEL_DIR)
        target_models = get_target_models(MODEL_DIR)
        
        print("Available local models:")
        for model in available_models:
            status = "TARGET" if model in target_models else ""
            print(f"  {status} {model}")
        
        print(f"\nTarget models for testing: {len(target_models)}")
        for model in target_models:
            print(f"  {model}")
        return
    
    # Load test cases
    test_cases = []
    if TEST_CASES_FILE.exists():
        try:
            with open(TEST_CASES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            test_cases = data.get("test_cases", [])
            logger.info(f"Loaded {len(test_cases)} test cases")
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            sys.exit(1)
    else:
        logger.error(f"Test cases file not found: {TEST_CASES_FILE}")
        sys.exit(1)
    
    if args.all:
        # Test all models
        print(f"Testing all available local models with {len(test_cases)} test cases...")
        all_results = test_all_models(test_cases, args.verbose)
        
        if all_results:
            # Save comparison results
            comparison_file = save_comparison_results(all_results)
            
            # Print comparison summary
            print_comparison_summary(all_results)
            
            if comparison_file:
                print(f"\nComparison results saved to: {comparison_file}")
        else:
            logger.error("No models were successfully tested")
            sys.exit(1)
    
    else:
        # Test single model
        model_id = args.model_id or MODEL_ID
        
        # Initialize tester
        tester = AccuracyTester(model_id)
        
        # Load model
        if not tester.load_model():
            logger.error("Failed to load model")
            sys.exit(1)
        
        # Run test
        print(f"Running accuracy test with {len(test_cases)} cases...")
        test_results = tester.run_accuracy_test(test_cases, args.verbose)
        
        if test_results is None:
            logger.error("Failed to run accuracy test")
            sys.exit(1)
        
        # Print summary
        tester.print_summary(test_results)
        
        # Show detailed results if requested
        if args.show_details:
            print(f"\n{'='*60}")
            print(f"DETAILED RESULTS")
            print(f"{'='*60}")
            
            for result in test_results["detailed_results"]:
                status = "PASS" if result["passed"] else "FAIL"
                print(f"\n[{result['id']}] {status} - {result['category']}")
                print(f"Prompt: {result['prompt']}")
                print(f"Expected: {result['expected']}")
                print(f"Score: {result['score']:.2f} ({result['matches']}/{len(result['expected'])} matches)")
                if result["error"]:
                    print(f"Error: {result['error']}")
                elif result["generated"]:
                    print(f"Generated: {result['generated'][:100]}...")
        
        # Save results
        output_file = tester.save_results(test_results, args.output_file)
        if output_file:
            print(f"\nResults saved to: {output_file}")
        
        # Exit with appropriate code
        accuracy = test_results["summary"]["overall_accuracy"]
        if accuracy >= 0.7:
            print(f"\nSUCCESS: Model achieved {accuracy*100:.1f}% accuracy (>= 70%)")
            sys.exit(0)
        else:
            print(f"\nWARNING: Model achieved {accuracy*100:.1f}% accuracy (< 70%)")
            sys.exit(1)

if __name__ == "__main__":
    main()