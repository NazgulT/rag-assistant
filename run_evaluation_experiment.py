#!/usr/bin/env python3
"""
Run RAGAS evaluation experiment for RAG system.

This script evaluates the RAG system on the hf_doc_qa_eval.csv dataset,
measuring relevancy, faithfulness, precision, and recall.

Results are saved to evaluation_results/ with detailed metrics and MLFlow trace URLs.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

# Disable tokenizers parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from evaluation_experiment import run_evaluation_experiment


def main():
    """Run the evaluation experiment."""
    print("="*70)
    print("RAG System - RAGAS Evaluation Experiment")
    print("="*70)
    print()
    
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation experiment"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="app/evaluation/datasets/hf_doc_qa_eval.csv",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Execute samples concurrently",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent workers when async",
    )
    args = parser.parse_args()

    dataset_path = args.csv
    output_dir = args.output

    # Check dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"📊 Dataset: {dataset_path}")
    print(f"📁 Output Directory: {output_dir}")
    print()

    print("⏳ Starting evaluation... This may take a few minutes.")
    print()

    try:
        # Run evaluation
        results_file = run_evaluation_experiment(
            dataset_path,
            output_dir,
            async_mode=args.async_mode,
            max_concurrency=args.concurrency,
        )
        
        print()
        print("="*70)
        print("✅ Evaluation Complete!")
        print("="*70)
        print()
        print(f"📊 Results saved to: {results_file}")
        print()
        print("📈 To view results:")
        print(f"   1. Open CSV: {results_file}")
        print()
        print("🔍 To view detailed traces:")
        print("   1. Start MLFlow UI: mlflow ui")
        print("   2. Open browser: http://127.0.0.1:5000")
        print("   3. Click on experiment: RAG_RAGAS_Evaluation")
        print("   4. Click individual runs to see traces and metrics")
        print()
        print("📋 Results CSV includes:")
        print("   - Question and expected answer")
        print("   - Generated answer from RAG system")
        print("   - PASSED/FAILED status")
        print("   - Key information match percentage")
        print("   - Average relevancy score")
        print("   - Context support indication")
        print("   - Retrieval, generation, and total time")
        print("   - mlflow_trace_id and mlflow_trace_url for deep-dive analysis")
        print()
        
    except Exception as e:
        print()
        print("="*70)
        print(f"❌ Evaluation Failed: {str(e)}")
        print("="*70)
        print()
        print("Troubleshooting:")
        print("  1. Ensure MLFlow server is running: mlflow ui")
        print("  2. Verify dataset exists: app/evaluation/datasets/hf_doc_qa_eval.csv")
        print("  3. Check logs for detailed error messages")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
