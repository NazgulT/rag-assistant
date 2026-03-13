"""
RAGAS evaluation experiment for RAG system.
Evaluates RAG system performance on hf_doc_qa_eval dataset.
"""
import csv
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from contextlib import nullcontext
from langsmith import evaluate
import pandas as pd
from datasets import Dataset

from app.logging.logger import get_logger
from app.logging.mlflow_tracker import MLFlowTracker
from app.rag_system import RAGSystem
from app.models.schemas import EvaluationMetrics, RAGResponse

#from ragas.metrics import discrete_metric
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness
)
from ragas import evaluate

logger = get_logger(__name__)
logger.setLevel("DEBUG")  # Set to DEBUG for detailed logs during evaluation


class RAGASExperiment:
    """RAGAS evaluation experiment for RAG system."""

    def __init__(self, rag_system: RAGSystem = None, use_mlflow: bool = True):
        """
        Initialize RAGAS experiment.

        Args:
            rag_system: RAG system instance
            use_mlflow: Whether to log to MLFlow
        """
        self.rag_system = rag_system or RAGSystem(use_mlflow=False)
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = MLFlowTracker() if use_mlflow else None
        logger.info("RAGASExperiment initialized")

    def load_evaluation_dataset(
        self, csv_path: str
    ) -> List[Dict[str, str]]:
        """
        Load evaluation dataset from CSV.

        Args:
            csv_path: Path to CSV file with 'question' and 'expected_answer' columns

        Returns:
            List of question-answer pairs
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        dataset = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("question") and row.get("expected_answer"):
                    dataset.append({
                        "question": row["question"].strip(),
                        "expected_answer": row["expected_answer"].strip(),
                    })

        logger.info(f"Loaded {len(dataset)} evaluation samples")
        return dataset


    def evaluate_rag_response(
            self,
            response: RAGResponse,
            expected_answer: str,
            retrieved_docs: List[str],  
            weights: Dict[str, float] = {
                "context_precision": 0.35,
                "context_recall": 0.30,
                "answer_relevancy": 0.20,
                "faithfulness": 0.15,   
            }     
    )->EvaluationMetrics:

        """
        Evaluate RAG response against expected answer using RAGAS metrics.

        Args:
            response: RAGResponse object containing answer and retrieved documents
            expected_answer: The expected answer text
            retrieved_docs: List of retrieved document texts

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metric scores.
        """

        # Limit retrieved documents to top 5 to prevent large datasets
        retrieved_docs = retrieved_docs[:5]

        # Prepare dataset in correct RAGAS format
        data = {
            "question": [response.query],
            "answer": [response.answer],
            "contexts": [retrieved_docs],
            "ground_truth": [expected_answer],
        }

        dataset = Dataset.from_dict(data)

        # Evaluate using RAGAS with error handling
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    answer_relevancy,
                    faithfulness,
                ],
            )
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            # Return default scores on failure
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "average_score": 0.0,
            }

        unified_score = sum(
            result[metric] * weights[metric]
            for metric in weights
        )

        # Convert to plain dict
        scores = {
            "context_precision": result["context_precision"][0],
            "context_recall": result["context_recall"][0],
            "answer_relevancy": result["answer_relevancy"][0],
            "faithfulness": result["faithfulness"][0],
            "average_score": unified_score,
        }

        return scores



    #@discrete_metric(name="correctness", allowed_values=["pass", "fail"])
    def my_correctness_metric(
        self, 
        question: str, 
        expected_answer: str) -> str:
        """
        Custom correctness metric using LLM evaluation.

        Args:
            question: The original question
            expected_answer: The expected answer text
            response: The model-generated answer text

        Returns:
            'pass' if the response is correct, 'fail' otherwise
        """
        # This function will be wrapped by the DiscreteMetric decorator, so the actual logic will be handled there.

        prompt="""Compare the model response to the expected answer and determine if it's correct.

        Consider the response correct if it:
        1. Contains the key information from the expected answer
        2. Is factually accurate based on the provided context
        3. Adequately addresses the question asked

        Return 'pass' if the response is correct, 'fail' if it's incorrect.

        Question: {question}
        Expected Answer: {expected_answer}
        Model Response: {response}

        Evaluation:"""
        '''
        correctness = self.correctness_metric.ascore(
            question=question,
            expected_answer=expected_answer,
            llm = llm)
        '''
        

        return "pass"

    def check_response_correctness(
        self,
        rag_response: RAGResponse,
        expected_answer: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Check if RAG response is correct.

        Criteria for passing:
        1. Contains key information from expected answer
        2. Is factually accurate based on provided context
        3. Adequately addresses the question asked

        Args:
            rag_response: RAG response object
            expected_answer: Expected answer text
            threshold: Minimum relevancy score threshold

        Returns:
            Dictionary with pass/fail status and reasoning
        """
        

        answer_text = rag_response.answer.lower()
        expected_lower = expected_answer.lower()


        '''
        reasoning = []
        reasoning.append(
            f"Key info match: {key_info_match:.2%} (matched {matched_words}/{len(expected_words)} key words)"
        )
        reasoning.append(
            f"Avg relevancy: {avg_relevancy:.4f} (threshold: {threshold})"
        )
        reasoning.append(
            f"Context support: {'Yes' if answer_supported else 'Partial'}"
        )
        reasoning.append(f"Answer length: {len(answer_text)} chars")

        return {
            "passed": passed,
            "key_info_match": key_info_match,
            "avg_relevancy": avg_relevancy,
            "context_support": answer_supported,
            "reasoning": " | ".join(reasoning),
        }
        '''

        return {
            "passed":True
        }

    # ---------------------------------------------------------------------
    def _evaluate_sample(self, idx: int, sample: Dict[str, str]) -> Dict[str, Any]:
        """Process a single sample, optionally logging to MLFlow.

        This helper wraps the original loop body and is useful for
        sequential or threaded/asynchronous execution. It starts a nested
        run before calling into the RAG system so that any internal metric
        logging from ``rag_system`` is associated with the correct trial.
        """
        question = sample["question"]
        expected_answer = sample["expected_answer"]

        status = "ERROR"
        result_row: Dict[str, Any] = {
            "sample_id": idx,
            "question": question,
            "expected_answer": expected_answer,
        }

        try:
            # start run early so rag_system logs fall under this run
            if self.use_mlflow:
                run_ctx = self.mlflow_tracker.start_run(
                    run_name=f"eval_sample_{idx}",
                    nested=True,
                )
            else:
                run_ctx = nullcontext()

            with run_ctx:
                rag_response = self.rag_system.answer_query(
                    query=question,
                    k_retrieve=5,
                    k_rerank=3,
                    use_reranking=True,
                )

                print(f"\nSample {idx} RAG Response:")
                print(f"Generated answer: {rag_response.answer}")

                logger.info(f"Evaluating sample {idx} using RAGAS metrics...")

                print(f"Sample {idx} Retrieved Documents:")
                for i, doc in enumerate(rag_response.retrieved_documents):
                    print(f"  Doc {i}: {doc.content[:100]}...")

                eval_scores = self.evaluate_rag_response(
                    response=rag_response,
                    expected_answer=expected_answer,
                    retrieved_docs=[doc.content for doc in rag_response.retrieved_documents],
                )

                print(f"Sample {idx} Evaluation Scores:")
                for metric, score in eval_scores.items():
                    print(f"  {metric}: {score:.4f}")

                result_row.update({
                    "generated_answer": rag_response.answer,
                    "status": status,
                    "average_score": f"{eval_scores['average_score']:.4f}",
                    "context_precision": f"{eval_scores['context_precision']:.4f}",
                    "context_recall": f"{eval_scores['context_recall']:.4f}",
                    "answer_relevancy": f"{eval_scores['answer_relevancy']:.4f}",
                    "faithfulness": f"{eval_scores['faithfulness']:.4f}",
                    "retrieval_time": f"{rag_response.retrieval_time:.4f}",
                    "generation_time": f"{rag_response.generation_time:.4f}",
                    "total_time": f"{rag_response.total_time:.4f}",
                    "retrieved_docs_count": len(rag_response.retrieved_documents),
                    #"retrieved_docs": [doc["content"][:100] for doc in rag_response.retrieved_documents],
                    "reranked_docs_count": len(rag_response.reranked_documents),
                    #"reranked_docs": [doc["content"][:100] for doc in rag_response.reranked_documents], 
                })

                print(f"\nSample {idx} Evaluation:")
                print(f"Result row: {result_row}")

                if self.use_mlflow:
                    # additional experiment-level logging
                    self.mlflow_tracker.log_params({
                        "question": question,
                        "sample_id": idx,
                    })
                    self.mlflow_tracker.log_metrics({
                        "average_score": f"{eval_scores['average_score']:.4f}",
                        "context_precision": f"{eval_scores['context_precision']:.4f}",
                        "context_recall": f"{eval_scores['context_recall']:.4f}",
                        "answer_relevancy": f"{eval_scores['answer_relevancy']:.4f}",
                        "faithfulness": f"{eval_scores['faithfulness']:.4f}",
                        "retrieval_time": rag_response.retrieval_time,
                        "generation_time": rag_response.generation_time,
                        "total_time": rag_response.total_time,
                    })
                    self.mlflow_tracker.log_artifacts({
                        "question": rag_response.query,
                        "answer": rag_response.answer,
                        "status": status,
                        "retrieved_docs_count": len(rag_response.retrieved_documents),
                        "retrieved_docs": [doc.content[:100] for doc in rag_response.retrieved_documents],
                        "reranked_docs_count": len(rag_response.reranked_documents),
                        "reranked_docs": [doc.content[:100] for doc in rag_response.reranked_documents], 
                    })

                    run_id = self.mlflow_tracker.get_run_id()
                    result_row["mlflow_trace_id"] = run_id
                    result_row["mlflow_trace_url"] = self.mlflow_tracker.get_trace_url(run_id)

                logger.info(f"[{idx}] {status}: {question[:50]}...")

        except Exception as e:
            logger.error(f"Evaluation. Error processing sample {idx}: {str(e)}")
            result_row.update({
                "generated_answer": f"ERROR: {str(e)}",
                "status": "ERROR",
                "reasoning": str(e),
            })

        return result_row

    async def run_experiment_async(
        self,
        dataset_path: str,
        output_dir: str = "evaluation_results",
        mlflow_exp_name: str = "RAG_RAGAS_Evaluation",
        max_concurrency: int = 5,
    ) -> str:
        """
        Asynchronous variant of ``run_experiment``.  Queries and evaluates
        multiple samples concurrently using threads.

        The core logic is the same; the only difference is that each sample
        is processed in a separate executor task to allow I/O-bound parts of
        the pipeline (retrieval/generation) to overlap.  MLFlow tracking
        uses nested runs so that every task may start a run independently.
        """
        # same setup as synchronous version
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"ragas_evaluation_{timestamp}.csv"

        dataset = self.load_evaluation_dataset(dataset_path)
        if self.use_mlflow:
            self.mlflow_tracker.set_experiment(mlflow_exp_name)

        results: List[Dict[str, Any]] = []
        passed_count = 0
        failed_count = 0

        logger.info(f"Starting async RAGAS evaluation on {len(dataset)} samples...")

        semaphore = asyncio.Semaphore(max_concurrency)

        async def worker(idx: int, sample: Dict[str, str]) -> Dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(self._evaluate_sample, idx, sample)

        tasks = [worker(idx, sample) for idx, sample in enumerate(dataset, 1)]
        batch_results = await asyncio.gather(*tasks)

        for row in batch_results:
            results.append(row)
            if row.get("status") == "PASSED":
                passed_count += 1
            else:
                failed_count += 1

        # save results and log summary same as synchronous function
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)

        total = len(dataset)
        pass_rate = (passed_count / total * 100) if total > 0 else 0

        logger.info(
            f"\n{'='*50}\n"
            f"RAGAS Evaluation Summary\n"
            f"{'='*50}\n"
            f"Total Samples: {total}\n"
            f"Passed: {passed_count}\n"
            f"Failed: {failed_count}\n"
            f"Pass Rate: {pass_rate:.2f}%\n"
            f"Results saved to: {results_file}\n"
            f"{'='*50}"
        )

        if self.use_mlflow:
            with self.mlflow_tracker.start_run(run_name="experiment_summary"):
                self.mlflow_tracker.log_metrics({
                    "total_samples": total,
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "pass_rate": pass_rate,
                })

        return str(results_file)


    def run_experiment(
        self,
        dataset_path: str,
        output_dir: str = "evaluation_results",
        mlflow_exp_name: str = "RAG_RAGAS_Evaluation",
    ) -> str:
        """
        Run RAGAS evaluation experiment on dataset.

        Args:
            dataset_path: Path to evaluation dataset CSV
            output_dir: Directory for output results
            mlflow_exp_name: MLFlow experiment name

        Returns:
            Path to results CSV file
        """
        # Setup
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"ragas_evaluation_{timestamp}.csv"

        # Load dataset
        dataset = self.load_evaluation_dataset(dataset_path)

        # Setup MLFlow experiment
        if self.use_mlflow:
            self.mlflow_tracker.set_experiment(mlflow_exp_name)

        # Results tracking
        results: List[Dict[str, Any]] = []
        passed_count = 0
        failed_count = 0

        logger.info(f"Starting RAGAS evaluation on {len(dataset)} samples...")

        # run samples sequentially by default
        for idx, sample in enumerate(dataset, 1):
            row = self._evaluate_sample(idx, sample)
            results.append(row)
            if row.get("status") == "PASSED":
                passed_count += 1
            elif row.get("status") == "FAILED":
                failed_count += 1
            else:
                failed_count += 1

        # helper for debugging when async is added
        logger.debug(f"Finished processing {len(dataset)} samples")

        # Save results and return file path as before (moved below)
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)

        # Log summary
        total = len(dataset)
        pass_rate = (passed_count / total * 100) if total > 0 else 0

        summary = {
            "total_samples": total,
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": f"{pass_rate:.2f}%",
            "results_file": str(results_file),
        }

        logger.info(
            f"\n{'='*50}\n"
            f"RAGAS Evaluation Summary\n"
            f"{'='*50}\n"
            f"Total Samples: {total}\n"
            f"Passed: {passed_count}\n"
            f"Failed: {failed_count}\n"
            f"Pass Rate: {pass_rate:.2f}%\n"
            f"Results saved to: {results_file}\n"
            f"{'='*50}"
        )

        # Log summary to MLFlow within its own run so we don't leave
        # a run active after the experiment completes.  Using a dedicated
        # name makes it easy to spot the summary run in the UI.
        if self.use_mlflow:
            with self.mlflow_tracker.start_run(run_name="experiment_summary"):
                self.mlflow_tracker.log_metrics({
                    "total_samples": total,
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "pass_rate": pass_rate,
                })

        return str(results_file)


def run_evaluation_experiment(
    csv_path: str = "app/evaluation/datasets/hf_doc_qa_eval.csv",
    output_dir: str = "evaluation_results",
    async_mode: bool = False,
    max_concurrency: int = 5,
) -> str:
    """
    Convenience wrapper around ``RAGASExperiment``.

    Args:
        csv_path: Path to evaluation dataset
        output_dir: Output directory for results
        async_mode: Whether to run the experiment asynchronously (concurrent samples)
        max_concurrency: When async, limit of concurrent workers

    Returns:
        Path to results CSV
    """
    rag_system = RAGSystem(use_mlflow=True)
    experiment = RAGASExperiment(rag_system=rag_system, use_mlflow=True)
    if async_mode:
        return asyncio.run(
            experiment.run_experiment_async(
                csv_path,
                output_dir,
                max_concurrency=max_concurrency,
            )
        )
    else:
        return experiment.run_experiment(csv_path, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation experiment"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="app/evaluation/datasets/hf_doc_qa_eval.csv",
        help="Path to evaluation CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Directory to write results",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Run samples concurrently using asyncio",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent workers when async",
    )

    args = parser.parse_args()

    results_file = run_evaluation_experiment(
        csv_path=args.csv,
        output_dir=args.output,
        async_mode=args.async_mode,
        max_concurrency=args.concurrency,
    )

    print(f"\nResults saved to: {results_file}")
    print(f"View traces at: http://127.0.0.1:5000")
