"""
Evaluation module for RAGAS-based assessment.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.logging.logger import get_logger
from app.config.settings import settings


import urllib.request
from pathlib import Path
#from ragas import Dataset
import pandas as pd

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluation."""

    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    answer_relevancy: Optional[float] = None
    faithfulness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "answer_relevancy": self.answer_relevancy,
            "faithfulness": self.faithfulness,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    def get_average_score(self) -> Optional[float]:
        """Get average of all computed metrics."""
        scores = [
            self.answer_relevancy,
            self.faithfulness,
            self.context_precision,
            self.context_recall,
        ]
        scores = [s for s in scores if s is not None]
        if scores:
            return sum(scores) / len(scores)
        return None


class RAGEvaluator:
    """Evaluator for RAG system using RAGAS metrics."""

    def __init__(self):
        """Initialize RAGAS evaluator."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            )

            self.evaluate = evaluate
            self.metrics = {
                "answer_relevancy": answer_relevancy,
                "faithfulness": faithfulness,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            logger.info("RAGEvaluator initialized with RAGAS metrics")
        except ImportError:
            logger.warning(
                "RAGAS not installed. Install with: pip install ragas"
            )
            self.evaluate = None
            self.metrics = {}

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.

        Args:
            question: Question asked
            answer: Generated answer
            contexts: Retrieved context passages
            ground_truth: Optional ground truth answer
            metrics: List of metrics to compute (default: all)

        Returns:
            EvaluationResult with computed metrics
        """
        if self.evaluate is None:
            logger.warning("RAGAS evaluator not available, returning empty evaluation")
            return EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

        try:
            metrics = metrics or list(self.metrics.keys())
            result = EvaluationResult(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )

            # Compute each metric
            for metric_name in metrics:
                if metric_name not in self.metrics:
                    logger.warning(f"Unknown metric: {metric_name}")
                    continue

                try:
                    # Create dataset entry
                    from datasets import Dataset

                    dataset = Dataset.from_dict(
                        {
                            "question": [question],
                            "answer": [answer],
                            "contexts": [contexts],
                        }
                    )

                    # Evaluate
                    metric_obj = self.metrics[metric_name]
                    evaluation_result = self.evaluate(
                        dataset,
                        metrics=[metric_obj],
                    )

                    # Extract score
                    score = evaluation_result[metric_name][0]
                    setattr(result, metric_name, float(score))

                    logger.debug(f"Computed {metric_name}: {score:.4f}")
                except Exception as e:
                    logger.warning(f"Error computing {metric_name}: {str(e)}")

            return result
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple RAG responses.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context passages for each question
            ground_truths: Optional list of ground truth answers
            metrics: List of metrics to compute

        Returns:
            List of EvaluationResults
        """
        if self.evaluate is None:
            logger.warning("RAGAS evaluator not available")
            return [
                EvaluationResult(
                    question=q,
                    answer=a,
                    contexts=c,
                    ground_truth=gt,
                )
                for q, a, c, gt in zip(
                    questions,
                    answers,
                    contexts,
                    ground_truths or [None] * len(questions),
                )
            ]

        try:
            results = []
            for q, a, c, gt in zip(
                questions,
                answers,
                contexts,
                ground_truths or [None] * len(questions),
            ):
                result = self.evaluate_response(
                    q, a, c, ground_truth=gt, metrics=metrics
                )
                results.append(result)

            logger.info(f"Evaluated {len(results)} responses")
            return results
        except Exception as e:
            logger.error(f"Error in batch evaluation: {str(e)}")
            raise

    def compute_aggregate_metrics(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics across multiple results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of aggregate metrics
        """
        aggregate = {}

        for metric_name in self.metrics.keys():
            scores = [
                getattr(r, metric_name)
                for r in results
                if getattr(r, metric_name) is not None
            ]
            if scores:
                aggregate[f"{metric_name}_mean"] = sum(scores) / len(scores)
                aggregate[f"{metric_name}_min"] = min(scores)
                aggregate[f"{metric_name}_max"] = max(scores)

        # Compute average score
        avg_scores = [r.get_average_score() for r in results]
        avg_scores = [s for s in avg_scores if s is not None]
        if avg_scores:
            aggregate["average_score_mean"] = sum(avg_scores) / len(avg_scores)

        logger.info(f"Computed aggregate metrics: {len(aggregate)} values")
        return aggregate

'''
    def create_ragas_dataset(dataset_path: Path) -> Dataset:
        dataset = Dataset(name="hf_doc_qa_eval", backend="local/csv", root_dir=".")
        df = pd.read_csv(dataset_path)

        for _, row in df.iterrows():
            dataset.append({"question": row["question"], "expected_answer": row["expected_answer"]})

        dataset.save()
        return dataset
'''
