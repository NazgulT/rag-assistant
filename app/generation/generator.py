"""
LLM generation module for RAG system.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from app.logging.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class LLMGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text based on prompt."""
        pass


class HuggingFaceGenerator(LLMGenerator):
    """LLM generator using HuggingFace models."""

    def __init__(
        self,
        model_name: str = settings.LLM_MODEL,
        device: str = settings.LLM_DEVICE,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        top_p: float = settings.LLM_TOP_P,
    ):
        """
        Initialize HuggingFace LLM generator.

        Args:
            model_name: Name of the model
            device: Device to use (cpu, cuda, mps)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
        """
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        try:
            # Map device names
            device_map = {
                "mps": 0,  # Will use CPU on Mac with MPS
                "cuda": 0,
                "cpu": -1,
            }
            actual_device = device_map.get(device, -1)

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            if device == "cpu" or device == "mps":
                self.model = self.model.to(device)

            logger.info(
                f"Loaded LLM model: {model_name} on device: {device}"
            )
        except Exception as e:
            logger.error(f"Error loading LLM model {model_name}: {str(e)}")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text based on prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        try:

            # Apply chat template
            input_text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize input
            #inputs = self.tokenizer(input_text, return_tensors="pt")

            # Handle device placement
            #device = next(self.model.parameters()).device
            #inputs = {k: v.to(device) for k, v in inputs.items()}

            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens
            ).to(self.model.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Decode
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            generated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Remove prompt from output
            #if generated_text.startswith(prompt):
               # generated_text = generated_text[len(prompt):].strip()

            logger.debug(f"Generated text of length {len(generated_text)}")
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise


class RAGGenerator:
    """Generator for RAG system that creates context-aware prompts."""

    def __init__(self, llm_generator: LLMGenerator = None):
        """
        Initialize RAG generator.

        Args:
            llm_generator: LLM generator instance
        """
        self.llm_generator = llm_generator or HuggingFaceGenerator()
        logger.info("RAGGenerator initialized")

    def _create_prompt(self, query: str, context: List[str]) -> str:
        """
        Create a prompt with context for the LLM.

        Args:
            query: User query
            context: List of context passages

        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join([f"Context {i+1}:\n{passage}" for i, passage in enumerate(context)])

        # Chat-style prompt (important for instruct models)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Answer strictly using the provided context. Do not add any new knowledge."
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{query}"
            },
        ]

        return messages

    def generate(
        self,
        query: str,
        context: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer based on query and context.

        Args:
            query: User query
            context: List of context passages
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            Generation result with metadata
        """
        start_time = time.time()

        try:
            # Create prompt
            prompt = self._create_prompt(query, context)

            # Generate answer
            answer = self.llm_generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            generation_time = time.time() - start_time
            tokens_generated = len(answer.split())  # Rough estimate

            logger.info(
                f"Generated answer in {generation_time:.2f}s with {tokens_generated} tokens"
            )

            return {
                "answer": answer,
                "context_used": context,
                "generation_time": generation_time,
                "model_name": self.llm_generator.model_name,
                "tokens_generated": tokens_generated,
            }
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            raise
