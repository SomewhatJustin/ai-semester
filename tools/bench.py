#!/usr/bin/env python3
"""
LLM Benchmarking Tool
Compares OpenAI and Ollama models on evaluation prompts
"""

import os
import json
import yaml
import time
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bench_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMBenchmark:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'gemma3:latest')
        self.results = []

    def load_prompts(self, prompts_path: str) -> List[Dict[str, Any]]:
        """Load evaluation prompts from YAML file"""
        with open(prompts_path, 'r') as f:
            prompts = yaml.safe_load(f)
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")
        return prompts

    def query_openai(self, prompt: str) -> Dict[str, Any]:
        """Query OpenAI model and return response with metadata"""
        start_time = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1000
            )

            end_time = time.time()
            return {
                "response": response.choices[0].message.content,
                "model": self.openai_model,
                "provider": "openai",
                "response_time": round(end_time - start_time, 2),
                "tokens_used": response.usage.total_tokens if response.usage else None,
                "success": True,
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            logger.error(f"OpenAI query failed: {e}")
            return {
                "response": None,
                "model": self.openai_model,
                "provider": "openai",
                "response_time": round(end_time - start_time, 2),
                "tokens_used": None,
                "success": False,
                "error": str(e)
            }

    def query_ollama(self, prompt: str) -> Dict[str, Any]:
        """Query Ollama model and return response with metadata"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()

            end_time = time.time()
            data = response.json()

            # Get the response content and handle encoding properly
            raw_response = data.get("message", {}).get("content", "")

            # Handle both Unicode escape sequences and UTF-8 encoding issues
            try:
                # First try to decode Unicode escape sequences
                if '\\u' in raw_response:
                    decoded_response = raw_response.encode().decode('unicode_escape')
                else:
                    decoded_response = raw_response

                # Then ensure proper UTF-8 encoding
                if isinstance(decoded_response, str):
                    # Fix common UTF-8 mojibake issues
                    decoded_response = decoded_response.encode('latin1').decode('utf-8', errors='replace')
            except (UnicodeDecodeError, UnicodeEncodeError):
                # If all else fails, use the original response
                decoded_response = raw_response

            return {
                "response": decoded_response,
                "model": self.ollama_model,
                "provider": "ollama",
                "response_time": round(end_time - start_time, 2),
                "tokens_used": None,  # Ollama doesn't always provide token counts
                "success": True,
                "error": None
            }
        except Exception as e:
            end_time = time.time()
            logger.error(f"Ollama query failed: {e}")
            return {
                "response": None,
                "model": self.ollama_model,
                "provider": "ollama",
                "response_time": round(end_time - start_time, 2),
                "tokens_used": None,
                "success": False,
                "error": str(e)
            }

    def run_benchmark(self, prompts_path: str = "data/prompts.yaml"):
        """Run benchmark on all prompts with both models"""
        prompts = self.load_prompts(prompts_path)

        logger.info(f"Starting benchmark with {len(prompts)} prompts")
        logger.info(f"OpenAI Model: {self.openai_model}")
        logger.info(f"Ollama Model: {self.ollama_model}")

        for i, prompt_data in enumerate(prompts, 1):
            prompt_id = prompt_data['id']
            task = prompt_data['task']
            prompt = prompt_data['prompt']

            logger.info(f"[{i}/{len(prompts)}] Running prompt {prompt_id}: {task}")

            # Test with OpenAI
            logger.info(f"  Querying OpenAI ({self.openai_model})...")
            openai_result = self.query_openai(prompt)

            # Test with Ollama
            logger.info(f"  Querying Ollama ({self.ollama_model})...")
            ollama_result = self.query_ollama(prompt)

            # Store results
            result = {
                "prompt_id": prompt_id,
                "task": task,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "openai": openai_result,
                "ollama": ollama_result
            }

            self.results.append(result)

            # Log responses for immediate review
            if openai_result["success"]:
                logger.info(f"  OpenAI response ({openai_result['response_time']}s): {openai_result['response'][:100]}...")
            else:
                logger.error(f"  OpenAI failed: {openai_result['error']}")

            if ollama_result["success"]:
                logger.info(f"  Ollama response ({ollama_result['response_time']}s): {ollama_result['response'][:100]}...")
            else:
                logger.error(f"  Ollama failed: {ollama_result['error']}")

            logger.info(f"  Completed prompt {prompt_id}")

        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save detailed results to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/results/benchmark_results_{timestamp}.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['provider', 'model', 'prompt_id', 'latency_ms', 'output_chars', 'error_flag', 'prompt', 'response'])

            # Write data rows
            for result in self.results:
                prompt_id = result['prompt_id']
                prompt_text = result['prompt']

                # OpenAI row
                openai_result = result['openai']
                writer.writerow([
                    'openai',
                    openai_result['model'],
                    prompt_id,
                    int(openai_result['response_time'] * 1000),  # Convert to ms
                    len(openai_result['response'] or '') if openai_result['response'] else 0,
                    not openai_result['success'],  # error_flag is True when success is False
                    prompt_text,
                    openai_result['response'] or ''
                ])

                # Ollama row
                ollama_result = result['ollama']
                writer.writerow([
                    'ollama',
                    ollama_result['model'],
                    prompt_id,
                    int(ollama_result['response_time'] * 1000),  # Convert to ms
                    len(ollama_result['response'] or '') if ollama_result['response'] else 0,
                    not ollama_result['success'],  # error_flag is True when success is False
                    prompt_text,
                    ollama_result['response'] or ''
                ])

        logger.info(f"Results saved to {filename}")

    def print_summary(self):
        """Print benchmark summary"""
        total_prompts = len(self.results)
        openai_successes = sum(1 for r in self.results if r["openai"]["success"])
        ollama_successes = sum(1 for r in self.results if r["ollama"]["success"])

        avg_openai_time = sum(r["openai"]["response_time"] for r in self.results) / total_prompts
        avg_ollama_time = sum(r["ollama"]["response_time"] for r in self.results) / total_prompts

        logger.info("\n" + "="*60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*60)
        logger.info(f"Total prompts: {total_prompts}")
        logger.info(f"OpenAI ({self.openai_model}): {openai_successes}/{total_prompts} successful, avg {avg_openai_time:.2f}s")
        logger.info(f"Ollama ({self.ollama_model}): {ollama_successes}/{total_prompts} successful, avg {avg_ollama_time:.2f}s")
        logger.info("="*60)

def main():
    benchmark = LLMBenchmark()
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
