#!/usr/bin/env python3
"""
Lab 2: Decoding Parameter Sweep
Tests how different decoding parameters (temperature, top_k, top_p) influence LLM output
"""

import os
import json
import yaml
import time
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import re

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def tokens(s):
    """Tokenize string into words and punctuation"""
    return re.findall(r"\w+|\S", s.lower())

def distinct_n(texts, n=1):
    """Calculate distinct n-gram ratio"""
    total, uniq = 0, set()
    for t in texts:
        toks = tokens(t)
        grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
        total += len(grams)
        uniq.update(grams)
    return (len(uniq)/total) if total else 0.0

def repetition_rate(texts, n=2):
    """Calculate repetition rate for n-grams"""
    reps, total = 0, 0
    for t in texts:
        toks = tokens(t)
        grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
        c = Counter(grams)
        reps += sum(v-1 for v in c.values() if v > 1)
        total += len(grams)
    return (reps/total) if total else 0.0

def self_bleu(texts):
    """Calculate self-BLEU score"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        refs = [tokens(t) for t in texts]
        cc = SmoothingFunction().method1
        scores = []
        for i, hyp in enumerate(refs):
            refset = refs[:i] + refs[i+1:]
            if refset:  # Only calculate if there are references
                scores.append(sentence_bleu(refset, hyp, smoothing_function=cc))
        return sum(scores)/len(scores) if scores else 0.0
    except ImportError:
        logger.warning("NLTK not available, self-BLEU will be 0")
        return 0.0

class DecodingSweep:
    def __init__(self):
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model = 'gemma3:12b'
        self.results = []

        # Parameter configurations
        self.temperature_values = [0, 0.4, 0.8, 1.5, 5]
        self.top_k_values = [1, 20, 40, 80, 500]
        self.top_p_values = [0.1, 0.7, 0.9, 0.95, 0.995]

    def load_prompts(self, prompts_path: str = "data/prompts.yaml") -> List[Dict[str, Any]]:
        """Load evaluation prompts from YAML file"""
        with open(prompts_path, 'r') as f:
            prompts = yaml.safe_load(f)
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_path}")
        return prompts

    def query_ollama(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query Ollama model with specific options"""
        start_time = time.time()
        time_to_first_token = None

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True  # Enable streaming to measure time to first token
            }

            if options:
                payload["options"] = options

            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=120,
                stream=True
            )
            response.raise_for_status()

            # Process streaming response
            full_response = ""
            first_token_received = False

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            if content and not first_token_received:
                                time_to_first_token = time.time() - start_time
                                first_token_received = True
                            full_response += content

                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue

            end_time = time.time()

            return {
                "response": full_response,
                "response_time": round(end_time - start_time, 2),
                "time_to_first_token": round(time_to_first_token, 3) if time_to_first_token else 0,
                "success": True,
                "error": None
            }

        except Exception as e:
            end_time = time.time()
            logger.error(f"Ollama query failed: {e}")
            return {
                "response": "",
                "response_time": round(end_time - start_time, 2),
                "time_to_first_token": 0,
                "success": False,
                "error": str(e)
            }

    def run_temperature_sweep(self, prompts: List[Dict[str, Any]]):
        """Run decoding sweep varying temperature"""
        logger.info("Running temperature sweep...")

        for temp in self.temperature_values:
            logger.info(f"Testing temperature: {temp}")
            options = {"temperature": temp}

            for prompt_data in prompts:
                result = self._run_single_test(prompt_data, "temperature", temp, options)
                self.results.append(result)

    def run_top_k_sweep(self, prompts: List[Dict[str, Any]]):
        """Run decoding sweep varying top_k"""
        logger.info("Running top_k sweep...")

        for top_k in self.top_k_values:
            logger.info(f"Testing top_k: {top_k}")
            options = {"top_k": top_k}

            for prompt_data in prompts:
                result = self._run_single_test(prompt_data, "top_k", top_k, options)
                self.results.append(result)

    def run_top_p_sweep(self, prompts: List[Dict[str, Any]]):
        """Run decoding sweep varying top_p"""
        logger.info("Running top_p sweep...")

        for top_p in self.top_p_values:
            logger.info(f"Testing top_p: {top_p}")
            options = {"top_p": top_p}

            for prompt_data in prompts:
                result = self._run_single_test(prompt_data, "top_p", top_p, options)
                self.results.append(result)

    def _run_single_test(self, prompt_data: Dict[str, Any], param_name: str, param_value: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test with given parameters"""
        prompt_id = prompt_data['id']
        task = prompt_data['task']
        prompt = prompt_data['prompt']

        logger.info(f"  Running {prompt_id} with {param_name}={param_value}")

        # Generate multiple responses for self-BLEU calculation
        responses = []
        total_time = 0
        first_token_times = []

        for i in range(3):  # Generate 3 responses for self-BLEU
            response_data = self.query_ollama(prompt, options)
            if response_data['success']:
                responses.append(response_data['response'])
                total_time += response_data['response_time']
                first_token_times.append(response_data['time_to_first_token'])
            else:
                # If any response fails, fall back to single response approach
                logger.warning(f"Response {i+1} failed, falling back to single response")
                break

        # If we didn't get multiple responses, try once more for at least one response
        if not responses:
            response_data = self.query_ollama(prompt, options)
            if response_data['success']:
                responses = [response_data['response']]
                total_time = response_data['response_time']
                first_token_times = [response_data['time_to_first_token']]

        if responses:
            # Use first response as primary, but calculate metrics across all
            primary_response = responses[0]

            # Calculate metrics
            distinct_1 = distinct_n(responses, n=1)
            distinct_2 = distinct_n(responses, n=2)
            repetition_2 = repetition_rate(responses, n=2)
            repetition_3 = repetition_rate(responses, n=3)
            self_bleu_score = self_bleu(responses) if len(responses) > 1 else 0.0

            result = {
                "prompt_id": prompt_id,
                "task": task,
                "parameter": param_name,
                "parameter_value": param_value,
                "generated_text": primary_response,
                "distinct_1": round(distinct_1, 4),
                "distinct_2": round(distinct_2, 4),
                "repetition_2": round(repetition_2, 4),
                "repetition_3": round(repetition_3, 4),
                "self_bleu": round(self_bleu_score, 4),
                "length": len(primary_response),
                "time_to_first_token": round(sum(first_token_times) / len(first_token_times), 3),
                "total_time": round(total_time / len(responses), 2),  # Average time per response
                "success": True,
                "error": None
            }
        else:
            result = {
                "prompt_id": prompt_id,
                "task": task,
                "parameter": param_name,
                "parameter_value": param_value,
                "generated_text": "",
                "distinct_1": 0,
                "distinct_2": 0,
                "repetition_2": 0,
                "repetition_3": 0,
                "self_bleu": 0,
                "length": 0,
                "time_to_first_token": 0,
                "total_time": response_data['response_time'],
                "success": False,
                "error": response_data['error']
            }

        return result

    def save_results(self):
        """Save results to CSV file"""
        output_path = "labs/lab_2_results.csv"

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'prompt_id', 'task', 'parameter', 'parameter_value',
                'generated_text', 'distinct_1', 'distinct_2',
                'repetition_2', 'repetition_3', 'self_bleu',
                'length', 'time_to_first_token', 'total_time',
                'success', 'error'
            ])

            # Write data rows
            for result in self.results:
                writer.writerow([
                    result['prompt_id'],
                    result['task'],
                    result['parameter'],
                    result['parameter_value'],
                    result['generated_text'],
                    result['distinct_1'],
                    result['distinct_2'],
                    result['repetition_2'],
                    result['repetition_3'],
                    result['self_bleu'],
                    result['length'],
                    result['time_to_first_token'],
                    result['total_time'],
                    result['success'],
                    result['error'] or ''
                ])

        logger.info(f"Results saved to {output_path}")

    def print_summary(self):
        """Print summary of results"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])

        logger.info("\n" + "="*60)
        logger.info("DECODING SWEEP SUMMARY")
        logger.info("="*60)
        logger.info(f"Model: {self.model}")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful tests: {successful_tests}/{total_tests}")
        logger.info(f"Temperature values tested: {self.temperature_values}")
        logger.info(f"Top-k values tested: {self.top_k_values}")
        logger.info(f"Top-p values tested: {self.top_p_values}")
        logger.info("="*60)

    def run_full_sweep(self):
        """Run complete decoding parameter sweep"""
        logger.info(f"Starting decoding sweep with model: {self.model}")

        # Load prompts
        prompts = self.load_prompts()

        # Run each parameter sweep
        self.run_temperature_sweep(prompts)
        self.run_top_k_sweep(prompts)
        self.run_top_p_sweep(prompts)

        # Save results and print summary
        self.save_results()
        self.print_summary()

def main():
    # Create labs directory if it doesn't exist
    Path("labs").mkdir(exist_ok=True)

    sweep = DecodingSweep()
    sweep.run_full_sweep()

if __name__ == "__main__":
    main()
