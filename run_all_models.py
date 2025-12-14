#!/usr/bin/env python3
"""
Wrapper script to run all available Ollama models through both Enneagram test versions.

Automatically discovers models via `ollama list` and runs:
1. enneagram_runner_v3-2_3run.py (standard version)
2. enneagram_runner_v3-2_3run_NoContext.py (context-clearing version)

For each model with --runs-per-test 3.
"""

import subprocess
import sys
import re
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


class BatchLogger:
    """Handles both text and JSON logging for batch test runs."""

    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create log file paths
        self.text_log_path = self.log_dir / f"batch_run_{self.timestamp}.log"
        self.json_log_path = self.log_dir / f"batch_run_{self.timestamp}.json"

        # Initialize JSON log structure
        self.json_data = {
            "session": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration_seconds": None,
                "command_line_args": sys.argv[1:]
            },
            "models": {
                "discovered": [],
                "excluded": [],
                "tested": []
            },
            "tests": [],
            "summary": {
                "total_tests": 0,
                "completed": 0,
                "failed": 0
            }
        }

        # Initialize text log
        self._write_text(f"{'='*80}\n")
        self._write_text(f"Enneagram LLM Test Suite - Batch Run Log\n")
        self._write_text(f"{'='*80}\n")
        self._write_text(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._write_text(f"Text log: {self.text_log_path}\n")
        self._write_text(f"JSON log: {self.json_log_path}\n")
        self._write_text(f"{'='*80}\n\n")

        self.start_time = time.time()

    def _write_text(self, message):
        """Append message to text log file."""
        with open(self.text_log_path, 'a', encoding='utf-8') as f:
            f.write(message)

    def log(self, message, to_console=True):
        """Log message to both text file and optionally console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        self._write_text(log_line)
        if to_console:
            print(message)

    def log_models_discovered(self, models):
        """Log discovered models."""
        self.json_data["models"]["discovered"] = models
        self.log(f"Discovered {len(models)} model(s): {', '.join(models)}")

    def log_models_excluded(self, excluded):
        """Log excluded models."""
        self.json_data["models"]["excluded"] = excluded
        if excluded:
            self.log(f"Excluded {len(excluded)} model(s): {', '.join(excluded)}")

    def log_models_to_test(self, models):
        """Log models that will be tested."""
        self.json_data["models"]["tested"] = models
        self.log(f"Testing {len(models)} model(s): {', '.join(models)}")

    def log_test_start(self, model, script, test_num, total_tests):
        """Log the start of a test."""
        self.log(f"\n{'='*80}")
        self.log(f"Test {test_num}/{total_tests}: {script} with {model}")
        self.log(f"{'='*80}")
        return time.time()  # Return start time for duration calculation

    def log_test_end(self, model, script, success, start_time, error_msg=None, stdout=None, stderr=None):
        """Log the end of a test."""
        duration = time.time() - start_time

        test_entry = {
            "model": model,
            "script": script,
            "success": success,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat()
        }

        if not success:
            test_entry["error"] = error_msg or "Unknown error"
            if stdout:
                test_entry["stdout"] = stdout
            if stderr:
                test_entry["stderr"] = stderr

        self.json_data["tests"].append(test_entry)

        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.log(f"{status} - Duration: {duration:.2f}s")

        if not success and error_msg:
            self._write_text(f"    Error: {error_msg}\n")
            if stderr:
                self._write_text(f"    stderr: {stderr}\n")

    def finalize(self, total_tests, completed, failed):
        """Write final summary and close logs."""
        end_time = time.time()
        duration = end_time - self.start_time

        # Update JSON summary
        self.json_data["session"]["end_time"] = datetime.now().isoformat()
        self.json_data["session"]["duration_seconds"] = round(duration, 2)
        self.json_data["summary"]["total_tests"] = total_tests
        self.json_data["summary"]["completed"] = completed
        self.json_data["summary"]["failed"] = failed

        # Write final text summary
        self.log(f"\n{'='*80}")
        self.log(f"FINAL SUMMARY")
        self.log(f"{'='*80}")
        self.log(f"Total tests: {total_tests}")
        self.log(f"Completed: {completed}")
        self.log(f"Failed: {failed}")
        self.log(f"Duration: {duration/60:.2f} minutes ({duration:.0f} seconds)")
        self.log(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"{'='*80}")

        # Write JSON log
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.json_data, f, indent=2)

        self.log(f"\nLogs saved:")
        self.log(f"  Text: {self.text_log_path}")
        self.log(f"  JSON: {self.json_log_path}")


def get_available_models():
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        # Parse the output - skip header line, extract model names
        lines = result.stdout.strip().split('\n')
        models = []

        for line in lines[1:]:  # Skip header
            # Extract model name (first column)
            parts = line.split()
            if parts:
                model_name = parts[0]
                models.append(model_name)

        return models

    except subprocess.CalledProcessError as e:
        print(f"Error running 'ollama list': {e}", file=sys.stderr)
        print(f"Make sure Ollama is installed and running.", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("Error: 'ollama' command not found.", file=sys.stderr)
        print("Make sure Ollama is installed and in your PATH.", file=sys.stderr)
        return []


def run_test_script(script_name, model_name, runs_per_test=3):
    """Run a single test script for a given model.

    Returns tuple: (success: bool, error_msg: str, stdout: str, stderr: str)
    """
    cmd = [
        "python3",
        script_name,
        "--model", model_name,
        "--runs-per-test", str(runs_per_test)
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=7200  # 2 hour timeout per test
        )
        return True, None, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code: {e.returncode}"
        print(f"❌ Error: {error_msg}", file=sys.stderr)
        return False, error_msg, e.stdout, e.stderr
    except subprocess.TimeoutExpired as e:
        error_msg = "Timeout after 2 hours"
        print(f"❌ {error_msg}", file=sys.stderr)
        return False, error_msg, e.stdout if e.stdout else "", e.stderr if e.stderr else ""


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run Enneagram tests on all available Ollama models"
    )
    parser.add_argument(
        "--exclude",
        "--omit",
        action="append",
        dest="exclude",
        metavar="MODEL",
        help="Exclude specific model(s) from testing (can be specified multiple times)"
    )
    args = parser.parse_args()

    # Initialize logger
    logger = BatchLogger()

    logger.log("="*80)
    logger.log("Enneagram LLM Test Suite - Batch Runner")
    logger.log("="*80)

    # Get available models
    logger.log("Discovering available Ollama models...")
    all_models = get_available_models()

    if not all_models:
        logger.log("No models found. Exiting.")
        sys.exit(1)

    logger.log_models_discovered(all_models)

    # Filter out excluded models
    excluded_models = args.exclude or []
    models = [m for m in all_models if m not in excluded_models]

    logger.log_models_excluded(excluded_models)

    if not models:
        logger.log("No models remaining after exclusions. Exiting.")
        sys.exit(1)

    logger.log_models_to_test(models)
    logger.log("")

    # Test scripts to run
    test_scripts = [
        "enneagram_runner_v3-2_3run.py",
        "enneagram_runner_v3-2_3run_NoContext.py"
    ]

    # Track results
    total_tests = len(models) * len(test_scripts)
    completed = 0
    failed = 0
    results = []

    # Run tests for each model
    for model_idx, model in enumerate(models, 1):
        logger.log(f"\n{'#'*80}")
        logger.log(f"# MODEL {model_idx}/{len(models)}: {model}")
        logger.log(f"{'#'*80}")

        for script_idx, script in enumerate(test_scripts, 1):
            test_num = (model_idx - 1) * len(test_scripts) + script_idx

            # Log test start
            start_time = logger.log_test_start(model, script, test_num, total_tests)

            # Run the test
            success, error_msg, stdout, stderr = run_test_script(script, model, runs_per_test=3)

            # Log test end
            logger.log_test_end(model, script, success, start_time, error_msg, stdout, stderr)

            results.append({
                "model": model,
                "script": script,
                "success": success
            })

            if success:
                completed += 1
            else:
                failed += 1

    # Finalize logs with summary
    logger.finalize(total_tests, completed, failed)

    # Detailed results to console
    if failed > 0:
        print("\nFailed tests:")
        for r in results:
            if not r["success"]:
                print(f"  ❌ {r['model']} - {r['script']}")

    print("\nSuccessful tests:")
    for r in results:
        if r["success"]:
            print(f"  ✅ {r['model']} - {r['script']}")

    print("\nAll results have been saved to the results/ directory.")
    print("="*80)

    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(130)
