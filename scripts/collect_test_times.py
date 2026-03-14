#!/usr/bin/env python3
"""
Run EngineTests and collect a sorted table of execution times per test.

Usage:
    python3 scripts/collect_test_times.py [path_to_EngineTests]

If no path is given, defaults to ./build/EngineTests.
Produces a markdown table sorted by execution time (longest first).
"""

import json
import os
import subprocess
import sys
import tempfile


def run_tests(engine_tests_path):
    """Run EngineTests with JSON output and return parsed results."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_output_path = f.name

    try:
        cmd = [
            engine_tests_path,
            f"--gtest_output=json:{json_output_path}",
        ]
        print(f"Running: {' '.join(cmd)}")
        print("This may take several minutes...\n")

        result = subprocess.run(cmd, capture_output=False, timeout=900)

        with open(json_output_path, "r") as f:
            return json.load(f), result.returncode
    finally:
        if os.path.exists(json_output_path):
            os.unlink(json_output_path)


def collect_test_times(data):
    """Extract test names and execution times from GTest JSON output."""
    tests = []
    for suite in data.get("testsuites", []):
        suite_name = suite["name"]
        for test in suite.get("testsuite", []):
            test_name = test["name"]
            full_name = f"{suite_name}.{test_name}"
            time_str = test.get("time", "0")
            time_sec = float(time_str.rstrip("s")) if time_str.endswith("s") else float(time_str)
            status = test.get("result", "UNKNOWN")
            tests.append((full_name, time_sec, status))
    return tests


def print_sorted_table(tests):
    """Print a markdown table of tests sorted by execution time (descending)."""
    tests_sorted = sorted(tests, key=lambda x: x[1], reverse=True)

    if not tests_sorted:
        print("\nNo test results found.")
        return

    max_name_len = max(len(t[0]) for t in tests_sorted)
    max_name_len = max(max_name_len, len("Test Name"))

    print(f"\n{'='*80}")
    print(f"  EngineTests Execution Times (sorted by duration, longest first)")
    print(f"  Total tests: {len(tests_sorted)}")
    total_time = sum(t[1] for t in tests_sorted)
    print(f"  Total time: {total_time:.3f}s")
    print(f"{'='*80}\n")

    print(f"| {'#':>4} | {'Test Name':<{max_name_len}} | {'Time (s)':>10} | {'Status':>10} |")
    print(f"|{'-'*6}|{'-'*(max_name_len+2)}|{'-'*12}|{'-'*12}|")

    for i, (name, time_sec, status) in enumerate(tests_sorted, 1):
        print(f"| {i:>4} | {name:<{max_name_len}} | {time_sec:>10.3f} | {status:>10} |")


def main():
    if len(sys.argv) > 1:
        engine_tests_path = sys.argv[1]
    else:
        engine_tests_path = os.path.join("build", "EngineTests")

    if not os.path.isfile(engine_tests_path):
        print(f"Error: EngineTests not found at '{engine_tests_path}'")
        print("Usage: python3 scripts/collect_test_times.py [path_to_EngineTests]")
        sys.exit(1)

    data, returncode = run_tests(engine_tests_path)
    tests = collect_test_times(data)
    print_sorted_table(tests)

    if returncode != 0:
        print(f"\nNote: EngineTests exited with code {returncode} (some tests may have failed)")


if __name__ == "__main__":
    main()
