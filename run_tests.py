"""
Lightweight Test Runner for Double Lenses.
Discovers and runs all test functions in tests/test_*.py.
"""

import importlib
import inspect
import os
import sys
import time
import traceback


def run_all_tests():
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    test_files = sorted([f[:-3] for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")])

    print("=" * 70)
    print("  RUNNING DOUBLE LENSES TEST SUITE")
    print("=" * 70)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    start_time = time.time()

    for mod_name in test_files:
        print(f"\n[*] Suite: tests.{mod_name}")
        try:
            mod = importlib.import_module(f"tests.{mod_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to import {mod_name}: {e}")
            traceback.print_exc()
            failed_tests += 1
            continue

        test_funcs = [
            (name, obj) for name, obj in inspect.getmembers(mod, inspect.isfunction) if name.startswith("test_")
        ]

        for func_name, func in test_funcs:
            total_tests += 1
            t0 = time.perf_counter()
            try:
                func()
                elapsed = (time.perf_counter() - t0) * 1000.0
                print(f"  [PASS] {func_name} ({elapsed:.1f} ms)")
                passed_tests += 1
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000.0
                print(f"  [FAIL] {func_name} ({elapsed:.1f} ms): {e}")
                traceback.print_exc()
                failed_tests += 1

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  TEST SUMMARY: {passed_tests}/{total_tests} Passed, {failed_tests} Failed (Time: {total_time:.2f}s)")
    print("=" * 70)

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
