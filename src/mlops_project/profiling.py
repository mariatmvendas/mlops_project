"""
This script provides tools for profiling Python scripts using cProfile, analyzing the results with pstats,
and visualizing the profiling data with SnakeViz. It automates the process of running cProfile on a target 
script, saving profiling outputs in both .txt and .prof formats, and optionally analyzing and visualizing 
the data.

Key Features:
- Profiles a Python script using cProfile.
- Saves the profiling data as a .txt file (human-readable) and a .prof file (for advanced visualization).
- Offers optional analysis of profiling data using the pstats module.
- Supports visualization of profiling data with SnakeViz for better insights.

Usage:
Run the script from the command line, providing the path to the target Python script and any additional
arguments to pass to the script being profiled. Optional flags allow for analysis with pstats and
visualization with SnakeViz.

e.g. python src/mlops_project/profiling.py src/mlops_project/train.py train --analyze --visualize
"""
#FIXME profiling results should be saved on appropriate directory

import os
import subprocess
import argparse

def run_cprofile(script_path, script_args, output_txt, output_prof):
    """
    Run cProfile on the given script and generate profiling outputs.

    Args:
        script_path (str): Path to the Python script to profile.
        script_args (list): Arguments to pass to the script (e.g., "train", "evaluate").
        output_txt (str): Path to save the cProfile output as a .txt file.
        output_prof (str): Path to save the cProfile output as a .prof file.
    """
    full_command = ["python", "-m", "cProfile", "-o", output_txt, script_path] + script_args
    print(f"Running cProfile for {script_path} with arguments {script_args}...")
    subprocess.run(full_command)
    print(f"Generated profile.txt: {output_txt}")

    full_command = ["python", "-m", "cProfile", "-o", output_prof, script_path] + script_args
    subprocess.run(full_command)
    print(f"Generated profile_results.prof: {output_prof}")

def analyze_with_pstats(profile_txt):
    """
    Analyze the profiling data with pstats.

    Args:
        profile_txt (str): Path to the profiling .txt file.
    """
    import pstats
    print(f"Analyzing {profile_txt} with pstats...")
    p = pstats.Stats(profile_txt)
    p.sort_stats("cumulative").print_stats(10)

def visualize_with_snakeviz(profile_prof):
    """
    Visualize the profiling data with SnakeViz.

    Args:
        profile_prof (str): Path to the profiling .prof file.
    """
    print(f"Visualizing {profile_prof} with SnakeViz...")
    os.system(f"snakeviz {profile_prof}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a Python script using cProfile.")
    parser.add_argument("script", help="Path to the script to profile.")
    parser.add_argument(
        "script_args",
        nargs="*",
        help="Additional arguments to pass to the script being profiled (e.g., 'train', 'evaluate').",
    )
    parser.add_argument(
        "--txt", default="profile.txt", help="Path to save the .txt profiling output."
    )
    parser.add_argument(
        "--prof", default="profile_results.prof", help="Path to save the .prof profiling output."
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze the profiling output with pstats."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the profiling output with SnakeViz."
    )
    args = parser.parse_args()

    # Run cProfile
    run_cprofile(args.script, args.script_args, args.txt, args.prof)

    # Optional analysis with pstats
    if args.analyze:
        analyze_with_pstats(args.txt)

    # Optional visualization with SnakeViz
    if args.visualize:
        visualize_with_snakeviz(args.prof)
